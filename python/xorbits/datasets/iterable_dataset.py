# Copyright 2022-2023 XProbe Inc.
# derived from copyright 1999-2021 Alibaba Group Holding Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import dataclasses
import json
import os.path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from queue import Queue
from typing import Dict, List, Optional, Tuple, Union

import fsspec
import numpy as np
import pyarrow as pa

from .._mars.utils import lazy_import

torch = lazy_import("torch")
_TorchIterableDataset = object if torch is None else torch.utils.data.IterableDataset


@dataclasses.dataclass(init=False)
class _GroupInfo:
    path: str
    index: pa.Table
    schema: pa.Schema
    info: dict


class IterableDataset(_TorchIterableDataset):
    def __init__(
        self,
        path: Union[str, os.PathLike],
        storage_options: Optional[dict] = None,
        prefetch: int = 2,
        fetch_retry: int = 2,
        fetch_timeout: float = 60,
        shuffle: bool = False,
        shuffle_seed: int = 0,
        distributed_rank: Optional[int] = None,
        distributed_world_size: Optional[int] = None,
        num_threads: Optional[int] = None,
    ):
        self._info, self._group_infos = self._get_infos(path, storage_options)
        self._path = path
        self._storage_options = storage_options
        self._prefetch = prefetch
        self._fetch_retry = fetch_retry
        self._fetch_timeout = fetch_timeout
        self._shuffle = shuffle
        self._shuffle_seed = shuffle_seed
        self._distributed_rank = distributed_rank
        self._distributed_world_size = distributed_world_size
        self._num_threads = num_threads
        self._epoch = 0

    def __iter__(self):
        rank, world_size = self._get_rank(), self._get_world_size()
        if rank is None or world_size is None:
            rank = 0
            world_size = 1
        else:
            assert rank < world_size

        rng = self._get_rng_generator()
        total_chunks = len(self._group_infos[0].index)
        index_of_index = np.arange(total_chunks)
        if self._shuffle:
            rng.shuffle(index_of_index)
        index_of_index = index_of_index[rank:total_chunks:world_size]

        fs_token_paths = fsspec.get_fs_token_paths(
            self._path, storage_options=self._storage_options
        )
        fs: fsspec.AbstractFileSystem = fs_token_paths[0]
        finish_sentinel = object()
        prefetch_queue = Queue(maxsize=self._prefetch)
        format_queue = Queue()

        def _load_arrow_table(filepath):
            # TODO(codingl2k1): mmap if local.
            with fs.open(filepath, "rb") as f:
                with pa.ipc.RecordBatchStreamReader(f) as reader:
                    return reader.read_all()

        def _format_row(tables, index):
            row = {}
            for t in tables:
                d = t.take([index]).to_pydict()
                row.update(d)
            return row

        def _prefetcher(_executor: ThreadPoolExecutor):
            for idx in index_of_index:
                group_files = [
                    os.path.join(group.path, group.index[0][idx].as_py())
                    for group in self._group_infos
                ]
                tables = _executor.map(_load_arrow_table, group_files)
                prefetch_queue.put(tables)
            prefetch_queue.put(finish_sentinel)

        def _formatter(_executor: ThreadPoolExecutor):
            while tables := prefetch_queue.get():
                if tables is finish_sentinel:
                    break
                tables = list(tables)
                in_table_index = np.arange(len(tables[0]))
                rng.shuffle(in_table_index)
                for idx in in_table_index:
                    format_queue.put(_executor.submit(_format_row, tables, idx))
            format_queue.put(finish_sentinel)

        with ThreadPoolExecutor(
            max_workers=self._num_threads,
            thread_name_prefix=self.__iter__.__qualname__,
        ) as executor:
            prefetch_thread = threading.Thread(target=_prefetcher, args=(executor,))
            prefetch_thread.start()
            format_thread = threading.Thread(target=_formatter, args=(executor,))
            format_thread.start()

            while fut := format_queue.get():
                if fut is finish_sentinel:
                    break
                yield fut.result()

            prefetch_thread.join()
            format_thread.join()

    def __len__(self):
        return self._info["num_rows"]

    @staticmethod
    def _get_infos(path, storage_options) -> Tuple[Dict, List[_GroupInfo]]:
        # TODO(codingl2k1): Merge group meta files into one.
        group_infos = []
        futures = []
        with ThreadPoolExecutor(
            thread_name_prefix=IterableDataset._get_infos.__qualname__
        ) as executor, ExitStack() as es:
            fs_token_paths = fsspec.get_fs_token_paths(
                path, storage_options=storage_options
            )
            fs: fsspec.AbstractFileSystem = fs_token_paths[0]
            path = fs_token_paths[2][0]
            es.callback(fs.clear_instance_cache)

            with fs.open(os.path.join(path, "info.json"), "r") as f:
                info = json.load(f)

            for name in info["groups"]:
                group_path = os.path.join(path, name)
                group_meta_path = os.path.join(group_path, ".meta")
                group_info = _GroupInfo()
                group_info.path = group_path
                group_infos.append(group_info)

                def _fill_index(_path, _group_info):
                    with fs.open(_path, "rb") as f:
                        with pa.ipc.RecordBatchStreamReader(f) as reader:
                            _group_info.index = reader.read_all()

                fut = executor.submit(
                    _fill_index,
                    os.path.join(group_meta_path, "index.arrow"),
                    group_info,
                )
                futures.append(fut)

                def _fill_schema(_path, _group_info):
                    with fs.open(_path, "rb") as f:
                        with pa.ipc.RecordBatchStreamReader(f) as reader:
                            _group_info.schema = reader.schema

                fut = executor.submit(
                    _fill_schema,
                    os.path.join(group_meta_path, "schema.arrow"),
                    group_info,
                )
                futures.append(fut)

                def _fill_info(_path, _group_info):
                    with fs.open(_path, "r") as f:
                        _group_info.info = json.load(f)

                fut = executor.submit(
                    _fill_info, os.path.join(group_meta_path, "info.json"), group_info
                )
                futures.append(fut)
            # Raise exception if exists.
            for fut in as_completed(futures):
                fut.result()
        return info, group_infos

    def _get_rank(self) -> int:
        """Returns the rank of the current process, which is on ``[0; WORLD_SIZE - 1]``.

        Returns:
            int: The rank.
        """
        rank = self._distributed_rank
        if rank is None:
            return int(os.environ.get("RANK", 0))
        return rank

    def _get_world_size(self) -> int:
        """Returns the world size, which is the number of processes participating in this training run.

        Returns:
            int: The world size.
        """
        world_size = self._distributed_world_size
        if world_size is None:
            return int(os.environ.get("WORLD_SIZE", 1))
        return world_size

    def _get_rng_generator(self):
        if self._shuffle and self._epoch == 0:
            return np.random.default_rng(self._shuffle_seed)
        elif self._shuffle:
            # Create new seed using self._epoch (we subtract in order to avoid overflow in long_scalars)
            epoch_seed = (
                np.random.default_rng(self._shuffle_seed).integers(0, 1 << 63)
                - self._epoch
            )
            epoch_seed = (1 << 63) + epoch_seed if epoch_seed < 0 else epoch_seed
            return np.random.default_rng(epoch_seed)
        else:
            raise ValueError("This dataset is not shuffled")

    def column_groups(self):
        return self._info["groups"]

    def schema(self):
        return pa.unify_schemas([gi.schema for gi in self._group_infos])

    def set_epoch(self, epoch: int):
        self._epoch = epoch

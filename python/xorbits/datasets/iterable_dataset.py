# Copyright 2022-2023 XProbe Inc.
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
import functools
import json
import os.path
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from queue import Queue
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from .._mars.utils import lazy_import

torch = lazy_import("torch")
if TYPE_CHECKING or torch is None:
    _TorchIterableDataset = object
else:
    _TorchIterableDataset = torch.utils.data.IterableDataset


@dataclasses.dataclass(init=False)
class GroupInfo:
    name: str
    path: str
    index: pa.Table
    schema: pa.Schema
    info: dict


@dataclasses.dataclass
class _FormatColumnInfo:
    as_py_columns: List[int] = dataclasses.field(default_factory=list)
    column_name_to_decoder: Dict[str, Callable] = dataclasses.field(
        default_factory=dict
    )


@dataclasses.dataclass
class _FinishSentinel:
    exception: Optional[Exception] = None


class Formatter:
    def format_batch(self, tables: List[pa.Table], indices: List[int]):
        result = {}
        for t in tables:
            d = t.take(indices).to_pydict()
            result.update(d)
        return result


class FeatureFormatter(Formatter):
    def __init__(self, features: Dict[str, Any], table_schemas: List[pa.Schema]):
        table_format_infos = []
        for ts in table_schemas:
            info = _FormatColumnInfo()
            for idx, name in enumerate(ts.names):
                if feat := features.get(name):
                    info.column_name_to_decoder[name] = feat.decode_example
                else:
                    info.as_py_columns.append(idx)
            table_format_infos.append(info)
        self._table_format_infos = table_format_infos

    def format_batch(self, tables: List[pa.Table], indices: List[int]):
        result = {}
        for info, table in zip(self._table_format_infos, tables):
            d = (
                table.select(info.as_py_columns).take(indices).to_pydict()
                if info.as_py_columns
                else {}
            )
            for name, decoder in info.column_name_to_decoder.items():
                chunked_array = table[name]
                d[name] = [
                    decoder(
                        {"bytes": chunked_array[idx]["bytes"].as_buffer(), "path": None}
                    )
                    for idx in indices
                ]
            result.update(d)
        return result


def _result_or_cancel(fut, timeout=None):
    try:
        try:
            return fut.result(timeout)
        finally:
            fut.cancel()
    finally:
        # Break a reference cycle with the exception in self._exception
        del fut


def map_retry(executor, fn, *iterables, timeout=None, retry=0):
    """Returns an iterator equivalent to map(fn, iter).

    Args:
        executor: A ThreadPoolExecutor instance.
        fn: A callable that will take as many arguments as there are
            passed iterables.
        timeout: The maximum number of seconds to wait. If None, then there
            is no limit on the wait time.
        retry: The retry times.

    Returns:
        An iterator equivalent to: map(func, *iterables) but the calls may
        be evaluated out-of-order.

    Raises:
        TimeoutError: If the entire result iterator could not be generated
            before the given timeout.
        Exception: If fn(*args) raises for any values.
    """
    if timeout is not None:
        end_time = timeout + time.monotonic()

    fs = []
    arg_list = []
    for args in zip(*iterables):
        arg_list.append(args)
        fs.append(executor.submit(fn, *args))

    # Yield must be hidden in closure so that the futures are submitted
    # before the first iterator value is required.
    def result_iterator():
        try:
            # reverse to keep finishing order
            fs.reverse()
            while fs:
                curr_args = arg_list.pop()
                # Careful not to keep a reference to the popped future
                for retry_time in range(retry + 1):
                    try:
                        if timeout is None:
                            yield _result_or_cancel(fs.pop())
                        else:
                            yield _result_or_cancel(
                                fs.pop(), end_time - time.monotonic()
                            )
                        break
                    except Exception as e:
                        if retry_time == retry:
                            raise e
                        fs.append(executor.submit(fn, *curr_args))
        finally:
            for future in fs:
                future.cancel()

    return result_iterator()


class IterableDataset(_TorchIterableDataset):
    _FORMAT_BATCH_SIZE = 10
    _DEFAULT_VERSION = "0.0.0"
    _META_DIR = ".meta"
    _FILE_NAME_FORMATTER = "{}_{}.arrow"

    def __init__(
        self,
        path: Union[str, os.PathLike],
        storage_options: Optional[dict] = None,
        prefetch: int = 2,
        fetch_retry: int = 2,
        fetch_timeout: float = 60,
        shuffle: bool = False,
        shuffle_seed: int = 9176,
        distributed_rank: Optional[int] = None,
        distributed_world_size: Optional[int] = None,
        worker_of_rank: Optional[int] = None,
        workers_per_rank: Optional[int] = None,
        num_threads: Optional[int] = None,
        version: Optional[str] = None,
    ):
        """
        An IterableDataset from the export path.

        Parameters
        ----------
        path: str
            The export path of Dataset include version.
        storage_options: dict
            Key/value pairs to be passed on to the caching file-system backend, if any.
        prefetch: int
            Prefetch chunks, default is 2.
        fetch_retry: int
            Fetch data retry times.
        fetch_timeout: float
            Fetch data timeout seconds.
        shuffle: bool
            Whether shuffle or not, default is False.
        shuffle_seed: int
            Shuffle seed, default is a fixed value.
        distributed_rank: int
            Distributed rank, if not set, use `RANK` from environment var.
        distributed_world_size: int
            Distributed world size, if not set, use `WORLD_SIZE` from environment var.
        worker_of_rank: int
            The worker id of rank, if not set, try to get from `torch.utils.data.get_worker_info`.
        workers_per_rank: int
            The num workers per rank, if not set, try to get from `torch.utils.data.get_worker_info`.
        num_threads: int
            The max worker threads for __iter__.
        version: str
            The version string, default is 0.0.0.
        """
        path = os.path.join(path, version or self._DEFAULT_VERSION)
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
        self._worker_of_rank = worker_of_rank
        self._workers_per_rank = workers_per_rank
        self._num_threads = num_threads
        self._epoch = 0

    def __iter__(self):
        import fsspec

        worker_index = self._get_worker_index()
        formatter = self._get_formatter()

        # Create a new fs instance in __iter__ make sure the fs instance
        # is bound to worker.
        fs_token_paths = fsspec.get_fs_token_paths(
            self._path, storage_options=self._storage_options
        )
        fs: fsspec.AbstractFileSystem = fs_token_paths[0]
        # The flag to stop the _formatter thread.
        finish = False
        # A sentinel to mark the upstream is finish.
        finish_sentinel = _FinishSentinel()
        # The queue _formatter -> __iter__
        format_queue = Queue(
            maxsize=self._info["max_chunk_rows"] // self._FORMAT_BATCH_SIZE + 1
        )

        def _load_arrow_table(filepath):
            # TODO(codingl2k1): mmap if local.
            with fs.open(filepath, "rb") as f:
                with pa.ipc.RecordBatchStreamReader(f) as reader:
                    return reader.read_all()

        def _prefetcher(_executor: ThreadPoolExecutor):
            for idx in worker_index:
                group_files = [
                    os.path.join(
                        group.path,
                        self._FILE_NAME_FORMATTER.format(*group.index[0][idx].as_py()),
                    )
                    for group in self._group_infos
                ]
                yield map_retry(
                    _executor,
                    _load_arrow_table,
                    group_files,
                    timeout=self._fetch_timeout,
                    retry=self._fetch_retry,
                )

        def _formatter(_output_queue: Queue):
            # TODO(codingl2k1): Use as_completed to optimize shuffle True.
            rng = (
                np.random.default_rng(self._shuffle_seed + self._epoch)
                if self._shuffle
                else None
            )
            with ThreadPoolExecutor(
                max_workers=self._num_threads,
                thread_name_prefix=self.__iter__.__qualname__,
            ) as executor:
                try:
                    prefetch = _prefetcher(executor)
                    buffer = [next(prefetch, None) for _ in range(self._prefetch)]
                    idx = 0
                    buffer_len = len(buffer)
                    while not finish:
                        tables = buffer[idx]
                        if tables is None:
                            break
                        buffer[idx] = next(prefetch, None)
                        idx += 1
                        idx %= buffer_len

                        tables = list(tables)
                        num_chunk_rows = len(tables[0])
                        in_chunk_index = np.arange(num_chunk_rows)
                        if rng is not None:
                            rng.shuffle(in_chunk_index)
                        for indices in np.array_split(
                            in_chunk_index,
                            num_chunk_rows // self._FORMAT_BATCH_SIZE + 1,
                        ):
                            _output_queue.put(
                                executor.submit(formatter.format_batch, tables, indices)
                            )
                except Exception as ex:
                    finish_sentinel.exception = ex
                finally:
                    _output_queue.put(finish_sentinel)

        format_thread = threading.Thread(target=_formatter, args=(format_queue,))
        format_thread.start()
        try:
            while fut := format_queue.get():
                if fut is finish_sentinel:
                    break
                for example in self._batch_to_examples(fut.result()):
                    yield example
        except Exception as e:
            finish = True
            finish_sentinel.exception = e
        finally:
            format_thread.join()
        # Reraise the exception.
        if finish_sentinel.exception is not None:
            raise finish_sentinel.exception

    def __len__(self):
        """Get data length according to rank, world size, worker id and num workers."""
        if self._get_world_size() > 1:
            worker_index = self._get_worker_index()
            group = self._group_infos[0]
            num_rows_series = group.index[f"{group.name}.num_rows"]
            return pc.sum(num_rows_series.take(worker_index)).as_py()
        else:
            return self._info["num_rows"]

    @staticmethod
    def _batch_to_examples(
        batch: Dict[str, list]
    ) -> Generator[Dict[str, Any], None, None]:
        """Convert a batch (dict of examples) to examples list"""
        n_examples = len(batch[next(iter(batch))])
        for i in range(n_examples):
            yield {col: array[i] for col, array in batch.items()}

    @classmethod
    def _get_infos(cls, path, storage_options) -> Tuple[Dict, List[GroupInfo]]:
        import fsspec

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
                group_meta_path = os.path.join(group_path, cls._META_DIR)
                group_info = GroupInfo()
                group_info.name = name
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

    def _get_worker_info(self):
        """Get worker id and num workers."""
        worker_id = self._worker_of_rank
        num_workers = self._workers_per_rank
        if worker_id is None or num_workers is None:
            worker_info = torch and torch.utils.data.get_worker_info()
            if worker_info is None:
                worker_id = 0
                num_workers = 1
            else:
                worker_id = worker_info.id
                num_workers = worker_info.num_workers
        return worker_id, num_workers

    def _get_worker_index(self):
        rank, world_size = self._get_rank(), self._get_world_size()
        assert rank < world_size

        # Generate an index for the Dataset group index.
        num_group_chunks = len(self._group_infos[0].index)
        rank_index = np.arange(num_group_chunks)
        if self._shuffle:
            rng = np.random.default_rng(self._shuffle_seed + self._epoch)
            rng.shuffle(rank_index)
        # Get each worker index by rank and world size.
        rank_index = rank_index[rank:num_group_chunks:world_size]
        worker_id, num_workers = self._get_worker_info()
        return (
            rank_index[worker_id : len(rank_index) : num_workers]
            if num_workers > 1
            else rank_index
        )

    def _get_formatter(self) -> Formatter:
        """
        Get formatters from metadata.

        Returns
        -------
            The Formatter instance.
        """
        from datasets.features.audio import Audio
        from datasets.features.features import generate_from_dict
        from datasets.features.image import Image

        # TODO(codingl2k1): Impl our decoders.
        metadata = self.schema.metadata.get(b"huggingface")
        if metadata:
            metadata = json.loads(metadata.decode())
            metadata = generate_from_dict(metadata)
            features = metadata["info"]["features"]
            requires_decoding = {
                k: v for k, v in features.items() if isinstance(v, (Audio, Image))
            }
            if requires_decoding:
                return FeatureFormatter(
                    requires_decoding,
                    [group.schema for group in self._group_infos],
                )
        # Fast path without decoding.
        return Formatter()

    @functools.cached_property
    def groups(self) -> List[str]:
        return self._info["groups"]

    @functools.cached_property
    def schema(self) -> pa.Schema:
        return pa.unify_schemas([gi.schema for gi in self._group_infos])

    @functools.cached_property
    def shape(self) -> Tuple:
        return len(self), len(self.schema.names)

    def info(self) -> Dict:
        return self._info

    def group_infos(self) -> List[GroupInfo]:
        """Groups has order, so returns a list."""
        return self._group_infos

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def epoch(self) -> int:
        return self._epoch

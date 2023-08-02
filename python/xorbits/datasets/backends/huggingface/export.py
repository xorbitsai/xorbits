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
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from ...._mars.core.entity import OutputType
from ...._mars.serialization.serializables import DictField, Int32Field, StringField
from ...operand import DataOperand, DataOperandMixin


class HuggingfaceExport(DataOperand, DataOperandMixin):
    path: str = StringField("path")
    storage_options: Dict = DictField("storage_options")
    version: str = StringField("version")
    max_chunk_rows: int = Int32Field("max_chunk_rows")

    def __call__(self, dataset):
        ArrowWriter.ensure_path(self.path, self.storage_options)
        return self.new_tileable([dataset], **dataset.params)

    @classmethod
    def tile(cls, op: "HuggingfaceExport"):
        assert len(op.inputs) == 1
        inp = op.inputs[0]
        out = op.outputs[0]
        out_chunks = []
        for chunk in inp.chunks:
            chunk_op = op.copy().reset_key()
            out_chunk = chunk_op.new_chunk([chunk], index=chunk.index)
            out_chunks.append(out_chunk)
        return op.copy().new_tileable(
            op.inputs,
            chunks=out_chunks,
            nsplits=((np.nan,) * len(out_chunks), (np.nan,)),
            **out.params,
        )

    @classmethod
    def execute(cls, ctx, op: "HuggingfaceExport"):
        inp = ctx[op.inputs[0].key]
        out = op.outputs[0]
        r = ArrowWriter.ensure_path(op.path, op.storage_options).write(
            dataset=inp, max_chunk_rows=op.max_chunk_rows, chunk_index=out.index[0]
        )
        ctx[out.key] = r


class ArrowWriter:
    _WRITER_CLASS = pa.RecordBatchStreamWriter
    _DEFAULT_VERSION = "0.0.0"
    _DEFAULT_MAX_CHUNK_ROWS = 100
    _META_DIR = ".meta"
    _MULTIMEDIA_PREFIX = "mdata"
    _DATA_PREFIX = "data"
    _FILE_NAME_FORMATTER = "{prefix}_{chunk_index}_{index}.arrow"

    def __init__(
        self,
        fs: fsspec.AbstractFileSystem,
        path: str,
    ):
        self._fs = fs
        self._path = path

    @classmethod
    def ensure_path(
        cls,
        path: str,
        storage_options: Optional[dict] = None,
        create_if_not_exists: bool = True,
    ):
        fs_token_paths = fsspec.get_fs_token_paths(
            path, storage_options=storage_options
        )
        fs: fsspec.AbstractFileSystem = fs_token_paths[0]
        path = fs_token_paths[2][0]
        if not fs.exists(path):
            if create_if_not_exists:
                fs.mkdirs(path, exist_ok=True)
            else:
                raise Exception(f"The path {path} does not exist.")
        elif not fs.isdir(path):
            raise Exception(f"The path {path} should be a dir.")
        return cls(fs, path)

    def _embed_and_write_table(self, file: str, pa_table: pa.Table, features, info):
        """Write a Table to file.

        Args:
            example: the Table to add.
        """
        from datasets.features.features import require_storage_embed
        from datasets.table import embed_array_storage

        with self._fs.open(file, "wb") as stream:
            # TODO(codingl2k1): embed uid to table to check the different parts
            # of data are matched.
            schema = pa_table.schema.remove_metadata()
            with self._WRITER_CLASS(stream, schema) as writer:
                arrays = [
                    embed_array_storage(pa_table[name], feature)
                    if require_storage_embed(feature)
                    else pa_table[name]
                    for name, feature in features.items()
                ]
                pa_table = pa.Table.from_arrays(arrays, schema=schema)
                # Update embeded num_bytes
                info["num_bytes"] = pa_table.nbytes
                writer.write(pa_table)

    def _write_meta_by_table_column(
        self, meta_path: str, pa_table: pa.Table, column: str
    ):
        self._fs.mkdirs(meta_path, exist_ok=True)
        pa_table = pa_table.select([column]).flatten()
        index_file = os.path.join(meta_path, "index.arrow")
        with self._fs.open(index_file, "wb") as stream:
            with self._WRITER_CLASS(stream, pa_table.schema) as writer:
                writer.write(pa_table)
        info = {
            "num_bytes": pc.sum(pa_table[f"{column}.num_bytes"]).as_py(),
            "num_rows": pc.sum(pa_table[f"{column}.num_rows"]).as_py(),
            "num_files": pa_table.num_rows,
        }
        info_file = os.path.join(meta_path, "info.json")
        with self._fs.open(info_file, "w") as f:
            json.dump(info, f)
        return info

    def write_meta(
        self,
        meta: pa.Table,
        num_threads: Optional[int] = None,
        version: Optional[str] = None,
    ):
        path = os.path.join(self._path, version or self._DEFAULT_VERSION)

        futures = []
        with ThreadPoolExecutor(
            max_workers=num_threads, thread_name_prefix=self.write_meta.__qualname__
        ) as executor:
            for name in meta.column_names:
                meta_path = os.path.join(path, name, self._META_DIR)
                futures.append(
                    executor.submit(
                        self._write_meta_by_table_column,
                        meta_path,
                        meta,
                        name,
                    )
                )

        # Raise exception if exists.
        return dict(zip(meta.column_names, (fut.result() for fut in futures)))

    def write(
        self,
        dataset,
        chunk_index: int,
        max_chunk_rows: Optional[int] = None,
        column_groups: Optional[dict] = None,
        num_threads: Optional[int] = None,
        version: Optional[str] = None,
    ):
        from datasets import Dataset
        from datasets.features.audio import Audio
        from datasets.features.image import Image

        assert isinstance(dataset, Dataset)

        if column_groups is None:
            # Auto split columns to multimedia and data groups.
            multimedia_columns = []
            data_columns = []
            for idx, ft in enumerate(dataset.features.values()):
                if isinstance(ft, (Image, Audio)):
                    multimedia_columns.append(idx)
                else:
                    data_columns.append(idx)
            column_groups = {}
            if multimedia_columns:
                column_groups[self._MULTIMEDIA_PREFIX] = multimedia_columns
            if data_columns:
                column_groups[self._DATA_PREFIX] = data_columns

        feature_groups = []
        column_names = dataset.column_names
        for name, columns in column_groups.items():
            features = {}
            for col in columns:
                col_name = column_names[col]
                features[col_name] = dataset.features[col_name]
            feature_groups.append(features)

        path = os.path.join(self._path, version or self._DEFAULT_VERSION)
        for name in column_groups.keys():
            self._fs.mkdirs(os.path.join(path, name), exist_ok=True)

        info = defaultdict(list)
        futures = []
        with ThreadPoolExecutor(
            max_workers=num_threads, thread_name_prefix=self.write.__qualname__
        ) as executor:
            for idx, pa_table in enumerate(
                dataset.with_format("arrow").iter(
                    max_chunk_rows or self._DEFAULT_MAX_CHUNK_ROWS
                )
            ):
                pa_table = pa_table.combine_chunks()
                for (name, columns), features in zip(
                    column_groups.items(), feature_groups
                ):
                    pa_group_table = pa_table.select(columns)
                    # The schema is not changed, don't need to table_cast.
                    filename = self._FILE_NAME_FORMATTER.format(
                        prefix=name, chunk_index=chunk_index, index=idx
                    )
                    file = os.path.join(path, name, filename)
                    file_info = {
                        "file": filename,
                        "num_rows": pa_table.num_rows,
                        "num_bytes": pa_group_table.nbytes,
                    }
                    info[name].append(file_info)
                    futures.append(
                        executor.submit(
                            self._embed_and_write_table,
                            file,
                            pa_group_table,
                            features,
                            file_info,
                        )
                    )
        # Raise exception if exists.
        for fut in as_completed(futures):
            fut.result()
        # to_pandas() to walk-around issue: https://github.com/xorbitsai/xorbits/issues/638
        meta = pa.Table.from_pydict(info).to_pandas()
        return meta


def export(
    dataset,
    path: str,
    storage_options: Optional[dict] = None,
    max_chunk_rows: Optional[int] = None,
):
    op = HuggingfaceExport(
        output_types=[OutputType.object],
        path=path,
        storage_options=storage_options,
        max_chunk_rows=max_chunk_rows,
    )
    meta = op(dataset).execute().fetch()
    meta = pa.Table.from_pandas(meta, preserve_index=False)
    info = ArrowWriter.ensure_path(path, storage_options).write_meta(meta)
    return info

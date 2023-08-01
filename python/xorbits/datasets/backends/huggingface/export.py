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

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional

import fsspec
import numpy as np
import pyarrow as pa

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
            inp, op.max_chunk_rows
        )
        ctx[out.key] = r


class ArrowWriter:
    _WRITER_CLASS = pa.RecordBatchStreamWriter
    _DEFAULT_VERSION = "0.0.0"
    _DEFAULT_MAX_CHUNK_ROWS = 100
    _MULTIMEDIA_PREFIX = "mdata"
    _DATA_PREFIX = "data"
    _FILE_NAME_FORMATTER = "{prefix}_{index}.arrow"

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

    def _embed_and_write_table(self, file: str, pa_table: pa.Table, features):
        """Write a Table to file.

        Args:
            example: the Table to add.
        """
        from datasets.features.features import require_storage_embed
        from datasets.table import embed_array_storage

        with self._fs.open(file, "wb") as stream:
            schema = pa_table.schema.remove_metadata()
            with self._WRITER_CLASS(stream, schema) as writer:
                arrays = [
                    embed_array_storage(pa_table[name], feature)
                    if require_storage_embed(feature)
                    else pa_table[name]
                    for name, feature in features.items()
                ]
                pa_table = pa.Table.from_arrays(arrays, schema=schema)
                writer.write(pa_table)

    def write(
        self,
        dataset,
        max_chunk_rows: Optional[int] = None,
        version: Optional[str] = None,
        column_groups: Optional[dict] = None,
        max_threads: Optional[int] = None,
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

        column_names = dataset.column_names
        feature_groups = []
        for name, columns in column_groups.items():
            features = {}
            for col in columns:
                col_name = column_names[col]
                features[col_name] = dataset.features[col_name]
            feature_groups.append(features)

        data_path = os.path.join(self._path, version or self._DEFAULT_VERSION, "data")
        self._fs.mkdirs(data_path, exist_ok=True)

        num_files = num_bytes = num_rows = 0
        futures = []
        with ThreadPoolExecutor(
            max_workers=max_threads, thread_name_prefix=ArrowWriter.__qualname__
        ) as executor:
            for idx, pa_table in enumerate(
                dataset.with_format("arrow").iter(
                    max_chunk_rows or self._DEFAULT_MAX_CHUNK_ROWS
                )
            ):
                pa_table = pa_table.combine_chunks()
                num_rows += pa_table.num_rows
                for (name, columns), features in zip(
                    column_groups.items(), feature_groups
                ):
                    pa_group_table = pa_table.select(columns)
                    # The schema is not changed, don't need to table_cast.
                    filename = self._FILE_NAME_FORMATTER.format(prefix=name, index=idx)
                    file = os.path.join(data_path, filename)
                    num_files += 1
                    num_bytes += pa_group_table.nbytes
                    futures.append(
                        executor.submit(
                            self._embed_and_write_table,
                            file,
                            pa_group_table,
                            features,
                        )
                    )
        for fut in as_completed(futures):
            fut.result()
        return {
            "num_files": [num_files],
            "num_bytes": [num_bytes],
            "num_rows": [num_rows],
        }


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
    return op(dataset).execute().fetch()

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
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import cloudpickle
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from ...._mars.core.entity import OutputType
from ...._mars.serialization.serializables import DictField, Int32Field, StringField
from ...._mars.utils import lazy_import
from ...operand import DataOperand, DataOperandMixin

if TYPE_CHECKING:
    from fsspec import AbstractFileSystem
else:
    AbstractFileSystem = lazy_import("fsspec.AbstractFileSystem")


class HuggingfaceExport(DataOperand, DataOperandMixin):
    path: str = StringField("path")
    storage_options: Dict = DictField("storage_options")
    max_chunk_rows: int = Int32Field("max_chunk_rows")
    column_groups: dict = DictField("column_groups")
    num_threads: int = Int32Field("num_threads")
    version: str = StringField("version")

    def __call__(self, dataset):
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
            dataset=inp,
            chunk_index=out.index[0],
            max_chunk_rows=op.max_chunk_rows,
            column_groups=op.column_groups,
            num_threads=op.num_threads,
            version=op.version,
        )
        ctx[out.key] = r


@dataclasses.dataclass
class SchemaInfo:
    schema: pa.Schema
    column_groups: Dict
    max_chunk_rows: int


class ArrowWriter:
    _WRITER_CLASS = pa.RecordBatchStreamWriter
    _DEFAULT_VERSION = "0.0.0"
    _DEFAULT_MAX_CHUNK_ROWS = 100
    _META_DIR = ".meta"
    _MULTIMEDIA_PREFIX = "mdata"
    _DATA_PREFIX = "data"
    _FILE_NAME_FORMATTER = "{chunk_index}_{index}.arrow"
    _META_SCHEMA_INFO_KEY = b"xdataset_schema_info"

    def __init__(
        self,
        fs: AbstractFileSystem,
        path: Union[str, os.PathLike],
    ):
        self._fs = fs
        self._path = path

    @classmethod
    def ensure_path(
        cls,
        path: Union[str, os.PathLike],
        storage_options: Optional[dict] = None,
        create_if_not_exists: Optional[bool] = True,
    ):
        import fsspec

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

    def check_version(
        self, version: Optional[str] = None, overwrite: Optional[bool] = True
    ):
        version = version or self._DEFAULT_VERSION
        version_path = os.path.join(self._path, version)
        if self._fs.exists(version_path):
            if overwrite:
                self._fs.rm(version_path, recursive=True)
            else:
                raise Exception(f"The version {version_path} already exists.")

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
                # Update embedded table nbytes to num_bytes
                info["num_bytes"] = pa_table.nbytes
                writer.write(pa_table)

    def _write_group_meta(
        self,
        q: Queue,
        meta_path: str,
        meta_table: pa.Table,
        name: str,
        schema_info: SchemaInfo,
    ):
        self._fs.mkdirs(meta_path, exist_ok=True)
        meta_table = meta_table.select([name]).flatten()
        group_schema_table = schema_info.schema.empty_table().select(
            schema_info.column_groups[name]
        )
        info = {
            "num_bytes": pc.sum(meta_table[f"{name}.num_bytes"]).as_py(),
            "num_rows": pc.sum(meta_table[f"{name}.num_rows"]).as_py(),
            "num_columns": group_schema_table.num_columns,
            "num_files": meta_table.num_rows,
            "schema_string": group_schema_table.schema.to_string(
                show_field_metadata=False, show_schema_metadata=False
            ),
        }
        q.put(info)
        info_file = os.path.join(meta_path, "info.json")
        with self._fs.open(info_file, "w") as f:
            json.dump(info, f)
        # Write index arrow table.
        index_file = os.path.join(meta_path, "index.arrow")
        with self._fs.open(index_file, "wb") as stream:
            with self._WRITER_CLASS(stream, meta_table.schema) as writer:
                writer.write(meta_table)
        # Write schema empty table.
        schema_file = os.path.join(meta_path, "schema.arrow")
        with self._fs.open(schema_file, "wb") as stream:
            with self._WRITER_CLASS(stream, group_schema_table.schema) as writer:
                writer.write(group_schema_table)
        return info

    def write_meta(
        self,
        meta_table: pa.Table,
        num_threads: Optional[int] = None,
        version: Optional[str] = None,
    ):
        version_path = os.path.join(self._path, version or self._DEFAULT_VERSION)
        schema_info = cloudpickle.loads(
            meta_table.schema.metadata[self._META_SCHEMA_INFO_KEY]
        )

        futures = []
        with ThreadPoolExecutor(
            max_workers=num_threads, thread_name_prefix=self.write_meta.__qualname__
        ) as executor:
            q: Queue = Queue()

            for name in meta_table.column_names:
                meta_path = os.path.join(version_path, name, self._META_DIR)
                futures.append(
                    executor.submit(
                        self._write_group_meta,
                        q,
                        meta_path,
                        meta_table,
                        name,
                        schema_info,
                    )
                )

            group_info = q.get()
            info = {
                "groups": meta_table.column_names,
                "num_rows": group_info["num_rows"],
                "max_chunk_rows": schema_info.max_chunk_rows,
            }
            info_file = os.path.join(version_path, "info.json")
            with self._fs.open(info_file, "w") as f:
                json.dump(info, f)

        # Raise exception if exists.
        return dict(zip(meta_table.column_names, (fut.result() for fut in futures)))

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
            # TODO(codingl2k1): Should we always split dataset?
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
        data_table = dataset.data
        for name, columns in column_groups.items():
            features = {}
            for col in columns:
                col_name = data_table.field(col).name
                features[col_name] = dataset.features[col_name]
            feature_groups.append(features)

        version_path = os.path.join(self._path, version or self._DEFAULT_VERSION)
        for name in column_groups.keys():
            self._fs.mkdirs(os.path.join(version_path, name), exist_ok=True)

        schema_info = None
        meta_info: Dict[str, List[Any]] = defaultdict(list)
        futures = []
        with ThreadPoolExecutor(
            max_workers=num_threads, thread_name_prefix=self.write.__qualname__
        ) as executor:
            # TODO(codingl2k1): split by max_chunk_rows may lead to rank data skew.
            # e.g. if 1000, 1000, 80, then the rank 2 world 3 worker read very few data.
            max_chunk_rows = max_chunk_rows or self._DEFAULT_MAX_CHUNK_ROWS
            for idx, pa_table in enumerate(
                dataset.with_format("arrow").iter(max_chunk_rows)
            ):
                if chunk_index == 0 and idx == 0:
                    schema_info = SchemaInfo(
                        pa_table.schema, column_groups, max_chunk_rows
                    )
                pa_table = pa_table.combine_chunks()
                for (name, columns), features in zip(
                    column_groups.items(), feature_groups
                ):
                    pa_group_table = pa_table.select(columns)
                    # The schema is not changed, don't need to table_cast.
                    filename = self._FILE_NAME_FORMATTER.format(
                        chunk_index=chunk_index, index=idx
                    )
                    file = os.path.join(version_path, name, filename)
                    file_info = {
                        "index": (chunk_index, idx),
                        "num_rows": pa_table.num_rows,
                        "num_bytes": pa_group_table.nbytes,
                    }
                    meta_info[name].append(file_info)
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

        struct_schema = pa.struct(
            [
                ("index", pa.list_(pa.int32(), 2)),
                ("num_rows", pa.int64()),
                ("num_bytes", pa.int64()),
            ]
        )
        meta_table = pa.Table.from_pydict(
            meta_info,
            schema=pa.schema(
                [pa.field(k, struct_schema) for k in meta_info],
                metadata=None
                if schema_info is None
                else {self._META_SCHEMA_INFO_KEY: cloudpickle.dumps(schema_info)},
            ),
        )
        return meta_table


def export(
    dataset,
    path: Union[str, os.PathLike],
    storage_options: Optional[dict] = None,
    create_if_not_exists: Optional[bool] = True,
    max_chunk_rows: Optional[int] = None,
    column_groups: Optional[dict] = None,
    num_threads: Optional[int] = None,
    version: Optional[str] = None,
    overwrite: Optional[bool] = True,
):
    arrow_writer = ArrowWriter.ensure_path(path, storage_options, create_if_not_exists)
    arrow_writer.check_version(version, overwrite)
    op = HuggingfaceExport(
        output_types=[OutputType.object],
        path=path,
        storage_options=storage_options,
        max_chunk_rows=max_chunk_rows,
        column_groups=column_groups,
        num_threads=num_threads,
        version=version,
    )
    meta_table = op(dataset).execute().fetch()
    info = arrow_writer.write_meta(meta_table)
    return info

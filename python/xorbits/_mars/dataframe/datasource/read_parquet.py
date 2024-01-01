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
import sys
import zipfile
from typing import Dict
from urllib.parse import urlparse

import numpy as np
import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq

    arrow_dtype = pa.DataType
except ImportError:
    pa = None
    arrow_dtype = None

try:
    import fastparquet
except ImportError:
    fastparquet = None

import fsspec

from ... import opcodes as OperandDef
from ...config import options
from ...serialization.serializables import (
    AnyField,
    BoolField,
    DictField,
    Int32Field,
    Int64Field,
    ListField,
    StringField,
)
from ...utils import is_object_dtype, lazy_import
from ..operands import OutputType
from ..utils import PD_VERSION_GREATER_THAN_2_10, arrow_dtype_kwargs, parse_index
from .core import (
    ColumnPruneSupportedDataSourceMixin,
    IncrementalIndexDatasource,
    IncrementalIndexDataSourceMixin,
    merge_small_files,
)
from .utils import convert_to_abspath

PARQUET_MEMORY_SCALE = 10
PARQUET_MEMORY_SCALE_WITH_ARROW_DTYPE = 3
STRING_FIELD_OVERHEAD = 50
cudf = lazy_import("cudf")


def check_engine(engine):
    if engine == "auto":
        if pa is not None:
            return "pyarrow"
        elif fastparquet is not None:  # pragma: no cover
            return "fastparquet"
        else:  # pragma: no cover
            raise RuntimeError("Please install either pyarrow or fastparquet.")
    elif engine == "pyarrow":
        if pa is None:  # pragma: no cover
            raise RuntimeError("Please install pyarrow first.")
        return engine
    elif engine == "fastparquet":
        if fastparquet is None:  # pragma: no cover
            raise RuntimeError("Please install fastparquet first.")
        return engine
    else:  # pragma: no cover
        raise RuntimeError("Unsupported engine {} to read parquet.".format(engine))


def get_engine(engine):
    if engine == "pyarrow":
        return ArrowEngine()
    elif engine == "fastparquet":
        return FastpaquetEngine()
    else:  # pragma: no cover
        raise RuntimeError("Unsupported engine {}".format(engine))


class ParquetEngine:
    """Read parquet by arrow / fastparquet instead of pandas is to read the
    parquet file by group, please refer to `groups_as_chunks`."""

    def get_row_num(self, f):
        raise NotImplementedError

    def read_dtypes(self, f, **kwargs):
        raise NotImplementedError

    def read_to_pandas(
        self, f, columns=None, nrows=None, use_arrow_dtype=None, **kwargs
    ):
        raise NotImplementedError

    def read_group_to_pandas(
        self, f, group_index, columns=None, nrows=None, use_arrow_dtype=None, **kwargs
    ):
        raise NotImplementedError

    def read_partitioned_to_pandas(
        self,
        f,
        partitions: Dict,
        partition_keys: Dict,
        columns=None,
        nrows=None,
        use_arrow_dtype=None,
        **kwargs,
    ):
        raw_df = self.read_to_pandas(
            f, columns=columns, nrows=nrows, use_arrow_dtype=use_arrow_dtype, **kwargs
        )
        for col, value in partition_keys.items():
            dictionary = partitions[col]
            raw_df[col] = pd.Series(
                value,
                dtype=pd.CategoricalDtype(categories=dictionary.tolist()),
                index=raw_df.index,
            )
        return raw_df

    def read_partitioned_dtypes(
        self, fs: fsspec.AbstractFileSystem, directory, storage_options
    ):
        # As ParquetDataset will iterate all files,
        # here we just find one file to infer dtypes
        current_path = directory
        partition_cols = []
        while fs.isdir(current_path):
            _, dirs, files = next(fs.walk(current_path))
            dirs = [d for d in dirs if not d.startswith(".")]
            files = [f for f in files if not f.startswith(".")]
            if len(files) == 0:
                # directory as partition
                partition_cols.append(dirs[0].split("=", 1)[0])
                current_path = os.path.join(current_path, dirs[0])
            elif len(dirs) == 0:
                # parquet files in deepest directory
                current_path = os.path.join(current_path, files[0])
            else:  # pragma: no cover
                raise ValueError(
                    "Files and directories are mixed in an intermediate directory"
                )

        # current path is now a parquet file
        of = fsspec.open(current_path, storage_options=storage_options)
        with of as f:
            dtypes = self.read_dtypes(f)
        for partition in partition_cols:
            dtypes[partition] = pd.CategoricalDtype()
        return dtypes


def _parse_prefix(path):
    path_prefix = ""
    if isinstance(path, str):
        parsed_path = urlparse(path)
        if parsed_path.scheme:
            path_prefix = f"{parsed_path.scheme}://{parsed_path.netloc}"
    return path_prefix


class ArrowEngine(ParquetEngine):
    def get_row_num(self, f):
        file = pq.ParquetFile(f)
        return file.metadata.num_rows

    def read_dtypes(self, f, **kwargs):
        types_mapper = kwargs.pop("types_mapper", None)
        file = pq.ParquetFile(f)
        return (
            file.schema_arrow.empty_table().to_pandas(types_mapper=types_mapper).dtypes
        )

    @classmethod
    def _table_to_pandas(cls, t, nrows=None, use_arrow_dtype=None):
        if nrows is not None:
            t = t.slice(0, nrows)
        if use_arrow_dtype:
            df = t.to_pandas(types_mapper=pd.ArrowDtype)
        else:
            df = t.to_pandas()
        return df

    def read_to_pandas(
        self, f, columns=None, nrows=None, use_arrow_dtype=None, **kwargs
    ):
        file = pq.ParquetFile(f)
        t = file.read(columns=columns, **kwargs)
        return self._table_to_pandas(t, nrows=nrows, use_arrow_dtype=use_arrow_dtype)

    def read_group_to_pandas(
        self, f, group_index, columns=None, nrows=None, use_arrow_dtype=None, **kwargs
    ):
        file = pq.ParquetFile(f)
        t = file.read_row_group(group_index, columns=columns, **kwargs)
        return self._table_to_pandas(t, nrows=nrows, use_arrow_dtype=use_arrow_dtype)


class FastpaquetEngine(ParquetEngine):
    def get_row_num(self, f):
        file = fastparquet.ParquetFile(f)
        return file.count()

    def read_dtypes(self, f, **kwargs):
        file = fastparquet.ParquetFile(f)
        dtypes_dict = file._dtypes()
        return pd.Series(dict((c, dtypes_dict[c]) for c in file.columns))

    def read_to_pandas(
        self, f, columns=None, nrows=None, use_arrow_dtype=None, **kwargs
    ):
        file = fastparquet.ParquetFile(f)
        df = file.to_pandas(columns, **kwargs)
        if nrows is not None:
            df = df.head(nrows)
        return df


class CudfEngine:
    @classmethod
    def read_to_cudf(cls, file, columns: list = None, nrows: int = None, **kwargs):
        df = cudf.read_parquet(file, columns=columns, **kwargs)
        if nrows is not None:
            df = df.head(nrows)
        return df

    def read_group_to_cudf(
        self, file, group_index: int, columns: list = None, nrows: int = None, **kwargs
    ):
        return self.read_to_cudf(
            file, columns=columns, nrows=nrows, row_groups=group_index, **kwargs
        )

    @classmethod
    def read_partitioned_to_cudf(
        cls,
        file,
        partitions: Dict,
        partition_keys: Dict,
        columns=None,
        nrows=None,
        **kwargs,
    ):
        # cudf will read entire partitions even if only one partition provided,
        # so we just read with pyarrow and convert to cudf DataFrame
        file = pq.ParquetFile(file)
        t = file.read(columns=columns, **kwargs)
        t = t.slice(0, nrows) if nrows is not None else t
        t = pa.table(t.columns, names=t.column_names)
        raw_df = cudf.DataFrame.from_arrow(t)
        for col, value in partition_keys.items():
            dictionary = partitions[col].tolist()
            codes = cudf.core.column.as_column(
                dictionary.index(value), length=len(raw_df)
            )
            raw_df[col] = cudf.core.column.build_categorical_column(
                categories=dictionary,
                codes=codes,
                size=codes.size,
                offset=codes.offset,
                ordered=False,
            )
        return raw_df


class DataFrameReadParquet(
    IncrementalIndexDatasource,
    ColumnPruneSupportedDataSourceMixin,
    IncrementalIndexDataSourceMixin,
):
    _op_type_ = OperandDef.READ_PARQUET

    path = AnyField("path")
    chunk_path = AnyField("chunk_path")
    engine = StringField("engine")
    columns = ListField("columns")
    use_arrow_dtype = BoolField("use_arrow_dtype")
    groups_as_chunks = BoolField("groups_as_chunks")
    group_index = Int32Field("group_index")
    read_kwargs = DictField("read_kwargs")
    incremental_index = BoolField("incremental_index")
    storage_options = DictField("storage_options")
    is_partitioned = BoolField("is_partitioned")
    merge_small_files = BoolField("merge_small_files")
    merge_small_file_options = DictField("merge_small_file_options")
    is_http_url = BoolField("is_http_url", None)
    # for chunk
    partitions = DictField("partitions", default=None)
    partition_keys = DictField("partition_keys", default=None)
    num_group_rows = Int64Field("num_group_rows", default=None)
    # as read meta may be too time-consuming when number of files is large,
    # thus we only read first file to get row number and raw file size
    first_chunk_row_num = Int64Field("first_chunk_row_num")
    first_chunk_raw_bytes = Int64Field("first_chunk_raw_bytes")

    def get_columns(self):
        return self.columns

    def set_pruned_columns(self, columns, *, keep_order=None):
        self.columns = columns

    @classmethod
    def _tile_partitioned(cls, op: "DataFrameReadParquet"):
        out_df = op.outputs[0]
        shape = (np.nan, out_df.shape[1])
        dtypes = out_df.dtypes
        dataset = pq.ParquetDataset(op.path, use_legacy_dataset=False)

        path_prefix = _parse_prefix(op.path)

        chunk_index = 0
        out_chunks = []
        first_chunk_row_num, first_chunk_raw_bytes = None, None
        for i, fragment in enumerate(dataset.fragments):
            chunk_op = op.copy().reset_key()
            chunk_op.path = chunk_path = path_prefix + fragment.path
            relpath = os.path.relpath(chunk_path, op.path)
            partition_keys = dict(
                tuple(s.split("=")) for s in relpath.split(os.sep)[:-1]
            )
            chunk_op.partition_keys = partition_keys
            chunk_op.partitions = dict(
                zip(
                    dataset.partitioning.schema.names, dataset.partitioning.dictionaries
                )
            )
            if i == 0:
                first_row_group = fragment.row_groups[0]
                first_chunk_raw_bytes = first_row_group.total_byte_size
                first_chunk_row_num = first_row_group.num_rows
            chunk_op.first_chunk_row_num = first_chunk_row_num
            chunk_op.first_chunk_raw_bytes = first_chunk_raw_bytes
            new_chunk = chunk_op.new_chunk(
                None,
                shape=shape,
                index=(chunk_index, 0),
                index_value=out_df.index_value,
                columns_value=out_df.columns_value,
                dtypes=dtypes,
            )
            out_chunks.append(new_chunk)
            chunk_index += 1

        new_op = op.copy()
        nsplits = ((np.nan,) * len(out_chunks), (out_df.shape[1],))
        return new_op.new_dataframes(
            None,
            out_df.shape,
            dtypes=dtypes,
            index_value=out_df.index_value,
            columns_value=out_df.columns_value,
            chunks=out_chunks,
            nsplits=nsplits,
        )

    @classmethod
    def _tile_no_partitioned(cls, op: "DataFrameReadParquet"):
        chunk_index = 0
        out_chunks = []
        out_df = op.outputs[0]

        dtypes = out_df.dtypes
        shape = (np.nan, out_df.shape[1])
        z = None
        fs, _, _ = fsspec.get_fs_token_paths(
            op.path, storage_options=op.storage_options
        )
        if isinstance(op.path, (tuple, list)):
            paths = op.path
        elif fs.isdir(op.path):
            paths = fs.ls(op.path)
            paths = sorted(paths)
            if not isinstance(fs, fsspec.implementations.local.LocalFileSystem):
                parsed_path = urlparse(op.path)
                paths = [f"{parsed_path.scheme}://{path}" for path in paths]
        elif isinstance(op.path, str) and op.path.endswith(".zip"):
            file = fs.open(op.path, storage_options=op.storage_options)
            z = zipfile.ZipFile(file)
            paths = z.namelist()
            paths = [
                path
                for path in paths
                if path.endswith(".parquet") and not path.startswith("__MACOSX")
            ]
        else:
            paths = fs.glob(op.path)
            if not isinstance(fs, fsspec.implementations.local.LocalFileSystem):
                parsed_path = urlparse(op.path)
                paths = [f"{parsed_path.scheme}://{path}" for path in paths]
        first_chunk_row_num, first_chunk_raw_bytes = None, None
        for i, pth in enumerate(paths):
            if i == 0:
                if z is not None:
                    with z.open(pth) as f:
                        first_chunk_row_num = get_engine(op.engine).get_row_num(f)
                        first_chunk_raw_bytes = sys.getsizeof(f)
                else:
                    of = fs.open(pth)
                    first_chunk_row_num = get_engine(op.engine).get_row_num(of)
                    first_chunk_raw_bytes = fsspec.get_fs_token_paths(
                        pth, storage_options=op.storage_options
                    )[0].size(pth)

            if op.groups_as_chunks:
                if z is not None:
                    with z.open(pth) as f:
                        num_row_groups = pq.ParquetFile(f).num_row_groups
                else:
                    num_row_groups = pq.ParquetFile(pth).num_row_groups
                for group_idx in range(num_row_groups):
                    chunk_op = op.copy().reset_key()
                    if z is not None:
                        chunk_op.path = op.path
                        chunk_op.chunk_path = pth
                    else:
                        chunk_op.path = pth
                    chunk_op.group_index = group_idx
                    chunk_op.first_chunk_row_num = first_chunk_row_num
                    chunk_op.first_chunk_raw_bytes = first_chunk_raw_bytes
                    chunk_op.num_group_rows = num_row_groups
                    new_chunk = chunk_op.new_chunk(
                        None,
                        shape=shape,
                        index=(chunk_index, 0),
                        index_value=out_df.index_value,
                        columns_value=out_df.columns_value,
                        dtypes=dtypes,
                    )
                    out_chunks.append(new_chunk)
                    chunk_index += 1
            else:
                chunk_op = op.copy().reset_key()
                if z is not None:
                    chunk_op.path = op.path
                    chunk_op.chunk_path = pth
                else:
                    chunk_op.path = pth
                chunk_op.first_chunk_row_num = first_chunk_row_num
                chunk_op.first_chunk_raw_bytes = first_chunk_raw_bytes
                new_chunk = chunk_op.new_chunk(
                    None,
                    shape=shape,
                    index=(chunk_index, 0),
                    index_value=out_df.index_value,
                    columns_value=out_df.columns_value,
                    dtypes=dtypes,
                )
                out_chunks.append(new_chunk)
                chunk_index += 1
        if z is not None:
            z.close()
        new_op = op.copy()
        nsplits = ((np.nan,) * len(out_chunks), (out_df.shape[1],))
        return new_op.new_dataframes(
            None,
            out_df.shape,
            dtypes=dtypes,
            index_value=out_df.index_value,
            columns_value=out_df.columns_value,
            chunks=out_chunks,
            nsplits=nsplits,
        )

    @classmethod
    def _tile_http_url(cls, op: "DataFrameReadParquet"):
        out_chunks = []
        out_df = op.outputs[0]
        z = None
        if op.path[0].endswith(".zip"):
            fs, _, _ = fsspec.get_fs_token_paths(op.path[0])
            zip_filename = fs.open(op.path[0])
            z = zipfile.ZipFile(zip_filename)
            paths = z.namelist()
            paths = [
                path
                for path in paths
                if path.endswith(".parquet") and not path.startswith("__MACOSX")
            ]
        else:
            paths = op.path
        for i, url in enumerate(paths):
            chunk_op = op.copy().reset_key()
            if z is not None:
                chunk_op.path = op.path[0]
                chunk_op.chunk_path = url
            else:
                chunk_op.path = url
            out_chunks.append(
                chunk_op.new_chunk(None, index=(i, 0), shape=(np.nan, np.nan))
            )
        if z is not None:
            z.close()
        new_op = op.copy()
        nsplits = ((np.nan,) * len(out_chunks), (np.nan,))
        return new_op.new_dataframes(
            None,
            out_df.shape,
            dtypes=out_df.dtypes,
            index_value=out_df.index_value,
            columns_value=out_df.columns_value,
            chunks=out_chunks,
            nsplits=nsplits,
        )

    @classmethod
    def _tile(cls, op: "DataFrameReadParquet"):
        if op.is_http_url:
            tiled = cls._tile_http_url(op)
        elif op.is_partitioned:
            tiled = cls._tile_partitioned(op)
        else:
            tiled = cls._tile_no_partitioned(op)
        if op.merge_small_files:
            tiled = [
                merge_small_files(tiled[0], **(op.merge_small_file_options or dict()))
            ]
        return tiled

    @classmethod
    def _execute_partitioned(cls, ctx, op: "DataFrameReadParquet"):
        out = op.outputs[0]
        engine = get_engine(op.engine)
        of = fsspec.open(op.path, storage_options=op.storage_options)
        with of as f:
            ctx[out.key] = engine.read_partitioned_to_pandas(
                f,
                op.partitions,
                op.partition_keys,
                columns=op.columns,
                nrows=op.nrows,
                use_arrow_dtype=op.use_arrow_dtype,
                **op.read_kwargs or dict(),
            )

    @classmethod
    def _pandas_read_parquet(cls, ctx: dict, op: "DataFrameReadParquet"):
        out = op.outputs[0]
        path = op.path
        z = None
        if op.is_http_url:
            if op.path.endswith(".zip"):
                fs, _, _ = fsspec.get_fs_token_paths(op.path)
                zip_filename = fs.open(op.path)
                z = zipfile.ZipFile(zip_filename)
                f = z.open(op.chunk_path)
            else:
                f = op.path
            read_kwargs = op.read_kwargs or dict()
            if op.use_arrow_dtype:
                read_kwargs.update(arrow_dtype_kwargs())
            r = pd.read_parquet(
                f,
                columns=op.columns,
                engine=op.engine,
                **read_kwargs,
            )
            if z is not None:
                z.close()
                f.close()
            ctx[out.key] = r
            return
        if op.partitions is not None:
            return cls._execute_partitioned(ctx, op)
        engine = get_engine(op.engine)
        z = None
        fs = fsspec.get_fs_token_paths(path, storage_options=op.storage_options)[0]
        if op.path.endswith(".zip"):
            file = fs.open(op.path, storage_options=op.storage_options)
            z = zipfile.ZipFile(file)
            f = z.open(op.chunk_path)
        else:
            f = fs.open(path, storage_options=op.storage_options)
        use_arrow_dtype = op.use_arrow_dtype
        if op.groups_as_chunks:
            df = engine.read_group_to_pandas(
                f,
                op.group_index,
                columns=op.columns,
                nrows=op.nrows,
                use_arrow_dtype=use_arrow_dtype,
                **op.read_kwargs or dict(),
            )
        else:
            df = engine.read_to_pandas(
                f,
                columns=op.columns,
                nrows=op.nrows,
                use_arrow_dtype=use_arrow_dtype,
                **op.read_kwargs or dict(),
            )
        ctx[out.key] = df
        if z is not None:
            z.close()
        f.close()

    @classmethod
    def _cudf_read_parquet(cls, ctx: dict, op: "DataFrameReadParquet"):
        out = op.outputs[0]
        path = op.path
        z = None
        fs = fsspec.get_fs_token_paths(path, storage_options=op.storage_options)[0]
        if path.endswith(".zip"):
            z = zipfile.ZipFile(path)
        engine = CudfEngine()
        if os.path.exists(path) and z is None:
            file = op.path
            close = lambda: None
        else:  # pragma: no cover
            if z is not None:
                file = z.open(op.chunk_path)
            else:
                file = fs.open(path, storage_options=op.storage_options)
            close = file.close
        try:
            if op.partitions is not None:
                ctx[out.key] = engine.read_partitioned_to_cudf(
                    file,
                    op.partitions,
                    op.partition_keys,
                    columns=op.columns,
                    nrows=op.nrows,
                    **op.read_kwargs or dict(),
                )
            else:
                if op.groups_as_chunks:
                    df = engine.read_group_to_cudf(
                        file,
                        op.group_index,
                        columns=op.columns,
                        nrows=op.nrows,
                        **op.read_kwargs or dict(),
                    )
                else:
                    df = engine.read_to_cudf(
                        file,
                        columns=op.columns,
                        nrows=op.nrows,
                        **op.read_kwargs or dict(),
                    )
                ctx[out.key] = df
        finally:
            close()

    @classmethod
    def execute(cls, ctx, op: "DataFrameReadParquet"):
        if not op.gpu:
            cls._pandas_read_parquet(ctx, op)
        else:
            cls._cudf_read_parquet(ctx, op)

    @classmethod
    def estimate_size(cls, ctx, op: "DataFrameReadParquet"):
        if op.is_http_url:
            return super().estimate_size(ctx, op)
        first_chunk_row_num = op.first_chunk_row_num
        first_chunk_raw_bytes = op.first_chunk_raw_bytes
        if isinstance(op.path, str) and op.path.endswith(".zip"):
            with fsspec.open(op.path, storage_options=op.storage_options) as zip_file:
                with zipfile.ZipFile(zip_file) as z:
                    with z.open(op.chunk_path) as f:
                        raw_bytes = sys.getsizeof(f)
        else:
            raw_bytes = fsspec.get_fs_token_paths(
                op.path, storage_options=op.storage_options
            )[0].size(op.path)
        if op.num_group_rows:
            raw_bytes = (
                np.ceil(np.divide(raw_bytes, op.num_group_rows)).astype(np.int64).item()
            )

        estimated_row_num = (
            np.ceil(first_chunk_row_num * (raw_bytes / first_chunk_raw_bytes))
            .astype(np.int64)
            .item()
        )
        if op.columns is not None:
            of = fsspec.open(op.path, storage_options=op.storage_options)
            with of as f:
                all_columns = list(get_engine(op.engine).read_dtypes(f).index)
        else:
            all_columns = list(op.outputs[0].dtypes.index)
        columns = op.columns if op.columns else all_columns
        if op.use_arrow_dtype:
            scale = op.memory_scale or PARQUET_MEMORY_SCALE_WITH_ARROW_DTYPE
        else:
            scale = op.memory_scale or PARQUET_MEMORY_SCALE
        phy_size = raw_bytes * scale * len(columns) / len(all_columns)
        n_strings = len(
            [
                dt
                for col, dt in op.outputs[0].dtypes.items()
                if col in columns and is_object_dtype(dt)
            ]
        )
        if op.use_arrow_dtype:
            pd_size = phy_size
        else:
            pd_size = phy_size + n_strings * estimated_row_num * STRING_FIELD_OVERHEAD
        ctx[op.outputs[0].key] = (pd_size, pd_size)

    def __call__(self, index_value=None, columns_value=None, dtypes=None):
        self._output_types = [OutputType.dataframe]
        if dtypes is not None:
            shape = (np.nan, len(dtypes))
        else:
            shape = (np.nan, np.nan)
        return self.new_dataframe(
            None,
            shape,
            dtypes=dtypes,
            index_value=index_value,
            columns_value=columns_value,
        )


def read_parquet(
    path,
    engine: str = "auto",
    columns: list = None,
    groups_as_chunks: bool = False,
    use_arrow_dtype: bool = None,
    incremental_index: bool = False,
    storage_options: dict = None,
    memory_scale: int = None,
    merge_small_files: bool = True,
    merge_small_file_options: dict = None,
    gpu: bool = None,
    **kwargs,
):
    """
    Load a parquet object from the file path, returning a DataFrame.

    Parameters
    ----------
    path : str, path object or file-like object
        Any valid string path is acceptable. The string could be a URL.
        For file URLs, a host is expected. A local file could be:
        ``file://localhost/path/to/table.parquet``.
        A file URL can also be a path to a directory that contains multiple
        partitioned parquet files. Both pyarrow and fastparquet support
        paths to directories as well as file URLs. A directory path could be:
        ``file://localhost/path/to/tables``.
        By file-like object, we refer to objects with a ``read()`` method,
        such as a file handler (e.g. via builtin ``open`` function)
        or ``StringIO``.
    engine : {'auto', 'pyarrow', 'fastparquet'}, default 'auto'
        Parquet library to use. The default behavior is to try 'pyarrow',
        falling back to 'fastparquet' if 'pyarrow' is unavailable.
    columns : list, default=None
        If not None, only these columns will be read from the file.
    groups_as_chunks : bool, default False
        if True, each row group correspond to a chunk.
        if False, each file correspond to a chunk.
        Only available for 'pyarrow' engine.
    incremental_index: bool, default False
        If index_col not specified, ensure range index incremental,
        gain a slightly better performance if setting False.
    use_arrow_dtype: bool, default None
        If True, use arrow dtype to store columns. Default enabled if pandas >= 2.1
    storage_options: dict, optional
        Options for storage connection.
    memory_scale: int, optional
        Scale that real memory occupation divided with raw file size.
    merge_small_files: bool, default True
        Merge small files whose size is small.
    merge_small_file_options: dict
        Options for merging small files
    **kwargs
        Any additional kwargs are passed to the engine.

    Returns
    -------
    Mars DataFrame
    """

    engine_type = check_engine(engine)
    engine = get_engine(engine_type)

    # We enable arrow dtype by default if pandas >= 2.1
    if use_arrow_dtype is None and engine_type == "pyarrow":
        use_arrow_dtype = PD_VERSION_GREATER_THAN_2_10

    single_path = path[0] if isinstance(path, list) else path
    is_partitioned = False
    if isinstance(single_path, str) and (
        single_path.startswith("http://")
        or single_path.startswith("https://")
        or single_path.startswith("ftp://")
        or single_path.startswith("sftp://")
    ):
        urls = path if isinstance(path, (list, tuple)) else [path]
        op = DataFrameReadParquet(
            path=urls,
            engine=engine_type,
            columns=columns,
            groups_as_chunks=groups_as_chunks,
            use_arrow_dtype=use_arrow_dtype,
            read_kwargs=kwargs,
            incremental_index=incremental_index,
            storage_options=storage_options,
            memory_scale=memory_scale,
            merge_small_files=merge_small_files,
            merge_small_file_options=merge_small_file_options,
            is_http_url=True,
            gpu=gpu,
        )
        return op()
    fs, _, _ = fsspec.get_fs_token_paths(single_path, storage_options=storage_options)
    if use_arrow_dtype is None:
        use_arrow_dtype = options.dataframe.use_arrow_dtype
    if use_arrow_dtype and engine_type != "pyarrow":
        raise ValueError(
            f"The 'use_arrow_dtype' argument is not supported for the {engine_type} engine"
        )
    # We enable arrow dtype by default if pandas >= 2.1
    if use_arrow_dtype is None:
        use_arrow_dtype = PD_VERSION_GREATER_THAN_2_10

    types_mapper = pd.ArrowDtype if use_arrow_dtype else None

    if fs.isdir(single_path):
        paths = fs.ls(path)
        if all(fs.isdir(p) for p in paths):
            # If all are directories, it is read as a partitioned dataset.
            dtypes = engine.read_partitioned_dtypes(fs, path, storage_options)
            is_partitioned = True
        else:
            with fs.open(paths[0], mode="rb") as f:
                dtypes = engine.read_dtypes(f, types_mapper=types_mapper)
    elif isinstance(path, str) and path.endswith(".zip"):
        with fsspec.open(path, "rb") as file:
            with zipfile.ZipFile(file) as z:
                with z.open(z.namelist()[0]) as f:
                    dtypes = engine.read_dtypes(f, types_mapper=types_mapper)
    else:
        if not isinstance(path, list):
            file_path = fs.glob(path)[0]
        else:
            file_path = path[0]
        with fs.open(file_path) as f:
            dtypes = engine.read_dtypes(f, types_mapper=types_mapper)
    if columns:
        dtypes = dtypes[columns]

    index_value = parse_index(pd.RangeIndex(-1))
    columns_value = parse_index(dtypes.index, store_data=True)

    # convert path to abs_path
    abs_path = convert_to_abspath(path, storage_options)
    op = DataFrameReadParquet(
        path=abs_path,
        engine=engine_type,
        columns=columns,
        groups_as_chunks=groups_as_chunks,
        use_arrow_dtype=use_arrow_dtype,
        read_kwargs=kwargs,
        incremental_index=incremental_index,
        storage_options=storage_options,
        is_partitioned=is_partitioned,
        memory_scale=memory_scale,
        merge_small_files=merge_small_files,
        merge_small_file_options=merge_small_file_options,
        gpu=gpu,
    )
    return op(index_value=index_value, columns_value=columns_value, dtypes=dtypes)

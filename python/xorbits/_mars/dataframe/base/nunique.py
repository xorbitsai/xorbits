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
import uuid
from collections import defaultdict
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ... import opcodes, options
from ...core import OutputType
from ...core.context import Context, get_context
from ...core.operand import MapReduceOperand, OperandStage
from ...serialization.serializables import (
    BoolField,
    Int8Field,
    Int32Field,
    Int64Field,
    StringField,
)
from ...utils import estimate_pandas_size
from ..groupby.aggregation import SizeRecorder
from ..operands import DataFrameOperandMixin, DataFrameShuffleProxy
from ..utils import build_concatenated_rows_frame, hash_dataframe_on, parse_index


class DataFrameNunique(MapReduceOperand, DataFrameOperandMixin):
    _op_type_ = opcodes.NUNIQUE

    axis = Int8Field("axis")
    dropna = BoolField("dropna")
    shuffle_size = Int32Field("shuffle_size")
    combine_size = Int64Field("combine_size")
    method = StringField("method")
    execution_stage = StringField("execution_stage", default=None)
    size_recorder_name = StringField("size_recorder_name")

    def __init__(self, **kw):
        super().__init__(**kw)

    def __call__(self, df: Union["pd.DataFrame", "pd.Series"]):
        if df.ndim == 2:
            shape = df.shape
            if shape[0] == 0:
                res_dtype = np.dtype(np.float64)
            else:
                res_dtype = np.dtype(np.int64)

            if self.axis == 0:
                res_shape = (len(df.columns),)
                res_index_value = df.columns
            else:
                res_shape = (len(df),)
                res_index_value = df.index

            return self.new_series(
                [df],
                shape=res_shape,
                dtype=res_dtype,
                index_value=parse_index(res_index_value),
                name=None,
            )
        else:
            return self.new_scalar([df], dtype=np.dtype(np.int64))

    @classmethod
    def _gen_chunk_params(cls, in_df: Union["pd.DataFrame", "pd.Series"]):
        output_type = OutputType.dataframe if in_df.ndim == 2 else OutputType.series
        shape = (np.nan, np.nan) if in_df.ndim == 2 else (np.nan,)
        index = (0, 0) if in_df.ndim == 2 else (0,)
        return output_type, shape, index

    @classmethod
    def _gen_tree_pre_chunks(
        cls,
        input_chunks: List,
        output_type: OutputType,
        shape: Tuple,
        op: "DataFrameNunique",
    ):
        chunks = []
        for c in input_chunks:
            _new_op = op.copy().reset_key()
            _new_op.execution_stage = "pre"
            _new_op.output_types = [output_type]
            chunks.append(
                _new_op.new_chunk(
                    inputs=[c],
                    shape=shape,
                    index=c.index,
                )
            )
        return chunks

    @classmethod
    def _gen_tree_agg_chunks(
        cls,
        chunks: List,
        output_type: OutputType,
        shape: Tuple,
        index: Tuple,
        op: "DataFrameNunique",
    ):
        combine_size = op.combine_size
        while len(chunks) > combine_size:
            new_chunks = []
            for i in range(0, len(chunks), combine_size):
                chks = chunks[i : i + combine_size]
                if len(chks) == 1:
                    chk = chks[0]
                else:
                    union_op = op.copy().reset_key()
                    union_op.output_types = [output_type]
                    union_op.execution_stage = "agg"
                    for j, c in enumerate(chks):
                        c._index = (j, 0)
                    chk = union_op.new_chunk(chks, shape=shape, index=index)
                new_chunks.append(chk)
            chunks = new_chunks

        if len(chunks) > 1:
            union_op = op.copy().reset_key()
            union_op.output_types = [output_type]
            union_op.execution_stage = "agg"
            agg_chunk = union_op.new_chunk(chunks, shape=shape, index=index)
        else:
            agg_chunk = chunks[0]
        return agg_chunk

    @classmethod
    def _tile_tree(
        cls, in_df: Union["pd.DataFrame", "pd.Series"], op: "DataFrameNunique"
    ):
        output_type, shape, index = cls._gen_chunk_params(in_df)
        pre_chunks = cls._gen_tree_pre_chunks(in_df.chunks, output_type, shape, op)
        agg_chunk = cls._gen_tree_agg_chunks(pre_chunks, output_type, shape, index, op)

        result_chunks = []
        new_op = op.copy().reset_key()
        new_op.execution_stage = "post"
        if in_df.ndim == 2:
            shape = (np.nan,)
            index_value = (
                parse_index(in_df.columns) if op.axis == 0 else parse_index(in_df.index)
            )
            params = dict(
                shape=shape,
                index=(0,),
                dtype=np.dtype(np.int64),
                index_value=index_value,
            )
            result_chunks.append(new_op.new_chunk([agg_chunk], **params))

            _new_op = op.copy()
            params = op.outputs[0].params.copy()
            params["nsplits"] = ((np.nan,) * len(result_chunks),)
            params["chunks"] = result_chunks
            return _new_op.new_seriess(op.inputs, **params)
        else:
            dtype = np.dtype(np.int64)
            params = dict(index=(0,), dtype=dtype, shape=())
            result_chunks.append(new_op.new_chunk([agg_chunk], **params))

            _new_op = op.copy()
            return _new_op.new_scalars(op.inputs, chunks=result_chunks, dtype=dtype)

    @classmethod
    def _tile_one_chunk_df(cls, in_df: "pd.DataFrame", op: "DataFrameNunique"):
        result_chunks = []
        new_op = op.copy().reset_key()
        shape = (np.nan,)
        index_value = (
            parse_index(in_df.columns) if op.axis == 0 else parse_index(in_df.index)
        )
        params = dict(
            shape=shape, index=(0,), dtype=np.dtype(np.int64), index_value=index_value
        )
        result_chunks.append(new_op.new_chunk(in_df.chunks, **params))

        _new_op = op.copy()
        params = op.outputs[0].params.copy()
        params["nsplits"] = ((np.nan,) * len(result_chunks),)
        params["chunks"] = result_chunks
        return _new_op.new_seriess(op.inputs, **params)

    @classmethod
    def _tile_one_chunk_series(cls, in_df: "pd.Series", op: "DataFrameNunique"):
        result_chunks = []
        new_op = op.copy().reset_key()
        dtype = np.dtype(np.int64)
        params = dict(index=(0,), dtype=dtype, shape=())
        result_chunks.append(new_op.new_chunk(in_df.chunks, **params))

        _new_op = op.copy()
        return _new_op.new_scalars(op.inputs, chunks=result_chunks, dtype=dtype)

    @classmethod
    def tile_one_chunk(
        cls, in_df: Union["pd.DataFrame", "pd.Series"], op: "DataFrameNunique"
    ):
        if in_df.ndim == 2:
            return cls._tile_one_chunk_df(in_df, op)
        else:
            return cls._tile_one_chunk_series(in_df, op)

    @classmethod
    def _tile_series_shuffle(cls, in_df: "pd.Series", op: "DataFrameNunique"):
        # generate map chunks
        map_chunks = []
        for chunk in in_df.chunks:
            map_op = op.copy().reset_key()
            map_op.stage = OperandStage.map
            map_op.shuffle_size = len(in_df.chunks)
            map_op._output_types = [OutputType.series]
            map_chunks.append(
                map_op.new_chunk(
                    [chunk],
                    shape=(np.nan,),
                    index=chunk.index,
                )
            )
        proxy_chunk = DataFrameShuffleProxy(output_types=[OutputType.series]).new_chunk(
            map_chunks, shape=()
        )

        # generate reduce chunks
        reduce_chunks = []
        for chunk in in_df.chunks:
            reduce_op = op.copy().reset_key()
            reduce_op._output_types = [OutputType.series]
            reduce_op.stage = OperandStage.reduce
            reduce_op.n_reducers = len(in_df.chunks)
            reduce_chunks.append(
                reduce_op.new_chunk([proxy_chunk], shape=(np.nan,), index=chunk.index)
            )

        out_chunks = []
        for chunk in reduce_chunks:
            new_op = op.copy().reset_key()
            params = dict(shape=(), index=(chunk.index[0],), dtype=np.dtype(np.int64))
            out_chunks.append(new_op.new_chunk([chunk], **params))

        combine_chunks = []
        combine_op = op.copy().reset_key()
        combine_op.stage = OperandStage.combine
        params = dict(shape=(), index=(0,), dtype=np.dtype(np.int64))
        combine_chunks.append(combine_op.new_chunk(out_chunks, **params))

        new_op = op.copy()
        return new_op.new_scalars(
            op.inputs, chunks=combine_chunks, dtype=np.dtype(np.int64)
        )

    @classmethod
    def _tile_df_shuffle(cls, in_df: "pd.DataFrame", op: "DataFrameNunique"):
        map_output_types = (
            [OutputType.series] if op.axis == 0 else [OutputType.dataframe]
        )
        # generate map chunks
        map_chunks = []
        for chunk in in_df.chunks:
            map_op = op.copy().reset_key()
            map_op.stage = OperandStage.map
            map_op.shuffle_size = len(in_df.chunks)
            map_op._output_types = map_output_types
            map_shape = (np.nan,) if op.axis == 0 else (np.nan, np.nan)
            chunk_inputs = [chunk]
            map_chunks.append(
                map_op.new_chunk(
                    chunk_inputs,
                    shape=map_shape,
                    index=chunk.index,
                )
            )
        proxy_chunk = DataFrameShuffleProxy(output_types=map_output_types).new_chunk(
            map_chunks, shape=()
        )

        # generate reduce chunks
        reduce_chunks = []
        for chunk in in_df.chunks:
            reduce_op = op.copy().reset_key()
            reduce_op._output_types = [OutputType.dataframe]
            reduce_op.stage = OperandStage.reduce
            reduce_op.n_reducers = len(in_df.chunks)
            reduce_chunks.append(
                reduce_op.new_chunk(
                    [proxy_chunk], shape=(np.nan, np.nan), index=chunk.index
                )
            )

        out_chunks = []
        for chunk in reduce_chunks:
            new_op = op.copy().reset_key()
            new_shape = (np.nan,)

            params = dict(
                shape=new_shape,
                index=(chunk.index[0],),
                dtype=np.dtype(np.int64),
                index_value=parse_index(None),
            )
            out_chunks.append(new_op.new_chunk([chunk], **params))

        combine_chunks = []
        combine_op = op.copy().reset_key()
        combine_shape = (np.nan,)
        combine_op.stage = OperandStage.combine
        params = dict(
            shape=combine_shape,
            index=(0,),
            dtype=np.dtype(np.int64),
            index_value=parse_index(None),
        )
        combine_chunks.append(combine_op.new_chunk(out_chunks, **params))

        new_op = op.copy()
        params = op.outputs[0].params.copy()
        params["nsplits"] = ((np.nan,) * len(combine_chunks),)
        params["chunks"] = combine_chunks
        return new_op.new_seriess(op.inputs, **params)

    @classmethod
    def _tile_shuffle(
        cls, in_df: Union["pd.DataFrame", "pd.Series"], op: "DataFrameNunique"
    ):
        if in_df.ndim == 1:
            return cls._tile_series_shuffle(in_df, op)
        else:
            return cls._tile_df_shuffle(in_df, op)

    @classmethod
    def _tile_auto(
        cls, in_df: Union["pd.DataFrame", "pd.Series"], op: "DataFrameNunique"
    ):
        output_type, shape, _ = cls._gen_chunk_params(in_df)

        ctx = get_context()
        combine_size = op.combine_size
        size_recorder_name = str(uuid.uuid4())
        size_recorder = ctx.create_remote_object(size_recorder_name, SizeRecorder)

        # collect the first pre chunk, run it get the size before and after pre stage
        pre_chunks = cls._gen_tree_pre_chunks(
            in_df.chunks[:combine_size], output_type, shape, op
        )
        for c in pre_chunks:
            c.op.size_recorder_name = size_recorder_name

        # yield chunks with source chunks to avoid submitting chunks repeatedly issue
        yield in_df.chunks[:combine_size] + pre_chunks

        raw_size, agg_size = size_recorder.get()
        # destroy size recorder
        ctx.destroy_remote_object(size_recorder_name)

        # here's a simple rule for deciding tree or shuffle, may improve it later
        if sum(agg_size) / sum(raw_size) > 0.1:
            return cls._tile_shuffle(in_df, op)
        else:
            return cls._tile_tree(in_df, op)

    @classmethod
    def tile(cls, op: "DataFrameNunique"):
        in_df = op.inputs[0]
        if in_df.ndim == 2:
            in_df = build_concatenated_rows_frame(in_df)
        if len(in_df.chunks) == 1:
            return cls.tile_one_chunk(in_df, op)
        if op.method == "auto":
            if len(in_df.chunks) <= op.combine_size:
                return cls._tile_tree(in_df, op)
            else:
                return (yield from cls._tile_auto(in_df, op))
        elif op.method == "tree":
            return cls._tile_tree(in_df, op)
        else:
            return cls._tile_shuffle(in_df, op)

    @classmethod
    def execute_map(cls, ctx: Union[dict, Context], op: "DataFrameNunique"):
        chunk = op.outputs[0]
        input = ctx[op.inputs[0].key]

        if input.ndim == 1:
            data = input.drop_duplicates()
            filters = hash_dataframe_on(data, data.index, op.shuffle_size)
            for index_idx, index_filter in enumerate(filters):
                reducer_index = (index_idx,)
                ctx[chunk.key, reducer_index] = (
                    ctx.get_current_chunk().index,
                    data.iloc[index_filter],
                )
        else:
            if op.axis == 0:
                reducer_index_to_series_list = defaultdict(list)
                for col, col_data in input.items():
                    drop_duplicates_data = col_data.drop_duplicates()
                    filters = hash_dataframe_on(
                        drop_duplicates_data,
                        drop_duplicates_data.index,
                        op.shuffle_size,
                    )
                    for index_idx, index_filter in enumerate(filters):
                        reducer_index = (index_idx, chunk.index[1])
                        part_data = drop_duplicates_data.iloc[index_filter]
                        reducer_index_to_series_list[reducer_index].append(part_data)
                for reducer_index, data in reducer_index_to_series_list.items():
                    ctx[chunk.key, reducer_index] = (
                        ctx.get_current_chunk().index,
                        data,
                    )

            else:
                unique_input = cls._drop_duplicates_by_row(input)
                # df that contains list element cannot be hashed,
                # so here we just mock a ``filters`` object,
                # since how to split is not important in ``axis=1`` case.
                mock_hash_series = pd.Series(range(len(unique_input)))
                mock_hash_series.index = unique_input.index
                idx_to_grouped = pd.RangeIndex(0, len(unique_input)).groupby(
                    mock_hash_series % op.shuffle_size
                )
                filters = [
                    idx_to_grouped.get(i, pd.Index([])) for i in range(op.shuffle_size)
                ]
                for index_idx, index_filter in enumerate(filters):
                    reducer_index = (index_idx, chunk.index[1])
                    ctx[chunk.key, reducer_index] = (
                        ctx.get_current_chunk().index,
                        unique_input.iloc[index_filter],
                    )

    @classmethod
    def execute_combine(cls, ctx: Union[dict, Context], op: "DataFrameNunique"):
        out = op.outputs[0]
        inputs = [ctx[i.key] for i in op.inputs]

        data = inputs[0]
        if isinstance(data, int):
            ctx[out.key] = sum(inputs)
        else:
            for s in inputs[1:]:
                data = data.add(s, fill_value=0)
            ctx[out.key] = data.astype(np.int64)

    @classmethod
    def execute_reduce(cls, ctx: Union[dict, Context], op: "DataFrameNunique"):
        chunk = op.outputs[0]
        input_idx_to_df = dict(op.iter_mapper_data(ctx))

        row_idxes = sorted(input_idx_to_df.keys())

        if op.axis is None or op.axis == 1:
            res = []
            for row_idx in row_idxes:
                row = input_idx_to_df.get(row_idx, None)
                if row is not None:
                    res.append(row)

            data = pd.concat(res, axis=0)
            ctx[chunk.key] = data.drop_duplicates() if op.axis is None else data
        else:
            column_to_series = {}
            for row_idx in row_idxes:
                series_list = input_idx_to_df.get(row_idx, None)
                if series_list is not None:
                    for series in series_list:
                        series_data = series.reset_index(drop=True)
                        if len(series_data) == 0:
                            continue
                        if series.name not in column_to_series and len(series_data) > 0:
                            column_to_series[series.name] = [series_data]
                        else:
                            column_to_series[series.name].append(series_data)

            if column_to_series:
                column_to_na_values = {}
                for name in column_to_series.keys():
                    col_value = (
                        pd.concat(column_to_series[name], axis=0)
                        .reset_index(drop=True)
                        .drop_duplicates()
                    )
                    col_value_has_na: bool = col_value.isnull().any()
                    if not col_value_has_na:
                        column_to_na_values[name] = col_value[0]
                    column_to_series[name] = col_value
                df = pd.concat(column_to_series.values(), axis=1).fillna(
                    value=column_to_na_values
                )
                ctx[chunk.key] = df
            else:
                ctx[chunk.key] = pd.DataFrame()

    @classmethod
    def _drop_duplicates_by_column(cls, df: "pd.DataFrame") -> "pd.DataFrame":
        data = []
        for col, col_data in df.items():
            data.append(col_data.drop_duplicates().tolist())
        res = pd.DataFrame([data])
        res.columns = df.columns
        return res

    @classmethod
    def _drop_duplicates_by_row(cls, df: "pd.DataFrame") -> "pd.DataFrame":
        res = pd.DataFrame(columns=[0])
        for i, row in df.iterrows():
            data = row.drop_duplicates().tolist()
            res.loc[i] = [data]
        return res

    @classmethod
    def execute_pre(cls, ctx: Union[dict, Context], op: "DataFrameNunique"):
        input = ctx[op.inputs[0].key]
        chunk = op.outputs[0]
        if input.ndim == 1:
            res = input.drop_duplicates()
            ctx[chunk.key] = res
        else:
            if op.axis == 0:
                res = cls._drop_duplicates_by_column(input)
                ctx[chunk.key] = res
            else:
                res = cls._drop_duplicates_by_row(input)
                ctx[chunk.key] = res

        if getattr(op, "size_recorder_name", None) is not None:
            # record_size
            raw_size = estimate_pandas_size(input)
            # when agg by a list of methods, agg_size should be sum
            agg_size = estimate_pandas_size(res)
            size_recorder = ctx.get_remote_object(op.size_recorder_name)
            size_recorder.record(raw_size, agg_size)

    @classmethod
    def execute_agg(cls, ctx: Union[dict, Context], op: "DataFrameNunique"):
        inputs = [ctx[inp.key] for inp in op.inputs]
        data = pd.concat(inputs, axis=0)
        if op.axis is None:
            res = data.drop_duplicates()
            ctx[op.outputs[0].key] = res
        else:
            if op.axis == 0:
                res = []
                for col, col_data in data.items():
                    res.append(col_data.explode().drop_duplicates().tolist())
                res = pd.DataFrame([res])
                res.columns = data.columns
                ctx[op.outputs[0].key] = res
            else:
                res = pd.DataFrame(columns=[0])
                for i, row in data.iterrows():
                    res.loc[i] = [row.explode().drop_duplicates().tolist()]
                ctx[op.outputs[0].key] = res

    @classmethod
    def _gen_post_result(
        cls, input: "pd.DataFrame", op: "DataFrameNunique"
    ) -> "pd.Series":
        res = []
        for i, row in input.iterrows():
            res.append(row.explode().nunique(dropna=op.dropna))
        res = pd.Series(res)
        res.index = input.index
        return res

    @classmethod
    def execute_post(cls, ctx: Union[dict, Context], op: "DataFrameNunique"):
        input = ctx[op.inputs[0].key]
        if input.ndim == 1:
            ctx[op.outputs[0].key] = input.nunique(dropna=op.dropna)
        else:
            if op.axis == 0:
                res = []
                for col, col_data in input.items():
                    res.append(col_data.explode().nunique(dropna=op.dropna))
                res = pd.Series(res)
                res.index = input.columns
                ctx[op.outputs[0].key] = res
            else:
                res = cls._gen_post_result(input, op)
                ctx[op.outputs[0].key] = res

    @classmethod
    def execute(cls, ctx: Union[dict, Context], op: "DataFrameNunique"):
        if op.execution_stage is not None:
            if op.execution_stage == "pre":
                cls.execute_pre(ctx, op)
            elif op.execution_stage == "agg":
                cls.execute_agg(ctx, op)
            else:
                cls.execute_post(ctx, op)
        else:
            if op.stage == OperandStage.map:
                cls.execute_map(ctx, op)
            elif op.stage == OperandStage.reduce:
                cls.execute_reduce(ctx, op)
            elif op.stage == OperandStage.combine:
                cls.execute_combine(ctx, op)
            else:
                chunk = op.outputs[0]
                df = ctx[op.inputs[0].key]
                if df.ndim == 2:
                    if op.axis == 0:
                        ctx[chunk.key] = df.nunique(axis=op.axis, dropna=op.dropna)
                    else:
                        ctx[chunk.key] = (
                            df.nunique(axis=op.axis, dropna=op.dropna)
                            if len(df) == 0
                            else cls._gen_post_result(df, op)
                        )
                else:
                    ctx[chunk.key] = df.nunique(dropna=op.dropna)


def nunique_dataframe(
    df,
    axis: Union[int, str] = 0,
    dropna: bool = True,
    method: str = "auto",
    combine_size: Optional[int] = None,
):
    """
    Count distinct observations over requested axis.

    Return Series with number of distinct observations. Can ignore NaN
    values.

    Parameters
    ----------
    axis : {0 or 'index', 1 or 'columns'}, default 0
        The axis to use. 0 or 'index' for row-wise, 1 or 'columns' for
        column-wise.
    dropna : bool, default True
        Don't include NaN in the counts.
    method : str, default 'auto'
        execute via tree or shuffle way.
    combine_size : int, default None
        combine chunk sizes when executing via tree way.

    Returns
    -------
    Series

    See Also
    --------
    Series.nunique: Method nunique for Series.
    DataFrame.count: Count non-NA cells for each column or row.

    Examples
    --------
    >>> import mars.dataframe as md
    >>> df = md.DataFrame({'A': [1, 2, 3], 'B': [1, 1, 1]})
    >>> df.nunique().execute()
    A    3
    B    1
    dtype: int64

    >>> df.nunique(axis=1).execute()
    0    1
    1    2
    2    2
    dtype: int64
    """
    if not (axis in (0, "index", 1, "columns")):
        raise ValueError(f"No axis named {axis} for object type DataFrame")
    if method not in ("tree", "shuffle", "auto"):
        raise ValueError(f"the method input `{method}` is not allowed.")
    if axis == "index":
        axis = 0
    elif axis == "columns":
        axis = 1

    if combine_size is None:
        combine_size = options.combine_size

    op = DataFrameNunique(
        axis=axis, dropna=dropna, method=method, combine_size=combine_size
    )
    return op(df)


def nunique_series(
    series,
    dropna: bool = True,
    method: str = "auto",
    combine_size: Optional[int] = None,
):
    """
    Return number of unique elements in the object.

    Excludes NA values by default.

    Parameters
    ----------
    dropna : bool, default True
        Don't include NaN in the count.
    method : str, default 'auto'
        execute via tree or shuffle way.
    combine_size : int, default None
        combine chunk sizes when executing via tree way.

    Returns
    -------
    int

    See Also
    --------
    DataFrame.nunique: Method nunique for DataFrame.
    Series.count: Count non-NA/null observations in the Series.

    Examples
    --------
    >>> import mars.dataframe as md
    >>> s = md.Series([1, 3, 5, 7, 7])
    >>> s.execute()
    0    1
    1    3
    2    5
    3    7
    4    7
    dtype: int64

    >>> s.nunique().execute()
    4
    """
    if method not in ("tree", "shuffle", "auto"):
        raise ValueError(f"the method input `{method}` is not allowed.")

    if combine_size is None:
        combine_size = options.combine_size

    op = DataFrameNunique(
        axis=None, dropna=dropna, method=method, combine_size=combine_size
    )
    return op(series)

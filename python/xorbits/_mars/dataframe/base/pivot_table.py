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
import itertools
from typing import Any, Union

import numpy as np
import pandas as pd

from ... import opcodes
from ...core import OutputType, recursive_tile
from ...core.context import Context
from ...core.operand import MapReduceOperand, OperandStage
from ...core.custom_log import redirect_custom_log
from ...serialization.serializables import AnyField, BoolField, Int32Field, StringField, ListField
from ...utils import enter_current_session, get_func_token, quiet_stdio
from ..operands import DataFrameOperandMixin, DataFrameShuffleProxy
from ..utils import (
    build_concatenated_rows_frame,
    hash_dataframe_on,
    parse_index,
    build_empty_df,
)


class DataFramePivotTable(MapReduceOperand, DataFrameOperandMixin):
    _op_type_ = opcodes.PIVOT_TABLE

    values = AnyField("values", default=None)
    index = AnyField("index", default=None)
    columns = AnyField("columns", default=None)
    aggfunc = AnyField("aggfunc", default="mean")
    fill_value = AnyField("fill_value", default=None)
    margins = BoolField("margins", default=False)
    dropna = BoolField("drapna", default=True)
    margins_name = StringField("margins_name", default="All")
    observed = BoolField("observed", default=False)
    sort = BoolField("sort", default=True)
    shuffle_size = Int32Field("shuffle_size")
    total_dtypes = ListField("total_dtypes", default=None)

    @classmethod
    def execute_map(cls, ctx: Union[dict, Context], op: "DataFramePivotTable"):
        chunk = op.outputs[0]
        df = ctx[op.inputs[0].key]
        filters = hash_dataframe_on(df, op.index, op.shuffle_size)
        
        for index_idx, index_filter in enumerate(filters):
            reducer_index = (index_idx, chunk.index[1])
            ctx[chunk.key, reducer_index] = (
                ctx.get_current_chunk().index,
                df.iloc[index_filter],
            )

    @classmethod
    def execute_reduce(cls, ctx: Union[dict, Context], op: "DataFramePivotTable"):
        chunk = op.outputs[0]
        input_idx_to_df = dict(op.iter_mapper_data(ctx))
        row_idxes = sorted(input_idx_to_df.keys())

        res = []
        for row_idx in row_idxes:
            row_df = input_idx_to_df.get(row_idx, None)
            if row_df is not None:
                res.append(row_df)

        ctx[chunk.key] = pd.concat(res, axis=0)

    @classmethod
    def execute_combine(cls, ctx: Union[dict, Context], op: "DataFramePivotTable"):
        input_data = ctx[op.inputs[0].key]
        out = op.outputs[0]
        print(input_data.shape)
        for dtype in op.total_dtypes:
            if dtype not in input_data.dtypes.index:
                input_data[dtype] = np.nan
        print(input_data.shape)
        ctx[out.key] = input_data

    @classmethod
    @redirect_custom_log
    @enter_current_session
    def execute(cls, ctx: Union[dict, Context], op: "DataFramePivotTable"):
        if op.stage == OperandStage.map:
            cls.execute_map(ctx, op)
        elif op.stage == OperandStage.reduce:
            cls.execute_reduce(ctx, op)
        elif op.stage == OperandStage.combine:
            print("execute combine")
            cls.execute_combine(ctx, op)
        else:
            input_data = ctx[op.inputs[0].key]
            out = op.outputs[0]
            result = input_data.pivot_table(
                values=op.values,
                index=op.index,
                columns=op.columns,
                aggfunc=op.aggfunc,
                fill_value=op.fill_value,
                margins=op.margins,
                dropna=op.dropna,
                observed=op.observed,
                sort=op.sort,
            )
            ctx[out.key] = result

    @classmethod
    def tile(cls, op: "DataFramePivotTable"):
        in_df = build_concatenated_rows_frame(op.inputs[0])
        out_df = op.outputs[0]
        output_type = OutputType.dataframe
        chunk_shape = in_df.chunk_shape

        # generate map chunks
        map_chunks = []
        for chunk in in_df.chunks:
            map_op = op.copy().reset_key()
            map_op.stage = OperandStage.map
            map_op.shuffle_size = chunk_shape[0]
            map_op._output_types = [output_type]
            chunk_inputs = [chunk]
            map_chunks.append(
                map_op.new_chunk(
                    chunk_inputs,
                    shape=(np.nan, np.nan),
                    index=chunk.index,
                )
            )
        proxy_chunk = DataFrameShuffleProxy(output_types=[output_type]).new_chunk(
            map_chunks, shape=()
        )

        # generate reduce chunks
        reduce_chunks = []
        out_indices = list(itertools.product(*(range(s) for s in chunk_shape)))
        for ordinal, out_idx in enumerate(out_indices):
            reduce_op = op.copy().reset_key()
            reduce_op._output_types = [output_type]
            reduce_op.stage = OperandStage.reduce
            reduce_op.reducer_ordinal = ordinal
            reduce_op.n_reducers = len(out_indices)
            reduce_chunks.append(
                reduce_op.new_chunk(
                    [proxy_chunk], shape=(np.nan, np.nan), index=out_idx
                )
            )

        out_chunks = []
        for chunk in reduce_chunks:
            new_op = op.copy().reset_key()
            new_shape = (np.nan, np.nan)
            params = dict(shape=new_shape, index=chunk.index)
            params.update(
                dict(
                    dtypes=in_df.dtypes,
                    columns_value=in_df.columns_value,
                    index_value=parse_index(None, chunk.key, proxy_chunk.key),
                )
            )
            out_chunks.append(new_op.new_chunk([chunk], **params))
        yield out_chunks 
        filtered_chunks = [chunk for chunk in out_chunks if chunk.shape[1] != 0]
        
        for i, chunk in enumerate(filtered_chunks):
            chunk._index = (i, 0)
    
        total_dtypes = set().union(*[set(chunk.dtypes.index) for chunk in filtered_chunks])
        dtypes_series = pd.concat([chunk.dtypes for chunk in filtered_chunks])
        dtypes_series = dtypes_series.groupby(dtypes_series.index).first()

        out_chunks = []
        for chunk in filtered_chunks:
            combine_op = op.copy().reset_key()
            combine_op.stage = OperandStage.combine
            combine_op.total_dtypes = total_dtypes
            params = dict(shape=(chunk.shape[0], len(total_dtypes)), index=chunk.index)
            params.update(
                dict(
                    dtypes=dtypes_series,
                    columns_value=in_df.columns_value,
                    index_value=chunk.index_value,
                )
            )
            out_chunks.append(combine_op.new_chunk([chunk], **params))

        new_op = op.copy()
        kw = out_df.params.copy()
        new_nsplits = (tuple(chunk.shape[0] for chunk in out_chunks), (2, ))
        kw.update(dict(chunks=out_chunks, nsplits=new_nsplits))

        return new_op.new_tileables(op.inputs, **kw)

    def __call__(self, df):
        return self.new_dataframe([df])


def df_pivot_table(
    df: pd.DataFrame,
    values: Any = None,
    index: Any = None,
    columns: Any = None,
    aggfunc="mean",
    fill_value=None,
    margins: bool = False,
    dropna: bool = True,
    margins_name="All",
    observed: bool = False,
    sort: bool = True,
) -> pd.DataFrame:
    """
    Create a spreadsheet-style pivot table as a DataFrame.

    The levels in the pivot table will be stored in MultiIndex objects
    (hierarchical indexes) on the index and columns of the result DataFrame.

    Parameters
    ----------
    values : list-like or scalar, optional
        Column or columns to aggregate.
    index : column, Grouper, array, or list of the previous
        If an array is passed, it must be the same length as the data. The
        list can contain any of the other types (except list).
        Keys to group by on the pivot table index.  If an array is passed,
        it is being used as the same manner as column values.
    columns : column, Grouper, array, or list of the previous
        If an array is passed, it must be the same length as the data. The
        list can contain any of the other types (except list).
        Keys to group by on the pivot table column.  If an array is passed,
        it is being used as the same manner as column values.
    aggfunc : function, list of functions, dict, default numpy.mean
        If list of functions passed, the resulting pivot table will have
        hierarchical columns whose top level are the function names
        (inferred from the function objects themselves)
        If dict is passed, the key is column to aggregate and value
        is function or list of functions. If ``margin=True``,
        aggfunc will be used to calculate the partial aggregates.
    fill_value : scalar, default None
        Value to replace missing values with (in the resulting pivot table,
        after aggregation).
    margins : bool, default False
        If ``margins=True``, special ``All`` columns and rows
        will be added with partial group aggregates across the categories
        on the rows and columns.
    dropna : bool, default True
        Do not include columns whose entries are all NaN. If True,
        rows with a NaN value in any column will be omitted before
        computing margins.
    margins_name : str, default 'All'
        Name of the row / column that will contain the totals
        when margins is True.
    observed : bool, default False
        This only applies if any of the groupers are Categoricals.
        If True: only show observed values for categorical groupers.
        If False: show all values for categorical groupers.

    sort : bool, default True
        Specifies if the result should be sorted.

        .. versionadded:: 1.3.0

    Returns
    -------
    DataFrame
        An Excel style pivot table.

    See Also
    --------
    DataFrame.pivot : Pivot without aggregation that can handle
        non-numeric data.
    DataFrame.melt: Unpivot a DataFrame from wide to long format,
        optionally leaving identifiers set.
    wide_to_long : Wide panel to long format. Less flexible but more
        user-friendly than melt.

    Notes
    -----
    Reference :ref:`the user guide <reshaping.pivot>` for more examples.

    Examples
    --------
    >>> df = pd.DataFrame({"A": ["foo", "foo", "foo", "foo", "foo",
    ...                          "bar", "bar", "bar", "bar"],
    ...                    "B": ["one", "one", "one", "two", "two",
    ...                          "one", "one", "two", "two"],
    ...                    "C": ["small", "large", "large", "small",
    ...                          "small", "large", "small", "small",
    ...                          "large"],
    ...                    "D": [1, 2, 2, 3, 3, 4, 5, 6, 7],
    ...                    "E": [2, 4, 5, 5, 6, 6, 8, 9, 9]})
    >>> df
            A    B      C  D  E
    0  foo  one  small  1  2
    1  foo  one  large  2  4
    2  foo  one  large  2  5
    3  foo  two  small  3  5
    4  foo  two  small  3  6
    5  bar  one  large  4  6
    6  bar  one  small  5  8
    7  bar  two  small  6  9
    8  bar  two  large  7  9

    This first example aggregates values by taking the sum.

    >>> table = pd.pivot_table(df, values='D', index=['A', 'B'],
    ...                        columns=['C'], aggfunc=np.sum)
    >>> table
    C        large  small
    A   B
    bar one    4.0    5.0
        two    7.0    6.0
    foo one    4.0    1.0
        two    NaN    6.0

    We can also fill missing values using the `fill_value` parameter.

    >>> table = pd.pivot_table(df, values='D', index=['A', 'B'],
    ...                        columns=['C'], aggfunc=np.sum, fill_value=0)
    >>> table
    C        large  small
    A   B
    bar one      4      5
        two      7      6
    foo one      4      1
        two      0      6

    The next example aggregates by taking the mean across multiple columns.

    >>> table = pd.pivot_table(df, values=['D', 'E'], index=['A', 'C'],
    ...                        aggfunc={'D': np.mean, 'E': np.mean})
    >>> table
                    D         E
    A   C
    bar large  5.500000  7.500000
        small  5.500000  8.500000
    foo large  2.000000  4.500000
        small  2.333333  4.333333

    We can also calculate multiple types of aggregations for any given
    value column.

    >>> table = pd.pivot_table(df, values=['D', 'E'], index=['A', 'C'],
    ...                        aggfunc={'D': np.mean,
    ...                                 'E': [min, max, np.mean]})
    >>> table
                        D   E
                    mean max      mean  min
    A   C
    bar large  5.500000   9  7.500000    6
        small  5.500000   9  8.500000    8
    foo large  2.000000   5  4.500000    4
        small  2.333333   6  4.333333    2
    """
    if margins:
        raise NotImplementedError("margins is not supported yet in xorbits.")
    
    op = DataFramePivotTable(
        values=values,
        index=index,
        columns=columns,
        aggfunc=aggfunc,
        fill_value=fill_value,
        margins=margins,
        dropna=dropna,
        margins_name=margins_name,
        observed=observed,
        sort=sort,
    )
    return op(df)

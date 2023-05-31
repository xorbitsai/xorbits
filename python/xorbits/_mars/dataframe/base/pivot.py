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
from ...core import OutputType
from ...core.context import Context
from ...core.operand import MapReduceOperand, OperandStage
from ...serialization.serializables import AnyField, Int32Field, ListField
from ..operands import DataFrameOperandMixin, DataFrameShuffleProxy
from ..utils import build_concatenated_rows_frame, hash_dataframe_on, parse_index


class DataFramePivot(MapReduceOperand, DataFrameOperandMixin):
    _op_type_ = opcodes.PIVOT

    columns = AnyField("columns", default=None)
    index = AnyField("index", default=None)
    values = AnyField("values", default=None)
    shuffle_size = Int32Field("shuffle_size")
    output_columns = ListField("output_columns", default=None)

    @classmethod
    def execute_map(cls, ctx: Union[dict, Context], op: "DataFramePivot"):
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
    def execute_reduce(cls, ctx: Union[dict, Context], op: "DataFramePivot"):
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
    def execute_combine(cls, ctx: Union[dict, Context], op: "DataFramePivot"):
        input_data = ctx[op.inputs[0].key]
        out = op.outputs[0]

        for dtype in op.output_columns:
            if dtype not in input_data.dtypes.index:
                input_data[dtype] = np.nan

        ctx[out.key] = input_data

    @classmethod
    def execute(cls, ctx: Union[dict, Context], op: "DataFramePivot"):
        if op.stage == OperandStage.map:
            cls.execute_map(ctx, op)
        elif op.stage == OperandStage.reduce:
            cls.execute_reduce(ctx, op)
        elif op.stage == OperandStage.combine:
            cls.execute_combine(ctx, op)
        else:
            input_data = ctx[op.inputs[0].key]
            out = op.outputs[0]
            kwargs = {}
            if op.index is not None:
                kwargs["index"] = op.index
            if op.values is not None:
                kwargs["values"] = op.values
            result = input_data.pivot(columns=op.columns, **kwargs)
            ctx[out.key] = result

    @classmethod
    def tile_one_chunk(cls, op: "DataFramePivot"):
        in_df = op.inputs[0]
        out_df = op.outputs[0]

        chunks = []
        for c in in_df.chunks:
            new_op = op.copy().reset_key()
            new_op.tileable_op_key = op.key
            chunks.append(
                new_op.new_chunk(
                    [c],
                    shape=(np.nan, np.nan),
                    index=c.index,
                    dtypes=out_df.dtypes,
                    index_value=c.index_value,
                    columns_value=c.columns_value,
                )
            )

        new_op = op.copy()
        kw = out_df.params.copy()
        new_nsplits = ((np.nan,), (np.nan,))
        kw.update(dict(chunks=chunks, nsplits=new_nsplits))
        return new_op.new_tileables(op.inputs, **kw)

    @classmethod
    def tile(cls, op: "DataFramePivot"):
        in_df = build_concatenated_rows_frame(op.inputs[0])

        if len(in_df.chunks) == 1:
            return cls.tile_one_chunk(op)

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

        combined_dtypes = pd.concat(
            [chunk.dtypes for chunk in filtered_chunks]
        ).to_dict()
        output_columns = list(combined_dtypes.keys())
        output_dtypes = pd.Series(combined_dtypes)

        # generate combine chunks
        combine_chunks = []
        for chunk in filtered_chunks:
            if chunk.shape[1] < len(output_columns):
                combine_op = op.copy().reset_key()
                combine_op.stage = OperandStage.combine
                combine_op.output_columns = output_columns
                params = dict(
                    shape=(chunk.shape[0], len(output_columns)), index=chunk.index
                )
                params.update(
                    dict(
                        dtypes=output_dtypes,
                        columns_value=in_df.columns_value,
                        index_value=chunk.index_value,
                    )
                )
                combine_chunks.append(combine_op.new_chunk([chunk], **params))
            else:
                combine_chunks.append(chunk)

        new_op = op.copy()
        kw = out_df.params.copy()
        new_nsplits = (
            tuple(chunk.shape[0] for chunk in combine_chunks),
            (len(output_columns),),
        )
        kw.update(dict(chunks=combine_chunks, nsplits=new_nsplits))

        return new_op.new_tileables(op.inputs, **kw)

    def __call__(self, df):
        return self.new_dataframe([df])


def df_pivot(
    df: pd.DataFrame, columns: Any, *, index: Any = None, values: Any = None
) -> pd.DataFrame:
    """
    Return reshaped DataFrame organized by given index / column values.

    Reshape data (produce a "pivot" table) based on column values. Uses
    unique values from specified `index` / `columns` to form axes of the
    resulting DataFrame. This function does not support data
    aggregation, multiple values will result in a MultiIndex in the
    columns. See the :ref:`User Guide <reshaping>` for more on reshaping.

    Parameters
    ----------
    columns : str or object or a list of str
        Column to use to make new frame's columns.

        .. versionchanged:: 1.1.0
            Also accept list of columns names.

    index : str or object or a list of str, optional
        Column to use to make new frame's index. If not given, uses existing index.

        .. versionchanged:: 1.1.0
            Also accept list of index names.

    values : str, object or a list of the previous, optional
        Column(s) to use for populating new frame's values. If not
        specified, all remaining columns will be used and the result will
        have hierarchically indexed columns.

    Returns
    -------
    DataFrame
        Returns reshaped DataFrame.

    Raises
    ------
    ValueError:
        When there are any `index`, `columns` combinations with multiple
        values. `DataFrame.pivot_table` when you need to aggregate.

    See Also
    --------
    DataFrame.pivot_table : Generalization of pivot that can handle
        duplicate values for one index/column pair.
    DataFrame.unstack : Pivot based on the index values instead of a
        column.
    wide_to_long : Wide panel to long format. Less flexible but more
        user-friendly than melt.

    Notes
    -----
    For finer-tuned control, see hierarchical indexing documentation along
    with the related stack/unstack methods.

    Reference :ref:`the user guide <reshaping.pivot>` for more examples.

    Examples
    --------
    >>> df = pd.DataFrame({'foo': ['one', 'one', 'one', 'two', 'two',
    ...                            'two'],
    ...                    'bar': ['A', 'B', 'C', 'A', 'B', 'C'],
    ...                    'baz': [1, 2, 3, 4, 5, 6],
    ...                    'zoo': ['x', 'y', 'z', 'q', 'w', 't']})
    >>> df
        foo   bar  baz  zoo
    0   one   A    1    x
    1   one   B    2    y
    2   one   C    3    z
    3   two   A    4    q
    4   two   B    5    w
    5   two   C    6    t

    >>> df.pivot(index='foo', columns='bar', values='baz')
    bar  A   B   C
    foo
    one  1   2   3
    two  4   5   6

    >>> df.pivot(index='foo', columns='bar')['baz']
    bar  A   B   C
    foo
    one  1   2   3
    two  4   5   6

    >>> df.pivot(index='foo', columns='bar', values=['baz', 'zoo'])
            baz       zoo
    bar   A  B  C   A  B  C
    foo
    one   1  2  3   x  y  z
    two   4  5  6   q  w  t

    You could also assign a list of column names or a list of index names.

    >>> df = pd.DataFrame({
    ...        "lev1": [1, 1, 1, 2, 2, 2],
    ...        "lev2": [1, 1, 2, 1, 1, 2],
    ...        "lev3": [1, 2, 1, 2, 1, 2],
    ...        "lev4": [1, 2, 3, 4, 5, 6],
    ...        "values": [0, 1, 2, 3, 4, 5]})
    >>> df
        lev1 lev2 lev3 lev4 values
    0   1    1    1    1    0
    1   1    1    2    2    1
    2   1    2    1    3    2
    3   2    1    2    4    3
    4   2    1    1    5    4
    5   2    2    2    6    5

    >>> df.pivot(index="lev1", columns=["lev2", "lev3"], values="values")
    lev2    1         2
    lev3    1    2    1    2
    lev1
    1     0.0  1.0  2.0  NaN
    2     4.0  3.0  NaN  5.0

    >>> df.pivot(index=["lev1", "lev2"], columns=["lev3"], values="values")
            lev3    1    2
    lev1  lev2
        1     1  0.0  1.0
                2  2.0  NaN
        2     1  4.0  3.0
                2  NaN  5.0

    A ValueError is raised if there are any duplicates.

    >>> df = pd.DataFrame({"foo": ['one', 'one', 'two', 'two'],
    ...                    "bar": ['A', 'A', 'B', 'C'],
    ...                    "baz": [1, 2, 3, 4]})
    >>> df
        foo bar  baz
    0  one   A    1
    1  one   A    2
    2  two   B    3
    3  two   C    4

    Notice that the first two rows are the same for our `index`
    and `columns` arguments.

    >>> df.pivot(index='foo', columns='bar', values='baz')
    Traceback (most recent call last):
        ...
    ValueError: Index contains duplicate entries, cannot reshape
    """
    op = DataFramePivot(columns=columns, index=index, values=values)
    return op(df)

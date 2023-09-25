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

import numpy as np
import pandas as pd

from ...core import OutputType, get_output_types, recursive_tile
from ...serialization.serializables import DictField, IndexField, StringField
from ..core import IndexValue
from ..operands import DataFrameOperand, DataFrameOperandMixin
from ..utils import build_concatenated_rows_frame, parse_index


class GroupByNthSelector(DataFrameOperand, DataFrameOperandMixin):
    _op_module_ = "dataframe.groupby"

    groupby_params = DictField("groupby_params")
    n = IndexField("n")
    dropna = StringField("dropna", default=None)

    def __call__(self, groupby):
        df = groupby
        while df.op.output_types[0] not in (OutputType.dataframe, OutputType.series):
            df = df.inputs[0]
        selection = groupby.op.groupby_params.pop("selection", None)
        if df.ndim > 1 and selection:
            if isinstance(selection, tuple) and selection not in df.dtypes:
                selection = list(selection)

            result_df = df[selection]
        else:
            result_df = df

        self._output_types = (
            [OutputType.dataframe] if result_df.ndim == 2 else [OutputType.series]
        )
        params = result_df.params
        params["shape"] = (np.nan,) + result_df.shape[1:]
        if isinstance(df.index_value.value, IndexValue.RangeIndex):
            params["index_value"] = parse_index(pd.RangeIndex(-1), df.key)

        return self.new_tileable([df], **params)

    @classmethod
    def tile(cls, op: "GroupByNthSelector"):
        in_df = op.inputs[0]
        groupby_params = op.groupby_params.copy()
        selection = groupby_params.pop("selection", None)
        if len(in_df.shape) > 1:
            in_df = build_concatenated_rows_frame(in_df)
        out_df = op.outputs[0]
        # if there is only one chunk, tile with a single chunk
        if len(in_df.chunks) <= 1:
            new_shape = (np.nan,)
            new_nsplits = ((np.nan,),)
            if out_df.ndim > 1:
                new_shape += (out_df.shape[1],)
                new_nsplits += ((out_df.shape[1],),)
            c = in_df.chunks[0]
            chunk_op = op.copy().reset_key()
            params = out_df.params
            params["shape"] = new_shape
            params["index"] = (0,) * out_df.ndim
            out_chunk = chunk_op.new_chunk([c], **params)

            tileable_op = op.copy().reset_key()
            return tileable_op.new_tileables(
                [in_df], nsplits=new_nsplits, chunks=[out_chunk], **params
            )

        if in_df.ndim > 1 and selection:
            if isinstance(selection, tuple) and selection not in in_df.dtypes:
                selection = list(selection)

            if not isinstance(selection, list):
                pre_selection = [selection]
            else:
                pre_selection = list(selection)

            if isinstance(groupby_params.get("by"), list):
                pre_selection += [
                    el for el in groupby_params["by"] if el not in pre_selection
                ]

            if len(pre_selection) != in_df.shape[1]:
                in_df = yield from recursive_tile(in_df[pre_selection])

        # pre chunks
        pre_chunks = []
        for c in in_df.chunks:
            pre_op = op.copy().reset_key()
            pre_op._output_types = get_output_types(c)
            pre_op.groupby_params = op.groupby_params.copy()
            pre_op.groupby_params.pop("selection", None)
            params = c.params
            params["shape"] = (np.nan,) + c.shape[1:]
            pre_chunks.append(pre_op.new_chunk([c], **params))

        new_op = op.copy().reset_key()
        new_op._output_types = get_output_types(in_df)
        new_nsplits = ((np.nan,) * len(in_df.nsplits[0]),) + in_df.nsplits[1:]
        pre_tiled = new_op.new_tileable(
            [in_df], chunks=pre_chunks, nsplits=new_nsplits, **in_df.params
        )
        # generate groupby
        grouped = yield from recursive_tile(pre_tiled.groupby(**groupby_params))
        if selection:
            grouped = yield from recursive_tile(grouped[selection])

        # generate post chunks
        post_chunks = []
        for c in grouped.chunks:
            post_op = op.copy().reset_key()
            post_op.groupby_params = op.groupby_params.copy()
            post_op.groupby_params.pop("selection", None)
            if op.output_types[0] == OutputType.dataframe:
                index = c.index
            else:
                index = (c.index[0],)
            params = out_df.params
            params["index"] = index
            post_chunks.append(post_op.new_chunk([c], **params))

        new_op = op.copy().reset_key()
        new_nsplits = ((np.nan,) * len(in_df.nsplits[0]),)
        if out_df.ndim > 1:
            new_nsplits += ((out_df.shape[1],),)
        return new_op.new_tileables(
            [in_df], chunks=post_chunks, nsplits=new_nsplits, **out_df.params
        )

    @classmethod
    def execute(cls, ctx, op: "GroupByNthSelector"):
        in_data = ctx[op.inputs[0].key]
        params = op.groupby_params.copy()
        selection = params.pop("selection", None)

        if hasattr(in_data, "groupby"):
            grouped = in_data.groupby(**params)
        else:
            grouped = in_data
        if selection:
            grouped = grouped[selection]
        result = grouped.nth(op.n, op.dropna)
        ctx[op.outputs[0].key] = result


def nth(groupby, n, dropna=None):
    """
    Take the nth row from each group if n is an int, or a subset of rows
    if n is a list of ints.

    If dropna, will take the nth non-null row, dropna is either
    Truthy (if a Series) or 'all', 'any' (if a DataFrame);
    this is equivalent to calling dropna(how=dropna) before the
    groupby.

    Parameters
    ----------
    n : int or list of ints
        a single nth value for the row or a list of nth values
    dropna : None or str, optional
        apply the specified dropna operation before counting which row is
        the nth row. Needs to be None, 'any' or 'all'

    Examples
    --------
    >>> import mars.dataframe as md
    >>> df = md.DataFrame({'A': [1, 1, 2, 1, 2],
    ...                    'B': [np.nan, 2, 3, 4, 5]}, columns=['A', 'B'])
    >>> g = df.groupby('A')
    >>> g.nth(0).execute()
            B
    A
    1  NaN
    2  3.0
    >>> g.nth(1).execute()
            B
    A
    1  2.0
    2  5.0
    >>> g.nth(-1).execute()
            B
    A
    1  4.0
    2  5.0
    >>> g.nth([0, 1]).execute()
            B
    A
    1  NaN
    1  2.0
    2  3.0
    2  5.0

    Specifying ``dropna`` allows count ignoring NaN

    >>> g.nth(0, dropna='any').execute()
            B
    A
    1  2.0
    2  3.0

    NaNs denote group exhausted when using dropna

    >>> g.nth(3, dropna='any').execute()
        B
    A
    1 NaN
    2 NaN

    Specifying ``as_index=False`` in ``groupby`` keeps the original index.

    >>> df.groupby('A', as_index=False).nth(1).execute()
        A    B
    1  1  2.0
    4  2  5.0
    """
    groupby_params = groupby.op.groupby_params.copy()
    groupby_params.pop("as_index", None)
    op = GroupByNthSelector(n=n, dropna=dropna, groupby_params=groupby_params)
    return op(groupby)

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

from collections import OrderedDict

import numpy as np
import pandas as pd

from ... import opcodes
from ...core import OutputType
from ...serialization.serializables import (
    AnyField,
    BoolField,
    DictField,
    Int32Field,
    Int64Field,
    KeyField,
    Serializable,
    StringField,
    TupleField,
)
from ..operands import DataFrameOperand, DataFrameOperandMixin
from ..utils import parse_index, validate_axis

_PAIRWISE_AGG = ["corr", "cov"]


class GroupByRolling(Serializable):
    input = KeyField("input")
    window = AnyField("window")
    min_periods = Int64Field("min_periods")
    center = BoolField("center")
    win_type = StringField("win_type")
    on = StringField("on")
    axis = Int32Field("axis")
    closed = StringField("closed")

    def validate(self):
        """
        leverage pandas itself to validate the parameters.
        """
        groupby = self.input
        groupby.op.build_mock_groupby().rolling(**self.params)

    @property
    def params(self):
        p = OrderedDict()
        args = [
            "window",
            "min_periods",
            "center",
            "win_type",
            "axis",
            "on",
            "closed",
        ]
        for attr in args:
            p[attr] = getattr(self, attr)
        return p

    def aggregate(self, func, *args, **kwargs):
        params = self.params
        if func in _PAIRWISE_AGG:
            # for convenience, since pairwise aggregations are axis irrelevant.
            params["axis"] = 0

        op = GroupbyRollingAgg(func=func, func_args=args, func_kwargs=kwargs, **params)
        return op(self)

    def agg(self, func, *args, **kwargs):
        return self.aggregate(func, *args, **kwargs)

    def count(self):
        return self.aggregate("count")

    def sum(self, *args, **kwargs):
        return self.aggregate("sum", *args, **kwargs)

    def mean(self, *args, **kwargs):
        return self.aggregate("mean", *args, **kwargs)

    def median(self, **kwargs):
        return self.aggregate("median", **kwargs)

    def var(self, ddof=1, *args, **kwargs):
        return self.aggregate("var", ddof=ddof, *args, **kwargs)

    def std(self, ddof=1, *args, **kwargs):
        return self.aggregate("std", ddof=ddof, *args, **kwargs)

    def min(self, *args, **kwargs):
        return self.aggregate("min", *args, **kwargs)

    def max(self, *args, **kwargs):
        return self.aggregate("max", *args, **kwargs)

    def skew(self, **kwargs):
        return self.aggregate("skew", **kwargs)

    def kurt(self, **kwargs):
        return self.aggregate("kurt", **kwargs)

    def corr(self, **kwargs):
        return self.aggregate("corr", **kwargs)

    def cov(self, **kwargs):
        return self.aggregate("cov", **kwargs)


class GroupbyRollingAgg(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = opcodes.GROUPBY_ROLLING_AGG

    input = KeyField("input")
    window = AnyField("window")
    min_periods = Int64Field("min_periods")
    center = BoolField("center")
    win_type = StringField("win_type")
    on = StringField("on")
    axis = Int32Field("axis")
    closed = StringField("closed")
    func = AnyField("func")
    func_args = TupleField("func_args")
    func_kwargs = DictField("func_kwargs")

    def __call__(self: "GroupbyRollingAgg", r: "GroupByRolling"):
        groupby = r.input
        mock_obj = (
            groupby.op.build_mock_groupby()
            .rolling(**r.params)
            .agg(self.func, *self.func_args, **self.func_kwargs)
        )

        index_value = parse_index(mock_obj.index)
        if isinstance(mock_obj, pd.DataFrame):
            return self.new_dataframe(
                [groupby],
                shape=(np.nan, mock_obj.shape[1]),
                dtypes=mock_obj.dtypes,
                index_value=index_value,
                columns_value=parse_index(mock_obj.dtypes.index, store_data=True),
            )
        elif isinstance(mock_obj, pd.Series):
            return self.new_series(
                [groupby],
                shape=(np.nan,),
                dtype=mock_obj.dtype,
                index_value=index_value,
                name=mock_obj.name,
            )

    @classmethod
    def tile(cls, op: "GroupbyRollingAgg"):
        in_groupby = op.inputs[0]
        out_df = op.outputs[0]

        chunks = []
        for c in in_groupby.chunks:
            inp_chunks = [c]

            new_op = op.copy().reset_key()
            new_op.tileable_op_key = op.key
            if op.output_types[0] == OutputType.dataframe:
                chunks.append(
                    new_op.new_chunk(
                        inp_chunks,
                        index=c.index,
                        shape=(np.nan, len(out_df.dtypes)),
                        dtypes=out_df.dtypes,
                        columns_value=out_df.columns_value,
                        index_value=out_df.index_value,
                    )
                )
            else:
                chunks.append(
                    new_op.new_chunk(
                        inp_chunks,
                        name=out_df.name,
                        index=(c.index[0],),
                        shape=(np.nan,),
                        dtype=out_df.dtype,
                        index_value=out_df.index_value,
                    )
                )

        new_op = op.copy()
        kw = out_df.params.copy()
        kw["chunks"] = chunks
        if op.output_types[0] == OutputType.dataframe:
            kw["nsplits"] = ((np.nan,) * len(chunks), (out_df.shape[1],))
        else:
            kw["nsplits"] = ((np.nan,) * len(chunks),)
        return new_op.new_tileable([in_groupby], **kw)

    @classmethod
    def execute(cls, ctx, op: "GroupbyRollingAgg"):
        inp = ctx[op.inputs[0].key]

        r = inp.rolling(
            window=op.window,
            min_periods=op.min_periods,
            center=op.center,
            win_type=op.win_type,
            on=op.on,
            axis=op.axis,
            closed=op.closed,
        )
        result = r.aggregate(op.func, *op.func_args, **op.func_kwargs)

        ctx[op.outputs[0].key] = result


def rolling(
    groupby,
    window,
    min_periods=None,
    center=False,
    win_type=None,
    on=None,
    axis=0,
    closed=None,
) -> GroupByRolling:
    axis = validate_axis(axis, groupby)
    r = GroupByRolling(
        input=groupby,
        window=window,
        min_periods=min_periods,
        center=center,
        win_type=win_type,
        on=on,
        axis=axis,
        closed=closed,
    )
    r.validate()

    return r

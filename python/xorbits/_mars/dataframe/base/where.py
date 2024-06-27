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
from typing import Callable, Union

import cloudpickle
import numpy as np
import pandas as pd

from ... import opcodes
from ...core import OutputType
from ...core.context import Context
from ...core.custom_log import redirect_custom_log
from ...serialization.serializables import AnyField, DictField, StringField
from ...utils import enter_current_session, quiet_stdio
from ..operands import DataFrameOperand, DataFrameOperandMixin
from ..utils import build_df, build_empty_df, parse_index


class DataFrameWhere(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = opcodes.WHERE

    condition = AnyField("condition")
    other = AnyField("other")
    func_token = StringField("func_token", default=None)
    kwds = DictField("kwds", default=None)

    def __init__(self, **kw):
        super().__init__(**kw)

    def _load_condition(self):
        if isinstance(self.condition, bytes):
            return cloudpickle.loads(self.condition)
        else:
            return self.condition

    def _load_other(self):
        if isinstance(self.other, bytes):
            return cloudpickle.loads(self.other)
        else:
            return self.other

    @classmethod
    @redirect_custom_log
    @enter_current_session
    def execute(cls, ctx: Union[dict, Context], op: "DataFrameWhere"):
        condition = op._load_condition()
        other = op._load_other()
        input_data = ctx[op.inputs[0].key]
        out = op.outputs[0]
        if len(input_data) == 0:
            ctx[out.key] = build_empty_df(out.dtypes)

        result = input_data.where(
            condition,
            other=other,
            **op.kwds,
        )
        ctx[out.key] = result

    @classmethod
    def tile(cls, op: "DataFrameWhere"):
        in_df = op.inputs[0]
        out_df = op.outputs[0]
        chunks = []
        for c in in_df.chunks:
            new_shape = c.shape

            new_index_value, new_columns_value = c.index_value, c.columns_value

            new_dtypes = out_df.dtypes

            new_op = op.copy().reset_key()
            new_op.tileable_op_key = op.key
            chunks.append(
                new_op.new_chunk(
                    [c],
                    shape=tuple(new_shape),
                    index=c.index,
                    dtypes=new_dtypes,
                    index_value=new_index_value,
                    columns_value=new_columns_value,
                )
            )
        new_nsplits = list(in_df.nsplits)
        new_op = op.copy()
        kw = out_df.params.copy()
        new_nsplits = tuple(new_nsplits)
        kw.update(dict(chunks=chunks, nsplits=new_nsplits))
        return new_op.new_tileables(op.inputs, **kw)

    def __call__(self, df: pd.DataFrame, skip_infer: bool = False):
        condition = self._load_condition()
        other = self._load_other()

        if skip_infer:
            condition = cloudpickle.dumps(condition)
            other = cloudpickle.dumps(other)
            return self.new_dataframe([df])

        dtypes, index_value = self._infer_df_func_returns(df)

        if index_value is None:
            index_value = parse_index(None, (df.key, df.index_value.key))
        for arg, desc in zip((self.output_types, dtypes), ("output_types", "dtypes")):
            if arg is None:
                raise TypeError(
                    f"Cannot determine {desc} by calculating with enumerate data, "
                    "please specify it as arguments"
                )

        if index_value == "inherit":
            index_value = df.index_value

        shape = df.shape

        condition = cloudpickle.dumps(condition)
        other = cloudpickle.dumps(other)

        return self.new_dataframe(
            [df],
            shape=shape,
            dtypes=dtypes,
            index_value=index_value,
            columns_value=parse_index(dtypes.index, store_data=True),
        )

    def _infer_df_func_returns(self, df: pd.DataFrame):
        condition = self._load_condition()
        other = self._load_other()
        # judge instance of (condition) like if condition is a bool/series/DataFrame
        if isinstance(condition, np.ufunc):
            output_type = OutputType.dataframe
            new_dtypes = None
            index_value = "inherit"
        else:
            output_type = new_dtypes = index_value = None

        try:
            empty_df = build_df(df, size=2)
            with np.errstate(all="ignore"), quiet_stdio():
                infer_df = empty_df.where(
                    condition,
                    other=other,
                    **self.kwds,
                )
            if index_value is None:
                if infer_df.index is empty_df.index:
                    index_value = "inherit"
                else:
                    index_value = parse_index(pd.RangeIndex(-1))

            output_type = output_type or OutputType.dataframe
            new_dtypes = new_dtypes or infer_df.dtypes
        except:
            pass

        self.output_types = (
            [output_type] if not self.output_types else self.output_types
        )
        dtypes = new_dtypes
        return dtypes, index_value


def df_where(
    df: pd.DataFrame,
    cond: Callable,
    other: Union[pd.DataFrame, pd.Series] = np.nan,
    skip_infer: bool = False,
    **kwds,
):
    """
    Apply a condition to a DataFrame elementwise.

    This method applies a condition that returns a boolean value to every element
    of a DataFrame. If the condition is True, the original value is retained; if
    False, the value is replaced with 'other'.

    Parameters
    ----------
    cond : callable
        Python function that returns a boolean value for each element.

    other : DataFrame or Series, default np.nan
        The value to replace elements where the condition is False. This can be
        a DataFrame with the same shape as `df`, or a scalar value.

    skip_infer: bool, default False
        Whether infer dtypes when dtypes or output_type is not specified.

    **kwds
        Additional keyword arguments to pass as keywords arguments to
        `condition`.

    Returns
    -------
    DataFrame
        Transformed DataFrame.

    See Also
    --------
    DataFrame.applymap : Apply a function elementwise to a DataFrame.

    Examples
    --------
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> df
       A  B
    0  1  4
    1  2  5
    2  3  6

    >>> condition = lambda x: x % 2 == 0
    >>> df_where(df, condition)
       A    B
    0  1  NaN
    1  2  5.0
    2  3  NaN

    You can also specify an 'other' value to replace elements that do not satisfy
    the condition:

    >>> df_where(df, condition, other=-1)
       A  B
    0  1 -1
    1  2  5
    2  3 -1

    Note that the 'other' parameter can be a scalar value or a DataFrame with the
    same shape as the input DataFrame `df`.
    """
    inplace = kwds.pop("inplace", None)

    op = DataFrameWhere(condition=cond, other=other, kwds=kwds)
    new_df = op(df, skip_infer=skip_infer)

    if inplace:
        df.data = new_df.data
    else:
        return new_df

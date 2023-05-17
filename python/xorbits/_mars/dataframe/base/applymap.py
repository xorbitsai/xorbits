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
from ...utils import enter_current_session, get_func_token, quiet_stdio
from ..operands import DataFrameOperand, DataFrameOperandMixin
from ..utils import build_df, build_empty_df, parse_index


class DataFrameApplymap(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = opcodes.APPLYMAP

    func = AnyField("func")
    na_action = StringField("na_action", default=None)
    func_token = StringField("func_token", default=None)
    kwds = DictField("kwds", default=None)

    def __init__(self, **kw):
        super().__init__(**kw)

    def _load_func(self):
        if isinstance(self.func, bytes):
            return cloudpickle.loads(self.func)
        else:
            return self.func

    @classmethod
    @redirect_custom_log
    @enter_current_session
    def execute(cls, ctx: Union[dict, Context], op: "DataFrameApplymap"):
        func = op._load_func()
        input_data = ctx[op.inputs[0].key]
        out = op.outputs[0]
        if len(input_data) == 0:
            ctx[out.key] = build_empty_df(out.dtypes)

        result = input_data.applymap(
            func,
            na_action=op.na_action,
            **op.kwds,
        )
        ctx[out.key] = result

    @classmethod
    def tile(cls, op: "DataFrameApplymap"):
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

    def _infer_df_func_returns(self, df: pd.DataFrame):
        func = self._load_func()
        if isinstance(func, np.ufunc):
            output_type = OutputType.dataframe
            new_dtypes = None
            index_value = "inherit"
        else:
            output_type = new_dtypes = index_value = None

        try:
            empty_df = build_df(df, size=2)
            with np.errstate(all="ignore"), quiet_stdio():
                infer_df = empty_df.applymap(
                    func,
                    na_action=self.na_action,
                    **self.kwds,
                )
            if index_value is None:
                if infer_df.index is empty_df.index:
                    index_value = "inherit"
                else:  # pragma: no cover
                    index_value = parse_index(pd.RangeIndex(-1))

            output_type = output_type or OutputType.dataframe
            new_dtypes = new_dtypes or infer_df.dtypes
        except:  # noqa: E722  # nosec
            pass

        self.output_types = (
            [output_type] if not self.output_types else self.output_types
        )
        dtypes = new_dtypes
        return dtypes, index_value

    def __call__(self, df: pd.DataFrame, skip_infer: bool = False):
        self.func_token = get_func_token(self.func)
        if skip_infer:
            self.func = cloudpickle.dumps(self.func)
            return self.new_dataframe([df])

        # for backward compatibility
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

        self.func = cloudpickle.dumps(self.func)

        return self.new_dataframe(
            [df],
            shape=shape,
            dtypes=dtypes,
            index_value=index_value,
            columns_value=parse_index(dtypes.index, store_data=True),
        )


def df_applymap(
    df: pd.DataFrame,
    func: Callable,
    na_action: str = None,
    skip_infer: bool = False,
    **kwds,
):
    """
    Apply a function to a Dataframe elementwise.

    This method applies a function that accepts and returns a scalar
    to every element of a DataFrame.

    Parameters
    ----------
    func : callable
        Python function, returns a single value from a single value.

    na_action : {None, 'ignore'}, default None
        If 'ignore', propagate NaN values, without passing them to func.

    skip_infer: bool, default False
        Whether infer dtypes when dtypes or output_type is not specified.

    **kwds
        Additional keyword arguments to pass as keywords arguments to
        `func`.

    Returns
    -------
    DataFrame
        Transformed DataFrame.

    See Also
    --------
    DataFrame.apply : Apply a function along input axis of DataFrame.

    Examples
    --------
    >>> df = pd.DataFrame([[1, 2.12], [3.356, 4.567]])
    >>> df
            0      1
    0  1.000  2.120
    1  3.356  4.567

    >>> df.applymap(lambda x: len(str(x)))
        0  1
    0  3  4
    1  5  5

    Like Series.map, NA values can be ignored:

    >>> df_copy = df.copy()
    >>> df_copy.iloc[0, 0] = pd.NA
    >>> df_copy.applymap(lambda x: len(str(x)), na_action='ignore')
            0  1
    0  NaN  4
    1  5.0  5

    Note that a vectorized version of `func` often exists, which will
    be much faster. You could square each number elementwise.

    >>> df.applymap(lambda x: x**2)
                0          1
    0   1.000000   4.494400
    1  11.262736  20.857489

    But it's better to avoid applymap in that case.

    >>> df ** 2
                0          1
    0   1.000000   4.494400
    1  11.262736  20.857489
    """
    if na_action not in {"ignore", None}:
        raise ValueError(f"na_action must be 'ignore' or None. Got {repr(na_action)}")

    op = DataFrameApplymap(func=func, na_action=na_action, kwds=kwds)
    return op(df, skip_infer=skip_infer)

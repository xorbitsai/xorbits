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

import inspect

import cloudpickle
import numpy as np
import pandas as pd

from ... import opcodes
from ...config import options
from ...core import OutputType, recursive_tile
from ...core.custom_log import redirect_custom_log
from ...core.operand import OperatorLogicKeyGeneratorMixin
from ...serialization.serializables import (
    AnyField,
    BoolField,
    DictField,
    StringField,
    TupleField,
)
from ...utils import enter_current_session, get_func_token, quiet_stdio
from ..arrays import ArrowArray
from ..operands import DataFrameOperand, DataFrameOperandMixin
from ..utils import (
    build_df,
    build_empty_df,
    build_empty_series,
    build_series,
    make_dtype,
    make_dtypes,
    parse_index,
    validate_axis,
    validate_output_types,
)


class ApplyOperandLogicKeyGeneratorMixin(OperatorLogicKeyGeneratorMixin):
    def _get_logic_key_token_values(self):
        return super()._get_logic_key_token_values() + [
            self.convert_dtype,
            self.result_type,
            self.func_token,
        ]


class ApplymapOperand(
    DataFrameOperand, DataFrameOperandMixin, ApplyOperandLogicKeyGeneratorMixin
):
    _op_type_ = opcodes.APPLYMAP

    func = AnyField("func")
    func_token = StringField("func_token", default=None)
    convert_dtype = BoolField("convert_dtype", default=None)
    result_type = StringField("result_type", default=None)
    logic_key = StringField("logic_key", default=None)
    args = TupleField("args", default=None)
    kwds = DictField("kwds", default=None)

    def __init__(self, output_types=None, **kw):
        super().__init__(_output_types=output_types, **kw)

    def _load_func(self):
        if isinstance(self.func, bytes):
            return cloudpickle.loads(self.func)
        else:
            return self.func

    @classmethod
    @redirect_custom_log
    @enter_current_session
    def execute(cls, ctx, op):
        func = op._load_func()
        input_data = ctx[op.inputs[0].key]
        out = op.outputs[0]
        if len(input_data) == 0:
            if op.output_types[0] == OutputType.dataframe:
                ctx[out.key] = build_empty_df(out.dtypes)

        if isinstance(input_data, pd.DataFrame):
            result = input_data.applymap(
                func,
                result_type=op.result_type,
                args=op.args,
                **op.kwds,
            )
        ctx[out.key] = result

    @classmethod
    def _tile_df(cls, op):
        in_df = op.inputs[0]
        out_df = op.outputs[0]
        print(type(op))
        chunks = []
        for c in in_df.chunks:
            new_shape = c.shape
            
            new_index_value, new_columns_value = c.index_value, c.columns_value

            new_dtypes = out_df.dtypes

            print(f"new shape: {new_shape}")
            print(f"new index value: {new_index_value}")
            print(f"new columns value: {new_columns_value}")
            print(f"new dtypes: {new_dtypes}")
            
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
        print(chunks)
        new_nsplits = list(in_df.nsplits)
        print(new_nsplits)
        new_op = op.copy()
        kw = out_df.params.copy()
        new_nsplits = tuple(new_nsplits)
        kw.update(dict(chunks=chunks, nsplits=new_nsplits))
        return new_op.new_tileables(op.inputs, **kw)

    @classmethod
    def tile(cls, op):
        return (yield from cls._tile_df(op))

    def _infer_df_func_returns(self, df, dtypes, dtype=None, name=None, index=None):
        func = self._load_func()
        if isinstance(func, np.ufunc):
            output_type = OutputType.dataframe
            new_dtypes = None
            index_value = "inherit"
        else:
            if self.output_types is not None and (
                dtypes is not None or dtype is not None
            ):
                ret_dtypes = dtypes if dtypes is not None else (name, dtype)
                ret_index_value = parse_index(index) if index is not None else None
                return ret_dtypes, ret_index_value

            output_type = new_dtypes = index_value = None

        try:
            empty_df = build_df(df, size=2)
            with np.errstate(all="ignore"), quiet_stdio():
                infer_df = empty_df.applymap(
                    func,
                    **self.kwds,
                )
            if index_value is None:
                if infer_df.index is empty_df.index:
                    index_value = "inherit"
                else:
                    index_value = parse_index(pd.RangeIndex(-1))

            if isinstance(infer_df, pd.DataFrame):
                output_type = output_type or OutputType.dataframe
                new_dtypes = new_dtypes or infer_df.dtypes
        except:  # noqa: E722  # nosec
            pass

        self.output_types = (
            [output_type] if not self.output_types else self.output_types
        )
        dtypes = new_dtypes if dtypes is None else dtypes
        index_value = index_value if index is None else parse_index(index)
        return dtypes, index_value

    def _call_dataframe(self, df, dtypes=None, dtype=None, name=None, index=None):
        # for backward compatibility
        dtype = dtype if dtype is not None else dtypes
        dtypes, index_value = self._infer_df_func_returns(
            df, dtypes, dtype=dtype, name=name, index=index
        )

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

        print(df.shape)
        # if self.elementwise:
        shape = df.shape
        # elif self.output_types[0] == OutputType.dataframe:
        #     shape = [np.nan, np.nan]
        #     shape[1 - self.axis] = df.shape[1 - self.axis]
        #     shape = tuple(shape)
        # else:
        #     shape = (df.shape[1 - self.axis],)
        # serialize in advance to reduce overhead
        self.func = cloudpickle.dumps(self.func)

        if self.output_types[0] == OutputType.dataframe:
            # if self.axis == 0:
            return self.new_dataframe(
                [df],
                shape=shape,
                dtypes=dtypes,
                index_value=index_value,
                columns_value=parse_index(dtypes.index, store_data=True),
            )
            # else:
            #     return self.new_dataframe(
            #         [df],
            #         shape=shape,
            #         dtypes=dtypes,
            #         index_value=df.index_value,
            #         columns_value=parse_index(dtypes.index, store_data=True),
            #     )
        # else:
        #     name, dtype = dtypes
        #     return self.new_series(
        #         [df], shape=shape, name=name, dtype=dtype, index_value=index_value
        #     )

    def __call__(self, df_or_series, dtypes=None, dtype=None, name=None, index=None):
        dtypes = make_dtypes(dtypes)
        dtype = make_dtype(dtype)
        self.func_token = get_func_token(self.func)

        return self._call_dataframe(df_or_series, dtypes=dtypes, dtype=dtype, name=name, index=index)


def df_applymap(
    df,
    func,
    result_type=None,
    args=(),
    dtypes=None,
    dtype=None,
    name=None,
    output_type=None,
    index=None,
    skip_infer=False,
    **kwds,
):
    output_types = kwds.pop("output_types", None)
    object_type = kwds.pop("object_type", None)
    output_types = validate_output_types(
        output_type=output_type, output_types=output_types, object_type=object_type
    )
    output_type = output_types[0] if output_types else None
    if skip_infer and output_type is None:
        output_types = [OutputType.df_or_series]

    # calling member function
    if isinstance(func, str):
        func = getattr(df, func)
        return func(*args, **kwds)

    op = ApplymapOperand(
        func=func,
        result_type=result_type,
        args=args,
        kwds=kwds,
        output_types=output_types,
    )
    return op(df, dtypes=dtypes, dtype=dtype, name=name, index=index)

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

import logging

import numpy as np
import pandas as pd

from ... import opcodes
from ...core import recursive_tile
from ...core.custom_log import redirect_custom_log
from ...serialization.serializables import (
    DictField,
    FunctionField,
    Int32Field,
    KeyField,
    StringField,
    TupleField,
)
from ...utils import enter_current_session, has_unknown_shape, quiet_stdio
from ..operands import DataFrameOperand, OutputType
from ..utils import (
    build_df,
    build_empty_df,
    build_series,
    parse_index,
    validate_output_types,
)
from .core import DataFrameAutoMergeMixin

logger = logging.getLogger(__name__)


class DataFrameCartesianChunk(DataFrameOperand, DataFrameAutoMergeMixin):
    _op_type_ = opcodes.CARTESIAN_CHUNK

    left = KeyField("left")
    right = KeyField("right")
    func = FunctionField("func")
    args = TupleField("args")
    kwargs = DictField("kwargs")
    auto_merge = StringField("auto_merge")
    auto_merge_threshold = Int32Field("auto_merge_threshold")

    def __init__(self, output_types=None, **kw):
        super().__init__(_output_types=output_types, **kw)
        if self.memory_scale is None:
            self.memory_scale = 2.0

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self.left = self.inputs[0]
        self.right = self.inputs[1]

    @staticmethod
    def _build_test_obj(obj):
        return (
            build_df(obj, size=2)
            if obj.ndim == 2
            else build_series(obj, size=2, name=obj.name)
        )

    def __call__(self, left, right, index=None, dtypes=None):
        test_left = self._build_test_obj(left)
        test_right = self._build_test_obj(right)
        output_type = self.output_types[0] if self.output_types else None

        if output_type == OutputType.df_or_series:
            return self.new_df_or_series([left, right])

        # try run to infer meta
        try:
            with np.errstate(all="ignore"), quiet_stdio():
                obj = self.func(test_left, test_right, *self.args, **self.kwargs)
        except:  # noqa: E722  # nosec  # pylint: disable=bare-except
            if output_type == OutputType.series:
                obj = pd.Series([], dtype=np.dtype(object))
            elif output_type == OutputType.dataframe and dtypes is not None:
                obj = build_empty_df(dtypes)
            else:
                raise TypeError(
                    "Cannot determine `output_type`, "
                    "you have to specify it as `dataframe` or `series`, "
                    "for dataframe, `dtypes` is required as well "
                    "if output_type='dataframe'"
                )

        if getattr(obj, "ndim", 0) == 1 or output_type == OutputType.series:
            shape = self.kwargs.pop("shape", (np.nan,))
            if index is None:
                index = obj.index
            index_value = parse_index(
                index, left, right, self.func, self.args, self.kwargs
            )
            return self.new_series(
                [left, right],
                dtype=obj.dtype,
                shape=shape,
                index_value=index_value,
                name=obj.name,
            )
        else:
            dtypes = dtypes if dtypes is not None else obj.dtypes
            # dataframe
            shape = (np.nan, len(dtypes))
            columns_value = parse_index(dtypes.index, store_data=True)
            if index is None:
                index = obj.index
            index_value = parse_index(
                index, left, right, self.func, self.args, self.kwargs
            )
            return self.new_dataframe(
                [left, right],
                shape=shape,
                dtypes=dtypes,
                index_value=index_value,
                columns_value=columns_value,
            )

    @classmethod
    def tile(cls, op: "DataFrameCartesianChunk"):
        left = op.left
        right = op.right
        out = op.outputs[0]
        out_type = op.output_types[0]

        auto_merge_threshold = op.auto_merge_threshold
        auto_merge_before, auto_merge_after = cls._get_auto_merge_options(op.auto_merge)

        merge_before_res = yield from cls._merge_before(
            op, auto_merge_before, auto_merge_threshold, left, right, logger
        )
        left, right = merge_before_res[0], merge_before_res[1]

        if left.ndim == 2 and left.chunk_shape[1] > 1:
            if has_unknown_shape(left):
                yield
            # if left is a DataFrame, make sure 1 chunk on axis columns
            left = yield from recursive_tile(left.rechunk({1: left.shape[1]}))
        if right.ndim == 2 and right.chunk_shape[1] > 1:
            if has_unknown_shape(right):
                yield
            # if right is a DataFrame, make sure 1 chunk on axis columns
            right = yield from recursive_tile(right.rechunk({1: right.shape[1]}))

        out_chunks = []
        if out_type == OutputType.dataframe:
            nsplits = [[], [out.shape[1]]]
        elif out_type == OutputType.series:
            nsplits = [[]]
        else:
            # DataFrameOrSeries
            nsplits = None
        i = 0
        for left_chunk in left.chunks:
            for right_chunk in right.chunks:
                chunk_op = op.copy().reset_key()
                chunk_op.tileable_op_key = op.key
                if out_type == OutputType.df_or_series:
                    out_chunks.append(
                        chunk_op.new_chunk(
                            [left_chunk, right_chunk], index=(i, 0), collapse_axis=1
                        )
                    )
                elif out_type == OutputType.dataframe:
                    shape = (np.nan, out.shape[1])
                    index_value = parse_index(
                        out.index_value.to_pandas(),
                        left_chunk,
                        right_chunk,
                        op.func,
                        op.args,
                        op.kwargs,
                    )
                    out_chunk = chunk_op.new_chunk(
                        [left_chunk, right_chunk],
                        shape=shape,
                        index_value=index_value,
                        columns_value=out.columns_value,
                        dtypes=out.dtypes,
                        index=(i, 0),
                    )
                    out_chunks.append(out_chunk)
                    nsplits[0].append(out_chunk.shape[0])
                else:
                    shape = (np.nan,)
                    index_value = parse_index(
                        out.index_value.to_pandas(),
                        left_chunk,
                        right_chunk,
                        op.func,
                        op.args,
                        op.kwargs,
                    )
                    out_chunk = chunk_op.new_chunk(
                        [left_chunk, right_chunk],
                        shape=shape,
                        index_value=index_value,
                        dtype=out.dtype,
                        name=out.name,
                        index=(i,),
                    )
                    out_chunks.append(out_chunk)
                    nsplits[0].append(out_chunk.shape[0])
                i += 1

        params = out.params
        params["nsplits"] = tuple(tuple(ns) for ns in nsplits) if nsplits else nsplits
        params["chunks"] = out_chunks
        new_op = op.copy()
        ret = new_op.new_tileables(op.inputs, kws=[params])

        ret = yield from cls._merge_after(
            op, auto_merge_after, auto_merge_threshold, ret, logger
        )
        return ret

    @classmethod
    @redirect_custom_log
    @enter_current_session
    def execute(cls, ctx, op: "DataFrameCartesianChunk"):
        left, right = ctx[op.left.key], ctx[op.right.key]
        ctx[op.outputs[0].key] = op.func(left, right, *op.args, **(op.kwargs or dict()))


def cartesian_chunk(
    left,
    right,
    func,
    skip_infer=False,
    args=(),
    auto_merge: str = "both",
    auto_merge_threshold: int = 8,
    **kwargs,
):
    output_type = kwargs.pop("output_type", None)
    output_types = kwargs.pop("output_types", None)
    object_type = kwargs.pop("object_type", None)
    output_types = validate_output_types(
        output_type=output_type, output_types=output_types, object_type=object_type
    )
    output_type = output_types[0] if output_types else None
    if output_type:
        output_types = [output_type]
    elif skip_infer:
        output_types = [OutputType.df_or_series]
    index = kwargs.pop("index", None)
    dtypes = kwargs.pop("dtypes", None)
    memory_scale = kwargs.pop("memory_scale", None)
    if auto_merge not in ["both", "none", "before", "after"]:  # pragma: no cover
        raise ValueError(
            f"auto_merge can only be `both`, `none`, `before` or `after`, got {auto_merge}"
        )

    op = DataFrameCartesianChunk(
        left=left,
        right=right,
        func=func,
        args=args,
        kwargs=kwargs,
        output_types=output_types,
        memory_scale=memory_scale,
        auto_merge=auto_merge,
        auto_merge_threshold=auto_merge_threshold,
    )
    return op(left, right, index=index, dtypes=dtypes)

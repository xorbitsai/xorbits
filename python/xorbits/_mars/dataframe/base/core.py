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

from __future__ import annotations

import logging

from ...core import TileStatus
from ...core.context import get_context
from ...serialization.serializables import KeyField
from ...typing import OperandType, TileableType
from ..core import DATAFRAME_TYPE, SERIES_TYPE
from ..operands import DataFrameOperand, DataFrameOperandMixin
from ..utils import auto_merge_chunks


class DataFrameDeviceConversionBase(DataFrameOperand, DataFrameOperandMixin):
    _input = KeyField("input")

    @property
    def input(self):
        return self._input

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._input = inputs[0]

    def __call__(self, obj):
        if isinstance(obj, DATAFRAME_TYPE):
            return self.new_dataframe(
                [obj],
                shape=obj.shape,
                dtypes=obj.dtypes,
                index_value=obj.index_value,
                columns_value=obj.columns_value,
            )
        else:
            assert isinstance(obj, SERIES_TYPE)
            return self.new_series(
                [obj],
                shape=obj.shape,
                dtype=obj.dtype,
                index_value=obj.index_value,
                name=obj.name,
            )

    @classmethod
    def tile(cls, op):
        # Isolate ops on cpu from subsequent ops on gpu
        yield
        out_chunks = []
        for c in op.input.chunks:
            chunk_op = op.copy().reset_key()
            out_chunk = chunk_op.new_chunk([c], **c.params)
            out_chunks.append(out_chunk)

        new_op = op.copy().reset_key()
        out = op.outputs[0]
        return new_op.new_tileables(
            op.inputs, chunks=out_chunks, nsplits=op.inputs[0].nsplits, **out.params
        )


class DataFrameAutoMergeMixin(DataFrameOperandMixin):
    @classmethod
    def _get_auto_merge_options(cls, auto_merge: str) -> tuple[bool, bool]:
        if auto_merge == "both":
            return True, True
        elif auto_merge == "none":
            return False, False
        elif auto_merge == "before":
            return True, False
        else:
            assert auto_merge == "after"
            return False, True

    @classmethod
    def _merge_before(
        cls,
        op: OperandType,
        auto_merge_before: bool,
        auto_merge_threshold: int,
        left: TileableType,
        right: TileableType,
        logger: logging.Logger,
    ):
        ctx = get_context()

        if (
            auto_merge_before
            and len(left.chunks) + len(right.chunks) > auto_merge_threshold
        ):
            yield TileStatus([left, right] + left.chunks + right.chunks, progress=0.2)
            left_chunk_size = len(left.chunks)
            right_chunk_size = len(right.chunks)
            left = auto_merge_chunks(ctx, left)
            right = auto_merge_chunks(ctx, right)
            logger.info(
                "Auto merge before %s, left data shape: %s, chunk count: %s -> %s, "
                "right data shape: %s, chunk count: %s -> %s.",
                op,
                left.shape,
                left_chunk_size,
                len(left.chunks),
                right.shape,
                right_chunk_size,
                len(right.chunks),
            )
        else:
            logger.info(
                "Skip auto merge before %s, left data shape: %s, chunk count: %d, "
                "right data shape: %s, chunk count: %d.",
                op,
                left.shape,
                len(left.chunks),
                right.shape,
                len(right.chunks),
            )
        return [left, right]

    @classmethod
    def _merge_after(
        cls,
        op: OperandType,
        auto_merge_after: bool,
        auto_merge_threshold: int,
        ret: TileableType,
        logger: logging.Logger,
    ):
        if auto_merge_after and len(ret[0].chunks) > auto_merge_threshold:
            # if how=="inner", output data size will reduce greatly with high probability，
            # use auto_merge_chunks to combine small chunks.
            yield TileStatus(
                ret[0].chunks, progress=0.8
            )  # trigger execution for chunks
            merged = auto_merge_chunks(get_context(), ret[0])
            logger.info(
                "Auto merge after %s, data shape: %s, chunk count: %s -> %s.",
                op,
                merged.shape,
                len(ret[0].chunks),
                len(merged.chunks),
            )
            return [merged]
        else:
            logger.info(
                "Skip auto merge after %s, data shape: %s, chunk count: %d.",
                op,
                ret[0].shape,
                len(ret[0].chunks),
            )
            return ret

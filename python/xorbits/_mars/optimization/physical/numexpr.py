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
from typing import List, Tuple, Union

import numpy as np

from ...core import ChunkType
from ...tensor import arithmetic, reduction
from ...tensor.fuse import TensorNeFuseChunk
from ...tensor.fuse.numexpr import NUMEXPR_INSTALLED
from .core import REDUCTION, GraphTraversalOptimizer, _Fuse, register_optimizer

logger = logging.getLogger(__name__)


REDUCTION_OP = {
    reduction.TensorSum,
    reduction.TensorProd,
    reduction.TensorMax,
    reduction.TensorMin,
}
SUPPORT_OP = {
    arithmetic.TensorAdd,
    arithmetic.TensorSubtract,
    arithmetic.TensorMultiply,
    arithmetic.TensorDivide,
    arithmetic.TensorPower,
    arithmetic.TensorMod,
    arithmetic.TensorNegative,
    arithmetic.TensorAbs,
    arithmetic.TensorConj,
    arithmetic.TensorExp,
    arithmetic.TensorLog,
    arithmetic.TensorLog10,
    arithmetic.TensorExpm1,
    arithmetic.TensorLog1p,
    arithmetic.TensorSqrt,
    arithmetic.TensorEqual,
    arithmetic.TensorNotEqual,
    arithmetic.TensorLessThan,
    arithmetic.TensorLessEqual,
    arithmetic.TensorGreaterThan,
    arithmetic.TensorGreaterEqual,
    arithmetic.TensorSin,
    arithmetic.TensorCos,
    arithmetic.TensorTan,
    arithmetic.TensorArcsin,
    arithmetic.TensorArccos,
    arithmetic.TensorArctan,
    arithmetic.TensorSinh,
    arithmetic.TensorCosh,
    arithmetic.TensorTanh,
    arithmetic.TensorArcsinh,
    arithmetic.TensorArccosh,
    arithmetic.TensorArctanh,
    arithmetic.TensorLshift,
    arithmetic.TensorRshift,
    arithmetic.TensorTreeAdd,
    arithmetic.TensorTreeMultiply,
    arithmetic.TensorFloor,
    arithmetic.TensorCeil,
    arithmetic.TensorAnd,
    arithmetic.TensorOr,
    arithmetic.TensorNot,
    reduction.TensorSum,
    reduction.TensorProd,
    reduction.TensorMax,
    reduction.TensorMin,
}


@register_optimizer
class NumexprRuntimeOptimizer(GraphTraversalOptimizer):
    engine = "numexpr"

    def _can_fuse(self, node: ChunkType) -> Union[bool, object]:
        op = node.op
        op_type = type(op)
        if op_type in REDUCTION_OP:
            if len(op.axis) == 1 or len(op.axis) == node.ndim:
                return REDUCTION
            else:
                return False
        # return op_type in SUPPORT_OP
        if op_type not in SUPPORT_OP:
            return False
        if op_type in (arithmetic.TensorOr, arithmetic.TensorAnd):
            # numexpr only support logical and or:
            # https://numexpr.readthedocs.io/projects/NumExpr3/en/latest/user_guide.html#supported-operators
            if np.isscalar(op.lhs) or np.isscalar(op.rhs):  # pragma: no cover
                return False
        return True

    @classmethod
    def is_available(cls) -> bool:
        return NUMEXPR_INSTALLED

    def optimize(self) -> Tuple[List[_Fuse], List[ChunkType]]:
        fuses = self._graph_traverse(
            logger, "Refused fusing for numexpr because the tail node count > 1."
        )

        if fuses == ([], []):
            return [], []

        return self._fuse_nodes(fuses, TensorNeFuseChunk)

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

import logging
from typing import List, Tuple, Union

from ...core import ChunkType
from ...tensor.fuse import TensorJAXFuseChunk
from ...tensor.fuse.jax import (
    BINARY_ARITHMETIC_SUPPORT,
    JAX_INSTALLED,
    REDUCTION_SUPPORT,
    TREE_SUPPORT,
    UNARY_ARITHMETIC_SUPPORT,
)
from .core import REDUCTION, GraphTraversalOptimizer, _Fuse, register_optimizer

logger = logging.getLogger(__name__)


@register_optimizer
class JAXRuntimeOptimizer(GraphTraversalOptimizer):
    engine = "jax"

    def _can_fuse(self, node: ChunkType) -> Union[bool, object]:
        op = node.op
        op_type = type(op)
        if op_type in REDUCTION_SUPPORT:
            if len(op.axis) == 1 or len(op.axis) == node.ndim:
                return REDUCTION
            else:  # pragma: no cover
                return False
        if (
            op_type not in UNARY_ARITHMETIC_SUPPORT
            and op_type not in BINARY_ARITHMETIC_SUPPORT
            and op_type not in TREE_SUPPORT
        ):
            return False
        return True

    @classmethod
    def is_available(cls) -> bool:
        return JAX_INSTALLED

    def optimize(self) -> Tuple[List[_Fuse], List[ChunkType]]:
        fuses = self._graph_traverse(
            logger, "Refused fusing for jax because the tail node count > 1."
        )

        if fuses == ([], []):  # pragma: no cover
            return [], []

        return self._fuse_nodes(fuses, TensorJAXFuseChunk)

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

from ...core import ChunkType
from ...tensor import arithmetic
from ...tensor.fuse import TensorJAXFuseChunk
from ...tensor.fuse.jax import JAX_INSTALLED
from .core import GraphTraversalOptimizer, register_optimizer

logger = logging.getLogger(__name__)


SUPPORT_OP = {
    arithmetic.TensorAdd,
    arithmetic.TensorTreeAdd,
    arithmetic.TensorSubtract,
    arithmetic.TensorDivide,
}


@register_optimizer
class JAXRuntimeOptimizer(GraphTraversalOptimizer):
    engine = "jax"

    def _can_fuse(self, node: ChunkType):
        op = node.op
        op_type = type(op)
        if op_type not in SUPPORT_OP:
            return False
        return True

    @classmethod
    def is_available(cls) -> bool:
        return JAX_INSTALLED

    def optimize(self):
        fuses = self._graph_traverse(
            logger, "Refused fusing for jax because the tail node count > 1."
        )

        if fuses == ([], []):
            return [], []

        return self._fuse_nodes(fuses, TensorJAXFuseChunk)

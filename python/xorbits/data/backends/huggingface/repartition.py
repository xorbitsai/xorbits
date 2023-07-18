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

from ...._mars.serialization.serializables import Int32Field, DictField
from ...._mars.typing import OperandType
from ...operand import DataOperand, DataOperandMixin


class HuggingfaceRepartition(DataOperand, DataOperandMixin):
    num_blocks: int = Int32Field("num_blocks")
    block_index: int = Int32Field("block_index")
    kwargs = DictField("kwargs")

    def __call__(self, inp):
        self.output_types = inp.op.output_types
        return self.new_tileable([inp])

    @classmethod
    def tile(cls, op: OperandType):
        input_chunks = []
        for inp in op.inputs:
            input_chunks.extend(inp.chunks)

        # TODO(codingl2k1): support repartition multi partitions.
        assert len(input_chunks) == 1

        chunks = []
        for index in range(op.num_blocks):
            chunk_op = op.copy().reset_key()
            chunk_op.block_index = index
            c = chunk_op.new_chunk(inputs=input_chunks, index=index)
            chunks.append(c)

        return op.copy().new_tileable(op.inputs, chunks=chunks)

    @classmethod
    def execute(cls, ctx, op: OperandType):
        inp = ctx[op.inputs[0].key]
        out_key = op.outputs[0].key
        ctx[out_key] = inp.shard(op.num_blocks, op.block_index, **op.kwargs)


def repartition(dataset, num_blocks: int, **kwargs):
    op = HuggingfaceRepartition(num_blocks=num_blocks, kwargs=kwargs)
    return op(dataset)

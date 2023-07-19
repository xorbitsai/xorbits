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

from ...._mars.serialization.serializables import DictField, Int32Field
from ...._mars.typing import OperandType
from ...operand import DataOperand, DataOperandMixin


class HuggingfaceRepartition(DataOperand, DataOperandMixin):
    num_chunks: int = Int32Field("num_chunks")
    chunk_index: int = Int32Field("chunk_index")
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
        for index in range(op.num_chunks):
            chunk_op = op.copy().reset_key()
            chunk_op.chunk_index = index
            c = chunk_op.new_chunk(inputs=input_chunks, index=index)
            chunks.append(c)

        return op.copy().new_tileable(op.inputs, chunks=chunks)

    @classmethod
    def execute(cls, ctx, op: OperandType):
        inp = ctx[op.inputs[0].key]
        out_key = op.outputs[0].key
        ctx[out_key] = inp.shard(op.num_chunks, op.chunk_index, **op.kwargs)


def repartition(dataset, num_chunks: int, **kwargs):
    op = HuggingfaceRepartition(num_chunks=num_chunks, kwargs=kwargs)
    return op(dataset)

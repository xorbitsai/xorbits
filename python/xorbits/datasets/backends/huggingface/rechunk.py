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

import numpy as np

from ...._mars.core.entity import OutputType
from ...._mars.serialization.serializables import DictField, Int32Field
from ...operand import DataOperand, DataOperandMixin


class HuggingfaceRechunk(DataOperand, DataOperandMixin):
    num_chunks: int = Int32Field("num_chunks")
    chunk_index: int = Int32Field("chunk_index")
    hf_kwargs = DictField("hf_kwargs")

    def __call__(self, dataset):
        return self.new_tileable([dataset], **dataset.params)

    @classmethod
    def tile(cls, op: "HuggingfaceRechunk"):
        input_chunks = []
        for inp in op.inputs:
            input_chunks.extend(inp.chunks)
        out = op.outputs[0]

        # TODO(codingl2k1): support rechunk multi chunks.
        if len(input_chunks) == 1 and op.num_chunks != 1:
            chunks = []
            for index in range(op.num_chunks):
                chunk_op = op.copy().reset_key()
                chunk_op.chunk_index = index
                c = chunk_op.new_chunk(inputs=input_chunks, index=(index, 0))
                chunks.append(c)
            return op.copy().new_tileable(
                op.inputs,
                chunks=chunks,
                nsplits=((np.nan,) * len(chunks), (np.nan,)),
                **out.params
            )
        else:
            return op.copy().new_tileable(
                op.inputs,
                chunks=input_chunks,
                nsplits=((np.nan,) * len(input_chunks), (np.nan,)),
                **out.params
            )

    @classmethod
    def execute(cls, ctx, op: "HuggingfaceRechunk"):
        inp = ctx[op.inputs[0].key]
        out_key = op.outputs[0].key
        # Default split dataset by contiguous == True.
        hf_kwargs = {"contiguous": True}
        hf_kwargs.update(op.hf_kwargs)
        ctx[out_key] = inp.shard(op.num_chunks, op.chunk_index, **hf_kwargs)


def rechunk(dataset, num_chunks: int, **hf_kwargs):
    op = HuggingfaceRechunk(
        output_types=[OutputType.huggingface_dataset],
        num_chunks=num_chunks,
        hf_kwargs=hf_kwargs,
    )
    return op(dataset)

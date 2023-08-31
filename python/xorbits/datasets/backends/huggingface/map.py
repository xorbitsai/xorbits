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

import cloudpickle
import numpy as np

from ...._mars.core.entity import OutputType
from ...._mars.serialization.serializables import AnyField, DictField
from ...operand import DataOperand, DataOperandMixin


class HuggingfaceMap(DataOperand, DataOperandMixin):
    func = AnyField("func")
    # The arguments are passing to huggingface dataset map:
    # https://huggingface.co/docs/datasets/v2.13.1/en/package_reference/main_classes#datasets.Dataset.map
    hf_kwargs = DictField("hf_kwargs")

    def __call__(self, dataset):
        # serialize in advance to reduce overhead
        self.func = cloudpickle.dumps(self.func)
        return self.new_tileable([dataset], **dataset.params)

    @classmethod
    def tile(cls, op: "HuggingfaceMap"):
        assert len(op.inputs) == 1
        inp = op.inputs[0]
        out = op.outputs[0]
        out_chunks = []
        for chunk in inp.chunks:
            chunk_op = op.copy().reset_key()
            out_chunk = chunk_op.new_chunk([chunk], index=chunk.index)
            out_chunks.append(out_chunk)
        return op.copy().new_tileable(
            op.inputs,
            chunks=out_chunks,
            nsplits=((np.nan,) * len(out_chunks), (np.nan,)),
            **out.params,
        )

    @classmethod
    def execute(cls, ctx, op: "HuggingfaceMap"):
        func = cloudpickle.loads(op.func)
        inp = ctx[op.inputs[0].key]
        out = op.outputs[0]
        ctx[out.key] = inp.map(func, **op.hf_kwargs)


def map(dataset, fn, **hf_kwargs):
    op = HuggingfaceMap(
        output_types=[OutputType.huggingface_dataset], func=fn, hf_kwargs=hf_kwargs
    )
    return op(dataset)

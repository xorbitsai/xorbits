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

import cloudpickle

from ...operand import DataOperand, DataOperandMixin
from ...._mars.core.entity import OutputType
from ...._mars.serialization.serializables import AnyField, DictField


class HuggingfaceMap(DataOperand, DataOperandMixin):
    func = AnyField("func")
    kwargs = DictField("kwargs")

    def __init__(self, output_types=None, **kw):
        super().__init__(_output_types=output_types, **kw)

    def __call__(self, dataset):
        self.output_types = dataset.op.output_types
        # serialize in advance to reduce overhead
        self.func = cloudpickle.dumps(self.func)
        return self.new_tileable([dataset])

    @classmethod
    def tile(cls, op: "HuggingfaceMap"):
        assert len(op.inputs) == 1
        inp = op.inputs[0]
        out_chunks = []
        for chunk in inp.chunks:
            chunk_op = op.copy().reset_key()
            out_chunk = chunk_op.new_chunk([chunk], index=chunk.index)
            out_chunks.append(out_chunk)
        return op.copy().new_tileable(op.inputs, chunks=out_chunks)

    @classmethod
    def execute(cls, ctx, op: "HuggingfaceMap"):
        func = cloudpickle.loads(op.func)
        inp = ctx[op.inputs[0].key]
        out = op.outputs[0]
        ctx[out.key] = inp.map(func, **op.kwargs)


def map(dataset, fn, **kwargs):
    op = HuggingfaceMap(
        output_types=[OutputType.huggingface_data], func=fn, kwargs=kwargs
    )
    return op(dataset)

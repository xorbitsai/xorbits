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

from typing import Iterable, Union

from ...._mars.core.entity import OutputType
from ...._mars.serialization.serializables import AnyField
from ...._mars.utils import has_unknown_shape
from ...operand import DataOperand, DataOperandMixin


class HuggingfaceGetItem(DataOperand, DataOperandMixin):
    hf_getitem_key = AnyField("hf_getitem_key")

    def __call__(self, dataset):
        return self.new_tileable([dataset], **dataset.params)

    @classmethod
    def tile(cls, op: "HuggingfaceGetItem"):
        assert len(op.inputs) == 1
        inp = op.inputs[0]

        if isinstance(op.hf_getitem_key, str):
            out_chunks = []
            for chunk in inp.chunks:
                chunk_op = op.copy().reset_key()
                out_chunk = chunk_op.new_chunk([chunk], index=chunk.index)
                out_chunks.append(out_chunk)
            return op.copy().new_tileable(op.inputs, chunks=out_chunks)
        elif isinstance(op.hf_getitem_key, int):
            if has_unknown_shape(*op.inputs):
                yield
            key = op.hf_getitem_key
            for chunk in inp.chunks:
                if key < 0:
                    raise IndexError(f"Input key {op.hf_getitem_key} is out of bound.")
                if key >= chunk.shape[0]:
                    key -= chunk.shape[0]
                else:
                    chunk_op = op.copy().reset_key()
                    chunk_op.hf_getitem_key = key
                    out_chunk = chunk_op.new_chunk([chunk], index=chunk.index)
                    return op.copy().new_tileable(op.inputs, chunks=[out_chunk])
            raise IndexError(f"Input key {op.hf_getitem_key} is out of bound.")
        else:
            raise NotImplementedError(
                f"Not support getitem with key type: {type(op.hf_getitem_key)}"
            )

    @classmethod
    def execute(cls, ctx, op: "HuggingfaceGetItem"):
        inp = ctx[op.inputs[0].key]
        out = op.outputs[0]
        ctx[out.key] = inp.__getitem__(op.hf_getitem_key)


def getitem(dataset, key: Union[int, slice, str, Iterable[int]]):
    if not isinstance(key, (str, int)):
        raise NotImplementedError(f"Not support getitem with key type: {type(key)}")
    op = HuggingfaceGetItem(output_types=[OutputType.object], hf_getitem_key=key)
    return op(dataset)

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

from typing import Union

from ...._mars.core.entity import OutputType
from ...._mars.serialization.serializables import AnyField
from ...._mars.utils import has_unknown_shape, is_full_slice
from ...operand import DataOperand, DataOperandMixin


class HuggingfaceGetItem(DataOperand, DataOperandMixin):
    hf_getitem_key = AnyField("hf_getitem_key")

    def __call__(self, dataset):
        return self.new_tileable([dataset], **dataset.params)

    @classmethod
    def _gen_copy_output(cls, inp, op: "HuggingfaceGetItem"):
        out_chunks = []
        for chunk in inp.chunks:
            chunk_op = op.copy().reset_key()
            out_chunk = chunk_op.new_chunk([chunk], index=chunk.index)
            out_chunks.append(out_chunk)
        return op.copy().new_tileable(op.inputs, chunks=out_chunks)

    @classmethod
    def _gen_empty_output(cls, inp, op: "HuggingfaceGetItem"):
        first_chunk = inp.chunks[0]
        chunk_op = op.copy().reset_key()
        chunk_op.hf_getitem_key = slice(0, 0)
        out_chunk = chunk_op.new_chunk([first_chunk], index=first_chunk.index)
        return op.copy().new_tileable(op.inputs, chunks=[out_chunk])

    @classmethod
    def _find_chunk_index_by_key(cls, chunks, key):
        for idx, chunk in enumerate(chunks):
            if key < 0:  # pragma: no cover
                raise IndexError(f"Input key {key} is out of bound.")
            if key >= chunk.shape[0]:
                key -= chunk.shape[0]
            else:
                return idx, key
        # pragma: no cover
        raise IndexError(f"Input key {key} is out of bound.")

    @classmethod
    def tile(cls, op: "HuggingfaceGetItem"):
        assert len(op.inputs) == 1
        inp = op.inputs[0]

        if isinstance(op.hf_getitem_key, str):
            return cls._gen_copy_output(inp, op)
        elif isinstance(op.hf_getitem_key, int):
            if has_unknown_shape(*op.inputs):
                yield
            index, key = cls._find_chunk_index_by_key(inp.chunks, op.hf_getitem_key)
            chunk = inp.chunks[index]
            chunk_op = op.copy().reset_key()
            chunk_op.hf_getitem_key = key
            out_chunk = chunk_op.new_chunk([chunk], index=chunk.index)
            return op.copy().new_tileable(op.inputs, chunks=[out_chunk])
        elif isinstance(op.hf_getitem_key, slice):
            if is_full_slice(op.hf_getitem_key):
                return cls._gen_copy_output(inp, op)
            else:
                start = op.hf_getitem_key.start
                stop = op.hf_getitem_key.stop
                assert op.hf_getitem_key.step is None
                if start >= stop:
                    # For empty slice, e.g. s[3:1], s[3:3], we translate the
                    # execution to the first chunk[0:0].
                    return cls._gen_empty_output(inp, op)
                else:
                    if has_unknown_shape(*op.inputs):
                        yield
                    try:
                        start_index, start_key = cls._find_chunk_index_by_key(
                            inp.chunks, start
                        )
                    except IndexError:
                        return cls._gen_empty_output(inp, op)
                    try:
                        stop_index, stop_key = cls._find_chunk_index_by_key(
                            inp.chunks, stop
                        )
                    except IndexError:
                        stop_index = len(inp.chunks) - 1
                        stop_key = None
                    chunks = []
                    for index, chunk in enumerate(inp.chunks):
                        if start_index <= index <= stop_index:
                            chunk_op = op.copy().reset_key()
                            slice_start = start_key if index == start_index else None
                            slice_stop = stop_key if index == stop_index else None
                            chunk_op.hf_getitem_key = slice(
                                slice_start, slice_stop, None
                            )
                            out_chunk = chunk_op.new_chunk([chunk], index=chunk.index)
                            chunks.append(out_chunk)
                        elif index > stop_index:
                            break
                    return op.copy().new_tileable(op.inputs, chunks=chunks)
        else:  # pragma: no cover
            raise NotImplementedError(
                f"Not support getitem with key type: {type(op.hf_getitem_key)}"
            )

    @classmethod
    def execute(cls, ctx, op: "HuggingfaceGetItem"):
        inp = ctx[op.inputs[0].key]
        out = op.outputs[0]
        ctx[out.key] = inp.__getitem__(op.hf_getitem_key)


def getitem(dataset, key: Union[int, slice, str]):
    if not isinstance(key, (str, int, slice)):
        raise NotImplementedError(f"Not support getitem with key type: {type(key)}")
    if isinstance(key, slice) and key.step is not None:
        raise NotImplementedError(f"Not support getitem with slice and step: {key}")
    op = HuggingfaceGetItem(output_types=[OutputType.object], hf_getitem_key=key)
    return op(dataset).execute().fetch()

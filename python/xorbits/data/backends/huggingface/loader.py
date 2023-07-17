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

import inspect
import os.path

from ...operand import DataOperand, DataOperandMixin
from ...._mars.core.entity import OutputType
from ...._mars.serialization.serializables import (
    Int32Field,
    DictField,
    StringField,
)
from ...._mars.typing import OperandType


class HuggingfaceLoader(DataOperand, DataOperandMixin):
    path = StringField("path")
    kwargs = DictField("kwargs")
    single_data_file = StringField("single_data_file")
    num_blocks: int = Int32Field("num_blocks")
    block_index: int = Int32Field("block_index")
    cache_dir: str = StringField("cache_dir")

    def __call__(self):
        self.output_types = [OutputType.huggingface_data]
        return self.new_tileable([])

    @classmethod
    def _get_kwargs(cls, obj, kwargs):
        sig_builder = inspect.signature(obj)
        return {
            key: kwargs[key] for key in sig_builder.parameters.keys() if key in kwargs
        }

    @classmethod
    def tile(cls, op: OperandType):
        assert len(op.inputs) == 0

        import datasets

        builder_kwargs = cls._get_kwargs(datasets.load_dataset_builder, op.kwargs)
        builder = datasets.load_dataset_builder(op.path, **builder_kwargs)
        data_files = builder.config.data_files
        # TODO(codingl2k1): check data_files can be supported
        split = op.kwargs.get("split")
        # TODO(codingl2k1): support multiple splits

        chunks = []
        if data_files and split:
            data_files = data_files[split]
            for index, f in enumerate(data_files):
                chunk_op = op.copy().reset_key()
                assert f, "Invalid data file from DatasetBuilder."
                chunk_op.single_data_file = f
                chunk_op.num_blocks = len(data_files)
                chunk_op.block_index = index
                chunk_op.cache_dir = builder.cache_dir
                c = chunk_op.new_chunk(inputs=[], index=index)
                chunks.append(c)
            builder.config.data_files.clear()
        else:
            chunk_op = op.copy().reset_key()
            chunk_op.single_data_file = None
            chunks.append(chunk_op.new_chunk(inputs=[]))

        return op.copy().new_tileable(op.inputs, chunks=chunks)

    @classmethod
    def execute(cls, ctx, op: OperandType):
        import datasets

        builder_kwargs = cls._get_kwargs(datasets.load_dataset_builder, op.kwargs)

        # TODO(codingl2k1): not sure if it's OK to share one cache dir among workers.
        # if op.single_data_file:
        #     # TODO(codingl2k1): use xorbits cache dir
        #     new_cache_dir = os.path.join(op.cache_dir, f"part_{op.block_index}_{op.num_blocks}")
        #     builder_kwargs["cache_dir"] = new_cache_dir

        # load_dataset_builder from every worker may be slow, but it's error to
        # deserialized a builder instance in a clean process / node, e.g. raise
        # ModuleNotFoundError: No module named 'datasets_modules'.
        #
        # Please refer to issue: https://github.com/huggingface/transformers/issues/11565
        builder = datasets.load_dataset_builder(op.path, **builder_kwargs)
        download_and_prepare_kwargs = cls._get_kwargs(
            datasets.DatasetBuilder.download_and_prepare, op.kwargs
        )

        if op.single_data_file is not None:
            output_dir = builder._output_dir
            output_dir = output_dir if output_dir is not None else builder.cache_dir
            output_dir = os.path.join(
                output_dir, f"part_{op.block_index}_{op.num_blocks}"
            )
            download_and_prepare_kwargs["output_dir"] = output_dir
            download_and_prepare_kwargs[
                "verification_mode"
            ] = datasets.VerificationMode.NO_CHECKS
            split = op.kwargs["split"]
            split_data_files = builder.config.data_files[split]
            split_data_files[:] = [op.single_data_file]

        builder.download_and_prepare(**download_and_prepare_kwargs)
        as_dataset_kwargs = cls._get_kwargs(
            datasets.DatasetBuilder.as_dataset, op.kwargs
        )
        ds = builder.as_dataset(**as_dataset_kwargs)
        ctx[op.outputs[0].key] = ds


def load_dataset_from_huggingface(path: str, **kwargs):
    op = HuggingfaceLoader(
        output_types=[OutputType.huggingface_data], path=path, kwargs=kwargs
    )
    return op()

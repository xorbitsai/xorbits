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
import itertools
import os.path

import numpy as np

from ...._mars.core import recursive_tile
from ...._mars.core.context import get_context
from ...._mars.core.entity import OutputType
from ...._mars.serialization.serializables import (
    BoolField,
    DictField,
    Int32Field,
    ListField,
    StringField,
)
from ...operand import DataOperand, DataOperandMixin
from .rechunk import rechunk


class HuggingfaceLoader(DataOperand, DataOperandMixin):
    path = StringField("path")
    hf_kwargs = DictField("hf_kwargs")
    single_data_file = StringField("single_data_file")
    num_chunks: int = Int32Field("num_chunks")
    data_files = ListField("data_files")
    auto_rechunk: bool = BoolField("auto_rechunk")

    def __call__(self):
        from datasets import load_dataset_builder
        from datasets.packaged_modules.folder_based_builder.folder_based_builder import (
            FolderBasedBuilder,
        )

        builder_kwargs = self._get_kwargs(load_dataset_builder, self.hf_kwargs)
        builder = load_dataset_builder(self.path, **builder_kwargs)
        data_files = builder.config.data_files
        # TODO(codingl2k1): support multiple splits
        split = self.hf_kwargs["split"]
        # TODO(codingl2k1): not pass dtypes if no to_dataframe() called.
        if builder.info.features:
            dtypes = builder.info.features.arrow_schema.empty_table().to_pandas().dtypes
        else:
            dtypes = None
        if dtypes is not None and builder.info.splits:
            shape = (
                builder.info.splits[split].num_examples,
                len(dtypes),
            )
        else:
            shape = (np.nan, np.nan)
        # TODO(codingl2k1): check data_files if can be supported
        # e.g. the datasets mariosasko/test_imagefolder_with_metadata has multiple
        # data files, but some of them are meta, so we can't parallel load it.
        if not isinstance(builder, FolderBasedBuilder):
            if data_files and len(data_files[split]) > 1:
                data_files = list(data_files[split])
            else:
                data_files = None
        else:
            data_files = None
        self.data_files = data_files
        return self.new_tileable([], dtypes=dtypes, shape=shape)

    @classmethod
    def _get_kwargs(cls, obj, kwargs):
        sig_builder = inspect.signature(obj)
        return {
            key: kwargs[key] for key in sig_builder.parameters.keys() if key in kwargs
        }

    @classmethod
    def tile(cls, op: "HuggingfaceLoader"):
        assert len(op.inputs) == 0
        out = op.outputs[0]

        data_files = op.data_files
        # Set op.data_files to None, we don't want every chunk op copy this field.
        op.data_files = None

        chunks = []
        if data_files is not None:
            ctx = get_context()
            # TODO(codingl2k1): make expect worker binding stable for cache reuse.
            all_bands = [b for b in ctx.get_worker_bands() if b[1].startswith("numa-")]
            for index, (f, band) in enumerate(
                zip(data_files, itertools.cycle(all_bands))
            ):
                chunk_op = op.copy().reset_key()
                assert f, "Invalid data file from DatasetBuilder."
                chunk_op.single_data_file = f
                chunk_op.num_chunks = len(data_files)
                chunk_op.expect_band = band
                c = chunk_op.new_chunk(inputs=[], index=(index, 0))
                chunks.append(c)
        else:
            chunk_op = op.copy().reset_key()
            chunk_op.single_data_file = None
            chunks.append(chunk_op.new_chunk(inputs=[], index=(0, 0)))
            if op.auto_rechunk:
                ctx = get_context()
                cluster_cpu_count = int(ctx.get_total_n_cpu())
                if (
                    out.shape
                    and not np.isnan(out.shape[0])
                    and out.shape[0] >= cluster_cpu_count
                ):
                    inp = op.copy().new_tileable(
                        op.inputs,
                        chunks=chunks,
                        nsplits=((np.nan,) * len(chunks), (np.nan,)),
                        **out.params,
                    )
                    auto_rechunked = yield from recursive_tile(
                        rechunk(inp, cluster_cpu_count)
                    )
                    return auto_rechunked

        return op.copy().new_tileable(
            op.inputs,
            chunks=chunks,
            nsplits=((np.nan,) * len(chunks), (np.nan,)),
            **out.params,
        )

    @classmethod
    def execute(cls, ctx, op: "HuggingfaceLoader"):
        from datasets import DatasetBuilder, VerificationMode, load_dataset_builder

        builder_kwargs = cls._get_kwargs(load_dataset_builder, op.hf_kwargs)

        # TODO(codingl2k1): not sure if it's OK to share one cache dir among workers.
        # load_dataset_builder from every worker may be slow, but it's error to
        # deserialized a builder instance in a clean process / node, e.g. raise
        # ModuleNotFoundError: No module named 'datasets_modules'.
        #
        # Please refer to issue: https://github.com/huggingface/transformers/issues/11565
        builder = load_dataset_builder(op.path, **builder_kwargs)
        download_and_prepare_kwargs = cls._get_kwargs(
            DatasetBuilder.download_and_prepare, op.hf_kwargs
        )

        if op.single_data_file is not None:
            chunk = op.outputs[0]
            chunk_index = chunk.index[0]
            output_dir = builder._output_dir
            output_dir = output_dir if output_dir is not None else builder.cache_dir
            output_dir = os.path.join(output_dir, f"part_{chunk_index}_{op.num_chunks}")
            download_and_prepare_kwargs["output_dir"] = output_dir
            download_and_prepare_kwargs[
                "verification_mode"
            ] = VerificationMode.NO_CHECKS
            split = op.hf_kwargs["split"]
            split_data_files = builder.config.data_files[split]
            split_data_files[:] = [op.single_data_file]

        builder.download_and_prepare(**download_and_prepare_kwargs)
        as_dataset_kwargs = cls._get_kwargs(DatasetBuilder.as_dataset, op.hf_kwargs)
        ds = builder.as_dataset(**as_dataset_kwargs)
        ctx[op.outputs[0].key] = ds


def load_huggingface_dataset(path: str, auto_rechunk: bool = True, **hf_kwargs):
    op = HuggingfaceLoader(
        output_types=[OutputType.huggingface_dataset],
        path=path,
        auto_rechunk=auto_rechunk,
        hf_kwargs=hf_kwargs,
    )
    return op()

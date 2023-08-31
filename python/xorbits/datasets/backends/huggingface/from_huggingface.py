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

import inspect
import itertools
import os.path
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import numpy as np

try:
    # For type hint.
    from datasets import (
        DownloadConfig,
        DownloadMode,
        Features,
        Split,
        VerificationMode,
        Version,
    )
except ImportError:
    DownloadConfig = Any
    DownloadMode = Any
    Features = Any
    Split = Any
    VerificationMode = Any
    Version = Any

from ...._mars.core import recursive_tile
from ...._mars.core.context import get_context
from ...._mars.core.entity import OutputType
from ...._mars.serialization.serializables import (
    BoolField,
    DictField,
    ListField,
    StringField,
)
from ....utils import check_signature_compatible, get_non_default_kwargs
from ...dataset import Dataset
from ...operand import DataOperand, DataOperandMixin
from .rechunk import rechunk


class FromHuggingface(DataOperand, DataOperandMixin):
    path = StringField("path")
    hf_kwargs = DictField("hf_kwargs")
    cache_dir = StringField("cache_dir")
    single_data_file = StringField("single_data_file")
    data_files = ListField("data_files")
    auto_rechunk: bool = BoolField("auto_rechunk")

    def __call__(self):
        from datasets import load_dataset_builder
        from datasets.packaged_modules.folder_based_builder.folder_based_builder import (
            FolderBasedBuilder,
        )

        builder_kwargs = self._get_kwargs(load_dataset_builder, self.hf_kwargs)
        builder = load_dataset_builder(self.path, **builder_kwargs)
        assert "cache_dir" in inspect.signature(load_dataset_builder).parameters
        self.cache_dir = builder.cache_dir
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
    def tile(cls, op: "FromHuggingface"):
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
                chunk_op.expect_band = band
                # The cache dir can't be shared, because there will be a file lock
                # on the cache dir when initializing builder.
                chunk_op.hf_kwargs = dict(
                    op.hf_kwargs,
                    cache_dir=os.path.join(
                        op.cache_dir, f"part_{index}_{len(data_files)}"
                    ),
                )
                chunk_op.cache_dir = None
                c = chunk_op.new_chunk(inputs=[], index=(index, 0))
                chunks.append(c)
        else:
            chunk_op = op.copy().reset_key()
            chunk_op.single_data_file = None
            chunk_op.cache_dir = None
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
    def execute(cls, ctx, op: "FromHuggingface"):
        from datasets import DatasetBuilder, VerificationMode, load_dataset_builder

        # load_dataset_builder from every worker may be slow, but it's error to
        # deserialized a builder instance in a clean process / node, e.g. raise
        # ModuleNotFoundError: No module named 'datasets_modules'.
        #
        # Please refer to issue: https://github.com/huggingface/transformers/issues/11565
        builder_kwargs = cls._get_kwargs(load_dataset_builder, op.hf_kwargs)
        builder = load_dataset_builder(op.path, **builder_kwargs)
        download_and_prepare_kwargs = cls._get_kwargs(
            DatasetBuilder.download_and_prepare, op.hf_kwargs
        )

        if op.single_data_file is not None:
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


def from_huggingface(
    path: str,
    name: Optional[str] = None,
    data_dir: Optional[str] = None,
    data_files: Optional[
        Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]
    ] = None,
    split: Optional[Union[str, Split]] = None,
    cache_dir: Optional[str] = None,
    features: Optional[Features] = None,
    download_config: Optional[DownloadConfig] = None,
    download_mode: Optional[Union[DownloadMode, str]] = None,
    verification_mode: Optional[Union[VerificationMode, str]] = None,
    keep_in_memory: Optional[bool] = None,
    save_infos: bool = False,
    revision: Optional[Union[str, Version]] = None,
    token: Optional[Union[bool, str]] = None,
    streaming: bool = False,
    num_proc: Optional[int] = None,
    storage_options: Optional[Dict] = None,
    **config_kwargs,
) -> Dataset:
    """Create a dataset from a Hugging Face Datasets Dataset.

    This function is parallelized, and is intended to be used
    with Hugging Face Datasets that are loaded into memory (as opposed
    to memory-mapped).

    Args:
        path (`str`):
            Path or name of the dataset.
            Depending on `path`, the dataset builder that is used comes from a generic dataset script (JSON, CSV, Parquet, text etc.) or from the dataset script (a python file) inside the dataset directory.

            For local datasets:

            - if `path` is a local directory (containing data files only)
              -> load a generic dataset builder (csv, json, text etc.) based on the content of the directory
              e.g. `'./path/to/directory/with/my/csv/data'`.
            - if `path` is a local dataset script or a directory containing a local dataset script (if the script has the same name as the directory)
              -> load the dataset builder from the dataset script
              e.g. `'./dataset/squad'` or `'./dataset/squad/squad.py'`.

            For datasets on the Hugging Face Hub (list all available datasets with [`huggingface_hub.list_datasets`])

            - if `path` is a dataset repository on the HF hub (containing data files only)
              -> load a generic dataset builder (csv, text etc.) based on the content of the repository
              e.g. `'username/dataset_name'`, a dataset repository on the HF hub containing your data files.
            - if `path` is a dataset repository on the HF hub with a dataset script (if the script has the same name as the directory)
              -> load the dataset builder from the dataset script in the dataset repository
              e.g. `glue`, `squad`, `'username/dataset_name'`, a dataset repository on the HF hub containing a dataset script `'dataset_name.py'`.

        name (`str`, *optional*):
            Defining the name of the dataset configuration.
        data_dir (`str`, *optional*):
            Defining the `data_dir` of the dataset configuration. If specified for the generic builders (csv, text etc.) or the Hub datasets and `data_files` is `None`,
            the behavior is equal to passing `os.path.join(data_dir, **)` as `data_files` to reference all the files in a directory.
        data_files (`str` or `Sequence` or `Mapping`, *optional*):
            Path(s) to source data file(s).
        split (`Split` or `str`):
            Which split of the data to load.
            If `None`, will return a `dict` with all splits (typically `datasets.Split.TRAIN` and `datasets.Split.TEST`).
            If given, will return a single Dataset.
            Splits can be combined and specified like in tensorflow-datasets.
        cache_dir (`str`, *optional*):
            Directory to read/write data. Defaults to `"~/.cache/huggingface/datasets"`.
        features (`Features`, *optional*):
            Set the features type to use for this dataset.
        download_config ([`DownloadConfig`], *optional*):
            Specific download configuration parameters.
        download_mode ([`DownloadMode`] or `str`, defaults to `REUSE_DATASET_IF_EXISTS`):
            Download/generate mode.
        verification_mode ([`VerificationMode`] or `str`, defaults to `BASIC_CHECKS`):
            Verification mode determining the checks to run on the downloaded/processed dataset information (checksums/size/splits/...).

            .. versionadded:: 2.9.1
        keep_in_memory (`bool`, defaults to `None`):
            Whether to copy the dataset in-memory. If `None`, the dataset
            will not be copied in-memory unless explicitly enabled by setting `datasets.config.IN_MEMORY_MAX_SIZE` to
            nonzero. See more details in the [improve performance](../cache#improve-performance) section.
        save_infos (`bool`, defaults to `False`):
            Save the dataset information (checksums/size/splits/...).
        revision ([`Version`] or `str`, *optional*):
            Version of the dataset script to load.
            As datasets have their own git repository on the Datasets Hub, the default version "main" corresponds to their "main" branch.
            You can specify a different version than the default "main" by using a commit SHA or a git tag of the dataset repository.
        token (`str` or `bool`, *optional*):
            Optional string or boolean to use as Bearer token for remote files on the Datasets Hub.
            If `True`, or not specified, will get token from `"~/.huggingface"`.
        streaming (`bool`, defaults to `False`):
            If set to `True`, don't download the data files. Instead, it streams the data progressively while
            iterating on the dataset. An [`IterableDataset`] or [`IterableDatasetDict`] is returned instead in this case.

            Note that streaming works for datasets that use data formats that support being iterated over like txt, csv, jsonl for example.
            Json files may be downloaded completely. Also streaming from remote zip or gzip files is supported but other compressed formats
            like rar and xz are not yet supported. The tgz format doesn't allow streaming.
        num_proc (`int`, *optional*, defaults to `None`):
            Number of processes when downloading and generating the dataset locally.
            Multiprocessing is disabled by default.

            .. versionadded:: 2.7.0
        storage_options (`dict`, *optional*, defaults to `None`):
            **Experimental**. Key/value pairs to be passed on to the dataset file-system backend, if any.

            .. versionadded:: 2.11.0
        **config_kwargs (additional keyword arguments):
            Keyword arguments to be passed to the `BuilderConfig`
            and used in the [`DatasetBuilder`].

    Returns:
        Dataset
    """
    if split is None:
        raise Exception("Arg `split` is required for `from_huggingface`.")

    import datasets

    check_signature_compatible(
        from_huggingface,
        datasets.load_dataset,
        "Please use a compatible version of datasets.",
    )
    # For compatible different versions of API. e.g.
    # new API: from_huggingface(path, token=None)
    # old API: from_huggingface(path, use_auth_token=None)
    # then, from_huggingface("abc") can be forward from new API to old API
    # because non-compatible params are defaults values.
    kwargs = get_non_default_kwargs(locals(), from_huggingface)
    kwargs.pop("path")
    auto_rechunk = kwargs.pop("auto_rechunk", True)
    op = FromHuggingface(
        output_types=[OutputType.huggingface_dataset],
        path=path,
        auto_rechunk=auto_rechunk,
        hf_kwargs=kwargs,
    )
    return op().to_dataset()

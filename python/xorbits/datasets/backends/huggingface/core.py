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

from typing import Any, Dict, Mapping, Optional, Sequence, Union

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

from ...._mars.core import is_build_mode
from ...._mars.core.entity import (
    OutputType,
    register_fetch_class,
    register_output_types,
)
from ...._mars.core.entity.utils import refresh_tileable_shape
from ...._mars.core.operand.objects import ObjectFetch
from ...._mars.serialization.serializables import FieldTypes, ListField
from ....utils import check_signature_compatible, get_non_default_kwargs
from ...dataset import Dataset, DatasetChunk, DatasetChunkData, DatasetData
from .getitem import getitem
from .loader import load_huggingface_dataset
from .map import map
from .rechunk import rechunk
from .to_dataframe import to_dataframe


class HuggingfaceDatasetChunkData(DatasetChunkData):
    __slots__ = ()
    type_name = "HuggingfaceDatasetChunkData"

    @classmethod
    def get_params_from_data(cls, data) -> Dict[str, Any]:
        """For updating chunk shape from data."""
        return {"shape": data.shape}


class HuggingfaceDatasetChunk(DatasetChunk):
    __slots__ = ()
    _allow_data_type_ = (HuggingfaceDatasetChunkData,)
    type_name = "HuggingfaceDatasetChunk"


class HuggingfaceDatasetData(DatasetData):
    __slots__ = ()
    type_name = "Huggingface Dataset"

    _chunks = ListField(
        "chunks",
        FieldTypes.reference(HuggingfaceDatasetChunk),
        on_serialize=lambda x: [it.data for it in x] if x is not None else x,
        on_deserialize=lambda x: [HuggingfaceDatasetChunk(it) for it in x]
        if x is not None
        else x,
    )

    def __repr__(self):
        if is_build_mode() or len(self._executed_sessions) == 0:
            # in build mode, or not executed, just return representation
            return f"Huggingface Dataset <op={type(self.op).__name__}, key={self.key}>"
        else:
            try:
                return f"Dataset({{\n    features: {self.dtypes.index.values.tolist()},\n    num_rows: {self.shape[0]}\n}})"
            except:  # noqa: E722  # nosec  # pylint: disable=bare-except  # pragma: no cover
                return (
                    f"Huggingface Dataset <op={type(self.op).__name__}, key={self.key}>"
                )

    def refresh_params(self):
        refresh_tileable_shape(self)
        # TODO(codingl2k1): update dtypes.

    def rechunk(self, num_chunks: int, **kwargs):
        return rechunk(self, num_chunks, **kwargs)

    def map(self, fn, **kwargs):
        return map(self, fn, **kwargs)

    def to_dataframe(self, types_mapper=None):
        return to_dataframe(self, types_mapper)

    def __getitem__(self, item: Union[int, slice, str]):
        return getitem(self, item)


class HuggingfaceDataset(Dataset):
    __slots__ = ()
    _allow_data_type_ = (HuggingfaceDatasetData,)
    type_name = "Huggingface Dataset"

    def to_dataset(self):
        return Dataset(self.data)


register_output_types(
    OutputType.huggingface_dataset,
    (HuggingfaceDataset, HuggingfaceDatasetData),
    (HuggingfaceDatasetChunk, HuggingfaceDatasetChunkData),
)


class HuggingfaceDatasetFetch(ObjectFetch):
    _output_type_ = OutputType.huggingface_dataset


register_fetch_class(OutputType.huggingface_dataset, HuggingfaceDatasetFetch, None)


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
    return load_huggingface_dataset(**kwargs).to_dataset()

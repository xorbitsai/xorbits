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

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

from .repartition import repartition
from xorbits.data.dataset import DatasetData, Dataset
from xorbits.data.block import Block
from xorbits.data.operand import DataOperand, DataOperandMixin
from xorbits._mars.core.entity import OutputType
from xorbits._mars.remote import spawn
from xorbits._mars.typing import OperandType
from xorbits._mars.serialization.serializables import Int32Field


class HuggingfaceDatasetData(DatasetData):
    __slots__ = ()
    type_name = "Huggingface Dataset"

    def __repr__(self):
        return f"Huggingface Dataset <op={type(self.op).__name__}, key={self.key}>"

    def repartition(self, num_blocks: int, **kwargs):
        return repartition(self, num_blocks)

    def map(self, fn, **kwargs):
        pass


class HuggingfaceDataset(Dataset):
    __slots__ = ()
    _allow_data_type_ = (HuggingfaceDatasetData,)
    type_name = "Huggingface Dataset"

    def to_dataset(self):
        return Dataset(self.data)


def from_huggingface(path: str, **kwargs) -> Union[Dataset, Dict[str, Dataset]]:
    """Create a dataset from a Hugging Face Datasets Dataset.

    This function is not parallelized, and is intended to be used
    with Hugging Face Datasets that are loaded into memory (as opposed
    to memory-mapped).

    Example:

    .. doctest::

        >>> import ray
        >>> import datasets
        >>> hf_dataset = datasets.load_dataset("tweet_eval", "emotion")
        Downloading ...
        >>> ray_ds = ray.data.from_huggingface(hf_dataset)
        >>> ray_ds
        {'train': MaterializedDataset(
           num_blocks=...,
           num_rows=3257,
           schema={text: string, label: int64}
        ), 'test': MaterializedDataset(
           num_blocks=...,
           num_rows=1421,
           schema={text: string, label: int64}
        ), 'validation': MaterializedDataset(
           num_blocks=...,
           num_rows=374,
           schema={text: string, label: int64}
        )}
        >>> ray_ds = ray.data.from_huggingface(hf_dataset["train"])
        >>> ray_ds
        MaterializedDataset(
           num_blocks=...,
           num_rows=3257,
           schema={text: string, label: int64}
        )

    Args:
        dataset: A Hugging Face Dataset, or DatasetDict. IterableDataset is not
            supported. ``IterableDataset`` is not supported.

    Returns:
        Dataset holding Arrow records from the Hugging Face Dataset, or a dict of
            datasets in case dataset is a DatasetDict.
    """

    def _load_dataset():
        import datasets

        return datasets.load_dataset(path, **kwargs)

    return spawn(_load_dataset, output_types=[OutputType.huggingface_data]).to_dataset()

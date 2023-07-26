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

from typing import Callable

from .._mars.core.entity.objects import ObjectChunk, ObjectChunkData
from .._mars.core.entity.tileables import HasShapeTileable, HasShapeTileableData
from .._mars.serialization.serializables import FieldTypes, ListField, SeriesField


class DatasetChunkData(ObjectChunkData):
    __slots__ = ()
    type_name = "DatasetChunkData"


class DatasetChunk(ObjectChunk):
    __slots__ = ()
    _allow_data_type_ = (DatasetChunkData,)
    type_name = "DatasetChunk"


class DatasetData(HasShapeTileableData):
    __slots__ = ()
    type_name = "DatasetData"

    # optional fields
    dtypes = SeriesField("dtypes")
    _chunks = ListField(
        "chunks",
        FieldTypes.reference(DatasetChunk),
        on_serialize=lambda x: [it.data for it in x] if x is not None else x,
        on_deserialize=lambda x: [DatasetChunk(it) for it in x] if x is not None else x,
    )

    def __init__(self, op=None, nsplits=None, chunks=None, shape=None, **kw):
        super().__init__(_op=op, _nsplits=nsplits, _chunks=chunks, _shape=shape, **kw)

    def __repr__(self):
        return f"Dataset <op={type(self.op).__name__}, key={self.key}>"

    @property
    def params(self):
        d = super().params
        d.update({"dtypes": self.dtypes})
        return d

    @params.setter
    def params(self, params):
        self._shape = params.pop("shape", None)
        self.dtypes = params.pop("dtypes", None)

    def refresh_params(self):
        # refresh params when chunks updated
        # nothing needs to do for Object
        pass

    def rechunk(self, num_chunks: int, **kwargs):
        raise NotImplementedError

    def map(self, fn, **kwargs):
        raise NotImplementedError

    def to_dataframe(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError


class Dataset(HasShapeTileable):
    __slots__ = ()
    _allow_data_type_ = (DatasetData,)
    type_name = "Dataset"

    def rechunk(self, num_chunks: int, **kwargs):
        """Split the internal data into chunks.

        Currently, `rechunk()` only works for single chunk dataset.

        Parameters
        ----------
        num_chunks: int
            The number of chunks.
        kwargs
            Preserved.

        Returns
        -------
            Dataset
        """
        return self.data.rechunk(num_chunks, **kwargs)

    def map(self, fn: Callable, **kwargs):
        """Apply fn to each row of the dataset.

        Parameters
        ----------
        fn: Callable
            The callable object, the signature is `function(example: Dict[str, Any]) -> Dict[str, Any]`.
        kwargs:
            The kwargs are passed to the underlying engine, e.g. the **kwargs
            will be passed to `datasets.Dataset.map` if the Dataset is constructed
            from_huggingface, please refer to:
            `datasets.Dataset.map <https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.Dataset.map>`_.

        Returns
        -------
            Dataset
        Examples
        --------
        >>> import xorbits.datasets as xdatasets
        >>> ds = xdatasets.from_huggingface("rotten_tomatoes", split="validation")
        >>> def add_prefix(example):
        ...     example["text"] = "Review: " + example["text"]
        ...     return example
        >>> ds = ds.map(add_prefix)
        """
        return self.data.map(fn, **kwargs)

    def to_dataframe(self):
        """Convert the dataset to xorbits dataframe.

        The conversion will be chunk to chunk.

        Returns
        -------
            DataFrame
        Examples
        --------
        >>> import xorbits.datasets as xdatasets
        >>> ds = xdatasets.from_huggingface("rotten_tomatoes", split="train")
        >>> df = ds.to_dataframe()
        """
        return self.data.to_dataframe()

    def __getitem__(self, item):
        return self.data.__getitem__(item)

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

from typing import Any, Callable, Dict, Union

import numpy as np

from .._mars.core.entity.chunks import Chunk, ChunkData
from .._mars.core.entity.tileables import HasShapeTileable, HasShapeTileableData
from .._mars.serialization.serializables import (
    FieldTypes,
    ListField,
    SeriesField,
    TupleField,
)
from .._mars.utils import on_deserialize_shape, on_serialize_shape


# DatasetChunk and DatasetChunkData can't inherit ObjectChunk and ObjectChunkData,
# because _get_output_type_by_cls() in xorbits/_mars/core/entity/output_types.py
# will generate an incorrect object output type for the DatasetChunk and DatasetChunkData.
class DatasetChunkData(ChunkData):
    __slots__ = ()
    type_name = "DatasetChunkData"

    # required for get shape of chunk, e.g. getitem.
    shape = TupleField(
        "shape",
        FieldTypes.int64,
        on_serialize=on_serialize_shape,
        on_deserialize=on_deserialize_shape,
    )

    def __init__(self, op=None, index=None, shape=None, **kwargs):
        # CheckedTaskPreprocessor._check_nsplits may check shape,
        # so the shape can't be None.
        super().__init__(
            _op=op, _index=index, shape=shape or (np.nan, np.nan), **kwargs
        )

    @property
    def params(self):
        return {
            "shape": self.shape,
            "index": self.index,
        }

    @params.setter
    def params(self, params):
        params.pop("index", None)  # index not needed to update
        self.shape = params.pop("shape", None)

    @classmethod
    def get_params_from_data(cls, data: Any) -> Dict[str, Any]:
        return dict()


class DatasetChunk(Chunk):
    __slots__ = ()
    _allow_data_type_ = (DatasetChunkData,)
    type_name = "DatasetChunk"


class DatasetData(HasShapeTileableData):
    __slots__ = ()
    type_name = "DatasetData"

    # required for to_dataframe.
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

    def to_dataframe(self, types_mapper=None):
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

    def to_dataframe(self, types_mapper=None):
        """Convert the dataset to xorbits dataframe.

        The conversion will be chunk to chunk.

        Parameters
        ----------
        types_mapper: Callable
            The types mapper to pandas dataframe.

        Returns
        -------
            DataFrame
        Examples
        --------
        >>> import xorbits.datasets as xdatasets
        >>> ds = xdatasets.from_huggingface("rotten_tomatoes", split="train")
        >>> df = ds.to_dataframe()
        """
        return self.data.to_dataframe(types_mapper)

    def __getitem__(self, item: Union[int, slice, str]):
        """Get rows or columns from dataset.

        The result will be formatted dict or list.

        Parameters
        ----------
        item: Union[int, slice, str]
            The item index. slice does not support steps, e.g. ds[1:3:2] is not supported.

        Returns
        -------
            List of column values if item is str.
            Dict[column name, List[values of selected rows]] if item is int or slice.
        Examples
        --------
        >>> import xorbits.datasets as xdatasets
        >>> ds = xdatasets.from_huggingface("rotten_tomatoes", split="train")
        >>> ds[1:3]["text"]
        """
        return self.data.__getitem__(item)

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

import os
from typing import Any, Dict, Optional, Union

from ...._mars.core import is_build_mode
from ...._mars.core.entity import (
    OutputType,
    register_fetch_class,
    register_output_types,
)
from ...._mars.core.entity.utils import refresh_tileable_shape
from ...._mars.core.operand.objects import ObjectFetch
from ...._mars.serialization.serializables import FieldTypes, ListField
from ...dataset import Dataset, DatasetChunk, DatasetChunkData, DatasetData
from .export import export
from .getitem import getitem
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

    def export(
        self,
        path: Union[str, os.PathLike],
        storage_options: Optional[dict] = None,
        create_if_not_exists: Optional[bool] = True,
        max_chunk_rows: Optional[int] = None,
        column_groups: Optional[dict] = None,
        num_threads: Optional[int] = None,
        version: Optional[str] = None,
        overwrite: Optional[bool] = True,
    ):
        return export(
            self,
            path,
            storage_options,
            create_if_not_exists,
            max_chunk_rows,
            column_groups,
            num_threads,
            version,
            overwrite,
        )

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

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

from typing import Any, Dict

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


class ArrowDatasetChunkData(DatasetChunkData):
    __slots__ = ()
    type_name = "ArrowDatasetChunkData"

    @classmethod
    def get_params_from_data(cls, data) -> Dict[str, Any]:
        """For updating chunk shape from data."""
        return {"shape": data.shape}


class ArrowDatasetChunk(DatasetChunk):
    __slots__ = ()
    _allow_data_type_ = (ArrowDatasetChunkData,)
    type_name = "ArrowDatasetChunk"


class ArrowDatasetData(DatasetData):
    __slots__ = ()
    type_name = "Arrow Dataset"

    _chunks = ListField(
        "chunks",
        FieldTypes.reference(ArrowDatasetChunk),
        on_serialize=lambda x: [it.data for it in x] if x is not None else x,
        on_deserialize=lambda x: [ArrowDatasetChunk(it) for it in x]
        if x is not None
        else x,
    )

    def __repr__(self):
        if is_build_mode() or len(self._executed_sessions) == 0:
            # in build mode, or not executed, just return representation
            return f"Arrow Dataset <op={type(self.op).__name__}, key={self.key}>"
        else:
            try:
                return f"Dataset({{\n    features: {self.dtypes.index.values.tolist()},\n    num_rows: {self.shape[0]}\n}})"
            except:  # noqa: E722  # nosec  # pylint: disable=bare-except  # pragma: no cover
                return f"Arrow Dataset <op={type(self.op).__name__}, key={self.key}>"

    def refresh_params(self):
        refresh_tileable_shape(self)
        # TODO(codingl2k1): update dtypes.


class ArrowDataset(Dataset):
    __slots__ = ()
    _allow_data_type_ = (ArrowDatasetData,)
    type_name = "Arrow Dataset"

    def to_dataset(self):
        return Dataset(self.data)


register_output_types(
    OutputType.arrow_dataset,
    (ArrowDataset, ArrowDatasetData),
    (ArrowDatasetChunk, ArrowDatasetChunkData),
)


class ArrowDatasetFetch(ObjectFetch):
    _output_type_ = OutputType.arrow_dataset


register_fetch_class(OutputType.arrow_dataset, ArrowDatasetFetch, None)

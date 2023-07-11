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

from .block import Block
from .._mars.core.entity.objects import Object, ObjectData
from .._mars.serialization.serializables import FieldTypes, ListField


class DatasetData(ObjectData):
    __slots__ = ()
    type_name = "DatasetData"

    # optional fields
    _chunks = ListField(
        "chunks",
        FieldTypes.reference(Block),
        on_serialize=lambda x: [it.data for it in x] if x is not None else x,
        on_deserialize=lambda x: [Block(it) for it in x] if x is not None else x,
    )

    def __repr__(self):
        return f"Dataset <op={type(self.op).__name__}, key={self.key}>"

    def repartition(self, num_blocks: int, **kwargs):
        raise NotImplementedError

    def map(self, fn, **kwargs):
        raise NotImplementedError


class Dataset(Object):
    __slots__ = ()
    _allow_data_type_ = (DatasetData,)
    type_name = "Dataset"

    def repartition(self, num_blocks: int, **kwargs):
        return self.data.repartition(num_blocks, **kwargs)

    def map(self, fn, **kwargs):
        return self.data.map(fn, **kwargs)

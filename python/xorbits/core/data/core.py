# Copyright 2022 XProbe Inc.
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


from typing import Any

from core.adapter.mars import get_data_cls, get_data_ref_cls
from core.adapter.typing import DataType


class Data:
    def __init__(self, *args, **kwargs):
        self._data_type = self._parse_data_type(*args, **kwargs)
        self._impl = self._materialize(self._data_type, *args, **kwargs)

    @staticmethod
    def _parse_data_type(*args, **kwargs) -> DataType:
        # TODO implement
        return DataType.dataframe

    @staticmethod
    def _materialize(data_type: DataType, *args, **kwargs) -> Any:
        cls = get_data_cls(data_type)
        return cls(*args, **kwargs)

    @property
    def data_type(self) -> DataType:
        return self._data_type

    @property
    def impl(self) -> Any:
        return self._impl

    def __getattr__(self, item):
        return getattr(self._impl, item)


class DataReference:
    def __init__(self, data: Data, *args, **kwargs):
        self._data = data
        self._impl = self._materialize(data, *args, **kwargs)

    @staticmethod
    def _materialize(data: Data, *args, **kwargs):
        cls = get_data_ref_cls(data.data_type)
        return cls(data.impl, *args, **kwargs)

    def __getattr__(self, item):
        return getattr(self._impl, item)

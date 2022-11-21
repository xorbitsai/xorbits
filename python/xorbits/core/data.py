# -*- coding: utf-8 -*-
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

from enum import Enum
from typing import TYPE_CHECKING, Any, Iterable, List

if TYPE_CHECKING:  # pragma: no cover
    from ..core.adapter import MarsEntity


class DataType(Enum):
    object_ = 1
    scalar = 2
    tensor = 3
    dataframe = 4
    series = 5
    index = 6
    categorical = 7
    dataframe_groupby = 8
    series_groupby = 9


class Data:
    data_type: DataType

    __fields: List[str] = [
        "data_type",
        "_mars_entity",
    ]

    def __dir__(self) -> Iterable[str]:
        return dir(self._mars_entity)

    def __init__(self, *args, **kwargs):
        self.data_type: DataType = kwargs.pop("data_type")
        self._mars_entity = kwargs.pop("mars_entity", None)
        if len(args) > 0 or len(kwargs) > 0:
            raise TypeError(f"Unexpected args {args} or kwargs {kwargs}.")

    @classmethod
    def from_mars(cls, mars_entity: "MarsEntity") -> "Data":
        from ..core.adapter import (
            MARS_CATEGORICAL_TYPE,
            MARS_DATAFRAME_GROUPBY_TYPE,
            MARS_DATAFRAME_TYPE,
            MARS_INDEX_TYPE,
            MARS_SERIES_GROUPBY_TYPE,
            MARS_SERIES_TYPE,
            MARS_TENSOR_TYPE,
        )

        if isinstance(mars_entity, MARS_DATAFRAME_TYPE):
            data_type = DataType.dataframe
        elif isinstance(mars_entity, MARS_SERIES_TYPE):
            data_type = DataType.series
        elif isinstance(mars_entity, MARS_DATAFRAME_GROUPBY_TYPE):
            data_type = DataType.dataframe_groupby
        elif isinstance(mars_entity, MARS_SERIES_GROUPBY_TYPE):
            data_type = DataType.series_groupby
        elif isinstance(mars_entity, MARS_TENSOR_TYPE):
            data_type = DataType.tensor
        elif isinstance(mars_entity, MARS_INDEX_TYPE):
            data_type = DataType.index
        elif isinstance(mars_entity, MARS_CATEGORICAL_TYPE):
            data_type = DataType.categorical
        else:
            raise NotImplementedError(f"Unsupported mars type {type(mars_entity)}")
        return Data(mars_entity=mars_entity, data_type=data_type)

    def __getattr__(self, item: str):
        from .adapter import MarsProxy

        return MarsProxy.getattr(self.data_type, self._mars_entity, item)

    def __setattr__(self, key: str, value: Any):
        from .adapter import MarsProxy

        if key in self.__fields:
            object.__setattr__(self, key, value)
        else:
            return MarsProxy.setattr(self._mars_entity, key, value)

    def __str__(self):
        if self._mars_entity is not None:
            return self._mars_entity.__str__()
        else:
            return super().__str__()

    def __repr__(self):
        if self._mars_entity is not None:
            return self._mars_entity.__repr__()
        else:
            return super().__repr__()


class DataRef:
    data: Data

    __fields = ["data"]

    def __dir__(self) -> Iterable[str]:
        return dir(self.data)

    def __init__(self, data: Data):
        self.data = data

    def __getattr__(self, item):
        return getattr(self.data, item)

    def __setattr__(self, key, value):
        if key in self.__fields:
            object.__setattr__(self, key, value)
        else:
            self.data.__setattr__(key, value)

    def __str__(self):
        from .execution import execute

        execute(self)
        return self.data.__str__()

    def __repr__(self):
        from .execution import execute

        execute(self)
        return self.data.__repr__()

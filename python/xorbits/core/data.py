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

from collections import defaultdict
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Type

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


DATA_MEMBERS: Dict[DataType, Dict[str, Any]] = defaultdict(dict)


class Data:
    data_type: DataType

    __fields: List[str] = [
        "data_type",
        "_mars_entity",
    ]

    def __dir__(self) -> Iterable[str]:
        return list(DATA_MEMBERS[self.data_type])

    def __init__(self, *args, **kwargs):
        self.data_type: DataType = kwargs.pop("data_type")
        self._mars_entity = kwargs.pop("mars_entity", None)
        if len(args) > 0 or len(kwargs) > 0:  # pragma: no cover
            raise TypeError(f"Unexpected args {args} or kwargs {kwargs}.")

    @classmethod
    def from_mars(cls, mars_entity: "MarsEntity") -> "Data":
        from ..core.adapter import (
            MARS_CATEGORICAL_TYPE,
            MARS_DATAFRAME_GROUPBY_TYPE,
            MARS_DATAFRAME_TYPE,
            MARS_INDEX_TYPE,
            MARS_OBJECT_TYPE,
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
        elif isinstance(mars_entity, MARS_OBJECT_TYPE):
            data_type = DataType.object_
        else:
            raise NotImplementedError(f"Unsupported mars type {type(mars_entity)}")
        return Data(mars_entity=mars_entity, data_type=data_type)

    def __setattr__(self, key: str, value: Any):
        from .adapter import MemberProxy

        if key in self.__fields:
            object.__setattr__(self, key, value)
        else:
            return MemberProxy.setattr(self._mars_entity, key, value)

    def __str__(self):
        if self._mars_entity is not None:
            return self._mars_entity.__str__()
        else:  # pragma: no cover
            return super().__str__()

    def __repr__(self):
        if self._mars_entity is not None:
            return self._mars_entity.__repr__()
        else:
            return super().__repr__()


class DataRefMeta(type):
    """
    Used to bind methods to subclasses of DataRef according to their data type.
    """

    __cls_members: Dict[str, Any]

    def __new__(mcs, name, bases, dct):
        cls = super().__new__(mcs, name, bases, dct)
        if name == "DataFrame":
            setattr(cls, "__cls_members", DATA_MEMBERS[DataType.dataframe])
        elif name == "Series":
            setattr(cls, "__cls_members", DATA_MEMBERS[DataType.series])
        elif name == "Index":
            setattr(cls, "__cls_members", DATA_MEMBERS[DataType.index])
        elif name == "DataFrameGroupBy":
            setattr(cls, "__cls_members", DATA_MEMBERS[DataType.dataframe_groupby])
        elif name == "SeriesGroupBy":
            setattr(cls, "__cls_members", DATA_MEMBERS[DataType.series_groupby])
        elif name == "ndarray":
            setattr(cls, "__cls_members", DATA_MEMBERS[DataType.tensor])
        return cls

    def __getattr__(cls, item: str):
        members = object.__getattribute__(cls, "__cls_members")
        if item not in members:
            raise AttributeError(item)
        else:
            return members[item]

    def __instancecheck__(cls: Type, instance: Any) -> bool:
        if not issubclass(instance.__class__, DataRef):
            # not a DataRef instance.
            return False

        if cls is DataRef:
            # isinstance(x, DataRef).
            return cls in instance.__class__.__mro__
        else:
            # for subclass like isinstance(x, DataFrame),
            # check its data_type if match with cls
            data_type = instance.data.data_type
            try:
                return data_type == SUB_CLASS_TO_DATA_TYPE[cls]
            except KeyError:
                # subclassing DataRef subclasses is not allowed.
                raise TypeError(f"Illegal subclass {instance.__class__.__name__}")


class DataRef(metaclass=DataRefMeta):
    data: Data

    __fields = ["data"]

    def __dir__(self) -> Iterable[str]:
        return dir(self.data)

    def __init__(self, data: Data):
        self.data = data

    def __getattr__(self, item):
        from .adapter import MemberProxy

        return MemberProxy.getattr(self, item)

    def __setattr__(self, key, value):
        if key in self.__fields:
            object.__setattr__(self, key, value)
        else:
            self.data.__setattr__(key, value)

    def __str__(self):
        from .execution import run

        run(self)
        return self.data.__str__()

    def __repr__(self):
        from .execution import run

        run(self)
        return self.data.__repr__()


SUB_CLASS_TO_DATA_TYPE: Dict[Type[DataRef], DataType] = dict()


def register_cls_to_type(data_type: DataType) -> Callable:
    def _wrap_cls(cls: Type[DataRef]):
        SUB_CLASS_TO_DATA_TYPE[cls] = data_type
        return cls

    return _wrap_cls

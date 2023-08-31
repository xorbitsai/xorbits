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
import warnings
from collections import defaultdict
from enum import Enum
from itertools import count
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Type

from pandas.api.types import is_float_dtype, is_integer_dtype

from ..utils import safe_repr_str

if TYPE_CHECKING:  # pragma: no cover
    from ..core.adapter import MarsEntity

import numpy as np


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
    dataset = 10


DATA_MEMBERS: Dict[DataType, Dict[str, Any]] = defaultdict(dict)


class AutoConversionType(Enum):
    int_conversion = 1
    float_conversion = 2

    def convert(self, val):
        if self == AutoConversionType.int_conversion:
            return int(val)
        else:
            return float(val)

    def generate_error_msg(self, val):
        if self == AutoConversionType.int_conversion:
            return f"{val} object cannot be interpreted as an integer."
        else:
            return f"{val} object cannot be interpreted as a float."


class Data:
    data_type: DataType

    __fields: List[str] = [
        "data_type",
        "_mars_entity",
    ]

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

        # import MARS_DATASET_TYPE from datasets instead of core.adapter
        # to avoid recursive import.
        from ..datasets import MARS_DATASET_TYPE

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
        elif isinstance(mars_entity, MARS_DATASET_TYPE):
            data_type = DataType.dataset
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
        else:  # pragma: no cover
            return super().__repr__()

    def __array__(self):
        if self._mars_entity is not None:
            return self._mars_entity.__array__()
        else:  # pragma: no cover
            raise AttributeError("__array__")


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
        return list(DATA_MEMBERS[self.data.data_type])

    def __init__(self, data: Data):
        self.data = data

    def __getattr__(self, item):
        from .adapter import MemberProxy

        return MemberProxy.getattr(self, item)

    def __setattr__(self, key, value):
        try:
            if key in self.__fields:
                object.__setattr__(self, key, value)
            else:
                self.data.__setattr__(key, value)
        except AttributeError:
            if key in self.dtypes:
                self.__setitem__(key, value)
            else:
                warnings.warn(
                    "xorbits.pandas doesn't allow columns to be created via a new attribute name.",
                    UserWarning,
                )
                object.__setattr__(self, key, value)

    def _own_data(self):
        from .adapter import own_data

        return own_data(self.data._mars_entity)

    def __iter__(self):
        # Mars entity hasn't implemented __iter__, however `iter(mars_entity)`
        # still works, it's because iteration is supported by `__getitem__` that
        # accepts integers 0,1,.., it can be seen as a "legacy feature" that not
        # recommended. Here we implement __iter__ for some data types, others keep
        # behaviors with Mars.
        if self._own_data():
            # if own data, return iteration on data directly
            yield from iter(self.data._mars_entity.op.data)
        else:
            if self.data.data_type == DataType.dataframe:
                yield from iter(self.data._mars_entity.dtypes.index)
            elif self.data.data_type == DataType.series:
                for batch_data in self.data._mars_entity.iterbatch():
                    yield from batch_data.__iter__()
            elif self.data.data_type == DataType.index:
                for batch_data in self.data._mars_entity.to_series().iterbatch():
                    yield from batch_data.__iter__()
            else:

                def gen():
                    counter = count()
                    while True:
                        try:
                            yield self.__getitem__(next(counter))
                        except IndexError:
                            break

                yield from gen()

    def __len__(self):
        from .._mars.core import HasShapeTileable
        from .execution import run

        if isinstance(self.data._mars_entity, HasShapeTileable):
            try:
                return int(self.data._mars_entity.shape[0])
            except (IndexError, ValueError):
                # shape is unknown, execute it
                run(self)
                try:
                    return int(self.data._mars_entity.shape[0])
                except (IndexError, ValueError):  # happens when dimension is 0
                    raise TypeError(
                        f"object with shape {self.data._mars_entity.shape} has no len()"
                    )

        else:
            raise TypeError(f"object of type '{self.data.data_type}' has no len()")

    @property
    def shape(self):
        from .._mars.core import HasShapeTileable
        from .execution import run

        if isinstance(self.data._mars_entity, HasShapeTileable):
            if np.isnan(self.data._mars_entity.shape).any():
                run(self)
            return self.data._mars_entity.shape
        else:
            raise TypeError(f"object of type '{self.data.data_type}' has no shape")

    @safe_repr_str
    def __str__(self):
        from .execution import run

        if self._own_data():
            # skip execution if own data
            return self.data._mars_entity.op.data.__str__()
        else:
            run(self)
            return self.data.__str__()

    @safe_repr_str
    def __repr__(self):
        from .execution import run

        if self._own_data():
            # skip execution if own data
            return self.data._mars_entity.op.data.__repr__()
        else:
            run(self)
            return self.data.__repr__()

    def __array__(self):
        from .execution import run

        if self._own_data():
            return self.data._mars_entity.op.data.__array__()
        else:
            run(self)
            return self.data.__array__()

    def _to_int_or_float(self, conversion: AutoConversionType, cast: bool = False):
        from .execution import run

        data_type = self.data.data_type
        if (
            data_type == DataType.tensor
            and len(self.shape) == 0
            and (is_integer_dtype(self.dtype) or is_float_dtype(self.dtype))
        ):
            run(self)
            return conversion.convert(self.to_numpy())
        elif data_type == DataType.object_:
            run(self)
            data_object = self.to_object()
            if cast or isinstance(data_object, int) or isinstance(data_object, float):
                return conversion.convert(self.to_object())
            else:
                raise TypeError(conversion.generate_error_msg(self.data.data_type))
        else:
            raise TypeError(conversion.generate_error_msg(self.data.data_type))

    def __bool__(self):
        data_type = self.data.data_type
        from .execution import run

        if data_type == DataType.tensor:
            if len(self.shape) == 0:
                run(self)
                return bool(self.to_numpy())
            else:
                if self.__len__() <= 1:
                    run(self)
                    return bool(self.to_numpy())
                else:
                    raise ValueError(
                        f"ValueError: The truth value of a {data_type} with more than one element is ambiguous. "
                        f"Use a.any() or a.all()"
                    )
        elif data_type == DataType.object_:
            run(self)
            return bool(self.to_object())
        elif (
            data_type == DataType.dataframe
            or data_type == DataType.series
            or data_type == DataType.index
        ):
            raise ValueError(
                f"The truth value of a {data_type} is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all()."
            )
        else:
            raise ValueError(f"{data_type} cannot be converted to boolean values.")

    def __int__(self):
        return self._to_int_or_float(AutoConversionType.int_conversion, cast=True)

    def __float__(self):
        return self._to_int_or_float(AutoConversionType.float_conversion, cast=True)

    def __index__(self):
        try:
            val = self._to_int_or_float(AutoConversionType.int_conversion, cast=False)
            return val
        except TypeError:
            raise TypeError(
                f"{self.data.data_type} object cannot be interpreted as an integer."
            )


SUB_CLASS_TO_DATA_TYPE: Dict[Type[DataRef], DataType] = dict()


def register_cls_to_type(data_type: DataType) -> Callable:
    def _wrap_cls(cls: Type[DataRef]):
        SUB_CLASS_TO_DATA_TYPE[cls] = data_type
        return cls

    return _wrap_cls

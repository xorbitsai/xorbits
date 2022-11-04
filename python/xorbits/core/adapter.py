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

# To avoid possible naming conflict, mars functions and classes should be renamed.
# Functions should be renamed by adding a prefix 'mars_', and classes should be renamed
# by adding a prefix 'Mars'.

import functools
from collections import defaultdict
from typing import Any, Callable, Dict, List, Tuple, Type, Union

# For maintenance, any module wants to import from mars, it should import from here.
from .._mars import dataframe as mars_dataframe
from .._mars import execute as mars_execute
from .._mars import new_session as mars_new_session
from .._mars import stop_server as mars_stop_server
from .._mars.core import Entity as MarsEntity
from .._mars.dataframe.base.accessor import CachedAccessor as MarsCachedAccessor
from .._mars.dataframe.base.accessor import DatetimeAccessor as MarsDatetimeAccessor
from .._mars.dataframe.base.accessor import StringAccessor as MarsStringAccessor
from .._mars.dataframe.core import CATEGORICAL_TYPE as MARS_CATEGORICAL_TYPE
from .._mars.dataframe.core import DATAFRAME_GROUPBY_TYPE as MARS_DATAFRAME_GROUPBY_TYPE
from .._mars.dataframe.core import DATAFRAME_TYPE as MARS_DATAFRAME_TYPE
from .._mars.dataframe.core import INDEX_TYPE as MARS_INDEX_TYPE
from .._mars.dataframe.core import SERIES_GROUPBY_TYPE as MARS_SERIES_GROUPBY_TYPE
from .._mars.dataframe.core import SERIES_TYPE as MARS_SERIES_TYPE
from .._mars.dataframe.core import DataFrame as MarsDataFrame
from .._mars.dataframe.core import DataFrameGroupBy as MarsDataFrameGroupBy
from .._mars.dataframe.core import Index as MarsIndex
from .._mars.dataframe.core import Series as MarsSeries
from .._mars.dataframe.indexing.loc import DataFrameLoc as MarsDataFrameLoc
from .._mars.tensor.core import TENSOR_TYPE as MARS_TENSOR_TYPE
from .data import Data, DataRef, DataType

_mars_entity_type_to_execution_condition: Dict[
    str, List[Callable[["MarsEntity"], bool]]
] = defaultdict(list)


def register_execution_condition(
    mars_entity_type: str, condition: Callable[["MarsEntity"], bool]
):
    _mars_entity_type_to_execution_condition[mars_entity_type].append(condition)


_mars_type_to_converters: Dict[Type, Callable] = {}


def _get_converter(from_cls: Type):
    if from_cls in _mars_type_to_converters:
        return _mars_type_to_converters[from_cls]
    for k, v in _mars_type_to_converters.items():
        if issubclass(from_cls, k):
            _mars_type_to_converters[from_cls] = v
            return v
    return None


def register_converter(from_cls: Type):
    """
    A decorator for convenience of registering a class converter.
    """

    def decorate(cls: Type):
        assert from_cls not in _mars_type_to_converters
        _mars_type_to_converters[from_cls] = cls
        return cls

    return decorate


def wrap_magic_method(method_name: str) -> Callable[[Any], Any]:
    def wrapped(self: DataRef, *args, **kwargs):
        mars_entity = getattr(self.data, "_mars_entity", None)
        if (mars_entity is None) or (
            not hasattr(mars_entity, method_name)
        ):  # pragma: no cover
            raise AttributeError(
                f"'{self.data.data_type.name}' object has no attribute '{method_name}'"
            )
        else:
            return wrap_mars_callable(getattr(mars_entity, method_name))(
                *args, **kwargs
            )

    return wrapped


class MarsProxy:
    @classmethod
    def getattr(cls, data_type: DataType, mars_entity: MarsEntity, item: str):
        attr = getattr(mars_entity, item, None)

        if attr is None:
            # TODO: pandas implementation
            raise AttributeError(f"'{data_type.name}' object has no attribute '{item}'")
        elif callable(attr):
            return wrap_mars_callable(attr)
        else:
            # e.g. string accessor
            return from_mars(attr)

    @classmethod
    def setattr(cls, mars_entity: MarsEntity, key: str, value: Any):
        if type(getattr(type(mars_entity), key)) is property:
            # call the setter of the specified property.
            getattr(type(mars_entity), key).fset(mars_entity, to_mars(value))
        else:
            mars_entity.__setattr__(key, value)


def to_mars(inp: Union[DataRef, Tuple, List, Dict]):
    """
    Convert xorbits data references to mars entities and execute them if needed.
    """

    if isinstance(inp, DataRef):
        mars_entity = getattr(inp.data, "_mars_entity", None)
        if mars_entity is None:
            raise TypeError(f"Can't covert {inp} to mars entity")
        # trigger execution
        conditions = _mars_entity_type_to_execution_condition[
            type(mars_entity).__name__
        ]
        for cond in conditions:
            if cond(mars_entity):
                from .execution import execute

                execute(inp)
        return mars_entity
    elif isinstance(inp, tuple):
        return tuple(to_mars(i) for i in inp)
    elif isinstance(inp, list):
        return list(to_mars(i) for i in inp)
    elif isinstance(inp, dict):
        return dict((k, to_mars(v)) for k, v in inp.items())
    else:
        return inp


def from_mars(inp: Union[MarsEntity, tuple, list, dict]):
    """
    Convert mars entities to xorbits data references.
    """
    converter = _get_converter(type(inp))
    if converter is not None:
        return converter(inp)
    elif isinstance(inp, MarsEntity):
        return DataRef(Data.from_mars(inp))
    elif isinstance(inp, tuple):
        return tuple(from_mars(i) for i in inp)
    elif isinstance(inp, list):
        return list(from_mars(i) for i in inp)
    elif isinstance(inp, dict):
        return dict((k, from_mars(v)) for k, v in inp.items())
    else:
        return inp


def wrap_mars_callable(c):
    """
    A function wrapper that makes arguments of the wrapped method be mars compatible type and
    return value be xorbits compatible type.
    """

    @functools.wraps(c)
    def wrapped(*args, **kwargs):
        return from_mars(c(*to_mars(args), **to_mars(kwargs)))

    return wrapped

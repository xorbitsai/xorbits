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

import functools
from collections import defaultdict
from typing import Any, Callable, Dict, List, Tuple, Type, Union

from ..adapter.mars import MarsEntity
from .execution import execute


class Proxy:
    def __init__(self, proxied: Any):
        self._proxied = proxied

    @property
    def proxied(self):
        return self._proxied


class XorbitsData(Proxy):

    _mars_entity_type_to_execution_condition: Dict[
        str, List[Callable[["MarsEntity"], bool]]
    ] = defaultdict(list)

    _mars_type_to_converters: Dict[str, Callable] = {}

    # TODO: use registered methods for better performance and code completion.
    # _mars_entity_type_to_methods: Dict[str, Dict[str, Callable]] = defaultdict(dict)

    def __init__(self, mars_entity: MarsEntity):
        super().__init__(mars_entity)
        self._mars_entity_type = type(mars_entity).__name__

        # TODO
        # trigger execution
        conditions = self._mars_entity_type_to_execution_condition[
            type(self._proxied).__name__
        ]
        for cond in conditions:
            if cond(self._proxied):
                execute(self._proxied)

    def __getattr__(self, item):
        # TODO: use registered methods for better performance and code completion.
        # if item in self._mars_entity_type_to_methods[self._mars_entity_type]:
        #     return self._mars_entity_type_to_methods[self._mars_entity_type][item]

        attr = getattr(self._proxied, item)
        if isinstance(attr, MarsEntity):
            # e.g. DataFrameTranspose
            return from_mars(attr)
        elif type(attr).__name__ in self._mars_type_to_converters:
            # e.g. DataFrameLoc
            return self._mars_type_to_converters[type(attr).__name__](attr)
        elif hasattr(attr, "__call__"):
            return wrap_mars_callable(attr)
        else:
            # e.g. string accessor
            return attr

    def __str__(self):
        execute(self._proxied)
        return self._proxied.__str__()

    def __repr__(self):
        execute(self._proxied)
        return self._proxied.__repr__()

    @classmethod
    def register_execution_condition(
        cls, mars_entity_type: str, condition: Callable[["MarsEntity"], bool]
    ):
        cls._mars_entity_type_to_execution_condition[mars_entity_type].append(condition)

    @classmethod
    def register_converter(cls, class_name: str, converter: Callable):
        assert class_name not in cls._mars_type_to_converters
        cls._mars_type_to_converters[class_name] = converter

    # TODO: use registered methods for better performance and code completion.
    # @classmethod
    # def register_method(cls, mars_entity_type: str, method_name: str, method: Callable):
    #     cls._mars_entity_type_to_methods[mars_entity_type][
    #         method_name
    #     ] = wrap_mars_callable(method)


class XorbitsDataRef(Proxy):
    def __init__(self, data: XorbitsData):
        super().__init__(data)

    def __getattr__(self, item):
        return getattr(self._proxied, item)

    def __str__(self):
        return self._proxied.__str__()

    def __repr__(self):
        return self._proxied.__repr__()


def to_mars(inp: Union["XorbitsDataRef", Tuple, List, Dict]):
    """
    Convert xorbits data references to mars entities.
    """

    if isinstance(inp, XorbitsDataRef):
        return inp.proxied.proxied
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

    if isinstance(inp, MarsEntity):
        return XorbitsDataRef(data=XorbitsData(mars_entity=inp))
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


def register_converter(from_cls: Type):
    """
    A decorator for convenience of registering a class converter.
    """

    def decorate(cls: Type):
        XorbitsData.register_converter(from_cls.__name__, lambda x: cls(x))
        return cls

    return decorate

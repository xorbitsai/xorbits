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
from typing import Callable, Dict, List, Tuple, Type, Union

from ..adapter.mars import MarsEntity
from .data import Data, DataRef

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


@register_converter(from_cls=MarsEntity)
class DataRefMarsImpl(DataRef):
    def __init__(self, mars_entity: "MarsEntity"):
        super().__init__(data=DataMarsImpl(mars_entity))


class DataMarsImpl(Data):
    """
    A proxy of a mars entity.
    """

    def __init__(self, mars_entity: "MarsEntity"):
        self._mars_entity = mars_entity

    def __getattr__(self, item):
        attr = getattr(self._mars_entity, item)

        if callable(attr):
            return wrap_mars_callable(attr)
        else:
            # e.g. string accessor
            return from_mars(attr)

    def __str__(self):
        return self._mars_entity.__str__()

    def __repr__(self):
        return self._mars_entity.__repr__()

    @property
    def mars_entity(self):
        return self._mars_entity


def to_mars(inp: Union["DataRefMarsImpl", Tuple, List, Dict]):
    """
    Convert xorbits data references to mars entities and execute them if needed.
    """

    if isinstance(inp, DataRefMarsImpl):
        mars_entity = inp.data.mars_entity
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


def from_mars(inp: Union["MarsEntity", tuple, list, dict]):
    """
    Convert mars entities to xorbits data references.
    """
    converter = _get_converter(type(inp))
    if converter is not None:
        return converter(inp)
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

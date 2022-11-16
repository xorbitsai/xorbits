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
import inspect
from collections import defaultdict
from types import ModuleType
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

# For maintenance, any module wants to import from mars, it should import from here.
from .._mars import dataframe as mars_dataframe
from .._mars import execute as mars_execute
from .._mars import new_session as mars_new_session
from .._mars import stop_server as mars_stop_server
from .._mars import tensor as mars_tensor
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
from .._mars.dataframe.window.ewm.core import EWM as MarsEWM
from .._mars.dataframe.window.expanding.core import Expanding as MarsExpanding
from .._mars.dataframe.window.rolling.core import Rolling as MarsRolling
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


def register_converter(from_cls_list: List[Type]):
    """
    A decorator for convenience of registering a class converter.
    """

    def decorate(cls: Type):
        for from_cls in from_cls_list:
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


def wrap_generator(wrapped: Generator):
    for item in wrapped:
        yield from_mars(item)


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
    elif isinstance(inp, Generator):
        return wrap_generator(inp)
    else:
        return inp


def add_docstring_disclaimer(
    docstring_src_module: Optional[ModuleType],
    docstring_src: Optional[Callable],
    doc: Optional[str],
) -> Optional[str]:
    if doc is None:
        return None

    if (
        docstring_src is not None
        and hasattr(docstring_src, "__module__")
        and not docstring_src.__module__
    ):
        return (
            doc
            + f"\n\nThis docstring was copied from {docstring_src.__module__}.{docstring_src.__name__}"
        )
    elif docstring_src_module is not None:
        return (
            doc + f"\n\nThis docstring was copied from {docstring_src_module.__name__}"
        )
    else:
        return doc


def skip_doctest(doc: Optional[str]) -> Optional[str]:
    def skip_line(line):
        # NumPy docstring contains cursor and comment only example
        stripped = line.strip()
        if stripped == ">>>" or stripped.startswith(">>> #"):
            return line
        elif ">>>" in stripped and "+SKIP" not in stripped:
            if "# doctest:" in line:
                return line + ", +SKIP"
            else:
                return line + "  # doctest: +SKIP"
        else:
            return line

    if doc is None:
        return None
    return "\n".join([skip_line(line) for line in doc.split("\n")])


def add_arg_disclaimer(
    src: Optional[Any],
    dest: Optional[Any],
    doc: Optional[str],
) -> Optional[str]:
    import re

    def get_named_args(func: Callable) -> list[str]:
        s = inspect.signature(func)
        return [
            n
            for n, p in s.parameters.items()
            if p.kind in [p.POSITIONAL_OR_KEYWORD, p.POSITIONAL_ONLY, p.KEYWORD_ONLY]
        ]

    def mark_unsupported_args(doc: str, args: list[str]) -> str:
        import re

        lines = doc.split("\n")
        for arg in args:
            subset = [
                (i, line)
                for i, line in enumerate(lines)
                if re.match(r"^\s*" + arg + " ?:", line)
            ]
            if len(subset) == 1:
                [(i, line)] = subset
                lines[i] = line + "  (Not supported yet)"
        return "\n".join(lines)

    def add_extra_args(doc: str, original_doc: Optional[Any], args: list[str]) -> str:
        def count_leading_spaces(s: str):
            return len(s) - len(s.lstrip(" "))

        def get_param_name(s: str) -> Optional[str]:
            # example:
            # '    data : ndarray (structured or homogeneous), Iterable, dict, or DataFrame'
            m = re.match(r"\s*(\S+)\s*:.*", s)
            if m:
                return m.group(1)
            return None

        def is_param_description(s: str, num_leading_spaces: int) -> bool:
            if s.strip():
                return count_leading_spaces(s) == num_leading_spaces + 4
            else:
                return True

        if original_doc is None:
            return doc

        new_section = [""]

        lines = original_doc.splitlines()
        idx = 0
        parameter_pattern = re.compile(r"\s+Parameters")
        while idx < len(lines):
            if re.match(parameter_pattern, lines[idx]):
                num_leading_spaces = count_leading_spaces(lines[idx])
                new_section.append(" " * num_leading_spaces + "Extra Parameters")
                new_section.append(
                    " " * num_leading_spaces + "-" * len("Extra Parameters")
                )
                idx += 2
                while idx < len(lines) and get_param_name(lines[idx]) is not None:
                    param_name = get_param_name(lines[idx])
                    if param_name in args:
                        new_section.append(
                            " " * num_leading_spaces + lines[idx].strip()
                        )
                        idx += 1
                        while idx < len(lines) and is_param_description(
                            lines[idx], num_leading_spaces
                        ):
                            if lines[idx].strip():
                                new_section.append(
                                    " " * (num_leading_spaces + 4) + lines[idx].strip()
                                )
                            idx += 1
                    else:
                        idx += 1
                        while idx < len(lines) and is_param_description(
                            lines[idx], num_leading_spaces
                        ):
                            idx += 1
                new_section.append(" " * num_leading_spaces)
                break
            idx += 1

        return doc + "\n".join(new_section)

    if doc is None:
        return None
    if src is None or dest is None:
        return doc

    try:
        src_args = get_named_args(src)
        dest_args = get_named_args(dest)
        unsupported_args = [a for a in src_args if a not in dest_args]
        extra_args = [a for a in dest_args if a not in src_args]
        if unsupported_args:
            doc = mark_unsupported_args(doc, unsupported_args)
        if extra_args:
            doc = add_extra_args(doc, getattr(dest, "__doc__", None), extra_args)
    except ValueError:
        return doc

    return doc


def gen_docstring(
    docstring_src: Optional[Type],
    method_name: str,
) -> Optional[str]:
    if docstring_src is None:
        return None

    src_method = getattr(docstring_src, method_name, None)
    if src_method is None:
        return None

    doc = getattr(src_method, "__doc__", None)
    if isinstance(src_method, property):
        # some things like SeriesGroupBy.unique are generated.
        src_method = src_method.fget
        if not doc:
            doc = getattr(src_method, "__doc__", None)

    # pandas DataFrame/Series sometimes override methods without setting __doc__.
    if doc is None and docstring_src.__name__ in {"DataFrame", "Series"}:
        for cls in docstring_src.mro():
            src_method = getattr(cls, method_name, None)
            if src_method is not None and hasattr(src_method, "__doc__"):
                doc = src_method.__doc__
    return doc


def attach_docstring(
    docstring_src_module: ModuleType, docstring_src: Optional[Callable], func: Callable
) -> Callable:
    if docstring_src is None:
        func.__doc__ = ""
        return func

    doc = getattr(docstring_src, "__doc__", None)
    doc = skip_doctest(doc)
    doc = add_arg_disclaimer(docstring_src, func, doc)
    doc = add_docstring_disclaimer(docstring_src_module, docstring_src, doc)
    func.__doc__ = "" if doc is None else doc
    return func


def wrap_mars_callable(c):
    """
    A function wrapper that makes arguments of the wrapped method be mars compatible type and
    return value be xorbits compatible type.
    """

    @functools.wraps(c)
    def wrapped(*args, **kwargs):
        return from_mars(c(*to_mars(args), **to_mars(kwargs)))

    return wrapped

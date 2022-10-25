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

import inspect
from typing import TYPE_CHECKING, Any, Callable, Set, Type

from ...adapter.mars import (
    MarsDataFrameDataSource,
    MarsDataFrameGroupBy,
    mars_dataframe,
)
from ...core.mars_adaption import (
    XorbitsDataMarsImpl,
    XorbitsDataRefMarsImpl,
    register_execution_condition,
    to_mars,
    wrap_mars_callable,
)
from .core import (
    DataFrame,
    DataFrameData,
    DataFrameGroupBy,
    DataFrameGroupByData,
    Index,
    IndexData,
    Series,
    SeriesData,
)

if TYPE_CHECKING:
    from ...adapter.mars import MarsEntity


# functions and class constructors defined by mars dataframe
MARS_DATAFRAME_CALLABLES = {}


def _install_functions() -> None:
    # install class constructors.
    MARS_DATAFRAME_CALLABLES[mars_dataframe.DataFrame.__name__] = wrap_mars_callable(
        mars_dataframe.DataFrame
    )
    MARS_DATAFRAME_CALLABLES[mars_dataframe.Series.__name__] = wrap_mars_callable(
        mars_dataframe.Series
    )
    MARS_DATAFRAME_CALLABLES[mars_dataframe.Index.__name__] = wrap_mars_callable(
        mars_dataframe.Index
    )
    # install module functions
    for name, func in inspect.getmembers(mars_dataframe, inspect.isfunction):
        MARS_DATAFRAME_CALLABLES[name] = wrap_mars_callable(func)


_install_functions()
del _install_functions


def _install_magic_methods() -> None:
    def wrap_setattr(owner: Type) -> Callable[[Any, str, Any], None]:
        def wrapped(self: Any, name: str, value: Any) -> None:
            # print(f"__setattr__ was called with ({name}, {value})")
            if name.startswith("_"):
                object.__setattr__(self, name, value)
            else:
                if issubclass(owner, XorbitsDataRefMarsImpl):
                    getattr(self.data, "__setattr__")(name, value)
                elif issubclass(owner, XorbitsDataMarsImpl):
                    if type(getattr(type(self.mars_entity), name)) is property:
                        # call the setter of the specified property.
                        getattr(type(self.mars_entity), name).fset(
                            self.mars_entity, to_mars(value)
                        )
                    else:
                        getattr(self.mars_entity, "__setattr__")(name, value)
                else:
                    raise NotImplementedError(f"Unsupported type: {owner}")

        return wrapped

    def wrap_magic_method(method_name: str, owner: Type) -> Callable[[Any], Any]:
        def wrapped(self: Any, *args, **kwargs):
            # print(f"{method_name} was called on {type(self)}")
            if issubclass(owner, XorbitsDataRefMarsImpl):
                if not hasattr(self.data, method_name):
                    raise AttributeError(
                        f"'{type(self)}' object has no attribute '{method_name}'"
                    )
                return getattr(self.data, method_name)(*args, **kwargs)
            elif issubclass(owner, XorbitsDataMarsImpl):
                if not hasattr(self.mars_entity, method_name):
                    raise AttributeError(
                        f"'{type(self)}' object has no attribute '{method_name}'"
                    )
                return wrap_mars_callable(getattr(self.mars_entity, method_name))(
                    *args, **kwargs
                )
            else:
                raise NotImplementedError(f"Unsupported type: {owner}")

        return wrapped

    def install_magic_methods(mars_cls: Type, xorbits_cls: Type) -> None:
        for name in dir(mars_cls):
            if (
                name.startswith("__")
                and name.endswith("__")
                and name not in magic_methods_to_skip
            ):
                if name == "__setattr__":
                    setattr(xorbits_cls, "__setattr__", wrap_setattr(owner=xorbits_cls))
                else:
                    setattr(
                        xorbits_cls, name, wrap_magic_method(name, owner=xorbits_cls)
                    )

    magic_methods_to_skip: Set[str] = {
        "__class__",
        "__copy__",
        "__delattr__",
        "__getattr__",
        "__getattribute__",
        "__init__",
        "__init_subclass__",
        "__module__",
        "__new__",
        "__repr__",
        "__slots__",
        "__str__",
    }

    # dataframe.
    install_magic_methods(mars_dataframe.DataFrame, DataFrame)
    install_magic_methods(mars_dataframe.DataFrame, DataFrameData)
    # dataframe groupby.
    install_magic_methods(MarsDataFrameGroupBy, DataFrameGroupBy)
    install_magic_methods(MarsDataFrameGroupBy, DataFrameGroupByData)
    # series.
    install_magic_methods(mars_dataframe.Series, Series)
    install_magic_methods(mars_dataframe.Series, SeriesData)
    # index.
    install_magic_methods(mars_dataframe.Index, Index)
    install_magic_methods(mars_dataframe.Index, IndexData)


_install_magic_methods()
del _install_magic_methods


def _register_execution_conditions() -> None:
    def _on_dtypes_being_none(mars_entity: "MarsEntity"):
        if (
            not isinstance(mars_entity, MarsDataFrameDataSource)
            and hasattr(mars_entity, "dtypes")
            and mars_entity.dtypes is None
        ):
            return True
        return False

    register_execution_condition(
        mars_dataframe.DataFrame.__name__, _on_dtypes_being_none
    )


_register_execution_conditions()
del _register_execution_conditions

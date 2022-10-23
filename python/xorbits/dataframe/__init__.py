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
from typing import TYPE_CHECKING

# noinspection PyUnresolvedReferences
from pandas import (  # noqa: F401
    DateOffset,
    Interval,
    NaT,
    Timedelta,
    Timestamp,
    offsets,
)

from ..adapter.mars import MarsDataFrameDataSource, mars_dataframe
from ..core.data import register_execution_condition, wrap_mars_callable
from .loc import DataFrameLoc

try:
    from pandas import NA, NamedAgg  # noqa: F401
except ImportError:  # pragma: no cover
    pass

if TYPE_CHECKING:
    from ..adapter.mars import MarsEntity


# functions and class constructors defined by mars dataframe
_MARS_DATAFRAME_CALLABLES = {}


def _install_functions() -> None:
    # install classes
    _MARS_DATAFRAME_CALLABLES[mars_dataframe.DataFrame.__name__] = wrap_mars_callable(
        mars_dataframe.DataFrame
    )
    _MARS_DATAFRAME_CALLABLES[mars_dataframe.Series.__name__] = wrap_mars_callable(
        mars_dataframe.Series
    )
    # install functions
    for name, func in inspect.getmembers(mars_dataframe, inspect.isfunction):
        _MARS_DATAFRAME_CALLABLES[name] = wrap_mars_callable(func)


_install_functions()
del _install_functions

# TODO: use registered methods for better performance and code completion.
# def _register_dataframe_methods() -> None:
#     for name, method in inspect.getmembers(mars_dataframe.core.DataFrame, inspect.isfunction):
#         if not name.startswith("__"):
#             XorbitsData.register_method(mars_dataframe.DataFrame.__name__, name, method)
#
#
# _register_dataframe_methods()
# del _register_dataframe_methods
#
#
# def _install_series_methods() -> None:
#     for name, method in inspect.getmembers(mars_dataframe.core.Series, inspect.isfunction):
#         if not name.startswith("__"):
#             XorbitsData.register_method(mars_dataframe.Series.__name__, name, method)
#
#
# _install_series_methods()
# del _install_series_methods
#
#
# def _register_index_methods() -> None:
#     for name, method in inspect.getmembers(mars_dataframe.core.Index, inspect.isfunction):
#         if not name.startswith("__"):
#             XorbitsData.register_method(mars_dataframe.Index.__name__, name, method)
#
#
# _register_index_methods()
# del _register_index_methods


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


def __getattr__(name: str):
    if name in _MARS_DATAFRAME_CALLABLES:
        return _MARS_DATAFRAME_CALLABLES[name]
    else:
        # TODO  for functions not implemented fallback to pandas
        raise NotImplementedError

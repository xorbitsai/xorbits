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

import functools
import inspect
from types import ModuleType
from typing import Callable, Dict, Set, Type

import pandas

from ...core.adapter import (
    MARS_DATAFRAME_GROUPBY_TYPE,
    MARS_DATAFRAME_OR_SERIES_TYPE,
    MARS_DATAFRAME_TYPE,
    MARS_SERIES_GROUPBY_TYPE,
    MARS_SERIES_TYPE,
    MarsDataFrameToCSV,
    MarsDataFrameToParquet,
    MarsDataFrameToSQLTable,
    MarsDataFrameToVineyardChunk,
    MarsEntity,
    collect_cls_members,
    from_mars,
    mars_dataframe,
    own_data,
    register_from_mars_execution_condition,
    register_to_mars_execution_condition,
    to_mars,
    wrap_mars_callable,
)
from ...core.data import DataType
from ...core.utils.docstring import attach_cls_member_docstring


# functions and class constructors defined by mars dataframe
def _collect_module_callables() -> Dict[str, Callable]:
    mars_dataframe_callables = dict()

    # install class constructors.
    mars_dataframe_callables[mars_dataframe.DataFrame.__name__] = wrap_mars_callable(
        mars_dataframe.DataFrame,
        attach_docstring=True,
        is_cls_member=False,
        docstring_src_module=pandas,
        docstring_src=pandas.DataFrame,
    )
    mars_dataframe_callables[mars_dataframe.Series.__name__] = wrap_mars_callable(
        mars_dataframe.Series,
        attach_docstring=True,
        is_cls_member=False,
        docstring_src_module=pandas,
        docstring_src=pandas.Series,
    )
    mars_dataframe_callables[mars_dataframe.Index.__name__] = wrap_mars_callable(
        mars_dataframe.Index,
        attach_docstring=True,
        is_cls_member=False,
        docstring_src_module=pandas,
        docstring_src=pandas.Index,
    )
    # install module functions
    for name, func in inspect.getmembers(mars_dataframe, inspect.isfunction):
        mars_dataframe_callables[name] = wrap_mars_callable(
            func,
            attach_docstring=True,
            is_cls_member=False,
            docstring_src_module=pandas,
            docstring_src=getattr(pandas, name, None),
        )

    return mars_dataframe_callables


MARS_DATAFRAME_CALLABLES: Dict[str, Callable] = _collect_module_callables()


def _collect_dataframe_magic_methods() -> Set[str]:
    magic_methods = set()

    magic_methods_to_skip: Set[str] = {
        "__init__",
        "__dir__",
        "__repr__",
        "__str__",
        "__setattr__",
        "__getattr__",
        "__len__",
        "__array__",
    }
    all_cls = (
        MARS_DATAFRAME_TYPE
        + MARS_SERIES_TYPE
        + MARS_DATAFRAME_GROUPBY_TYPE
        + MARS_SERIES_GROUPBY_TYPE
    )
    for mars_cls in all_cls:
        for name, _ in inspect.getmembers(mars_cls, inspect.isfunction):
            if (
                name.startswith("__")
                and name.endswith("__")
                and name not in magic_methods_to_skip
            ):
                magic_methods.add(name)
    return magic_methods


MARS_DATAFRAME_MAGIC_METHODS: Set[str] = _collect_dataframe_magic_methods()


def _register_to_mars_execution_conditions() -> None:
    def _on_dtypes_being_none(mars_entity: "MarsEntity") -> bool:
        if hasattr(mars_entity, "dtypes") and mars_entity.dtypes is None:
            return True
        return False

    register_to_mars_execution_condition(
        mars_dataframe.DataFrame.__name__, _on_dtypes_being_none
    )


def _register_from_mars_execution_conditions() -> None:
    def _on_dataframe_export_functions_being_called(mars_entity: "MarsEntity") -> bool:
        return isinstance(
            mars_entity.op,
            (
                MarsDataFrameToParquet,
                MarsDataFrameToCSV,
                MarsDataFrameToSQLTable,
                MarsDataFrameToVineyardChunk,
            ),
        )

    def _on_series_export_functions_being_called(mars_entity: "MarsEntity") -> bool:
        return isinstance(mars_entity.op, (MarsDataFrameToCSV, MarsDataFrameToSQLTable))

    register_from_mars_execution_condition(
        mars_dataframe.DataFrame.__name__, _on_dataframe_export_functions_being_called
    )
    register_from_mars_execution_condition(
        mars_dataframe.Series.__name__, _on_series_export_functions_being_called
    )


def install_members(
    cls: Type, mars_cls: Type, docstring_src_module: ModuleType, docstring_src_cls: Type
):
    members = collect_cls_members(
        mars_cls,
        docstring_src_module=docstring_src_module,
        docstring_src_cls=docstring_src_cls,
    )
    for name in members:
        setattr(cls, name, members[name])


def wrap_user_defined_functions(
    c: Callable, member_name: str, data_type: DataType
) -> Callable:
    """
    A function wrapper for user defined functions.
    """

    @functools.wraps(c)
    def wrapped(*args, **kwargs):
        new_args = to_mars(args)
        new_kwargs = to_mars(kwargs)
        try:
            return from_mars(c(*new_args, **new_kwargs))
        except (TypeError, ValueError):
            # infer failed, add skip_infer=True manually
            new_kwargs["skip_infer"] = True
            ret = c(*new_args, **new_kwargs)
            if isinstance(ret, MARS_DATAFRAME_OR_SERIES_TYPE):
                ret = ret.ensure_data()
            return from_mars(ret)

    return attach_cls_member_docstring(wrapped, member_name, data_type=data_type)


def wrap_iteration_functions(
    member_method: Callable,
    member_name: str,
    data_type: DataType,
    attach_docstring: bool,
):
    """
    Wrapper for iteration functions.
    """

    @functools.wraps(member_method)
    def wrapped(self: MarsEntity, *args, **kwargs):
        if own_data(self):
            # if own data, return iteration on data directly
            return getattr(self.op.data, member_name)(*to_mars(args), **to_mars(kwargs))
        else:
            return from_mars(member_method(self, *to_mars(args), **to_mars(kwargs)))

    if attach_docstring:
        return attach_cls_member_docstring(wrapped, member_name, data_type)
    else:  # pragma: no cover
        wrapped.__doc__ = ""
        return wrapped

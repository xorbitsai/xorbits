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

import inspect
from types import ModuleType
from typing import Callable, Dict, List, Optional, Type

import xgboost

from ..._mars.learn.contrib.xgboost.classifier import (
    XGBClassifier as mars_XGBClassifier,
)
from ...core.adapter import collect_cls_members, mars_xgboost, wrap_mars_callable


class XGBClassifier(mars_XGBClassifier):
    pass


def _collect_module_callables(
    m: ModuleType,
    docstring_src_module: ModuleType,
    skip_members: Optional[List[str]] = None,
) -> Dict[str, Callable]:
    module_callables: Dict[str, Callable] = dict()

    module_callables[mars_xgboost.XGBClassifier.__name__] = XGBClassifier

    # install module functions.
    for name, func in inspect.getmembers(XGBClassifier, inspect.isfunction):
        if skip_members is not None and name in skip_members:
            continue

        setattr(
            XGBClassifier,
            name,
            wrap_mars_callable(
                func,
                attach_docstring=False,
                is_cls_member=False,
                docstring_src_module=xgboost.XGBClassifier,
                docstring_src=getattr(xgboost.XGBClassifier, name, None),
            ),
        )

    return module_callables


MARS_XGBOOST_CALLABLES: Dict[str, Callable] = _collect_module_callables(
    mars_xgboost,
    xgboost,
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


# functions and class constructors defined by mars dataframe
# def _collect_module_callables() -> Dict[str, Callable]:
#     module_callables: Dict[str, Callable] = dict()

#     # install class constructors.
#     mars_dataframe_callables[mars_dataframe.DataFrame.__name__] = wrap_mars_callable(
#         mars_dataframe.DataFrame,
#         attach_docstring=True,
#         is_cls_member=False,
#         docstring_src_module=pandas,
#         docstring_src=pandas.DataFrame,
#     )
#     mars_dataframe_callables[mars_dataframe.Series.__name__] = wrap_mars_callable(
#         mars_dataframe.Series,
#         attach_docstring=True,
#         is_cls_member=False,
#         docstring_src_module=pandas,
#         docstring_src=pandas.Series,
#     )
#     mars_dataframe_callables[mars_dataframe.Index.__name__] = wrap_mars_callable(
#         mars_dataframe.Index,
#         attach_docstring=True,
#         is_cls_member=False,
#         docstring_src_module=pandas,
#         docstring_src=pandas.Index,
#     )
#     # install module functions
#     for name, func in inspect.getmembers(mars_dataframe, inspect.isfunction):
#         mars_dataframe_callables[name] = wrap_mars_callable(
#             func,
#             attach_docstring=True,
#             is_cls_member=False,
#             docstring_src_module=pandas,
#             docstring_src=getattr(pandas, name, None),
#         )

#     return mars_dataframe_callables


# MARS_DATAFRAME_CALLABLES: Dict[str, Callable] = _collect_module_callables()

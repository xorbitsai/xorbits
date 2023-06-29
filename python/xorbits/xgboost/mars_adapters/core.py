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

try:
    import xgboost
except ImportError:  # pragma: no cover
    xgboost = None

MARS_XGBOOST_CALLABLES = {}

if xgboost is not None:
    import functools
    import inspect
    from typing import Callable, Dict, List, Optional

    from ..._mars.learn.contrib import xgboost as mars_xgboost
    from ..._mars.learn.contrib.xgboost.classifier import (
        XGBClassifier as MarsXGBClassifier,
    )
    from ..._mars.learn.contrib.xgboost.regressor import (
        XGBRegressor as MarsXGBRegressor,
    )
    from ...core.adapter import to_mars, wrap_mars_callable
    from ...core.utils.docstring import attach_module_callable_docstring

    class BaseXGB:
        def __init__(self, *args, **kwargs):
            self.mars_instance = self._mars_cls(*to_mars(args), **to_mars(kwargs))

    class XGBClassifier(BaseXGB):
        _mars_cls = MarsXGBClassifier

    class XGBRegressor(BaseXGB):
        _mars_cls = MarsXGBRegressor

    xgboost_class_mappings: Dict = {
        XGBClassifier: MarsXGBClassifier,
        XGBRegressor: MarsXGBRegressor,
    }

    def wrap_cls_func(mars_cls: Callable, name: str):
        @functools.wraps(getattr(mars_cls, name))
        def wrapped(self, *args, **kwargs):
            return getattr(self.mars_instance, name)(*args, **kwargs)

        return wrap_mars_callable(
            wrapped,
            member_name=name,
            attach_docstring=True,
            is_cls_member=True,
            docstring_src_module=xgboost,
            docstring_src_cls=getattr(xgboost, mars_cls.__name__, None),
        )

    def _collect_module_callables(
        skip_members: Optional[List[str]] = None,
    ) -> Dict[str, Callable]:
        module_callables: Dict[str, Callable] = dict()

        for name, func in inspect.getmembers(mars_xgboost, inspect.isfunction):
            if skip_members is not None and name in skip_members:
                continue

            if name == "MarsDMatrix":
                name = "DMatrix"

            module_callables[name] = wrap_mars_callable(
                func,
                attach_docstring=True,
                is_cls_member=False,
                docstring_src_module=xgboost,
                docstring_src=getattr(xgboost, name, None),
            )
        return module_callables

    def _install_cls_members(module_callables: Dict[str, Callable]):
        for k, v in xgboost_class_mappings.items():
            module_callables[k.__name__] = k
            for name, _ in inspect.getmembers(v, inspect.isfunction):
                if not name.startswith("_"):
                    setattr(k, name, wrap_cls_func(v, name))

    MARS_XGBOOST_CALLABLES = _collect_module_callables(skip_members=["register_op"])

    _install_cls_members(MARS_XGBOOST_CALLABLES)

    attach_module_callable_docstring(XGBClassifier, xgboost, xgboost.XGBClassifier)
    attach_module_callable_docstring(XGBRegressor, xgboost, xgboost.XGBRegressor)

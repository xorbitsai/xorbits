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
from typing import Callable, Dict

MARS_XGBOOST_CALLABLES = None

try:
    import xgboost
except ImportError:
    xgboost = None

if xgboost is not None:
    from ..._mars.learn.contrib.xgboost.classifier import (
        XGBClassifier as mars_XGBClassifier,
    )
    from ..._mars.learn.contrib.xgboost.dmatrix import MarsDMatrix
    from ..._mars.learn.contrib.xgboost.regressor import (
        XGBRegressor as mars_XGBRegressor,
    )
    from ...core.adapter import mars_xgboost, wrap_mars_callable

    class XGBClassifier(mars_XGBClassifier):
        pass

    class XGBRegressor(mars_XGBRegressor):
        pass

    class DMatrix:
        def __init__(self) -> None:
            pass

        def __call__(self, data, **kws):
            return MarsDMatrix(data, **kws)

    def _collect_module_callables() -> Dict[str, Callable]:
        module_callables: Dict[str, Callable] = dict()

        module_callables[xgboost.XGBClassifier.__name__] = XGBClassifier
        for name, func in inspect.getmembers(XGBClassifier, inspect.isfunction):
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

        module_callables[xgboost.XGBRegressor.__name__] = XGBRegressor
        for name, func in inspect.getmembers(XGBRegressor, inspect.isfunction):
            setattr(
                XGBRegressor,
                name,
                wrap_mars_callable(
                    func,
                    attach_docstring=True,
                    is_cls_member=False,
                    docstring_src_module=xgboost.XGBRegressor,
                    docstring_src=getattr(xgboost.XGBRegressor, name, None),
                ),
            )

        module_callables[xgboost.DMatrix.__name__] = DMatrix
        for name, func in inspect.getmembers(DMatrix, inspect.isfunction):
            setattr(
                DMatrix,
                name,
                wrap_mars_callable(
                    func,
                    attach_docstring=False,
                    is_cls_member=False,
                ),
            )

        for name, func in inspect.getmembers(mars_xgboost, inspect.isfunction):
            if name == "MarsDMatrix":
                continue
            module_callables[name] = wrap_mars_callable(
                func,
                attach_docstring=True,
                is_cls_member=False,
                docstring_src_module=xgboost,
                docstring_src=getattr(xgboost, name, None),
            )
        return module_callables

    MARS_XGBOOST_CALLABLES = _collect_module_callables()

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

from collections.abc import Iterable
from typing import Any

try:
    import xgboost
except ImportError:  # pragma: no cover
    xgboost = None

MARS_XGBOOST_CALLABLES = {}

if xgboost is not None:
    import inspect
    from typing import Callable, Dict, List, Optional

    from ..._mars.learn.contrib.xgboost.classifier import (
        XGBClassifier as MarsXGBClassifier,
    )
    from ..._mars.learn.contrib.xgboost.dmatrix import MarsDMatrix
    from ..._mars.learn.contrib.xgboost.regressor import (
        XGBRegressor as MarsXGBRegressor,
    )
    from ...core.adapter import mars_xgboost, wrap_mars_callable

    class BaseXGB:
        def __init__(self, *args, **kwargs):
            self.mars_instance = self.mars_cls(*args, **kwargs)

        def __dir__(self) -> Iterable[str]:
            return self.mars_instance.__dir__()

    class XGBClassifier(BaseXGB):
        mars_cls = MarsXGBClassifier

    class XGBRegressor(BaseXGB):
        mars_cls = MarsXGBRegressor

    def wrap_cls_func(func_name):
        def wrapper(self, *args, **kwargs):
            return wrap_mars_callable(
                getattr(self.mars_instance, func_name),
                attach_docstring=False,
                is_cls_member=False,
            )(*args, **kwargs)

        return wrapper

    class DMatrix:
        def __init__(self) -> None:
            pass

        def __call__(self, data, **kws) -> Any:
            return MarsDMatrix(data, **kws)

    def _collect_module_callables(
        skip_members: Optional[List[str]] = None,
    ) -> Dict[str, Callable]:
        module_callables: Dict[str, Callable] = dict()

        module_callables[xgboost.XGBClassifier.__name__] = XGBClassifier
        for name, func in inspect.getmembers(MarsXGBClassifier, inspect.isfunction):
            if not name.startswith("_"):
                setattr(XGBClassifier, name, wrap_cls_func(name))

        module_callables[xgboost.XGBRegressor.__name__] = XGBRegressor
        for name, func in inspect.getmembers(MarsXGBRegressor, inspect.isfunction):
            if not name.startswith("_"):
                setattr(XGBRegressor, name, wrap_cls_func(name))

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
            if skip_members is not None and name in skip_members:
                continue
            module_callables[name] = wrap_mars_callable(
                func,
                attach_docstring=True,
                is_cls_member=False,
                docstring_src_module=xgboost,
                docstring_src=getattr(xgboost, name, None),
            )
        return module_callables

    MARS_XGBOOST_CALLABLES = _collect_module_callables(
        skip_members=["MarsDMatrix", "register_op"]
    )

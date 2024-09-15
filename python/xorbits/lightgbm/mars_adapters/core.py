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
from typing import Callable, Dict, List, Optional

import lightgbm

from ..._mars.learn.contrib import lightgbm as mars_lightgbm
from ..._mars.learn.contrib.lightgbm.classifier import (
    LGBMClassifier as MarsLGBMClassifier,
)
from ..._mars.learn.contrib.lightgbm.ranker import LGBMRanker as MarsLGBMRanker
from ..._mars.learn.contrib.lightgbm.regressor import LGBMRegressor as MarsLGBMRegressor
from ...core.adapter import to_mars, wrap_mars_callable
from ...core.utils.docstring import attach_module_callable_docstring

MARS_LIGHTGBM_CALLABLES = {}


class LGBMBase:
    def __init__(self, *args, **kwargs):
        self.mars_instance = self._marscls(*to_mars(args), **to_mars(kwargs))


class LGBMRegressor(LGBMBase):
    _marscls = MarsLGBMRegressor


class LGBMClassifier(LGBMBase):
    _marscls = MarsLGBMClassifier


class LGBMRanker(LGBMBase):
    _marscls = MarsLGBMRanker


LGBM_CLS_MAP = {
    LGBMClassifier: MarsLGBMClassifier,
    LGBMRegressor: MarsLGBMRegressor,
    LGBMRanker: MarsLGBMRanker,
}


def wrap_cls_func(marscls: Callable, name: str):
    @functools.wraps(getattr(marscls, name))
    def wrapped(self, *args, **kwargs):
        return getattr(self.mars_instance, name)(*args, **kwargs)

    return wrap_mars_callable(
        wrapped,
        member_name=name,
        attach_docstring=True,
        is_cls_member=True,
        docstring_src_module=lightgbm,
        docstring_src_cls=getattr(lightgbm, marscls.__name__, None),
    )


def _collect_module_callables(
    skip_members: Optional[List[str]] = None,
) -> Dict[str, Callable]:
    module_callables: Dict[str, Callable] = dict()

    for name, func in inspect.getmembers(mars_lightgbm, inspect.isfunction):
        if skip_members is not None and name in skip_members:
            continue
        module_callables[name] = wrap_mars_callable(
            func,
            attach_docstring=True,
            is_cls_member=False,
            docstring_src_module=lightgbm,
            docstring_src=getattr(lightgbm, name, None),
        )
    return module_callables


def _install_cls_members(module_callables: Dict[str, Callable]):
    for x_cls, mars_cls in LGBM_CLS_MAP.items():
        module_callables[x_cls.__name__] = x_cls
        for name, _ in inspect.getmembers(mars_cls, inspect.isfunction):
            if not name.startswith("_"):
                setattr(x_cls, name, wrap_cls_func(mars_cls, name))


MARS_LIGHTGBM_CALLABLES = _collect_module_callables(skip_members=["register_op"])
_install_cls_members(MARS_LIGHTGBM_CALLABLES)
attach_module_callable_docstring(LGBMClassifier, lightgbm, lightgbm.LGBMClassifier)
attach_module_callable_docstring(LGBMRegressor, lightgbm, lightgbm.LGBMRegressor)
attach_module_callable_docstring(LGBMRanker, lightgbm, lightgbm.LGBMRanker)

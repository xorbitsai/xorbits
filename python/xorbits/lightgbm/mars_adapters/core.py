# Copyright 2022-2023 XProbe Inc.
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

from typing import Any

import lightgbm

MARS_LIGHGBM_CALLABLES = {}


import inspect
from typing import Callable, Dict, List, Optional

from ..._mars.learn.contrib.lightgbm.classifier import (
    LGBMClassifier as MarsLGBMClassifier,
)
from ..._mars.learn.contrib.lightgbm.ranker import LGBMRanker as MarsLGBMRanker
from ..._mars.learn.contrib.lightgbm.regressor import LGBMRegressor as MarsLGBMRegressor
from ...core.adapter import mars_lightgbm, to_mars, wrap_mars_callable
from ...core.utils.docstring import attach_module_callable_docstring


class LGBMBase:
    def __init__(self, *args, **kwargs):
        self.mars_instance = self.marscls(*to_mars(args), **to_mars(kwargs))

    def __getattr__(self, name: str) -> Any:
        if callable(getattr(self.mars_instance, name)):
            return wrap_mars_callable(
                getattr(self.mars_instance, name),
                attach_docstring=True,
                is_cls_member=False,
                docstring_src_module=self.LGBCls,
                docstring_src=getattr(self.LGBCls, name, None),
            )


class LGBMRegressor(LGBMBase):
    marscls = MarsLGBMRegressor
    LGBCls = lightgbm.LGBMRegressor


class LGBMClassifier(LGBMBase):
    marscls = MarsLGBMClassifier
    LGBCls = lightgbm.LGBMClassifier


class LGBMRanker(LGBMBase):
    marscls = MarsLGBMRanker
    LGBCls = lightgbm.LGBMRanker


def _collect_module_callables(
    skip_members: Optional[List[str]] = None,
) -> Dict[str, Callable]:
    module_callables: Dict[str, Callable] = dict()

    module_callables[lightgbm.LGBMClassifier.__name__] = LGBMClassifier

    module_callables[lightgbm.LGBMRegressor.__name__] = LGBMRegressor

    module_callables[lightgbm.LGBMRanker.__name__] = LGBMRanker

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


attach_module_callable_docstring(LGBMClassifier, lightgbm, lightgbm.LGBMClassifier)
attach_module_callable_docstring(LGBMRegressor, lightgbm, lightgbm.LGBMRegressor)
attach_module_callable_docstring(LGBMRanker, lightgbm, lightgbm.LGBMRanker)

MARS_LIGHGBM_CALLABLES = _collect_module_callables(skip_members=["register_op"])

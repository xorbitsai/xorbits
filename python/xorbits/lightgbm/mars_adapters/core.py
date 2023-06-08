# Copyright 2022-2023 XProbe Inc.
# derived from copyright 1999-2021 Alibaba Group Holding Ltd.
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
    import lightgbm
except ImportError:
    lightgbm = None

MARS_LIGHGBM_CALLABLES = {}

if lightgbm is not None:
    import inspect
    from typing import Callable, Dict, List, Optional

    from ..._mars.learn.contrib.lightgbm.classifier import (
        LGBMClassifier as MarsLGBMClassifier,
    )
    from ..._mars.learn.contrib.lightgbm.ranker import LGBMRanker as MarsLGBMRanker
    from ..._mars.learn.contrib.lightgbm.regressor import (
        LGBMRegressor as MarsLGBMRegressor,
    )
    from ...core.adapter import mars_lightgbm, wrap_mars_callable

    class LGBMClassifier(MarsLGBMClassifier):
        pass

    class LGBMRegressor(MarsLGBMRegressor):
        pass

    class LGBMRanker(MarsLGBMRanker):
        pass

    def _collect_module_callables(
        skip_members: Optional[List[str]] = None,
    ) -> Dict[str, Callable]:
        module_callables: Dict[str, Callable] = dict()

        module_callables[lightgbm.LGBMClassifier.__name__] = LGBMClassifier
        # install module functions.
        for name, func in inspect.getmembers(LGBMClassifier, inspect.isfunction):
            setattr(
                LGBMClassifier,
                name,
                wrap_mars_callable(
                    func,
                    attach_docstring=False,
                    is_cls_member=False,
                    docstring_src_module=lightgbm.LGBMClassifier,
                    docstring_src=getattr(lightgbm.LGBMClassifier, name, None),
                ),
            )

        module_callables[lightgbm.LGBMRegressor.__name__] = LGBMRegressor
        for name, func in inspect.getmembers(LGBMRegressor, inspect.isfunction):
            setattr(
                LGBMRegressor,
                name,
                wrap_mars_callable(
                    func,
                    attach_docstring=True,
                    is_cls_member=False,
                    docstring_src_module=lightgbm.LGBMRegressor,
                    docstring_src=getattr(lightgbm.LGBMRegressor, name, None),
                ),
            )
        module_callables[lightgbm.LGBMRanker.__name__] = LGBMRanker
        for name, func in inspect.getmembers(LGBMRanker, inspect.isfunction):
            setattr(
                LGBMRanker,
                name,
                wrap_mars_callable(
                    func,
                    attach_docstring=True,
                    is_cls_member=False,
                    docstring_src_module=lightgbm.LGBMRanker,
                    docstring_src=getattr(lightgbm.LGBMRanker, name, None),
                ),
            )
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

    MARS_LIGHGBM_CALLABLES = _collect_module_callables(skip_members=["register_op"])

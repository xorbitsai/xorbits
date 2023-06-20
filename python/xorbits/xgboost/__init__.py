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

from ..core.utils.fallback import unimplemented_func


def _install():
    """Nothing required for installing xgboost."""


def __dir__():
    from .mars_adapters import MARS_XGBOOST_CALLABLES

    return list(MARS_XGBOOST_CALLABLES.keys())


def __getattr__(name: str):
    from .mars_adapters import MARS_XGBOOST_CALLABLES

    if name in MARS_XGBOOST_CALLABLES:
        return MARS_XGBOOST_CALLABLES[name]
    else:
        try:
            import xgboost
        except ImportError:  # pragma: no cover
            xgboost = None

        if xgboost is not None:
            if not hasattr(xgboost, name):
                raise AttributeError(name)
            else:  # pragma: no cover
                if inspect.ismethod(getattr(xgboost, name)):
                    return unimplemented_func
                else:
                    raise AttributeError(name)

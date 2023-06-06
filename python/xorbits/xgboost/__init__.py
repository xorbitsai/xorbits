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


def _install():
    from .mars_adapters import _install as _install_mars_methods

    _install_mars_methods()


def __dir__():
    import xgboost

    from .mars_adapters import MARS_XGBOOST_CALLABLES

    return list(MARS_XGBOOST_CALLABLES.keys())


def __getattr__(name: str):
    from .mars_adapters import MARS_XGBOOST_CALLABLES

    if name in MARS_XGBOOST_CALLABLES:
        return MARS_XGBOOST_CALLABLES[name]
    else:
        raise NotImplementedError(f"Xorbits.XGBoost does not support {name} yet")

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
from ...core.utils.fallback import unimplemented_func


def _install():
    """Nothing required for installing sklearn."""


def __dir__():  # pragma: no cover
    try:
        import sklearn
    except ImportError:
        raise AttributeError("sklearn is required but not installed.")
    from .mars_adapters import MARS_SKLEARN_METRICS_CALLABLES

    return list(MARS_SKLEARN_METRICS_CALLABLES.keys())


def __getattr__(name: str):  # pragma: no cover
    import inspect

    try:
        import sklearn.metrics as sk_metrics
    except ImportError:
        raise AttributeError("sklearn is required but not installed.")
    from .mars_adapters import MARS_SKLEARN_METRICS_CALLABLES

    if name in MARS_SKLEARN_METRICS_CALLABLES:
        return MARS_SKLEARN_METRICS_CALLABLES[name]
    else:
        if not hasattr(sk_metrics, name):
            raise AttributeError(name)
        else:
            if inspect.ismethod(getattr(sk_metrics, name)):
                return unimplemented_func()
            else:
                raise AttributeError

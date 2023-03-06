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

from ...core.utils.fallback import unimplemented_func

METHODS = None


def __dir__():
    from ..mars_adapters import MARS_TENSOR_RANDOM_CALLABLES

    return list(MARS_TENSOR_RANDOM_CALLABLES.keys())


def __getattr__(name: str):
    from ..mars_adapters import MARS_TENSOR_RANDOM_CALLABLES
    from ..numpy_adapters import collect_numpy_module_members

    if name in MARS_TENSOR_RANDOM_CALLABLES:
        return MARS_TENSOR_RANDOM_CALLABLES[name]
    else:
        global METHODS
        import numpy

        if METHODS is None:
            METHODS = collect_numpy_module_members(numpy.random)

        if not hasattr(numpy.random, name):
            raise AttributeError(name)
        elif name in METHODS:
            return METHODS[name]
        else:  # pragma: no cover
            if inspect.ismethod(getattr(numpy.random, name)):
                return unimplemented_func
            else:
                raise AttributeError(name)

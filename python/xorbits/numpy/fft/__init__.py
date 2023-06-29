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
from typing import Any, Callable, Dict, Optional

from ...core.utils.fallback import unimplemented_func


def __dir__():
    from ..mars_adapters import MARS_TENSOR_FFT_CALLABLES
    from ..numpy_adapters.core import NUMPY_FFT_MEMBERS

    global NUMPY_FFT_METHODS
    import numpy

    return list(MARS_TENSOR_FFT_CALLABLES.keys()) + list(NUMPY_FFT_MEMBERS.keys())


def __getattr__(name: str):
    from ..mars_adapters import MARS_TENSOR_FFT_CALLABLES
    from ..numpy_adapters.core import NUMPY_FFT_MEMBERS

    if name in MARS_TENSOR_FFT_CALLABLES:
        return MARS_TENSOR_FFT_CALLABLES[name]
    else:  # pragma: no cover
        import numpy

        if not hasattr(numpy.fft, name):
            raise AttributeError(name)
        elif name in NUMPY_FFT_MEMBERS:
            return NUMPY_FFT_MEMBERS[name]
        else:
            if inspect.ismethod(getattr(numpy.fft, name)):
                return unimplemented_func
            else:
                raise AttributeError(name)

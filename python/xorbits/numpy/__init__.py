# Copyright 2022 XProbe Inc.
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

from numpy import (
    NAN,
    NINF,
    AxisError,
    Inf,
    NaN,
    bool_,
    bytes_,
    cfloat,
    character,
    complex64,
    complex128,
    complexfloating,
    datetime64,
    double,
    dtype,
    e,
    errstate,
    finfo,
    flexible,
    float16,
    float32,
    float64,
    float_,
    floating,
    generic,
    inexact,
    inf,
    int8,
    int16,
    int32,
    int64,
    int_,
    intc,
    integer,
    intp,
    nan,
    newaxis,
    number,
    object_,
    pi,
    signedinteger,
    timedelta64,
    uint,
    uint8,
    uint16,
    uint32,
    uint64,
    unicode_,
    unsignedinteger,
    void,
)
from numpy.lib.index_tricks import ndindex


def _install():
    from .mars_adapters import _install as _install_mars_adapters

    _install_mars_adapters()


try:
    import warnings

    # suppress numpy warnings on types
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.simplefilter("ignore", FutureWarning)
        # noinspection PyUnresolvedReferences
        from numpy import bool, float, int, object  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    pass
finally:
    del warnings


from . import fft, linalg, random
from .core import ndarray


def __dir__():
    from .mars_adapters import MARS_TENSOR_CALLABLES, MARS_TENSOR_OBJECTS

    return list(MARS_TENSOR_CALLABLES.keys()) + list((MARS_TENSOR_OBJECTS.keys()))


def __getattr__(name: str):
    from .mars_adapters import MARS_TENSOR_CALLABLES, MARS_TENSOR_OBJECTS

    if name in MARS_TENSOR_CALLABLES:
        return MARS_TENSOR_CALLABLES[name]
    elif name in MARS_TENSOR_OBJECTS:
        return MARS_TENSOR_OBJECTS[name]
    else:
        # TODO: fallback to numpy
        raise AttributeError(name)

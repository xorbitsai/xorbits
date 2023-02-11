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

from .all import TensorAll, all
from .allclose import allclose
from .any import TensorAny, any
from .argmax import TensorArgmax, argmax
from .argmin import TensorArgmin, argmin
from .array_equal import array_equal
from .count_nonzero import TensorCountNonzero, count_nonzero
from .cumprod import TensorCumprod, cumprod
from .cumsum import TensorCumsum, cumsum
from .max import TensorMax, max
from .mean import TensorMean, mean
from .min import TensorMin, min
from .nanargmax import TensorNanArgmax, nanargmax
from .nanargmin import TensorNanArgmin, nanargmin
from .nancumprod import TensorNanCumprod, nancumprod
from .nancumsum import TensorNanCumsum, nancumsum
from .nanmax import TensorNanMax, nanmax
from .nanmean import TensorNanMean, nanmean
from .nanmin import TensorNanMin, nanmin
from .nanprod import TensorNanProd, nanprod
from .nanstd import nanstd
from .nansum import TensorNanSum, nansum
from .nanvar import TensorNanMoment, TensorNanVar, nanvar
from .prod import TensorProd, prod
from .std import std
from .sum import TensorSum, sum
from .var import TensorMoment, TensorVar, var


def _install():
    from ..core import Tensor, TensorData

    for cls in (Tensor, TensorData):
        setattr(cls, "sum", sum)
        setattr(cls, "prod", prod)
        setattr(cls, "max", max)
        setattr(cls, "min", min)
        setattr(cls, "all", all)
        setattr(cls, "any", any)
        setattr(cls, "mean", mean)
        setattr(cls, "argmax", argmax)
        setattr(cls, "argmin", argmin)
        setattr(cls, "cumsum", cumsum)
        setattr(cls, "cumprod", cumprod)
        setattr(cls, "var", var)
        setattr(cls, "std", std)


_install()
del _install

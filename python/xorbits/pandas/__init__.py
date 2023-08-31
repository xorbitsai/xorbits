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
from typing import Callable, Dict, Optional

# noinspection PyUnresolvedReferences
from pandas import Timedelta  # noqa: F401
from pandas import DateOffset, Interval, NaT, Timestamp, offsets

try:
    from pandas import NA, NamedAgg  # noqa: F401
except ImportError:  # pragma: no cover
    pass

try:
    from pandas import ArrowDtype  # noqa: F401 pandas >= 1.5.x has this type mapper
except ImportError:
    ArrowDtype = None

from ..core.utils.fallback import unimplemented_func
from . import accessors, core, groupby, plotting, window
from ._config import (
    describe_option,
    get_option,
    option_context,
    reset_option,
    set_eng_float_format,
    set_option,
)
from .core import DataFrame, Index, Series


def _install():
    from .mars_adapters import _install as _install_mars_methods
    from .pandas_adapters import _install as _install_pandas_methods

    _install_mars_methods()
    _install_pandas_methods()


def __dir__():
    from .mars_adapters import MARS_DATAFRAME_CALLABLES
    from .pandas_adapters import collect_pandas_module_members

    return list(MARS_DATAFRAME_CALLABLES.keys()) + list(
        collect_pandas_module_members().keys()
    )


def __getattr__(name: str):
    from .mars_adapters import MARS_DATAFRAME_CALLABLES
    from .pandas_adapters import collect_pandas_module_members

    if name in MARS_DATAFRAME_CALLABLES:
        return MARS_DATAFRAME_CALLABLES[name]
    else:
        import pandas

        if not hasattr(pandas, name):
            raise AttributeError(name)
        elif name in collect_pandas_module_members():
            return collect_pandas_module_members()[name]
        else:  # pragma: no cover
            if inspect.ismethod(getattr(pandas, name)):
                return unimplemented_func
            else:
                raise AttributeError(name)

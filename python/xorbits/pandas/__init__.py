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

# noinspection PyUnresolvedReferences
from pandas import Timedelta  # noqa: F401
from pandas import DateOffset, Interval, NaT, Timestamp, offsets

try:
    from pandas import NA, NamedAgg  # noqa: F401
except ImportError:  # pragma: no cover
    pass

from .core import DataFrame


def unimplemented_func():
    """
    Not implemented yet.
    """
    raise NotImplementedError(f"This function is not implemented yet.")


def __dir__():
    from .mars_adapters import MARS_DATAFRAME_CALLABLES

    return list(MARS_DATAFRAME_CALLABLES.keys())


def __getattr__(name: str):
    from .mars_adapters import MARS_DATAFRAME_CALLABLES

    if name in MARS_DATAFRAME_CALLABLES:
        return MARS_DATAFRAME_CALLABLES[name]
    else:
        # TODO fallback to pandas
        import pandas

        if not hasattr(pandas, name):
            raise AttributeError(name)
        else:
            return unimplemented_func

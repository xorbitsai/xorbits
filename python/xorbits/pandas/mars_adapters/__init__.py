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

from . import loc
from .core import MARS_DATAFRAME_CALLABLES, MARS_DATAFRAME_MAGIC_METHODS


def _install():
    from ...core.adapter import (
        MARS_DATAFRAME_GROUPBY_TYPE,
        MARS_DATAFRAME_TYPE,
        MARS_INDEX_TYPE,
        MARS_SERIES_GROUPBY_TYPE,
        MARS_SERIES_TYPE,
        register_data_members,
        wrap_magic_method,
    )
    from ...core.data import DataRef, DataType
    from .core import _register_execution_conditions

    for method in MARS_DATAFRAME_MAGIC_METHODS:
        setattr(DataRef, method, wrap_magic_method(method))

    _register_execution_conditions()
    for cls in MARS_DATAFRAME_TYPE:
        register_data_members(DataType.dataframe, cls)
    for cls in MARS_SERIES_TYPE:
        register_data_members(DataType.series, cls)
    for cls in MARS_INDEX_TYPE:
        register_data_members(DataType.index, cls)
    for cls in MARS_DATAFRAME_GROUPBY_TYPE:
        register_data_members(DataType.dataframe_groupby, cls)
    for cls in MARS_SERIES_GROUPBY_TYPE:
        register_data_members(DataType.series_groupby, cls)

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

import pandas

from ..core import Data, DataRef, DataType
from ..core.adapter import MarsDataFrameGroupBy, MarsSeriesGroupBy, to_mars
from ..core.utils.docstring import attach_module_callable_docstring


class DataFrameGroupBy(DataRef):
    def __init__(self, *args, **kwargs):
        data = Data(
            data_type=DataType.dataframe_groupby,
            mars_entity=MarsDataFrameGroupBy(*to_mars(args), **to_mars(kwargs)),
        )
        super().__init__(data)


attach_module_callable_docstring(
    DataFrameGroupBy, pandas, pandas.core.groupby.DataFrameGroupBy
)


class SeriesGroupBy(DataRef):
    def __init__(self, *args, **kwargs):
        data = Data(
            data_type=DataType.series_groupby,
            mars_entity=MarsSeriesGroupBy(*to_mars(args), **to_mars(kwargs)),
        )
        super().__init__(data)


attach_module_callable_docstring(
    SeriesGroupBy, pandas, pandas.core.groupby.SeriesGroupBy
)

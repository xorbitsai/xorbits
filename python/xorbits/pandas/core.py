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
from pandas.core.accessor import CachedAccessor

from ..core import Data, DataRef, DataType
from ..core.adapter import MarsDataFrame, MarsIndex, MarsSeries, to_mars
from ..core.data import register_cls_to_type
from ..core.utils.docstring import attach_module_callable_docstring
from .accessors import DatetimeAccessor, StringAccessor
from .plotting import PlotAccessor


@register_cls_to_type(data_type=DataType.dataframe)
class DataFrame(DataRef):

    plot = CachedAccessor("plot", PlotAccessor)

    def __init__(self, *args, **kwargs):
        data = Data(
            data_type=DataType.dataframe,
            mars_entity=MarsDataFrame(*to_mars(args), **to_mars(kwargs)),
        )
        super().__init__(data)


attach_module_callable_docstring(DataFrame, pandas, pandas.DataFrame)


@register_cls_to_type(data_type=DataType.series)
class Series(DataRef):

    str = CachedAccessor("str", StringAccessor)
    dt = CachedAccessor("dt", DatetimeAccessor)
    plot = CachedAccessor("plot", PlotAccessor)

    def __init__(self, *args, **kwargs):
        data = Data(
            data_type=DataType.series,
            mars_entity=MarsSeries(*to_mars(args), **to_mars(kwargs)),
        )
        super().__init__(data)


attach_module_callable_docstring(Series, pandas, pandas.Series)


@register_cls_to_type(data_type=DataType.index)
class Index(DataRef):
    def __init__(self, *args, **kwargs):
        data = Data(
            data_type=DataType.index,
            mars_entity=MarsIndex(*to_mars(args), **to_mars(kwargs)),
        )
        super().__init__(data)


attach_module_callable_docstring(Index, pandas, pandas.Index)

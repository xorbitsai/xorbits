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

import pandas

from ..core import DataRef, DataType
from ..core.data import register_cls_to_type
from ..core.utils.docstring import attach_module_callable_docstring


@register_cls_to_type(DataType.dataframe_groupby)
class DataFrameGroupBy(DataRef):
    """This is a DataFrameGroupBy subclass"""


attach_module_callable_docstring(
    DataFrameGroupBy, pandas, pandas.core.groupby.DataFrameGroupBy
)


@register_cls_to_type(data_type=DataType.series_groupby)
class SeriesGroupBy(DataRef):
    """This is a SeriesGroupBy subclass"""


attach_module_callable_docstring(
    SeriesGroupBy, pandas, pandas.core.groupby.SeriesGroupBy
)

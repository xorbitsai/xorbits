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

from ..core.adapter import (
    MarsDataFrameAt,
    MarsDataFrameIat,
    MarsDataFrameIloc,
    MarsDataFrameLoc,
    MarsGetItemProxy,
    register_converter,
)
from ..core.utils.docstring import attach_module_callable_docstring


@register_converter(from_cls_list=[MarsDataFrameAt])
class DataFrameAt(MarsGetItemProxy):
    pass


attach_module_callable_docstring(DataFrameAt, pandas, pandas.core.indexing._AtIndexer)


@register_converter(from_cls_list=[MarsDataFrameIat])
class DataFrameIat(MarsGetItemProxy):
    pass


attach_module_callable_docstring(DataFrameIat, pandas, pandas.core.indexing._iAtIndexer)


@register_converter(from_cls_list=[MarsDataFrameLoc])
class DataFrameLoc(MarsGetItemProxy):
    pass


attach_module_callable_docstring(DataFrameLoc, pandas, pandas.core.indexing._LocIndexer)


@register_converter(from_cls_list=[MarsDataFrameIloc])
class DataFrameIloc(MarsGetItemProxy):
    pass


attach_module_callable_docstring(
    DataFrameIloc, pandas, pandas.core.indexing._iLocIndexer
)

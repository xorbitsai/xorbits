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
from typing import Union

import pandas

from ..core import DataRef
from ..core.adapter import (
    MarsDatetimeAccessor,
    MarsGetAttrProxy,
    MarsStringAccessor,
    register_converter,
    to_mars,
)
from ..core.utils.docstring import attach_module_callable_docstring
from .mars_adapters.core import install_members


@register_converter(from_cls_list=[MarsStringAccessor])
class StringAccessor(MarsGetAttrProxy):
    def __init__(self, obj: Union[DataRef, MarsStringAccessor]):
        if isinstance(obj, MarsStringAccessor):
            # when the owner of the accessor being xorbits.core.data.Data.
            super().__init__(obj)
        elif isinstance(obj, DataRef):
            # when the owner of the accessor being xorbits.pandas.core.Series.
            super().__init__(MarsStringAccessor(to_mars(obj)))


install_members(
    StringAccessor,
    MarsStringAccessor,
    pandas,
    pandas.core.strings.accessor.StringMethods,
)
attach_module_callable_docstring(
    StringAccessor, pandas, pandas.core.strings.accessor.StringMethods
)


@register_converter(from_cls_list=[MarsDatetimeAccessor])
class DatetimeAccessor(MarsGetAttrProxy):
    def __init__(self, obj: Union[DataRef, MarsDatetimeAccessor]):
        if isinstance(obj, MarsDatetimeAccessor):
            # when the owner of the accessor being xorbits.core.data.Data.
            super().__init__(obj)
        elif isinstance(obj, DataRef):
            # when the owner of the accessor being xorbits.pandas.core.Series.
            super().__init__(MarsDatetimeAccessor(to_mars(obj)))


install_members(
    DatetimeAccessor,
    MarsDatetimeAccessor,
    pandas,
    pandas.core.indexes.accessors.CombinedDatetimelikeProperties,
)
attach_module_callable_docstring(
    DatetimeAccessor,
    pandas,
    pandas.core.indexes.accessors.CombinedDatetimelikeProperties,
)

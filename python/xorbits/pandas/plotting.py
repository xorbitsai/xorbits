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

from ..core.adapter import MarsPlotAccessor, register_converter, to_mars
from ..core.utils.docstring import attach_module_callable_docstring
from .mars_adapters.core import MarsGetAttrProxy, install_members


@register_converter(from_cls_list=[MarsPlotAccessor])
class PlotAccessor(MarsGetAttrProxy):
    def __init__(self, obj):
        super().__init__(MarsPlotAccessor(to_mars(obj)))


install_members(
    PlotAccessor,
    MarsPlotAccessor,
    pandas,
    pandas.plotting._core.PlotAccessor,
)
attach_module_callable_docstring(
    PlotAccessor, pandas, pandas.plotting._core.PlotAccessor
)

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
    MarsEWM,
    MarsExpanding,
    MarsGetAttrProxy,
    MarsRolling,
    register_converter,
)
from ..core.utils.docstring import attach_module_callable_docstring
from .mars_adapters.core import install_members


@register_converter(from_cls_list=[MarsRolling])
class Rolling(MarsGetAttrProxy):
    pass


install_members(
    Rolling,
    MarsRolling,
    pandas,
    pandas.core.window.rolling.Rolling,
)
attach_module_callable_docstring(Rolling, pandas, pandas.core.window.rolling.Rolling)


@register_converter(from_cls_list=[MarsExpanding])
class Expanding(MarsGetAttrProxy):
    pass


install_members(
    Expanding,
    MarsExpanding,
    pandas,
    pandas.core.window.expanding.Expanding,
)
attach_module_callable_docstring(
    Expanding, pandas, pandas.core.window.expanding.Expanding
)


@register_converter(from_cls_list=[MarsEWM])
class ExponentialMovingWindow(MarsGetAttrProxy):
    pass


install_members(
    ExponentialMovingWindow,
    MarsEWM,
    pandas,
    pandas.core.window.ewm.ExponentialMovingWindow,
)
attach_module_callable_docstring(
    ExponentialMovingWindow, pandas, pandas.core.window.ewm.ExponentialMovingWindow
)

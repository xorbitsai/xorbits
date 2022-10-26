# -*- coding: utf-8 -*-
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
from xorbits.core.mars_adaption import from_mars, register_converter, wrap_mars_callable

from ...adapter.mars import MarsCachedAccessor, MarsDatetimeAccessor, MarsStringAccessor


class Accessor:
    def __init__(self, mars_obj):
        self._mars_obj = mars_obj

    def __getattr__(self, item):
        attr = getattr(self._mars_obj, item)
        if callable(attr):
            return wrap_mars_callable(attr)
        else:
            return from_mars(attr)


@register_converter(from_cls=MarsStringAccessor)
class StringAccessor(Accessor):
    pass


@register_converter(from_cls=MarsDatetimeAccessor)
class DatetimeAccessor(Accessor):
    pass


@register_converter(from_cls=MarsCachedAccessor)
class CachedAccessor(Accessor):
    pass

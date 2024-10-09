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
import numpy

from ...utils import is_numpy_2

if is_numpy_2():
    from numpy.lib import _index_tricks_impl as index_tricks
else:
    from numpy.lib import index_tricks

from ...core.adapter import (
    MarsCClass,
    MarsGetItemProxy,
    MarsMGridClass,
    MarsOGridClass,
    MarsRClass,
    register_converter,
)
from ...core.utils.docstring import attach_module_callable_docstring


@register_converter(from_cls_list=[MarsCClass])
class CClass(MarsGetItemProxy):
    def __init__(self):
        super().__init__(MarsCClass())


attach_module_callable_docstring(CClass, numpy, index_tricks.CClass)
c_ = CClass()


@register_converter(from_cls_list=[MarsRClass])
class RClass(MarsGetItemProxy):
    def __init__(self):
        super().__init__(MarsRClass())


attach_module_callable_docstring(RClass, numpy, index_tricks.RClass)
r_ = RClass()


@register_converter(from_cls_list=[MarsOGridClass])
class OGridClass(MarsGetItemProxy):
    def __init__(self):
        super().__init__(MarsOGridClass())


attach_module_callable_docstring(OGridClass, numpy, index_tricks.OGridClass)
ogrid = OGridClass()


@register_converter(from_cls_list=[MarsMGridClass])
class MGridClass(MarsGetItemProxy):
    def __init__(self):
        super().__init__(MarsMGridClass())


attach_module_callable_docstring(MGridClass, numpy, index_tricks.MGridClass)
mgrid = MGridClass()

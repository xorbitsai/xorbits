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
import inspect
from types import ModuleType
from typing import Any, Callable, Dict, Set

import numpy

from ...core.adapter import (
    MARS_TENSOR_TYPE,
    MarsAxisConcatenator,
    from_mars,
    mars_c_,
    mars_mgrid,
    mars_ogrid,
    mars_r_,
    mars_tensor,
    register_converter,
    to_mars,
    wrap_mars_callable,
)


def _collect_module_callables(
    m: ModuleType, docstring_src_module: ModuleType
) -> Dict[str, Callable]:
    module_callables: Dict[str, Callable] = dict()

    # install module functions.
    for name, func in inspect.getmembers(m, callable):
        module_callables[name] = wrap_mars_callable(
            func,
            attach_docstring=True,
            is_cls_member=False,
            docstring_src_module=docstring_src_module,
            docstring_src=getattr(docstring_src_module, name, None),
        )

    return module_callables


MARS_TENSOR_CALLABLES: Dict[str, Callable] = _collect_module_callables(
    mars_tensor, numpy
)
MARS_TENSOR_RANDOM_CALLABLES: Dict[str, Callable] = _collect_module_callables(
    mars_tensor.random, numpy.random
)
MARS_TENSOR_FFT_CALLABLES: Dict[str, Callable] = _collect_module_callables(
    mars_tensor.fft, numpy.fft
)
MARS_TENSOR_LINALG_CALLABLES: Dict[str, Callable] = _collect_module_callables(
    mars_tensor.linalg, numpy.linalg
)


@register_converter(from_cls_list=[MarsAxisConcatenator])
class MarsGetItemProxy:
    def __init__(self, mars_obj):
        self._mars_obj = mars_obj

    def __getitem__(self, item):
        return from_mars(self._mars_obj[to_mars(item)])


def _collect_tensor_objects():
    mars_tensor_objects: Dict[str, Any] = dict()
    mars_tensor_objects["mgrid"] = MarsGetItemProxy(mars_mgrid)
    mars_tensor_objects["ogrid"] = MarsGetItemProxy(mars_ogrid)
    mars_tensor_objects["c_"] = MarsGetItemProxy(mars_c_)
    mars_tensor_objects["r_"] = MarsGetItemProxy(mars_r_)
    return mars_tensor_objects


MARS_TENSOR_OBJECTS = _collect_tensor_objects()


def _collect_tensor_magic_methods() -> Set[str]:
    magic_methods: Set[str] = set()

    magic_methods_to_skip: Set[str] = {
        "__init__",
        "__dir__",
        "__repr__",
        "__str__",
        "__setattr__",
        "__getattr__",
    }

    for mars_cls in MARS_TENSOR_TYPE:
        for name, _ in inspect.getmembers(mars_cls, inspect.isfunction):
            if (
                name.startswith("__")
                and name.endswith("__")
                and name not in magic_methods_to_skip
            ):
                magic_methods.add(name)
    return magic_methods


MARS_TENSOR_MAGIC_METHODS = _collect_tensor_magic_methods()

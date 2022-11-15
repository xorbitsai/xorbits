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
import inspect
from types import ModuleType
from typing import Callable, Dict, Set

from ...core.adapter import MARS_TENSOR_TYPE, mars_tensor, wrap_mars_callable


def _collect_module_callables(m: ModuleType) -> Dict[str, Callable]:
    mars_tensor_callables: Dict[str, Callable] = dict()

    # incall module functions.
    for name, func in inspect.getmembers(m, callable):
        mars_tensor_callables[name] = wrap_mars_callable(func)
    return mars_tensor_callables


MARS_TENSOR_CALLABLES: Dict[str, Callable] = _collect_module_callables(mars_tensor)
MARS_TENSOR_RANDOM_CALLABLES: Dict[str, Callable] = _collect_module_callables(
    mars_tensor.random
)
MARS_TENSOR_FFT_CALLABLES: Dict[str, Callable] = _collect_module_callables(
    mars_tensor.fft
)
MARS_TENSOR_LINALG_CALLABLES: Dict[str, Callable] = _collect_module_callables(
    mars_tensor.linalg
)


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

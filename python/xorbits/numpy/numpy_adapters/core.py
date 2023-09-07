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

import inspect
import warnings
from functools import lru_cache
from types import ModuleType
from typing import Any, Callable, Dict, Type

import numpy as np
import scipy

from ..._mars.core import Entity as MarsEntity
from ...core import DataType
from ...core.adapter import (
    ClsMethodWrapper,
    MarsOutputType,
    get_cls_members,
    wrap_mars_callable,
)
from ...core.utils.fallback import wrap_fallback_module_method

_NO_ANNOTATION_FUNCS: Dict[Callable, MarsOutputType] = {
    np.busday_count: MarsOutputType.tensor,
    np.busday_offset: MarsOutputType.object,
    np.is_busday: MarsOutputType.object,
    np.isneginf: MarsOutputType.object,
    np.isposinf: MarsOutputType.object,
    np.einsum_path: MarsOutputType.object,
    np.kron: MarsOutputType.tensor,
    np.outer: MarsOutputType.tensor,
    np.trace: MarsOutputType.tensor,
    np.linalg.cond: MarsOutputType.tensor,
    np.linalg.det: MarsOutputType.tensor,
    np.linalg.eig: MarsOutputType.object,
    np.linalg.eigh: MarsOutputType.object,
    np.linalg.eigvals: MarsOutputType.tensor,
    np.linalg.eigvalsh: MarsOutputType.tensor,
    np.linalg.tensorsolve: MarsOutputType.tensor,
    np.linalg.multi_dot: MarsOutputType.tensor,
    np.linalg.matrix_power: MarsOutputType.tensor,
    np.linalg.matrix_rank: MarsOutputType.tensor,
    np.linalg.lstsq: MarsOutputType.object,
    np.linalg.slogdet: MarsOutputType.object,
    np.linalg.pinv: MarsOutputType.tensor,
    np.linalg.tensorinv: MarsOutputType.tensor,
    np.random.default_rng: MarsOutputType.object,
    np.random.Generator: MarsOutputType.object,
    np.random.PCG64: MarsOutputType.object,
    np.random.MT19937: MarsOutputType.object,
    np.random.Generator: MarsOutputType.object,
}


class NumpyClsMethodWrapper(ClsMethodWrapper):
    def _generate_fallback_data(self, mars_entity: MarsEntity):
        return mars_entity.to_numpy()

    def _generate_warning_msg(self, entity: MarsEntity, func_name: str):
        return f"{type(entity).__name__}.{func_name} will fallback to Numpy"

    def _get_output_type(self, func: Callable) -> MarsOutputType:
        try:  # pragma: no cover
            return_annotation = inspect.signature(func).return_annotation
            if return_annotation is inspect.Signature.empty:
                # mostly for python3.7 whose return_annotation is always empty
                return _NO_ANNOTATION_FUNCS.get(func, MarsOutputType.object)
            all_types = [t.strip() for t in return_annotation.split("|")]

            return (
                MarsOutputType.tensor
                if "ndarray" in all_types
                else MarsOutputType.object
            )
        except (
            ValueError
        ):  # some np methods return objects and inspect.signature throws a ValueError
            return _NO_ANNOTATION_FUNCS.get(func, MarsOutputType.object)

    def _get_docstring_src_module(self):
        return np


def _collect_numpy_cls_members(np_cls: Type, data_type: DataType):
    members = get_cls_members(data_type)
    for name, np_cls_member in inspect.getmembers(np_cls):
        if name not in members and not name.startswith("_"):
            numpy_cls_method_wrapper = NumpyClsMethodWrapper(
                library_cls=np_cls, func_name=name, fallback_warning=True
            )
            members[name] = numpy_cls_method_wrapper.get_wrapped()


def _collect_numpy_ndarray_members():
    _collect_numpy_cls_members(np.ndarray, DataType.tensor)


@lru_cache(maxsize=1)
def collect_numpy_module_members(np_mod: ModuleType) -> Dict[str, Any]:
    from ..mars_adapters.core import MARS_TENSOR_CALLABLES

    module_methods: Dict[str, Any] = dict()
    with warnings.catch_warnings():
        for name, mod_member in inspect.getmembers(np_mod):
            if (
                name not in MARS_TENSOR_CALLABLES
                and callable(mod_member)
                and not name.startswith("_")
            ):
                # avoid inconsistency: np.ramdom.__name__ is 'numpy.random' while np.ndarray.__name__ is 'ndarray'
                np_mod_str = (
                    np_mod.__name__
                    if "numpy" in np_mod.__name__
                    else "numpy." + np_mod.__name__
                )
                warning_str = f"xorbits.{np_mod_str}.{name} will fallback to NumPy"
                numpy_cls_method_wrapper = NumpyClsMethodWrapper()
                output_type = numpy_cls_method_wrapper._get_output_type(
                    func=getattr(np_mod, name)
                )

                module_methods[name] = wrap_mars_callable(
                    wrap_fallback_module_method(np_mod, name, output_type, warning_str),
                    attach_docstring=True,
                    is_cls_member=False,
                    docstring_src_module=np_mod,
                    docstring_src=getattr(np_mod, name, None),
                    fallback_warning=True,
                )

    return module_methods


NUMPY_MEMBERS = collect_numpy_module_members(np)
NUMPY_LINALG_MEMBERS = collect_numpy_module_members(np.linalg)
NUMPY_FFT_MEMBERS = collect_numpy_module_members(np.fft)
NUMPY_RANDOM_MEMBERS = collect_numpy_module_members(np.random)
NUMPY_SPECIAL_MEMBERS = collect_numpy_module_members(scipy.special)

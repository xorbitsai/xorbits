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
from types import ModuleType
from typing import Any, Callable, Dict

import numpy as np

from ...core.adapter import MarsOutputType, wrap_mars_callable
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


def _get_output_type(func: Callable) -> MarsOutputType:
    try:  # pragma: no cover
        return_annotation = inspect.signature(func).return_annotation
        if return_annotation is inspect.Signature.empty:
            # mostly for python3.7 whose return_annotation is always empty
            return _NO_ANNOTATION_FUNCS.get(func, MarsOutputType.object)
        all_types = [t.strip() for t in return_annotation.split("|")]

        return (
            MarsOutputType.tensor if "ndarray" in all_types else MarsOutputType.object
        )
    except (
        ValueError
    ):  # some np methods return objects and inspect.signature throws a ValueError
        return _NO_ANNOTATION_FUNCS.get(func, MarsOutputType.object)


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
                output_type = _get_output_type(getattr(np_mod, name))

                module_methods[name] = wrap_mars_callable(
                    wrap_fallback_module_method(np_mod, name, output_type, warning_str),
                    attach_docstring=True,
                    is_cls_member=False,
                    docstring_src_module=np_mod,
                    docstring_src=getattr(np_mod, name, None),
                    fallback_warning=True,
                )

    return module_methods

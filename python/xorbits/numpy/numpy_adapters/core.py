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

import functools
import inspect
import warnings
from typing import Any, Dict, Type, Callable

import numpy as np

from ...core.adapter import (
    MarsOutputType,
    mars_remote,
    wrap_mars_callable,
)


_NO_ANNOTATION_FUNCS: Dict[Callable, MarsOutputType] = {
    np.random.default_rng: MarsOutputType.object,
}


def _get_output_type(func: Callable) -> MarsOutputType:
    try:
        return_annotation = inspect.signature(func).return_annotation
        if return_annotation is inspect.Signature.empty:
            # mostly for python3.7 whose return_annotation is always empty
            return _NO_ANNOTATION_FUNCS.get(func, MarsOutputType.object)
        all_types = [t.strip() for t in return_annotation.split("|")]

        return MarsOutputType.tensor if 'ndarray' in all_types else MarsOutputType.object
    except TypeError: # some np methods return objects and inspect.signature throws a TypeError
        return _NO_ANNOTATION_FUNCS.get(func, MarsOutputType.object)


def wrap_numpy_module_method(np_cls: Type, func_name: str):
    # wrap np function
    @functools.wraps(getattr(np_cls, func_name))
    def _wrapped(*args, **kwargs):
        warnings.warn(
            f"xorbits.numpy.{func_name} will fallback to NumPy", RuntimeWarning
        )
        output_type = _get_output_type(func_name)

        # use mars remote to execute numpy functions
        def execute_func(f_name: str, *args, **kwargs):
            from xorbits.core.adapter import MarsEntity

            def _replace_data(nested):
                if isinstance(nested, dict):
                    vals = list(nested.values())
                else:
                    vals = list(nested)

                new_vals = []
                for val in vals:
                    if isinstance(val, (dict, list, tuple, set)):
                        new_val = _replace_data(val)
                    else:
                        if isinstance(val, MarsEntity):
                            new_val = val.fetch()
                        else:
                            new_val = val
                    new_vals.append(new_val)
                if isinstance(nested, dict):
                    return type(nested)((k, v) for k, v in zip(nested.keys(), new_vals))
                else:
                    return type(nested)(new_vals)

            return getattr(np_cls, f_name)(*_replace_data(args), **_replace_data(kwargs))

        new_args = (func_name,) + args
        ret = mars_remote.spawn(
            execute_func, args=new_args, kwargs=kwargs, output_types=output_type
        )
        if output_type == MarsOutputType.tensor:
            ret = ret.ensure_data()
        else:
            ret = ret.execute()
        return ret

    return _wrapped


def _collect_numpy_module_members(np_cls: Type) -> Dict[str, Any]:
    from ..mars_adapters.core import MARS_TENSOR_CALLABLES

    module_methods: Dict[str, Any] = dict()
    with warnings.catch_warnings():
        for name, cls_member in inspect.getmembers(np_cls):
            if (
                name not in MARS_TENSOR_CALLABLES
                and callable(cls_member)
                and not name.startswith("_")
            ):
                module_methods[name] = wrap_mars_callable(
                    wrap_numpy_module_method(np_cls, name),
                    attach_docstring=True,
                    is_cls_member=False,
                    docstring_src_module=np_cls,
                    docstring_src=getattr(np_cls, name, None),
                    fallback_warning=True,
                )
    return module_methods


NUMPY_MODULE_METHODS = _collect_numpy_module_members

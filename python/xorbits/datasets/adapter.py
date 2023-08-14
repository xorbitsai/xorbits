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

from typing import Callable, Dict

from .._mars.utils import lazy_import
from ..core.adapter import (
    collect_cls_members,
    register_data_members,
    wrap_mars_callable,
)
from ..core.data import DataType
from .backends.huggingface.from_huggingface import from_huggingface
from .dataset import Dataset

MARS_DATASET_TYPE = (Dataset,)


def _collect_module_callables() -> Dict[str, Callable]:
    module_callables = {}
    # TODO(fyrestone): Remove this hard code module
    hf_datasets = lazy_import("datasets")

    # install module functions
    for func in [from_huggingface]:
        module_callables[func.__name__] = wrap_mars_callable(
            func,
            attach_docstring=True,
            is_cls_member=False,
            docstring_src_module=hf_datasets,
            docstring_src=func,
        )
    return module_callables


MARS_DATASET_CALLABLES: Dict[str, Callable] = _collect_module_callables()


def _install():
    for cls in MARS_DATASET_TYPE:
        register_data_members(
            DataType.dataset, collect_cls_members(cls, DataType.dataset)
        )

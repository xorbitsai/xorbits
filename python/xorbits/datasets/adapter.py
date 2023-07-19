# Copyright 2022-2023 XProbe Inc.
# derived from copyright 1999-2021 Alibaba Group Holding Ltd.
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

from ..core.adapter import (
    collect_cls_members,
    register_data_members,
    wrap_mars_callable,
)
from ..core.data import DataType
from .dataset import Dataset
from .backends.huggingface.core import from_huggingface

MARS_DATASET_TYPE = (Dataset,)


def _collect_module_callables() -> Dict[str, Callable]:
    module_callables = {}

    # install module functions
    for func in [from_huggingface]:
        module_callables[func.__name__] = wrap_mars_callable(
            func,
            attach_docstring=False,
            is_cls_member=False,
        )
    return module_callables


MARS_DATASET_CALLABLES: Dict[str, Callable] = _collect_module_callables()


def _install():
    for cls in MARS_DATASET_TYPE:
        register_data_members(
            DataType.dataset, collect_cls_members(cls, DataType.dataset)
        )

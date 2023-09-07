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


from . import flatiter
from .core import (
    MARS_TENSOR_CALLABLES,
    MARS_TENSOR_FFT_CALLABLES,
    MARS_TENSOR_LINALG_CALLABLES,
    MARS_TENSOR_MAGIC_METHODS,
    MARS_TENSOR_OBJECTS,
    MARS_TENSOR_RANDOM_CALLABLES,
    MARS_TENSOR_SPECIAL_CALLABLES,
)


def _install():
    from ...core.adapter import (
        MARS_TENSOR_TYPE,
        collect_cls_members,
        register_data_members,
        wrap_magic_method,
    )
    from ...core.data import DataRef, DataType

    for method in MARS_TENSOR_MAGIC_METHODS:
        setattr(DataRef, method, wrap_magic_method(method))

    for cls in MARS_TENSOR_TYPE:
        register_data_members(
            DataType.tensor, collect_cls_members(cls, DataType.tensor)
        )

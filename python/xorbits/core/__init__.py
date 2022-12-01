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

from .data import Data, DataRef, DataRefMeta, DataType


def _register_magic_methods():
    from ..numpy.mars_adapters import MARS_TENSOR_MAGIC_METHODS
    from ..pandas.mars_adapters import MARS_DATAFRAME_MAGIC_METHODS
    from .adapter import wrap_magic_method

    magic_methods = MARS_TENSOR_MAGIC_METHODS.union(MARS_DATAFRAME_MAGIC_METHODS)

    for method in magic_methods:
        setattr(DataRef, method, wrap_magic_method(method))


_register_magic_methods()
del _register_magic_methods

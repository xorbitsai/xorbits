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

from xorbits.core import DataRef

from .... import numpy as np


def test_array_creation():
    # 1-d array creation.
    assert isinstance(np.arange(10), DataRef)
    # 2-d array creation.
    assert isinstance(np.eye(3), DataRef)
    # n-d array creation.
    assert isinstance(np.zeros((2, 3)), DataRef)


def test_indexing(dummy_int_1d_array):
    # test magic method __getitem__.
    assert isinstance(dummy_int_1d_array[0], DataRef)
    assert isinstance(dummy_int_1d_array[1:], DataRef)
    assert isinstance(dummy_int_1d_array[np.array([1, 2])], DataRef)


def test_arithmetic_op(dummy_int_1d_array):
    # test arithmetic magic methods.
    assert isinstance(dummy_int_1d_array + dummy_int_1d_array, DataRef)
    assert isinstance(dummy_int_1d_array - dummy_int_1d_array, DataRef)
    assert isinstance(dummy_int_1d_array * dummy_int_1d_array, DataRef)
    assert isinstance(dummy_int_1d_array / dummy_int_1d_array, DataRef)
    assert isinstance(dummy_int_1d_array**2, DataRef)
    assert isinstance(-dummy_int_1d_array, DataRef)


def test_comparison_op(dummy_int_1d_array):
    # test comparison magic methods.
    assert isinstance(dummy_int_1d_array > 0, DataRef)
    assert isinstance(dummy_int_1d_array == 0, DataRef)
    assert isinstance(dummy_int_1d_array < 0, DataRef)

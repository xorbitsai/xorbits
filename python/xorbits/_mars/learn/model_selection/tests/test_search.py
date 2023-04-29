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

import itertools
from typing import Iterable, Sized

import numpy as np
import pytest

from .... import tensor as mt
from .. import ParameterGrid


def test_parameter_grid(setup):
    with pytest.raises(TypeError) as e:
        invalid_param = 2
        _ = ParameterGrid(invalid_param)
        error_str = (
            f"Parameter grid should be a dict or a list, got: {invalid_param!r} of"
            f" type {type(invalid_param).__name__}"
        )
        assert error_str == e.value

    with pytest.raises(TypeError) as e:
        invalid_param = [2]
        _ = ParameterGrid(invalid_param)
        error_str = f"Parameter grid is not a dict ({invalid_param[0]!r})"
        assert error_str == e.value

    with pytest.raises(ValueError) as e:
        invalid_param = [{"foo": mt.ones((1, 2))}]
        _ = ParameterGrid(invalid_param)
        error_str = (
            f"Parameter array for foo should be one-dimensional, got:"
            f" {invalid_param[0]['foo']!r} with shape {invalid_param[0]['foo'].shape}"
        )
        assert error_str == e.value

    with pytest.raises(ValueError) as e:
        invalid_param = [{"foo": []}]
        _ = ParameterGrid(invalid_param)
        error_str = (
            f"Parameter grid for parameter foo need "
            f"to be a non-empty sequence, got: {[]!r}"
        )
        assert error_str == e.value

    arr1 = [1, 2, 3]
    params1 = {"foo": arr1}
    grid1 = ParameterGrid(params1)
    assert isinstance(grid1, Iterable)
    assert isinstance(grid1, Sized)
    assert len(grid1) == 3

    for i, param in enumerate(grid1):
        key, value = next(iter(param.items()))
        assert key == "foo"
        assert isinstance(value, mt.Tensor)
        np.testing.assert_equal(np.int32(value.execute().fetch()), np.int32(arr1[i]))

    arr2 = [np.int32(1), np.int32(2), np.int32(3)]
    params2 = {"foo": arr2}
    grid2 = ParameterGrid(params2)
    assert isinstance(grid2, Iterable)
    assert isinstance(grid2, Sized)
    assert len(grid2) == 3

    for i, param in enumerate(grid2):
        key, value = next(iter(param.items()))
        assert key == "foo"
        assert isinstance(value, mt.Tensor)
        np.testing.assert_equal(np.int32(value.execute().fetch()), np.int32(arr2[i]))

    arr3 = [mt.array(1), mt.array(2), mt.array(3)]
    params3 = {"foo": arr3}
    grid3 = ParameterGrid(params3)
    assert isinstance(grid3, Iterable)
    assert isinstance(grid3, Sized)
    assert len(grid3) == 3

    for i, param in enumerate(grid3):
        key, value = next(iter(param.items()))
        assert key == "foo"
        assert isinstance(value, mt.Tensor)
        np.testing.assert_equal(np.int32(value.execute().fetch()), np.int32(arr3[i]))

    params4 = {
        "k1": [1, np.int32(2), mt.array(3)],
        "k2": ["a", "b", "c"],
        "k3": [True, False],
    }
    grid4 = ParameterGrid(params4)
    assert isinstance(grid4, Iterable)
    assert isinstance(grid4, Sized)
    assert len(grid4) == 18

    expected = []
    for values in itertools.product(*params4.values()):
        new_dict = {key: value for key, value in zip(params4.keys(), values)}
        expected.append(new_dict)

    res = list(grid4)
    for expected_param, res_param in zip(expected, res):
        assert expected_param == res_param

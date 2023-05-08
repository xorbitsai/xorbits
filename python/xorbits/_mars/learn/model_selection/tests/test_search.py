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

    with pytest.raises(TypeError) as e:
        invalid_param = [{"foo": "s"}]
        _ = ParameterGrid(invalid_param)
        error_str = (
            f"Parameter grid for parameter foo needs to be a list or a"
            f" mars array, but got {'s'!r} (of type "
            f"{type('s').__name__}) instead. Single values "
            "need to be wrapped in a list with one element."
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
        np.testing.assert_equal(np.int32(grid1[i]["foo"].fetch()), np.int32(arr1[i]))

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
        np.testing.assert_equal(np.int32(grid2[i]["foo"].fetch()), np.int32(arr2[i]))

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
        np.testing.assert_equal(np.int32(grid3[i]["foo"].fetch()), np.int32(arr3[i]))

    arr4 = np.ones(5)
    params4 = {"foo": arr4}
    grid4 = ParameterGrid(params4)
    assert isinstance(grid4, Iterable)
    assert isinstance(grid4, Sized)
    assert len(grid4) == 5

    for i, param in enumerate(grid4):
        key, value = next(iter(param.items()))
        assert key == "foo"
        assert isinstance(value, mt.Tensor)
        np.testing.assert_equal(np.int32(value.execute().fetch()), np.int32(arr4[i]))
        np.testing.assert_equal(np.int32(grid4[i]["foo"].fetch()), np.int32(arr4[i]))

    params5 = {
        "k1": [1, np.int32(2), mt.array(3)],
        "k2": ["a", "b", "c"],
        "k3": [True, False],
    }
    grid5 = ParameterGrid(params5)
    assert isinstance(grid5, Iterable)
    assert isinstance(grid5, Sized)
    assert len(grid5) == 3 * 3 * 2

    expected = []
    for values in itertools.product(*params5.values()):
        new_dict = {key: value for key, value in zip(params5.keys(), values)}
        expected.append(new_dict)

    res = list(grid5)
    for expected_param, res_param in zip(expected, res):
        assert expected_param == res_param

    params6 = [
        {"k1": ["a"]},
        {"k1": ["b"], "k2": [1, 2]},
    ]
    grid6 = ParameterGrid(params6)
    assert isinstance(grid6, Iterable)
    assert isinstance(grid6, Sized)
    assert len(grid6) == 1 + 1 * 2

    expected = []
    for params in params6:
        for values in itertools.product(*params.values()):
            new_dict = {key: value for key, value in zip(params.keys(), values)}
            expected.append(new_dict)

    res = list(grid6)
    for expected_param, res_param in zip(expected, res):
        assert expected_param == res_param

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

import numpy as np
import pandas as pd
import pytest

from ..adapter import MarsEntity, from_mars, mars_dataframe, to_mars
from ..data import DataRef


def test_to_mars(setup, dummy_df):
    assert isinstance(to_mars(dummy_df), MarsEntity)

    # tuple.
    ins = (dummy_df, "foo")
    assert isinstance(to_mars(ins)[0], MarsEntity)
    assert not isinstance(to_mars(ins)[1], MarsEntity)
    # nested tuple.
    ins = ((dummy_df, "foo"), "bar")
    assert isinstance(to_mars(ins)[0][0], MarsEntity)
    assert not isinstance(to_mars(ins)[0][1], MarsEntity)
    assert not isinstance(to_mars(ins)[1], MarsEntity)
    ins = ([dummy_df, "foo"], "bar")
    assert isinstance(to_mars(ins)[0][0], MarsEntity)
    assert not isinstance(to_mars(ins)[0][1], MarsEntity)
    assert not isinstance(to_mars(ins)[1], MarsEntity)
    ins = ({"foo": dummy_df, "bar": "baz"}, "bar")
    assert isinstance(to_mars(ins)[0]["foo"], MarsEntity)
    assert not isinstance(to_mars(ins)[0]["bar"], MarsEntity)
    assert not isinstance(to_mars(ins)[1], MarsEntity)

    # list.
    ins = [dummy_df, "foo"]
    assert isinstance(to_mars(ins)[0], MarsEntity)
    assert not isinstance(to_mars(ins)[1], MarsEntity)
    # nested list.
    ins = [(dummy_df, "foo"), "bar"]
    assert isinstance(to_mars(ins)[0][0], MarsEntity)
    assert not isinstance(to_mars(ins)[0][1], MarsEntity)
    assert not isinstance(to_mars(ins)[1], MarsEntity)
    ins = [[dummy_df, "foo"], "bar"]
    assert isinstance(to_mars(ins)[0][0], MarsEntity)
    assert not isinstance(to_mars(ins)[0][1], MarsEntity)
    assert not isinstance(to_mars(ins)[1], MarsEntity)
    ins = [{"foo": dummy_df, "bar": "baz"}, "bar"]
    assert isinstance(to_mars(ins)[0]["foo"], MarsEntity)
    assert not isinstance(to_mars(ins)[0]["bar"], MarsEntity)
    assert not isinstance(to_mars(ins)[1], MarsEntity)

    # dict.
    ins = {"foo": dummy_df, "bar": "baz"}
    assert isinstance(to_mars(ins)["foo"], MarsEntity)
    assert not isinstance(to_mars(ins)["bar"], MarsEntity)
    # nested dict.
    ins = {"foo": (dummy_df, "foo"), "bar": "baz"}
    assert isinstance(to_mars(ins)["foo"][0], MarsEntity)
    assert not isinstance(to_mars(ins)["foo"][1], MarsEntity)
    assert not isinstance(to_mars(ins)["bar"], MarsEntity)
    ins = {"foo": [dummy_df, "foo"], "bar": "baz"}
    assert isinstance(to_mars(ins)["foo"][0], MarsEntity)
    assert not isinstance(to_mars(ins)["foo"][1], MarsEntity)
    assert not isinstance(to_mars(ins)["bar"], MarsEntity)
    ins = {"foo": {"bar": dummy_df}, "bar": "baz"}
    assert isinstance(to_mars(ins)["foo"]["bar"], MarsEntity)
    assert not isinstance(to_mars(ins)["bar"], MarsEntity)


def test_from_mars():
    mdf = mars_dataframe.DataFrame({"foo": (1, 2, 3), "bar": (4, 5, 6)})
    assert isinstance(from_mars(mdf), DataRef)

    # tuple.
    ins = (mdf, "foo")
    assert isinstance(from_mars(ins)[0], DataRef)
    assert not isinstance(from_mars(ins)[1], DataRef)
    # nested tuple.
    ins = ((mdf, "foo"), "bar")
    assert isinstance(from_mars(ins)[0][0], DataRef)
    assert not isinstance(from_mars(ins)[0][1], DataRef)
    assert not isinstance(from_mars(ins)[1], DataRef)
    ins = ([mdf, "foo"], "bar")
    assert isinstance(from_mars(ins)[0][0], DataRef)
    assert not isinstance(from_mars(ins)[0][1], DataRef)
    assert not isinstance(from_mars(ins)[1], DataRef)
    ins = ({"foo": mdf, "bar": "baz"}, "bar")
    assert isinstance(from_mars(ins)[0]["foo"], DataRef)
    assert not isinstance(from_mars(ins)[0]["bar"], DataRef)
    assert not isinstance(from_mars(ins)[1], DataRef)

    # list.
    ins = [mdf, "foo"]
    assert isinstance(from_mars(ins)[0], DataRef)
    assert not isinstance(from_mars(ins)[1], DataRef)
    # nested list.
    ins = [(mdf, "foo"), "bar"]
    assert isinstance(from_mars(ins)[0][0], DataRef)
    assert not isinstance(from_mars(ins)[0][1], DataRef)
    assert not isinstance(from_mars(ins)[1], DataRef)
    ins = [[mdf, "foo"], "bar"]
    assert isinstance(from_mars(ins)[0][0], DataRef)
    assert not isinstance(from_mars(ins)[0][1], DataRef)
    assert not isinstance(from_mars(ins)[1], DataRef)
    ins = [{"foo": mdf, "bar": "baz"}, "bar"]
    assert isinstance(from_mars(ins)[0]["foo"], DataRef)
    assert not isinstance(from_mars(ins)[0]["bar"], DataRef)
    assert not isinstance(from_mars(ins)[1], DataRef)

    # dict.
    ins = {"foo": mdf, "bar": "baz"}
    assert isinstance(from_mars(ins)["foo"], DataRef)
    assert not isinstance(from_mars(ins)["bar"], DataRef)
    # nested dict.
    ins = {"foo": (mdf, "foo"), "bar": "baz"}
    assert isinstance(from_mars(ins)["foo"][0], DataRef)
    assert not isinstance(from_mars(ins)["foo"][1], DataRef)
    assert not isinstance(from_mars(ins)["bar"], DataRef)
    ins = {"foo": [mdf, "foo"], "bar": "baz"}
    assert isinstance(from_mars(ins)["foo"][0], DataRef)
    assert not isinstance(from_mars(ins)["foo"][1], DataRef)
    assert not isinstance(from_mars(ins)["bar"], DataRef)
    ins = {"foo": {"bar": mdf}, "bar": "baz"}
    assert isinstance(from_mars(ins)["foo"]["bar"], DataRef)
    assert not isinstance(from_mars(ins)["bar"], DataRef)


def test_on_nonexistent_attr(dummy_df):
    with pytest.raises(
        AttributeError, match="'dataframe' object has no attribute 'nonexistent_attr'"
    ):
        dummy_df.nonexistent_attr


def test_on_nonexistent_magic_method(dummy_df):
    with pytest.raises(
        AttributeError, match="'index' object has no attribute '__add__'"
    ):
        dummy_df.index + dummy_df.index


def test_setattr(dummy_df):
    mdf = mars_dataframe.DataFrame({"foo": (1, 2, 3), "bar": (4, 5, 6)})
    dummy_df._data = mdf.data
    assert dummy_df.data._mars_entity.data is mdf.data


def test_iter(setup, dummy_df, dummy_str_series, dummy_int_1d_array):
    expected_df = pd.DataFrame({"foo": (1, 2, 3), "bar": (4, 5, 6)}) + 1
    added = dummy_df + 1
    for r, expected in zip(added, expected_df):
        assert r == expected

    expected_series = pd.Series(["foo", "bar", "baz"]) + "1"
    added = dummy_str_series + "1"
    for r, expected in zip(added, expected_series):
        assert r == expected

    expected_series = pd.Series(["foo", "bar", "baz"]) + "1"
    added = dummy_str_series + "1"
    for r, expected in zip(added, expected_series):
        assert r == expected

    expected_index = pd.Index(["i0", "i1", "i2"])
    dummy_df.index = expected_index
    for r, expected in zip(dummy_df.index, expected_index):
        assert r == expected

    np_array = np.array([0, 1, 2]) + 1
    for r, expected in zip(dummy_int_1d_array + 1, np_array):
        assert r.to_numpy() == expected

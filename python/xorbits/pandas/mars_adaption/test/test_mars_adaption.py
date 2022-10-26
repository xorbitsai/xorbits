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

import pandas as pd

from ....core.mars_adaption import DataRefMarsImpl
from ..accssor import DatetimeAccessor, StringAccessor
from ..loc import DataFrameLoc


def test_dataframe_loc(dummy_df):
    assert isinstance(dummy_df.loc, DataFrameLoc)

    xdf = dummy_df.loc[[0], ["foo"]]
    assert isinstance(xdf, DataRefMarsImpl)

    rows = list(xdf.iterrows())
    assert 1 == len(rows)
    row = rows[0]
    assert 2 == len(row)
    assert 0 == row[0]
    assert pd.Series({"foo": 0}, name=0).equals(row[1])


def test_string_accessor(dummy_str_series):
    assert isinstance(dummy_str_series.str, StringAccessor)
    s = dummy_str_series.str.fullmatch("foo")

    assert isinstance(s, DataRefMarsImpl)
    for i, val in s.iteritems():
        assert val == (str(dummy_str_series[i]) == "foo")


def test_datetime_accessor(dummy_dt_series):
    assert isinstance(dummy_dt_series.dt, DatetimeAccessor)
    s = dummy_dt_series.dt.second

    assert isinstance(s, DataRefMarsImpl)
    for i, val in s.iteritems():
        assert val == i


def test_dataframe_getitem(dummy_df):
    foo = dummy_df["foo"]
    assert isinstance(foo, DataRefMarsImpl)

    idx = 0
    for i, val in foo.iteritems():
        assert idx == i
        assert idx == val
        idx += 1


def test_dataframe_setitem(dummy_df):
    dummy_df["baz"] = (0.0, 1.0, 2.0)
    baz = dummy_df.baz
    assert isinstance(baz, DataRefMarsImpl)

    idx = 0
    for i, val in baz.iteritems():
        assert idx == i
        assert val == float(idx)
        idx += 1


def test_dataframe_getattr(dummy_df):
    foo = dummy_df.foo
    assert isinstance(foo, DataRefMarsImpl)

    idx = 0
    for i, val in foo.iteritems():
        assert idx == i
        assert idx == val
        idx += 1


def test_dataframe_setattr(dummy_df):
    assert isinstance(dummy_df.columns, DataRefMarsImpl)
    assert ["foo", "bar"] == list(dummy_df.dtypes.index)

    dummy_df.columns = ["c1", "c2"]
    assert ["c1", "c2"] == list(dummy_df.dtypes.index)

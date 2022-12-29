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

from .... import pandas as xpd
from ....core import DataRef
from ....core.adapter import MarsGetAttrProxy
from ...indexing import DataFrameAt, DataFrameIat, DataFrameIloc, DataFrameLoc


def test_dataframe_categorical(setup):
    c = xpd.qcut(range(5), 4)
    assert isinstance(c, DataRef)
    assert isinstance(c.dtype, pd.CategoricalDtype)


def test_dataframe_ewm(setup, dummy_df):
    e = dummy_df.foo.ewm(com=0.5)
    assert isinstance(e, MarsGetAttrProxy)
    df = e.mean()
    assert isinstance(df, DataRef)


def test_dataframe_expanding(setup, dummy_df):
    e = dummy_df.foo.expanding(1)
    assert isinstance(e, MarsGetAttrProxy)
    df = e.sum()
    assert isinstance(df, DataRef)


def test_dataframe_rolling(setup, dummy_df):
    r = dummy_df.foo.rolling(2)
    assert isinstance(r, MarsGetAttrProxy)
    df = r.sum()
    assert isinstance(df, DataRef)


def test_dataframe_indexing(setup, dummy_df):
    assert isinstance(dummy_df.loc, DataFrameLoc)

    xdf = dummy_df.loc[[0], ["foo"]]
    assert isinstance(xdf, DataRef)

    rows = list(xdf.iterrows())
    assert 1 == len(rows)
    row = rows[0]
    assert 2 == len(row)
    assert 0 == row[0]
    assert pd.Series({"foo": 0}, name=0).equals(row[1])

    assert isinstance(dummy_df.iloc, DataFrameIloc)

    xdf = dummy_df.iloc[[0], [0]]
    assert isinstance(xdf, DataRef)

    rows = list(xdf.iterrows())
    assert 1 == len(rows)
    row = rows[0]
    assert 2 == len(row)
    assert 0 == row[0]
    assert pd.Series({"foo": 0}, name=0).equals(row[1])

    assert isinstance(dummy_df.at, DataFrameAt)

    xdf = dummy_df.at[0, "foo"]
    assert isinstance(xdf, DataRef)

    assert 0 == len(xdf)
    assert 0 == xdf.to_numpy()

    assert isinstance(dummy_df.iat, DataFrameIat)

    xdf = dummy_df.iat[0, 0]
    assert isinstance(xdf, DataRef)

    assert 0 == len(xdf)
    assert 0 == xdf.to_numpy()


def test_series_indexing(setup, dummy_int_series):
    assert isinstance(dummy_int_series.loc, DataFrameLoc)
    res = dummy_int_series.loc[0]
    assert isinstance(res, DataRef)

    assert 0 == len(res)
    assert 1 == res.to_numpy()

    assert isinstance(dummy_int_series.iloc, DataFrameIloc)
    res = dummy_int_series.iloc[0]
    assert isinstance(res, DataRef)

    assert 0 == len(res)
    assert 1 == res.to_numpy()

    assert isinstance(dummy_int_series.at, DataFrameAt)
    res = dummy_int_series.at[1]
    assert isinstance(res, DataRef)

    assert 0 == len(res)
    assert 2 == res.to_numpy()

    assert isinstance(dummy_int_series.iat, DataFrameIat)
    res = dummy_int_series.iat[1]
    assert isinstance(res, DataRef)

    assert 0 == len(res)
    assert 2 == res.to_numpy()


def test_string_accessor(setup, dummy_str_series):
    assert isinstance(dummy_str_series.str, MarsGetAttrProxy)
    s = dummy_str_series.str.fullmatch("foo")

    assert isinstance(s, DataRef)
    for i, val in s.iteritems():
        assert val == (str(dummy_str_series[i]) == "foo")


def test_datetime_accessor(setup, dummy_dt_series):
    assert isinstance(dummy_dt_series.dt, MarsGetAttrProxy)
    s = dummy_dt_series.dt.second

    assert isinstance(s, DataRef)
    for i, val in s.iteritems():
        assert val == i


def test_dataframe_getitem(setup, dummy_df):
    foo = dummy_df["foo"]
    assert isinstance(foo, DataRef)

    idx = 0
    for i, val in foo.iteritems():
        assert idx == i
        assert idx == val
        idx += 1


def test_dataframe_setitem(setup, dummy_df):
    dummy_df["baz"] = (0.0, 1.0, 2.0)
    baz = dummy_df.baz
    assert isinstance(baz, DataRef)

    idx = 0
    for i, val in baz.iteritems():
        assert idx == i
        assert val == float(idx)
        idx += 1


def test_dataframe_getattr(setup, dummy_df):
    foo = dummy_df.foo
    assert isinstance(foo, DataRef)

    idx = 0
    for i, val in foo.iteritems():
        assert idx == i
        assert idx == val
        idx += 1


def test_dataframe_setattr(setup, dummy_df):
    assert isinstance(dummy_df.columns, DataRef)
    assert ["foo", "bar"] == list(dummy_df.dtypes.index)

    dummy_df.columns = ["c1", "c2"]
    assert ["c1", "c2"] == list(dummy_df.dtypes.index)


def test_dataframe_items(setup, dummy_df):
    for label, content in dummy_df.items():
        assert isinstance(content, DataRef)

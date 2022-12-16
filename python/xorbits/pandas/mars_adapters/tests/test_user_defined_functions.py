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

import numpy as np
import pandas as pd

from .... import pandas as xpd
from ....core import DataRef
from ....pandas import DataFrame, Series


def test_dataframe_apply(setup):
    df = pd.DataFrame({"col1": [1, 2, 3, 4], "col2": list("abcd")})
    xdf = xpd.DataFrame(df, chunk_size=2)

    res = xdf.apply(lambda x: x, axis=1)
    assert isinstance(res, DataRef)
    assert isinstance(res, DataFrame)
    pd.testing.assert_frame_equal(res.to_pandas(), df)

    # infer failed
    def apply_func(series):
        if series[1] not in "abcd":
            # make it fail when inferring
            raise TypeError
        else:
            return 1

    res = xdf.apply(apply_func, axis=1)
    assert isinstance(res, DataRef)
    assert isinstance(res, Series)
    pd.testing.assert_series_equal(res.to_pandas(), pd.Series([1] * 4))
    assert DataFrame.apply.__doc__


def test_series_map(setup):
    series = pd.Series(range(10))
    xseries = xpd.Series(series)

    res = xseries.map(lambda x: x + 1.5)

    assert isinstance(res, DataRef)
    assert isinstance(res, Series)
    pd.testing.assert_series_equal(res.to_pandas(), series + 1.5)

    # infer failed
    res = xseries.map({5: 10})

    assert isinstance(res, DataRef)
    assert isinstance(res, Series)
    expected = series.map({5: 10})
    pd.testing.assert_series_equal(res.to_pandas(), expected)
    assert Series.map.__doc__


def test_dataframe_transform(setup):
    cols = [chr(ord("A") + i) for i in range(10)]
    df = pd.DataFrame(dict((c, [i**2 for i in range(20)]) for c in cols))
    xdf = xpd.DataFrame(df, chunk_size=2)

    def f(s):
        if s[2] > 0:
            return s
        else:
            return pd.Series([s[2]] * len(s))

    res = xdf.transform(f)

    assert isinstance(res, DataRef)
    assert isinstance(res, DataFrame)
    expected = df.transform(f)
    pd.testing.assert_frame_equal(res.to_pandas(), expected)


def test_dataframe_map_chunk(setup):
    df = pd.DataFrame(np.random.rand(10, 5), columns=[f"col{i}" for i in range(5)])
    xdf = xpd.DataFrame(df, chunk_size=(5, 3))

    # infer failed
    def f1(pdf):
        return pdf.iloc[2, :2]

    res = xdf.map_chunk(f1)
    assert isinstance(res, DataRef)
    assert isinstance(res, Series)
    pd.testing.assert_series_equal(
        res.to_pandas(), pd.concat([df.iloc[2, :2], df.iloc[7, :2]])
    )

    res = xdf.map_chunk(lambda x: x + 1)
    assert isinstance(res, DataRef)
    assert isinstance(res, DataFrame)
    pd.testing.assert_frame_equal(res.to_pandas(), df + 1)


def test_dataframe_cartesian_chunk(setup):
    rs = np.random.RandomState(0)
    raw1 = pd.DataFrame({"a": range(10), "b": rs.rand(10)})
    raw2 = pd.DataFrame(
        {"c": rs.randint(3, size=10), "d": rs.rand(10), "e": rs.rand(10)}
    )
    df1 = xpd.DataFrame(raw1, chunk_size=(5, 1))
    df2 = xpd.DataFrame(raw2, chunk_size=(5, 1))

    def f1(c1, c2):
        return c1.iloc[[2, 4], :]

    res = df1.cartesian_chunk(df2, f1)
    assert isinstance(res, DataRef)
    assert isinstance(res, DataFrame)
    pd.testing.assert_frame_equal(
        res.to_pandas(), raw1.iloc[[2, 4] * 2 + [7, 9] * 2, :]
    )


def test_groupby_apply(setup):
    raw = pd.DataFrame(
        {
            "a": [3, 4, 5, 3, 5, 4, 1, 2, 3],
            "b": [6, 3, 3, 5, 6, 5, 4, 4, 4],
            "c": list("aabaabbbb"),
        }
    )
    df = xpd.DataFrame(raw, chunk_size=5)

    # infer failed
    def f1(df):
        return df.a.iloc[2]

    res = df.groupby("c").apply(f1)
    assert isinstance(res, DataRef)
    assert isinstance(res, Series)
    pd.testing.assert_series_equal(
        res.to_pandas().sort_index(), raw.groupby("c").apply(f1).sort_index()
    )


def test_groupby_transform(setup):
    df = pd.DataFrame(
        {
            "a": [3, 4, 5, 3, 5, 4, 1, 2, 3],
            "b": [1, 3, 4, 5, 6, 5, 4, 4, 4],
            "c": list("aabaabbba"),
        }
    )

    def f(in_df):
        if in_df.iloc[2] > 0:
            return in_df
        else:
            return in_df + in_df.max()

    pdf = xpd.DataFrame(df, chunk_size=5)
    res = pdf.groupby("c").transform(f)
    assert isinstance(res, DataRef)
    assert isinstance(res, DataFrame)
    pd.testing.assert_frame_equal(
        res.to_pandas().sort_index(),
        df.groupby("c").transform(f).sort_index(),
    )

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
import os
import tempfile
from itertools import product

import numpy as np
import pandas as pd
import pytest

from ..... import dataframe as md


def test_rolling_agg_execution(setup):
    rs = np.random.RandomState(0)
    raw = pd.DataFrame(
        {
            "a": rs.randint(100, size=(10,)),
            "b": rs.rand(10),
            "c": rs.randint(100, size=(10,)),
            "d": ["c" * i for i in rs.randint(4, size=10)],
        }
    )
    raw.iloc[1, ::4] = np.nan
    s = raw.iloc[:, 1]

    dfs = [
        md.DataFrame(raw, chunk_size=10),  # 1 chunk
        md.DataFrame(raw, chunk_size=3),  # multiple chunks on each axis
    ]
    funcs = ["min", ["max", "mean"], {"c": ["std"], "b": ["count", "min"]}]

    df2 = dfs[0].rolling(3).agg(funcs[2])

    # test 1 chunk
    result = df2.execute().fetch()
    expected = raw.rolling(3).agg(funcs[2])
    pd.testing.assert_frame_equal(result, expected)

    for window in [2, 5]:
        for center in [True, False]:
            for func in funcs:
                df2 = dfs[1].rolling(window, center=center).agg(func)

                result = df2.execute().fetch()
                expected = raw.rolling(window, center=center).agg(func)
                pd.testing.assert_frame_equal(result, expected)

    # test min_periods and win_type
    df2 = dfs[1].rolling(3, min_periods=1, win_type="triang").agg("sum")

    result = df2.execute().fetch()
    expected = raw.rolling(3, min_periods=1, win_type="triang").agg("sum")
    pd.testing.assert_frame_equal(result, expected)

    # test rolling getitem, series
    df2 = dfs[1].rolling(3)["b"].agg("sum")

    result = df2.execute().fetch()
    expected = raw.rolling(3)["b"].agg("sum")
    pd.testing.assert_series_equal(result, expected)

    # test rolling getitem, dataframe
    df2 = dfs[1].rolling(3)["c", "b"].agg("sum")

    result = df2.execute().fetch()
    expected = raw.rolling(3)["c", "b"].agg("sum")
    pd.testing.assert_frame_equal(result, expected)

    # test axis=1
    df2 = dfs[1].rolling(3, axis=1).agg("sum")

    result = df2.execute(
        extra_config=dict(check_all=False, check_nsplits=False)
    ).fetch()
    expected = raw.rolling(3, axis=1).agg("sum")
    pd.testing.assert_frame_equal(result, expected)

    # test window which is offset
    raw2 = raw.copy()
    raw2.reset_index(inplace=True, drop=True)
    raw2.index = pd.date_range("2020-2-25", periods=10)

    df = md.DataFrame(raw2, chunk_size=3)
    for func in funcs:
        df2 = df.rolling("2d").agg(func)

        result = df2.execute().fetch()
        expected = raw2.rolling("2d").agg(func)
        pd.testing.assert_frame_equal(result, expected)

    series = [md.Series(s, chunk_size=10), md.Series(s, chunk_size=4)]

    funcs = ["min", ["max", "mean"], {"c": "std", "b": "count"}]

    for series in series:
        for window in [2, 3, 5]:
            for center in [True, False]:
                for func in funcs:
                    series2 = series.rolling(window, center=center).agg(func)

                    result = series2.execute().fetch()
                    expected = s.rolling(window, center=center).agg(func)
                    if isinstance(expected, pd.Series):
                        pd.testing.assert_series_equal(result, expected)
                    else:
                        pd.testing.assert_frame_equal(result, expected)

    df = md.DataFrame(raw, chunk_size=3)
    df = df[df.a > 0.5]
    r = df.rolling(3).agg("max")

    result = r.execute().fetch()
    expected = raw[raw.a > 0.5].rolling(3).agg("max")
    pd.testing.assert_frame_equal(result, expected)

    series = md.Series(s, chunk_size=3)
    series = series[series > 0.5]
    r = series.rolling(3).agg("max")

    result = r.execute().fetch()
    expected = s[s > 0.5].rolling(3).agg("max")
    pd.testing.assert_series_equal(result, expected)

    # test agg functions
    df = md.DataFrame(raw, chunk_size=3)
    for func in ["count", "sum", "mean", "median", "min", "max", "skew", "kurt"]:
        r = getattr(df.rolling(4), func)()

        result = r.execute().fetch()
        expected = getattr(raw.rolling(4), func)()
        pd.testing.assert_frame_equal(result, expected)
    for func in ["std", "var"]:
        r = getattr(df.rolling(4), func)(ddof=0)

        result = r.execute().fetch()
        expected = getattr(raw.rolling(4), func)(ddof=0)
        pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "window,min_periods,center,on,axis,closed,pairwise,",
    list(
        product(
            [3, "3D"],
            [1, 3],
            [False, True],
            [None, "foo"],
            [0, 1],
            ["right", "left", "both", "neither"],
            [True, False],
        )
    ),
)
def test_datetime_indexed_df_rolling_corr_execution(
    window, min_periods, center, on, axis, closed, pairwise, setup
):
    # skipped for now due to https://github.com/pandas-dev/pandas/issues/52299
    if on is not None:
        return

    # skip invalid combinations.
    if (
        on is not None
        and isinstance(window, str)
        or center
        and isinstance(window, str)
        or axis == 1
        and isinstance(window, str)
    ):
        return

    with tempfile.TemporaryDirectory("mars_test_df_rolling_corr_execution") as tempdir:
        path = os.path.join(tempdir, "datetime_indexed_df.csv")
        index = pd.date_range(start="2023-01-01", end="2023-01-05", freq="D")
        pd.DataFrame(
            {
                "foo": list(range(0, 5)),
                "bar": list(range(1, 6)),
                "baz": list(range(2, 7)),
            },
            index=index,
        ).to_csv(path)

        mdf = md.read_csv(path, index_col=0, parse_dates=True, chunk_bytes=50)
        mr = mdf.rolling(
            window=window,
            min_periods=min_periods,
            center=center,
            on=on,
            axis=axis,
            closed=closed,
        ).corr(pairwise=pairwise)
        mr.execute()

        pdf = pd.read_csv(path, index_col=0, parse_dates=True)
        pr = pdf.rolling(
            window=window,
            min_periods=min_periods,
            center=center,
            on=on,
            axis=axis,
            closed=closed,
        ).corr(pairwise=pairwise)
        pd.testing.assert_frame_equal(pr, mr.fetch())


@pytest.mark.parametrize(
    "window,min_periods,center,on,axis,closed,pairwise,",
    list(
        product(
            [3],
            [1, 3],
            [False, True],
            [None, "foo"],
            [0, 1],
            ["right", "left", "both", "neither"],
            [True, False],
        )
    ),
)
def test_range_indexed_df_rolling_corr_execution(
    window, min_periods, center, on, axis, closed, pairwise, setup
):
    # skipped for now due to https://github.com/pandas-dev/pandas/issues/52299
    if on is not None:
        return

    with tempfile.TemporaryDirectory("mars_test_df_rolling_corr_execution") as tempdir:
        path = os.path.join(tempdir, "range_indexed_df.csv")
        pd.DataFrame(
            {
                "foo": list(range(0, 5)),
                "bar": list(range(1, 6)),
                "baz": list(range(2, 7)),
            },
            index=list(range(0, 5)),
        ).to_csv(path)

        mdf = md.read_csv(path, index_col=0, chunk_bytes=50)
        mr = mdf.rolling(
            window=window,
            min_periods=min_periods,
            center=center,
            on=on,
            axis=axis,
            closed=closed,
        ).corr(pairwise=pairwise)
        mr.execute()

        pdf = pd.read_csv(path, index_col=0)
        pr = pdf.rolling(
            window=window,
            min_periods=min_periods,
            center=center,
            on=on,
            axis=axis,
            closed=closed,
        ).corr(pairwise=pairwise)
        pd.testing.assert_frame_equal(pr, mr.fetch())

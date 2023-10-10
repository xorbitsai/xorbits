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
from collections import OrderedDict
from itertools import product

import numpy as np
import pandas as pd
import pytest

try:
    import pyarrow as pa
except ImportError:  # pragma: no cover
    pa = None

from .... import dataframe as md
from ....config import option_context
from ....core.operand import OperandStage
from ....tests.core import assert_groupby_equal, require_cudf, support_cuda
from ....utils import lazy_import, pd_release_version
from ...core import DATAFRAME_OR_SERIES_TYPE
from ...utils import is_pandas_2
from ..aggregation import DataFrameGroupByAgg
from ..rolling import _PAIRWISE_AGG

cudf = lazy_import("cudf")
pytestmark = pytest.mark.pd_compat

_agg_size_as_frame = pd_release_version[:2] > (1, 0)


class MockReduction1(md.CustomReduction):
    def agg(self, v1):
        return v1.sum()


def test_groupby(setup):
    rs = np.random.RandomState(0)
    data_size = 100
    data_dict = {
        "a": rs.randint(0, 10, size=(data_size,)),
        "b": rs.randint(0, 10, size=(data_size,)),
        "c": rs.choice(list("abcd"), size=(data_size,)),
    }

    # test groupby with DataFrames and RangeIndex
    df1 = pd.DataFrame(data_dict)
    mdf1 = md.DataFrame(df1, chunk_size=13)
    grouped = mdf1.groupby("b")
    assert_groupby_equal(grouped.execute().fetch(), df1.groupby("b"))

    # test groupby with string index with duplications
    df2 = pd.DataFrame(data_dict, index=["i" + str(i % 3) for i in range(data_size)])
    mdf2 = md.DataFrame(df2, chunk_size=13)
    grouped = mdf2.groupby("b")
    assert_groupby_equal(grouped.execute().fetch(), df2.groupby("b"))

    # test groupby with DataFrames by series
    grouped = mdf1.groupby(mdf1["b"])
    assert_groupby_equal(grouped.execute().fetch(), df1.groupby(df1["b"]))

    # test groupby with DataFrames by multiple series
    grouped = mdf1.groupby(by=[mdf1["b"], mdf1["c"]])
    assert_groupby_equal(
        grouped.execute().fetch(), df1.groupby(by=[df1["b"], df1["c"]])
    )

    # test groupby with DataFrames with MultiIndex
    df3 = pd.DataFrame(
        data_dict,
        index=pd.MultiIndex.from_tuples(
            [(i % 3, "i" + str(i)) for i in range(data_size)]
        ),
    )
    mdf = md.DataFrame(df3, chunk_size=13)
    grouped = mdf.groupby(level=0)
    assert_groupby_equal(grouped.execute().fetch(), df3.groupby(level=0))

    # test groupby with DataFrames by integer columns
    df4 = pd.DataFrame(list(data_dict.values())).T
    mdf = md.DataFrame(df4, chunk_size=13)
    grouped = mdf.groupby(0)
    assert_groupby_equal(grouped.execute().fetch(), df4.groupby(0))

    series1 = pd.Series(data_dict["a"])
    ms1 = md.Series(series1, chunk_size=13)
    grouped = ms1.groupby(lambda x: x % 3)
    assert_groupby_equal(grouped.execute().fetch(), series1.groupby(lambda x: x % 3))

    # test groupby series
    grouped = ms1.groupby(ms1)
    assert_groupby_equal(grouped.execute().fetch(), series1.groupby(series1))

    series2 = pd.Series(data_dict["a"], index=["i" + str(i) for i in range(data_size)])
    ms2 = md.Series(series2, chunk_size=13)
    grouped = ms2.groupby(lambda x: int(x[1:]) % 3)
    assert_groupby_equal(
        grouped.execute().fetch(), series2.groupby(lambda x: int(x[1:]) % 3)
    )


def test_groupby_getitem(setup):
    rs = np.random.RandomState(0)
    data_size = 100
    raw = pd.DataFrame(
        {
            "a": rs.randint(0, 10, size=(data_size,)),
            "b": rs.randint(0, 10, size=(data_size,)),
            "c": rs.choice(list("abcd"), size=(data_size,)),
        },
        index=pd.MultiIndex.from_tuples(
            [(i % 3, "i" + str(i)) for i in range(data_size)]
        ),
    )
    mdf = md.DataFrame(raw, chunk_size=13)

    r = mdf.groupby(level=0)[["a", "b"]]
    assert_groupby_equal(
        r.execute().fetch(), raw.groupby(level=0)[["a", "b"]], with_selection=True
    )

    for method in ("tree", "shuffle"):
        r = mdf.groupby(level=0)[["a", "b"]].sum(method=method)
        pd.testing.assert_frame_equal(
            r.execute().fetch().sort_index(),
            raw.groupby(level=0)[["a", "b"]].sum().sort_index(),
        )

    r = mdf.groupby(level=0)[["a", "b"]].apply(lambda x: x + 1)
    pd.testing.assert_frame_equal(
        r.execute().fetch().sort_index(),
        raw.groupby(level=0)[["a", "b"]].apply(lambda x: x + 1).sort_index(),
    )

    r = mdf.groupby("b")[["a", "b"]]
    assert_groupby_equal(
        r.execute().fetch(), raw.groupby("b")[["a", "b"]], with_selection=True
    )

    r = mdf.groupby("b")[["a", "c"]]
    assert_groupby_equal(
        r.execute().fetch(), raw.groupby("b")[["a", "c"]], with_selection=True
    )

    for method in ("tree", "shuffle"):
        r = mdf.groupby("b")[["a", "b"]].sum(method=method)
        pd.testing.assert_frame_equal(
            r.execute().fetch().sort_index(),
            raw.groupby("b")[["a", "b"]].sum().sort_index(),
        )

        r = mdf.groupby("b")[["a", "b"]].agg(["sum", "count"], method=method)
        pd.testing.assert_frame_equal(
            r.execute().fetch().sort_index(),
            raw.groupby("b")[["a", "b"]].agg(["sum", "count"]).sort_index(),
        )

        r = mdf.groupby("b")[["a", "c"]].agg(["sum", "count"], method=method)
        pd.testing.assert_frame_equal(
            r.execute().fetch().sort_index(),
            raw.groupby("b")[["a", "c"]].agg(["sum", "count"]).sort_index(),
        )

    r = mdf.groupby("b")[["a", "b"]].apply(lambda x: x + 1)
    pd.testing.assert_frame_equal(
        r.execute().fetch().sort_index(),
        raw.groupby("b")[["a", "b"]].apply(lambda x: x + 1).sort_index(),
        check_names=False,
    )

    r = mdf.groupby("b")[["a", "b"]].transform(lambda x: x + 1)
    pd.testing.assert_frame_equal(
        r.execute().fetch().sort_index(),
        raw.groupby("b")[["a", "b"]].transform(lambda x: x + 1).sort_index(),
    )

    r = mdf.groupby("b")[["a", "b"]].cumsum()
    pd.testing.assert_frame_equal(
        r.execute().fetch().sort_index(),
        raw.groupby("b")[["a", "b"]].cumsum().sort_index(),
    )

    r = mdf.groupby("b").a
    assert_groupby_equal(r.execute().fetch(), raw.groupby("b").a, with_selection=True)

    for method in ("shuffle", "tree"):
        r = mdf.groupby("b").a.sum(method=method)
        pd.testing.assert_series_equal(
            r.execute().fetch().sort_index(), raw.groupby("b").a.sum().sort_index()
        )

        r = mdf.groupby("b").a.agg(["sum", "mean", "var"], method=method)
        pd.testing.assert_frame_equal(
            r.execute().fetch().sort_index(),
            raw.groupby("b").a.agg(["sum", "mean", "var"]).sort_index(),
        )

        r = mdf.groupby("b", as_index=False).a.sum(method=method)
        pd.testing.assert_frame_equal(
            r.execute()
            .fetch()
            .sort_values("b", ignore_index=True)
            .reset_index(drop=True),
            raw.groupby("b", as_index=False)
            .a.sum()
            .sort_values("b", ignore_index=True),
        )

        r = mdf.groupby("b", as_index=False).b.count(method=method)
        result = r.execute().fetch().sort_values("b", ignore_index=True)
        try:
            expected = (
                raw.groupby("b", as_index=False)
                .b.count()
                .sort_values("b", ignore_index=True)
            )
        except ValueError:
            expected = raw.groupby("b").b.count().to_frame()
            expected.index.names = [None] * expected.index.nlevels
            expected = expected.sort_values("b", ignore_index=True)
        pd.testing.assert_frame_equal(result, expected)

        r = mdf.groupby("b", as_index=False).b.agg({"cnt": "count"}, method=method)
        result = (
            r.execute()
            .fetch()
            .sort_values("b", ignore_index=True)
            .reset_index(drop=True)
        )
        try:
            expected = (
                raw.groupby("b", as_index=False)
                .b.agg({"cnt": "count"})
                .sort_values("b", ignore_index=True)
            )
        except ValueError:
            expected = raw.groupby("b").b.agg({"cnt": "count"}).to_frame()
            expected.index.names = [None] * expected.index.nlevels
            expected = expected.sort_values("b", ignore_index=True)
        pd.testing.assert_frame_equal(result, expected)

    r = mdf.groupby("b").a.apply(lambda x: x + 1)
    pd.testing.assert_series_equal(
        r.execute().fetch().sort_index(),
        raw.groupby("b").a.apply(lambda x: x + 1).sort_index(),
        check_names=False,
    )

    r = mdf.groupby("b").a.transform(lambda x: x + 1)
    pd.testing.assert_series_equal(
        r.execute().fetch().sort_index(),
        raw.groupby("b").a.transform(lambda x: x + 1).sort_index(),
    )

    r = mdf.groupby("b").a.cumsum()
    pd.testing.assert_series_equal(
        r.execute().fetch().sort_index(), raw.groupby("b").a.cumsum().sort_index()
    )

    # special test for selection key == 0
    raw = pd.DataFrame(rs.rand(data_size, 10))
    raw[0] = 0
    mdf = md.DataFrame(raw, chunk_size=13)
    r = mdf.groupby(0, as_index=False)[0].agg({"cnt": "count"}, method="tree")
    pd.testing.assert_frame_equal(
        r.execute().fetch().sort_index(),
        raw.groupby(0, as_index=False)[0].agg({"cnt": "count"}),
    )

    # test groupby getitem then agg(#GH 2640)
    rs = np.random.RandomState(0)
    raw = pd.DataFrame(
        {
            "c1": rs.randint(0, 10, size=(100,)).astype(np.int64),
            "c2": rs.choice(["a", "b", "c"], (100,)),
            "c3": rs.rand(100),
            "c4": rs.rand(100),
        }
    )
    mdf = md.DataFrame(raw, chunk_size=20)

    r = mdf.groupby(["c2"])[["c1", "c3"]].agg({"c1": "max", "c3": "min"}, method="tree")
    pd.testing.assert_frame_equal(
        r.execute().fetch(),
        raw.groupby(["c2"])[["c1", "c3"]].agg({"c1": "max", "c3": "min"}),
    )

    mdf = md.DataFrame(raw.copy(), chunk_size=30)
    r = mdf.groupby(["c2"])[["c1", "c4"]].agg(
        {"c1": "max", "c4": "mean"}, method="shuffle"
    )
    pd.testing.assert_frame_equal(
        r.execute().fetch().sort_index(),
        raw.groupby(["c2"])[["c1", "c4"]].agg({"c1": "max", "c4": "mean"}),
    )

    # test anonymous function lists
    agg_funs = [lambda x: (x + 1).sum()]
    r = mdf.groupby(["c2"])["c1"].agg(agg_funs)
    pd.testing.assert_frame_equal(
        r.execute().fetch(), raw.groupby(["c2"])["c1"].agg(agg_funs)
    )

    # test group by multiple cols
    r = mdf.groupby(["c1", "c2"], as_index=False)["c3"].sum()
    expected = raw.groupby(["c1", "c2"], as_index=False)["c3"].sum()
    pd.testing.assert_frame_equal(
        r.execute().fetch().sort_values(["c1", "c2"]).reset_index(drop=True),
        expected.sort_values(["c1", "c2"]).reset_index(drop=True),
    )

    r = mdf.groupby(["c1", "c2"], as_index=False)["c3"].agg(["sum"])
    expected = raw.groupby(["c1", "c2"], as_index=False)["c3"].agg(["sum"])
    pd.testing.assert_frame_equal(
        r.execute().fetch().sort_values(["c1", "c2"]),
        expected.sort_values(["c1", "c2"]),
    )


def test_dataframe_groupby_agg(setup):
    agg_funs = [
        "std",
        "mean",
        "var",
        "max",
        "count",
        "size",
        "all",
        "any",
        "skew",
        "kurt",
        "sem",
        "nunique",
    ]

    rs = np.random.RandomState(0)
    raw = pd.DataFrame(
        {
            "c1": np.arange(100).astype(np.int64),
            "c2": rs.choice(["a", "b", "c"], (100,)),
            "c3": rs.rand(100),
        }
    )
    mdf = md.DataFrame(raw, chunk_size=13)

    for method in ["tree", "shuffle"]:
        for sort in [True, False]:
            r = mdf.groupby("c2").agg("size", method=method)
            pd.testing.assert_series_equal(
                r.execute().fetch().sort_index(),
                raw.groupby("c2").agg("size").sort_index(),
            )

            for agg_fun in agg_funs:
                if agg_fun == "size":
                    continue
                r = mdf.groupby("c2", sort=sort).agg(agg_fun, method=method)
                pd.testing.assert_frame_equal(
                    r.execute().fetch().sort_index(),
                    raw.groupby("c2").agg(agg_fun).sort_index(),
                )

            r = mdf.groupby("c2", sort=sort).agg(agg_funs, method=method)
            pd.testing.assert_frame_equal(
                r.execute().fetch().sort_index(),
                raw.groupby("c2").agg(agg_funs).sort_index(),
            )

            agg = OrderedDict([("c1", ["min", "mean"]), ("c3", "std")])
            r = mdf.groupby("c2", sort=sort).agg(agg, method=method)
            pd.testing.assert_frame_equal(
                r.execute().fetch().sort_index(),
                raw.groupby("c2").agg(agg).sort_index(),
            )

            agg = OrderedDict([("c1", "min"), ("c3", "sum")])
            r = mdf.groupby("c2", sort=sort).agg(agg, method=method)
            pd.testing.assert_frame_equal(
                r.execute().fetch().sort_index(),
                raw.groupby("c2").agg(agg).sort_index(),
            )

            r = mdf.groupby("c2", sort=sort).agg(
                {"c1": "min", "c3": "min"}, method=method
            )
            pd.testing.assert_frame_equal(
                r.execute().fetch().sort_index(),
                raw.groupby("c2").agg({"c1": "min", "c3": "min"}).sort_index(),
            )

            r = mdf.groupby("c2", sort=sort).agg({"c1": "min"}, method=method)
            pd.testing.assert_frame_equal(
                r.execute().fetch().sort_index(),
                raw.groupby("c2").agg({"c1": "min"}).sort_index(),
            )

            # test groupby series
            r = mdf.groupby(mdf["c2"], sort=sort).sum(method=method, numeric_only=True)
            pd.testing.assert_frame_equal(
                r.execute().fetch().sort_index(),
                raw.groupby(raw["c2"]).sum(numeric_only=True).sort_index(),
            )

    r = mdf.groupby("c2").size(method="tree")
    pd.testing.assert_series_equal(r.execute().fetch(), raw.groupby("c2").size())

    # test inserted kurt method
    r = mdf.groupby("c2").kurtosis(method="tree")
    pd.testing.assert_frame_equal(r.execute().fetch(), raw.groupby("c2").kurtosis())

    for agg_fun in agg_funs:
        if agg_fun == "size" or callable(agg_fun):
            continue
        r = getattr(mdf.groupby("c2"), agg_fun)(method="tree")
        pd.testing.assert_frame_equal(
            r.execute().fetch(), getattr(raw.groupby("c2"), agg_fun)()
        )

    # test as_index=False
    for method in ["tree", "shuffle"]:
        r = mdf.groupby("c2", as_index=False).agg("size", method=method)
        if _agg_size_as_frame:
            result = r.execute().fetch().sort_values("c2", ignore_index=True)
            expected = (
                raw.groupby("c2", as_index=False)
                .agg("size")
                .sort_values("c2", ignore_index=True)
            )
            pd.testing.assert_frame_equal(result, expected)
        else:
            result = r.execute().fetch().sort_index()
            expected = raw.groupby("c2", as_index=False).agg("size").sort_index()
            pd.testing.assert_series_equal(result, expected)

        r = mdf.groupby("c2", as_index=False).agg("mean", method=method)
        pd.testing.assert_frame_equal(
            r.execute().fetch().sort_values("c2", ignore_index=True),
            raw.groupby("c2", as_index=False)
            .agg("mean")
            .sort_values("c2", ignore_index=True),
        )
        assert r.op.groupby_params["as_index"] is False

        r = mdf.groupby(["c1", "c2"], as_index=False).agg("mean", method=method)
        pd.testing.assert_frame_equal(
            r.execute()
            .fetch()
            .sort_values(["c1", "c2"], ignore_index=True)
            .reset_index(drop=True),
            raw.groupby(["c1", "c2"], as_index=False)
            .agg("mean")
            .sort_values(["c1", "c2"], ignore_index=True)
            .reset_index(drop=True),
        )
        assert r.op.groupby_params["as_index"] is False

    # test as_index=False takes no effect
    r = mdf.groupby(["c1", "c2"], as_index=False).agg(["mean", "count"])
    pd.testing.assert_frame_equal(
        r.execute().fetch(),
        raw.groupby(["c1", "c2"], as_index=False).agg(["mean", "count"]),
    )

    r = mdf.groupby("c2").agg(["cumsum", "cumcount"])
    pd.testing.assert_frame_equal(
        r.execute().fetch().sort_index(),
        raw.groupby("c2").agg(["cumsum", "cumcount"]).sort_index(),
    )

    r = mdf.groupby("c2").agg(
        sum_c1=md.NamedAgg("c1", "sum"),
        min_c1=md.NamedAgg("c1", "min"),
        mean_c3=md.NamedAgg("c3", "mean"),
        method="tree",
    )
    pd.testing.assert_frame_equal(
        r.execute().fetch(),
        raw.groupby("c2").agg(
            sum_c1=md.NamedAgg("c1", "sum"),
            min_c1=md.NamedAgg("c1", "min"),
            mean_c3=md.NamedAgg("c3", "mean"),
        ),
    )


def test_dataframe_groupby_agg_sort(setup):
    agg_funs = [
        "std",
        "mean",
        "var",
        "max",
        "count",
        "size",
        "all",
        "any",
        "skew",
        "kurt",
        "sem",
        "nunique",
    ]

    rs = np.random.RandomState(0)
    raw = pd.DataFrame(
        {
            "c1": np.arange(100).astype(np.int64),
            "c2": rs.choice(["a", "b", "c"], (100,)),
            "c3": rs.rand(100),
        }
    )
    mdf = md.DataFrame(raw, chunk_size=13)

    for method in ["tree", "shuffle"]:
        r = mdf.groupby("c2").agg("size", method=method)
        pd.testing.assert_series_equal(
            r.execute().fetch(), raw.groupby("c2").agg("size")
        )

        for agg_fun in agg_funs:
            if agg_fun == "size":
                continue
            r = mdf.groupby("c2").agg(agg_fun, method=method)
            pd.testing.assert_frame_equal(
                r.execute().fetch(),
                raw.groupby("c2").agg(agg_fun),
            )

        r = mdf.groupby("c2").agg(agg_funs, method=method)
        pd.testing.assert_frame_equal(
            r.execute().fetch(),
            raw.groupby("c2").agg(agg_funs),
        )

        agg = OrderedDict([("c1", ["min", "mean"]), ("c3", "std")])
        r = mdf.groupby("c2").agg(agg, method=method)
        pd.testing.assert_frame_equal(r.execute().fetch(), raw.groupby("c2").agg(agg))

        agg = OrderedDict([("c1", "min"), ("c3", "sum")])
        r = mdf.groupby("c2").agg(agg, method=method)
        pd.testing.assert_frame_equal(r.execute().fetch(), raw.groupby("c2").agg(agg))

        r = mdf.groupby("c2").agg({"c1": "min", "c3": "min"}, method=method)
        pd.testing.assert_frame_equal(
            r.execute().fetch(),
            raw.groupby("c2").agg({"c1": "min", "c3": "min"}),
        )

        r = mdf.groupby("c2").agg({"c1": "min"}, method=method)
        pd.testing.assert_frame_equal(
            r.execute().fetch(),
            raw.groupby("c2").agg({"c1": "min"}),
        )

        # test groupby series
        r = mdf.groupby(mdf["c2"]).sum(method=method, numeric_only=True)
        pd.testing.assert_frame_equal(
            r.execute().fetch(), raw.groupby(raw["c2"]).sum(numeric_only=True)
        )

    r = mdf.groupby("c2").size(method="tree")
    pd.testing.assert_series_equal(r.execute().fetch(), raw.groupby("c2").size())

    # test inserted kurt method
    r = mdf.groupby("c2").kurtosis(method="tree")
    pd.testing.assert_frame_equal(r.execute().fetch(), raw.groupby("c2").kurtosis())

    for agg_fun in agg_funs:
        if agg_fun == "size" or callable(agg_fun):
            continue
        r = getattr(mdf.groupby("c2"), agg_fun)(method="tree")
        pd.testing.assert_frame_equal(
            r.execute().fetch(), getattr(raw.groupby("c2"), agg_fun)()
        )

    # test as_index=False takes no effect
    r = mdf.groupby(["c1", "c2"], as_index=False).agg(["mean", "count"])
    pd.testing.assert_frame_equal(
        r.execute().fetch(),
        raw.groupby(["c1", "c2"], as_index=False).agg(["mean", "count"]),
    )


def test_series_groupby_agg(setup):
    rs = np.random.RandomState(0)
    series1 = pd.Series(rs.rand(10))
    ms1 = md.Series(series1, chunk_size=3)

    agg_funs = [
        "std",
        "mean",
        "var",
        "max",
        "count",
        "size",
        "all",
        "any",
        "skew",
        "kurt",
        "sem",
    ]

    for method in ["tree", "shuffle"]:
        for agg_fun in agg_funs:
            r = ms1.groupby(lambda x: x % 2).agg(agg_fun, method=method)
            pd.testing.assert_series_equal(
                r.execute().fetch(), series1.groupby(lambda x: x % 2).agg(agg_fun)
            )

        r = ms1.groupby(lambda x: x % 2).agg(agg_funs, method=method)
        pd.testing.assert_frame_equal(
            r.execute().fetch(), series1.groupby(lambda x: x % 2).agg(agg_funs)
        )

        # test groupby series
        r = ms1.groupby(ms1).sum(method=method)
        pd.testing.assert_series_equal(
            r.execute().fetch().sort_index(),
            series1.groupby(series1).sum().sort_index(),
        )

        r = ms1.groupby(ms1).sum(method=method)
        pd.testing.assert_series_equal(
            r.execute().fetch().sort_index(),
            series1.groupby(series1).sum().sort_index(),
        )

    # test inserted kurt method
    r = ms1.groupby(ms1).kurtosis(method="tree")
    pd.testing.assert_series_equal(
        r.execute().fetch(), series1.groupby(series1).kurtosis()
    )

    for agg_fun in agg_funs:
        r = getattr(ms1.groupby(lambda x: x % 2), agg_fun)(method="tree")
        pd.testing.assert_series_equal(
            r.execute().fetch(), getattr(series1.groupby(lambda x: x % 2), agg_fun)()
        )

    r = ms1.groupby(lambda x: x % 2).agg(["cumsum", "cumcount"], method="tree")
    pd.testing.assert_frame_equal(
        r.execute().fetch().sort_index(),
        series1.groupby(lambda x: x % 2).agg(["cumsum", "cumcount"]).sort_index(),
    )

    r = ms1.groupby(lambda x: x % 2).agg(col_var="var", col_skew="skew", method="tree")
    pd.testing.assert_frame_equal(
        r.execute().fetch(),
        series1.groupby(lambda x: x % 2).agg(col_var="var", col_skew="skew"),
    )


def test_groupby_agg_auto_method(setup):
    rs = np.random.RandomState(0)
    raw = pd.DataFrame(
        {
            "c1": rs.randint(20, size=100),
            "c2": rs.choice(["a", "b", "c"], (100,)),
            "c3": rs.rand(100),
        }
    )
    mdf = md.DataFrame(raw, chunk_size=20)

    def _disallow_reduce(ctx, op):
        assert op.stage != OperandStage.reduce
        op.execute(ctx, op)

    r = mdf.groupby("c2").agg("sum")
    operand_executors = {DataFrameGroupByAgg: _disallow_reduce}
    result = r.execute(
        extra_config={"operand_executors": operand_executors, "check_all": False}
    ).fetch()
    pd.testing.assert_frame_equal(result.sort_index(), raw.groupby("c2").agg("sum"))

    r = mdf.groupby("c3").agg("min")
    operand_executors = {DataFrameGroupByAgg: _disallow_reduce}
    result = r.execute(
        extra_config={"operand_executors": operand_executors, "check_all": False}
    ).fetch()
    pd.testing.assert_frame_equal(result.sort_index(), raw.groupby("c3").agg("min"))

    def _disallow_combine_and_agg(ctx, op):
        assert op.stage != OperandStage.combine
        op.execute(ctx, op)

    with option_context({"chunk_store_limit": 1}):
        raw2 = pd.DataFrame(
            {
                "c1": rs.randint(20, size=100),
                "c2": rs.rand(100),
                "c3": rs.rand(100),
            }
        )
        mdf = md.DataFrame(raw2, chunk_size=20)
        r = mdf.groupby("c3").agg("min")
        operand_executors = {DataFrameGroupByAgg: _disallow_combine_and_agg}
        result = r.execute(
            extra_config={"operand_executors": operand_executors, "check_all": False}
        ).fetch()
        pd.testing.assert_frame_equal(
            result.sort_index(), raw2.groupby("c3").agg("min")
        )

    rs = np.random.RandomState(0)
    raw = pd.DataFrame(
        {
            "c1": list(range(4)) * 12,
            "c2": rs.choice(["a", "b", "c"], (48,)),
            "c3": rs.rand(48),
        }
    )

    mdf = md.DataFrame(raw, chunk_size=8)
    r = mdf.groupby("c1").agg("sum")
    operand_executors = {DataFrameGroupByAgg: _disallow_reduce}
    result = r.execute(
        extra_config={"operand_executors": operand_executors, "check_all": False}
    ).fetch()
    pd.testing.assert_frame_equal(result.sort_index(), raw.groupby("c1").agg("sum"))


@pytest.mark.skip_ray_dag  # _fetch_infos() is not supported by ray backend.
def test_distributed_groupby_agg(setup_cluster):
    rs = np.random.RandomState(0)
    raw = pd.DataFrame(rs.rand(50000, 10))
    df = md.DataFrame(raw, chunk_size=raw.shape[0] // 2)
    with option_context({"chunk_store_limit": 1024**2}):
        r = df.groupby(0).sum(combine_size=1)
    result = r.execute().fetch()
    pd.testing.assert_frame_equal(result, raw.groupby(0).sum())
    # test use shuffle
    assert len(r._fetch_infos()["memory_size"]) > 1

    rs = np.random.RandomState(0)
    raw = pd.DataFrame(
        {
            "c1": rs.randint(20, size=100),
            "c2": rs.choice(["a", "b", "c"], (100,)),
            "c3": rs.rand(100),
        }
    )
    mdf = md.DataFrame(raw, chunk_size=20)
    r = mdf.groupby("c2").sum().execute()
    pd.testing.assert_frame_equal(r.fetch(), raw.groupby("c2").sum())
    # test use tree
    assert len(r._fetch_infos()["memory_size"]) == 1

    rs = np.random.RandomState(0)
    raw = pd.DataFrame(
        {
            "c1": rs.randint(20, size=100),
            "c2": rs.choice(["a", "b", "c"], (100,)),
            "c3": rs.rand(100),
        }
    )
    mdf = md.DataFrame(raw, chunk_size=10)
    with option_context({"chunk_store_limit": 2048}):
        r = mdf.groupby("c2", sort=False).sum().execute()
    pd.testing.assert_frame_equal(
        r.fetch().sort_index(), raw.groupby("c2", sort=False).sum().sort_index()
    )
    # use tree and shuffle
    assert len(r._fetch_infos()["memory_size"]) == 3


def test_groupby_agg_str_cat(setup):
    agg_fun = lambda x: x.str.cat(sep="_", na_rep="NA")

    rs = np.random.RandomState(0)
    raw_df = pd.DataFrame(
        {
            "a": rs.choice(["A", "B", "C"], size=(100,)),
            "b": rs.choice([None, "alfa", "bravo", "charlie"], size=(100,)),
        }
    )

    mdf = md.DataFrame(raw_df, chunk_size=13)

    r = mdf.groupby("a").agg(agg_fun, method="tree")
    pd.testing.assert_frame_equal(r.execute().fetch(), raw_df.groupby("a").agg(agg_fun))

    raw_series = pd.Series(rs.choice([None, "alfa", "bravo", "charlie"], size=(100,)))

    ms = md.Series(raw_series, chunk_size=13)

    r = ms.groupby(lambda x: x % 2).agg(agg_fun, method="tree")
    pd.testing.assert_series_equal(
        r.execute().fetch(), raw_series.groupby(lambda x: x % 2).agg(agg_fun)
    )


@require_cudf
def test_gpu_groupby_agg(setup_gpu):
    rs = np.random.RandomState(0)
    df1 = pd.DataFrame(
        {"a": rs.choice([2, 3, 4], size=(100,)), "b": rs.choice([2, 3, 4], size=(100,))}
    )
    mdf = md.DataFrame(df1, chunk_size=13).to_gpu()

    r = mdf.groupby("a").sum()
    pd.testing.assert_frame_equal(
        r.execute().fetch(to_cpu=False).to_pandas(), df1.groupby("a").sum()
    )

    r = mdf.groupby("a").kurt()
    pd.testing.assert_frame_equal(
        r.execute().fetch(to_cpu=False).to_pandas(), df1.groupby("a").kurt()
    )

    r = mdf.groupby("a").agg(["sum", "var"])
    pd.testing.assert_frame_equal(
        r.execute().fetch(to_cpu=False).to_pandas(),
        df1.groupby("a").agg(["sum", "var"]),
    )

    rs = np.random.RandomState(0)
    idx = pd.Index(np.where(rs.rand(10) > 0.5, "A", "B"))
    series1 = pd.Series(rs.rand(10), index=idx)
    ms = md.Series(series1, index=idx, chunk_size=3).to_gpu().to_gpu()

    r = ms.groupby(level=0).sum()
    pd.testing.assert_series_equal(
        r.execute().fetch(to_cpu=False).to_pandas(), series1.groupby(level=0).sum()
    )

    r = ms.groupby(level=0).kurt()
    pd.testing.assert_series_equal(
        r.execute().fetch(to_cpu=False).to_pandas(), series1.groupby(level=0).kurt()
    )

    r = ms.groupby(level=0).agg(["sum", "var"])
    pd.testing.assert_frame_equal(
        r.execute().fetch(to_cpu=False).to_pandas(),
        series1.groupby(level=0).agg(["sum", "var"]),
    )


@support_cuda
def test_groupby_apply(setup_gpu, gpu):
    df1 = pd.DataFrame(
        {
            "a": [3, 4, 5, 3, 5, 4, 1, 2, 3],
            "b": [1, 3, 4, 5, 6, 5, 4, 4, 4],
            "c": list("aabaaddce"),
        }
    )

    def apply_df(df, ret_series=False):
        df = df.sort_index()
        df.a += df.b
        if len(df.index) > 0:
            if not ret_series:
                df = df.iloc[:-1, :]
            else:
                df = df.iloc[-1, :]
        return df

    def apply_series(s, truncate=True):
        s = s.sort_index()
        if truncate and len(s.index) > 0:
            s = s.iloc[:-1]
        return s

    mdf = md.DataFrame(df1, gpu=gpu, chunk_size=3)

    # Pandas is not compatible with the results of cudf in this case
    # cudf return a series, however, pandas returns empty dataframe.
    # So gpu is not tested here.
    if not gpu:
        applied = mdf.groupby("b").apply(lambda df: None)
        pd.testing.assert_frame_equal(
            applied.execute().fetch(), df1.groupby("b").apply(lambda df: None)
        )

    # For the index of result in this case, pandas is not compatible with cudf.
    # See ``Pandas Compatibility Note`` in cudf doc:
    # https://docs.rapids.ai/api/cudf/stable/api_docs/api/cudf.core.groupby.groupby.groupby.apply/
    applied = mdf.groupby("b").apply(apply_df)
    if gpu:
        cdf = cudf.DataFrame(df1)
        cudf.testing.assert_frame_equal(
            applied.execute().fetch(to_cpu=False).sort_index(),
            cdf.groupby("b").apply(apply_df).sort_index(),
        )
    else:
        pd.testing.assert_frame_equal(
            applied.execute().fetch().sort_index(),
            df1.groupby("b").apply(apply_df).sort_index(),
        )

    # For this case, cudf groupby apply method do not receive kwargs.
    # Also, cudf does not handle as_index is True.
    # So here only determine whether the results are consistent with cudf.
    if gpu:
        cdf = cudf.DataFrame(df1)
        applied = mdf.groupby("b").apply(apply_df, True)
        cudf.testing.assert_frame_equal(
            applied.execute().fetch(to_cpu=False).sort_index(),
            cdf.groupby("b").apply(apply_df, True).sort_index(),
        )
    else:
        applied = mdf.groupby("b").apply(apply_df, ret_series=True)
        pd.testing.assert_frame_equal(
            applied.execute().fetch().sort_index(),
            df1.groupby("b").apply(apply_df, ret_series=True).sort_index(),
        )

    # For this case, cudf does not handle as_index is True,
    # resulting in a mismatch between the output type and the actual type.
    # Therefore, just test cpu here.
    if not gpu:
        applied = mdf.groupby("b").apply(lambda df: df.a, output_type="series")
        pd.testing.assert_series_equal(
            applied.execute().fetch().sort_index(),
            df1.groupby("b").apply(lambda df: df.a).sort_index(),
        )

    applied = mdf.groupby("b").apply(lambda df: df.a.sum())
    pd.testing.assert_series_equal(
        applied.execute().fetch().sort_index(),
        df1.groupby("b").apply(lambda df: df.a.sum()).sort_index(),
    )

    series1 = pd.Series([3, 4, 5, 3, 5, 4, 1, 2, 3])
    ms1 = md.Series(series1, gpu=gpu, chunk_size=3)

    applied = ms1.groupby(lambda x: x % 3).apply(lambda df: None)
    pd.testing.assert_series_equal(
        applied.execute().fetch().sort_index(),
        series1.groupby(lambda x: x % 3).apply(lambda df: None).sort_index(),
    )

    # For this case, ``group_keys`` option does not take effect in cudf
    applied = ms1.groupby(lambda x: x % 3).apply(apply_series)
    if gpu:
        cs = cudf.Series(series1)
        cudf.testing.assert_series_equal(
            applied.execute().fetch(to_cpu=False).sort_index(),
            cs.groupby(lambda x: x % 3).apply(apply_series).sort_index(),
        )
    else:
        pd.testing.assert_series_equal(
            applied.execute().fetch().sort_index(),
            series1.groupby(lambda x: x % 3).apply(apply_series).sort_index(),
        )

    sindex2 = pd.MultiIndex.from_arrays([list(range(9)), list("ABCDEFGHI")])
    series2 = pd.Series(list("CDECEDABC"), index=sindex2)
    ms2 = md.Series(series2, gpu=gpu, chunk_size=3)

    # do not test multi index on gpu for now
    if not gpu:
        applied = ms2.groupby(lambda x: x[0] % 3).apply(apply_series)
        pd.testing.assert_series_equal(
            applied.execute().fetch().sort_index(),
            series2.groupby(lambda x: x[0] % 3).apply(apply_series).sort_index(),
        )


@support_cuda
def test_groupby_apply_with_df_or_series_output(setup_gpu, gpu):
    raw = pd.DataFrame(
        {
            "a": [3, 4, 5, 3, 5, 4, 1, 2, 3],
            "b": [6, 3, 3, 5, 6, 5, 4, 4, 4],
            "c": list("aabaabbbb"),
        }
    )
    mdf = md.DataFrame(raw, gpu=gpu, chunk_size=3)

    def f1(df):
        return df.a.iloc[2]

    with pytest.raises(TypeError):
        mdf.groupby("c").apply(f1)

    with pytest.raises(ValueError):
        mdf.groupby("c").apply(f1, output_types=["df_or_series"]).execute()

    for kwargs in [dict(output_type="df_or_series"), dict(skip_infer=True)]:
        mdf = md.DataFrame(raw, gpu=gpu, chunk_size=5)
        applied = mdf.groupby("c").apply(f1, **kwargs)
        assert isinstance(applied, DATAFRAME_OR_SERIES_TYPE)
        applied = applied.execute()
        assert applied.data_type == "series"
        assert not ("dtypes" in applied.data_params)
        assert applied.shape == (2,)
        pd.testing.assert_series_equal(
            applied.fetch().sort_index(), raw.groupby("c").apply(f1).sort_index()
        )

    def f2(df):
        return df[["a"]]

    mdf = md.DataFrame(raw, gpu=gpu, chunk_size=5)
    applied = mdf.groupby("c").apply(f2, output_types=["df_or_series"])
    assert isinstance(applied, DATAFRAME_OR_SERIES_TYPE)
    applied = applied.execute()
    assert applied.data_type == "dataframe"
    assert not ("dtype" in applied.data_params)
    assert applied.shape == (9, 1)
    expected = raw.groupby("c", as_index=True).apply(f2)
    pd.testing.assert_series_equal(applied.dtypes, expected.dtypes)
    pd.testing.assert_frame_equal(applied.fetch().sort_index(), expected.sort_index())


@support_cuda
def test_groupby_apply_closure(setup_gpu, gpu):
    # DataFrame
    df1 = pd.DataFrame(
        {
            "a": [3, 4, 5, 3, 5, 4, 1, 2, 3],
            "b": [1, 3, 4, 5, 6, 5, 4, 4, 4],
            "c": list("aabaaddce"),
        }
    )

    x, y = 10, 11

    def apply_closure_df(df):
        return df["a"].max() * x

    def apply_closure_series(s):
        return s.mean() * y

    class callable_df:
        def __init__(self):
            self.x = 10

        def __call__(self, df):
            return df["a"].max() * x

    class callable_series:
        def __init__(self):
            self.y = 11

        def __call__(self, s):
            return s.mean() * y

    mdf = md.DataFrame(df1, gpu=gpu, chunk_size=3)

    applied = mdf.groupby("b").apply(apply_closure_df)
    pd.testing.assert_series_equal(
        applied.execute().fetch().sort_index(),
        df1.groupby("b").apply(apply_closure_df).sort_index(),
    )

    cdf = callable_df()
    applied = mdf.groupby("b").apply(cdf)
    pd.testing.assert_series_equal(
        applied.execute().fetch().sort_index(),
        df1.groupby("b").apply(cdf).sort_index(),
    )

    # Series
    series1 = pd.Series([3, 4, 5, 3, 5, 4, 1, 2, 3])
    ms1 = md.Series(series1, gpu=gpu, chunk_size=3)

    applied = ms1.groupby(lambda x: x % 3).apply(apply_closure_series)
    pd.testing.assert_series_equal(
        applied.execute().fetch().sort_index(),
        series1.groupby(lambda x: x % 3).apply(apply_closure_series).sort_index(),
    )

    cs = callable_series()
    applied = ms1.groupby(lambda x: x % 3).apply(cs)
    pd.testing.assert_series_equal(
        applied.execute().fetch().sort_index(),
        series1.groupby(lambda x: x % 3).apply(cs).sort_index(),
    )


@support_cuda
@pytest.mark.parametrize(
    "chunked,as_index", [(True, True), (True, False), (False, True), (False, False)]
)
def test_groupby_apply_as_index(chunked, as_index, setup_gpu, gpu):
    df = pd.DataFrame(
        {
            "a": list(range(1, 11)),
            "b": list(range(1, 11))[::-1],
            "c": list("aabbccddac"),
        }
    )

    def udf(v):
        denominator = v["a"].sum() * v["a"].mean()
        v = v[v["c"] == "c"]
        numerator = v["a"].sum()
        return numerator / float(denominator)

    chunk_size = 3 if chunked else None
    mdf = md.DataFrame(df, gpu=gpu, chunk_size=chunk_size)
    applied = mdf.groupby("b", as_index=as_index).apply(udf)
    actual = applied.execute().fetch()
    expected = df.groupby("b", as_index=as_index).apply(udf)

    # cannot ensure the index for this case
    if chunked is True and as_index is False:
        actual = actual.sort_values(by="b").reset_index(drop=True)
        expected = expected.sort_values(by="b").reset_index(drop=True)

    if isinstance(expected, pd.DataFrame):
        pd.testing.assert_frame_equal(actual.sort_index(), expected.sort_index())
    else:
        pd.testing.assert_series_equal(actual.sort_index(), expected.sort_index())


def test_groupby_transform(setup):
    df1 = pd.DataFrame(
        {
            "a": [3, 4, 5, 3, 5, 4, 1, 2, 3],
            "b": [1, 3, 4, 5, 6, 5, 4, 4, 4],
            "c": list("aabaaddce"),
            "d": [3, 4, 5, 3, 5, 4, 1, 2, 3],
            "e": [1, 3, 4, 5, 6, 5, 4, 4, 4],
            "f": list("aabaaddce"),
        }
    )

    def transform_series(s, truncate=True):
        s = s.sort_index()
        if truncate and len(s.index) > 1:
            s = s.iloc[:-1].reset_index(drop=True)
        return s

    mdf = md.DataFrame(df1, chunk_size=3)

    r = mdf.groupby("b").transform(transform_series, truncate=False)
    pd.testing.assert_frame_equal(
        r.execute().fetch().sort_index(),
        df1.groupby("b").transform(transform_series, truncate=False).sort_index(),
    )

    df2 = pd.DataFrame(
        {
            "a": [3, 4, 5, 3, 5, 4, 1, 2, 3],
            "b": [1, 3, 4, 5, 6, 5, 4, 4, 4],
            "c": list("aabaabbba"),
        }
    )

    def f(df):
        if df.iloc[2]:
            return df
        else:
            return df + df.max()

    mdf2 = md.DataFrame(df2, chunk_size=5)
    with pytest.raises(TypeError):
        mdf2.groupby("c").transform(f)

    r = mdf2.groupby("c").transform(f, skip_infer=True)
    pd.testing.assert_frame_equal(
        r.execute().fetch().sort_index(),
        df2.groupby("c").transform(f).sort_index(),
    )

    if pd.__version__ != "1.1.0":
        df3 = pd.DataFrame(
            {
                "a": [3, 4, 5, 3, 5, 4, 1, 2, 3],
                "b": [1, 3, 4, 5, 6, 5, 4, 4, 4],
                "c": [1, 3, 4, 5, 6, 5, 4, 4, 4],
            }
        )
        mdf3 = md.DataFrame(df3, chunk_size=3)

        r = mdf3.groupby("b").transform(["cummax", "cumsum"], _call_agg=True)
        pd.testing.assert_frame_equal(
            r.execute().fetch().sort_index(),
            df3.groupby("b").agg(["cummax", "cumsum"]).sort_index(),
        )

        agg_list = ["cummax", "cumsum"]
        r = mdf3.groupby("b").transform(agg_list, _call_agg=True)
        pd.testing.assert_frame_equal(
            r.execute().fetch().sort_index(),
            df3.groupby("b").agg(agg_list).sort_index(),
        )

        agg_dict = OrderedDict([("a", "cummax"), ("c", "cumsum")])
        r = mdf3.groupby("b").transform(agg_dict, _call_agg=True)
        pd.testing.assert_frame_equal(
            r.execute().fetch().sort_index(),
            df3.groupby("b").agg(agg_dict).sort_index(),
        )

    agg_list = ["sum", lambda s: s.sum()]
    r = mdf.groupby("b").transform(agg_list, _call_agg=True)
    pd.testing.assert_frame_equal(
        r.execute().fetch().sort_index(), df1.groupby("b").agg(agg_list).sort_index()
    )

    series1 = pd.Series([3, 4, 5, 3, 5, 4, 1, 2, 3])
    ms1 = md.Series(series1, chunk_size=3)

    r = ms1.groupby(lambda x: x % 3).transform(lambda x: x + 1)
    pd.testing.assert_series_equal(
        r.execute().fetch().sort_index(),
        series1.groupby(lambda x: x % 3).transform(lambda x: x + 1).sort_index(),
    )

    r = ms1.groupby(lambda x: x % 3).transform("cummax", _call_agg=True)
    pd.testing.assert_series_equal(
        r.execute().fetch().sort_index(),
        series1.groupby(lambda x: x % 3).agg("cummax").sort_index(),
    )

    agg_list = ["cummax", "cumcount"]
    r = ms1.groupby(lambda x: x % 3).transform(agg_list, _call_agg=True)
    pd.testing.assert_frame_equal(
        r.execute().fetch().sort_index(),
        series1.groupby(lambda x: x % 3).agg(agg_list).sort_index(),
    )


def test_groupby_cum(setup):
    df1 = pd.DataFrame(
        {
            "a": [3, 5, 2, 7, 1, 2, 4, 6, 2, 4],
            "b": [8, 3, 4, 1, 8, 2, 2, 2, 2, 3],
            "c": [1, 8, 8, 5, 3, 5, 0, 0, 5, 4],
        }
    )
    mdf = md.DataFrame(df1, chunk_size=3)

    for fun in ["cummin", "cummax", "cumprod", "cumsum"]:
        r1 = getattr(mdf.groupby("b"), fun)()
        pd.testing.assert_frame_equal(
            r1.execute().fetch().sort_index(),
            getattr(df1.groupby("b"), fun)().sort_index(),
        )

        r2 = getattr(mdf.groupby("b"), fun)(axis=1)
        pd.testing.assert_frame_equal(
            r2.execute().fetch().sort_index(),
            getattr(df1.groupby("b"), fun)(axis=1).sort_index(),
        )

    r3 = mdf.groupby("b").cumcount()
    pd.testing.assert_series_equal(
        r3.execute().fetch().sort_index(), df1.groupby("b").cumcount().sort_index()
    )

    series1 = pd.Series([3, 4, 5, 3, 5, 4, 1, 2, 3])
    ms1 = md.Series(series1, chunk_size=3)

    for fun in ["cummin", "cummax", "cumprod", "cumsum", "cumcount"]:
        r1 = getattr(ms1.groupby(lambda x: x % 2), fun)()
        pd.testing.assert_series_equal(
            r1.execute().fetch().sort_index(),
            getattr(series1.groupby(lambda x: x % 2), fun)().sort_index(),
        )


def test_groupby_fill(setup):
    df1 = pd.DataFrame(
        [
            [1, 1, 10],
            [1, 1, np.nan],
            [1, 1, np.nan],
            [1, 2, np.nan],
            [1, 2, 20],
            [1, 2, np.nan],
            [1, 3, np.nan],
            [1, 3, np.nan],
        ],
        columns=["one", "two", "three"],
    )
    mdf = md.DataFrame(df1, chunk_size=3)
    r1 = getattr(mdf.groupby(["one", "two"]), "ffill")()
    pd.testing.assert_frame_equal(
        r1.execute().fetch().sort_index(),
        getattr(df1.groupby(["one", "two"]), "ffill")().sort_index(),
    )

    r2 = getattr(mdf.groupby("two"), "bfill")()
    pd.testing.assert_frame_equal(
        r2.execute().fetch().sort_index(),
        getattr(df1.groupby("two"), "bfill")().sort_index(),
    )

    r3 = getattr(mdf.groupby("one"), "fillna")(5)
    pd.testing.assert_frame_equal(
        r3.execute().fetch().sort_index(),
        getattr(df1.groupby("one"), "fillna")(5).sort_index(),
    )

    if not is_pandas_2():
        r4 = getattr(mdf.groupby("two"), "backfill")()
        pd.testing.assert_frame_equal(
            r4.execute().fetch().sort_index(),
            getattr(df1.groupby("two"), "backfill")().sort_index(),
        )

    s1 = pd.Series([4, 3, 9, np.nan, np.nan, 7, 10, 8, 1, 6])
    ms1 = md.Series(s1, chunk_size=3)

    r1 = getattr(ms1.groupby(lambda x: x % 2), "ffill")()
    pd.testing.assert_series_equal(
        r1.execute().fetch().sort_index(),
        getattr(s1.groupby(lambda x: x % 2), "ffill")().sort_index(),
    )

    r2 = getattr(ms1.groupby(lambda x: x % 2), "bfill")()
    pd.testing.assert_series_equal(
        r2.execute().fetch().sort_index(),
        getattr(s1.groupby(lambda x: x % 2), "bfill")().sort_index(),
    )

    if not is_pandas_2():
        r4 = getattr(ms1.groupby(lambda x: x % 2), "backfill")()
        pd.testing.assert_series_equal(
            r4.execute().fetch().sort_index(),
            getattr(s1.groupby(lambda x: x % 2), "backfill")().sort_index(),
        )


def test_groupby_head(setup):
    df1 = pd.DataFrame(
        {
            "a": [3, 5, 2, 7, 1, 2, 4, 6, 2, 4],
            "b": [8, 3, 4, 1, 8, 2, 2, 2, 2, 3],
            "c": [1, 8, 8, 5, 3, 5, 0, 0, 5, 4],
            "d": [9, 7, 6, 3, 6, 3, 2, 1, 5, 8],
        }
    )
    # test single chunk
    mdf = md.DataFrame(df1)

    r = mdf.groupby("b").head(1)
    pd.testing.assert_frame_equal(
        r.execute().fetch().sort_index(), df1.groupby("b").head(1)
    )
    r = mdf.groupby("b").head(-1)
    pd.testing.assert_frame_equal(
        r.execute().fetch().sort_index(), df1.groupby("b").head(-1)
    )
    r = mdf.groupby("b")[["a", "c"]].head(1)
    pd.testing.assert_frame_equal(
        r.execute().fetch().sort_index(), df1.groupby("b")[["a", "c"]].head(1)
    )

    # test multiple chunks
    mdf = md.DataFrame(df1, chunk_size=3)

    r = mdf.groupby("b").head(1)
    pd.testing.assert_frame_equal(
        r.execute().fetch().sort_index(), df1.groupby("b").head(1)
    )

    r = mdf.groupby("b").head(-1)
    pd.testing.assert_frame_equal(
        r.execute().fetch().sort_index(), df1.groupby("b").head(-1)
    )

    # test head with selection
    r = mdf.groupby("b")[["a", "d"]].head(1)
    pd.testing.assert_frame_equal(
        r.execute().fetch().sort_index(), df1.groupby("b")[["a", "d"]].head(1)
    )
    r = mdf.groupby("b")[["c", "a", "d"]].head(1)
    pd.testing.assert_frame_equal(
        r.execute().fetch().sort_index(), df1.groupby("b")[["c", "a", "d"]].head(1)
    )
    r = mdf.groupby("b")["c"].head(1)
    pd.testing.assert_series_equal(
        r.execute().fetch().sort_index(), df1.groupby("b")["c"].head(1)
    )

    # test single chunk
    series1 = pd.Series([3, 4, 5, 3, 5, 4, 1, 2, 3])
    ms = md.Series(series1)

    r = ms.groupby(lambda x: x % 2).head(1)
    pd.testing.assert_series_equal(
        r.execute().fetch().sort_index(), series1.groupby(lambda x: x % 2).head(1)
    )
    r = ms.groupby(lambda x: x % 2).head(-1)
    pd.testing.assert_series_equal(
        r.execute().fetch().sort_index(), series1.groupby(lambda x: x % 2).head(-1)
    )

    # test multiple chunk
    ms = md.Series(series1, chunk_size=3)

    r = ms.groupby(lambda x: x % 2).head(1)
    pd.testing.assert_series_equal(
        r.execute().fetch().sort_index(), series1.groupby(lambda x: x % 2).head(1)
    )

    # test with special index
    series1 = pd.Series([3, 4, 5, 3, 5, 4, 1, 2, 3], index=[4, 1, 2, 3, 5, 8, 6, 7, 9])
    ms = md.Series(series1, chunk_size=3)

    r = ms.groupby(lambda x: x % 2).head(1)
    pd.testing.assert_series_equal(
        r.execute().fetch().sort_index(),
        series1.groupby(lambda x: x % 2).head(1).sort_index(),
    )


def test_groupby_sample(setup):
    rs = np.random.RandomState(0)
    sample_count = 10
    src_data_list = []
    for b in range(5):
        data_count = int(rs.randint(20, 100))
        src_data_list.append(
            pd.DataFrame(
                {
                    "a": rs.randint(0, 100, size=data_count),
                    "b": np.array([b] * data_count),
                    "c": rs.randint(0, 100, size=data_count),
                    "d": rs.randint(0, 100, size=data_count),
                }
            )
        )
    df1 = pd.concat(src_data_list)
    shuffle_idx = np.arange(len(df1))
    rs.shuffle(shuffle_idx)
    df1 = df1.iloc[shuffle_idx].reset_index(drop=True)

    # test single chunk
    mdf = md.DataFrame(df1)

    r1 = mdf.groupby("b").sample(sample_count, random_state=rs)
    result1 = r1.execute().fetch()
    r2 = mdf.groupby("b").sample(sample_count, random_state=rs)
    result2 = r2.execute().fetch()
    pd.testing.assert_frame_equal(result1, result2)
    assert not (result1.groupby("b").count() - sample_count).any()[0]

    r1 = mdf.groupby("b").sample(
        sample_count, weights=df1["c"] / df1["c"].sum(), random_state=rs
    )
    result1 = r1.execute().fetch()
    r2 = mdf.groupby("b").sample(
        sample_count, weights=df1["c"] / df1["c"].sum(), random_state=rs
    )
    result2 = r2.execute().fetch()
    pd.testing.assert_frame_equal(result1, result2)
    assert not (result1.groupby("b").count() - sample_count).any()[0]

    r1 = mdf.groupby("b")[["b", "c"]].sample(sample_count, random_state=rs)
    result1 = r1.execute().fetch()
    r2 = mdf.groupby("b")[["b", "c"]].sample(sample_count, random_state=rs)
    result2 = r2.execute().fetch()
    pd.testing.assert_frame_equal(result1, result2)
    assert len(result1.columns) == 2
    assert not (result1.groupby("b").count() - sample_count).any()[0]

    r1 = mdf.groupby("b").c.sample(sample_count, random_state=rs)
    result1 = r1.execute().fetch()
    r2 = mdf.groupby("b").c.sample(sample_count, random_state=rs)
    result2 = r2.execute().fetch()
    pd.testing.assert_series_equal(result1, result2)

    r1 = mdf.groupby("b").c.sample(len(df1), random_state=rs)
    result1 = r1.execute().fetch()
    assert len(result1) == len(df1)

    with pytest.raises(ValueError):
        r1 = mdf.groupby("b").c.sample(len(df1), random_state=rs, errors="raises")
        r1.execute().fetch()

    # test multiple chunks
    mdf = md.DataFrame(df1, chunk_size=47)

    r1 = mdf.groupby("b").sample(sample_count, random_state=rs)
    result1 = r1.execute().fetch()
    r2 = mdf.groupby("b").sample(sample_count, random_state=rs)
    result2 = r2.execute().fetch()
    pd.testing.assert_frame_equal(result1, result2)
    assert not (result1.groupby("b").count() - sample_count).any()[0]

    r1 = mdf.groupby("b").sample(
        sample_count, weights=df1["c"] / df1["c"].sum(), random_state=rs
    )
    result1 = r1.execute().fetch()
    r2 = mdf.groupby("b").sample(
        sample_count, weights=df1["c"] / df1["c"].sum(), random_state=rs
    )
    result2 = r2.execute().fetch()
    pd.testing.assert_frame_equal(result1, result2)
    assert not (result1.groupby("b").count() - sample_count).any()[0]

    r1 = mdf.groupby("b")[["b", "c"]].sample(sample_count, random_state=rs)
    result1 = r1.execute().fetch()
    r2 = mdf.groupby("b")[["b", "c"]].sample(sample_count, random_state=rs)
    result2 = r2.execute().fetch()
    pd.testing.assert_frame_equal(result1, result2)
    assert len(result1.columns) == 2
    assert not (result1.groupby("b").count() - sample_count).any()[0]

    r1 = mdf.groupby("b").c.sample(sample_count, random_state=rs)
    result1 = r1.execute().fetch()
    r2 = mdf.groupby("b").c.sample(sample_count, random_state=rs)
    result2 = r2.execute().fetch()
    pd.testing.assert_series_equal(result1, result2)

    r1 = mdf.groupby("b").c.sample(len(df1), random_state=rs)
    result1 = r1.execute().fetch()
    assert len(result1) == len(df1)

    with pytest.raises(ValueError):
        r1 = mdf.groupby("b").c.sample(len(df1), random_state=rs, errors="raises")
        r1.execute().fetch()


@pytest.mark.skipif(pa is None, reason="pyarrow not installed")
def test_groupby_agg_with_arrow_dtype(setup):
    table = pa.table({"a": [1, 2, 1], "b": ["a", "b", "a"]})
    df1 = table.to_pandas(types_mapper=pd.ArrowDtype)
    mdf = md.DataFrame(df1)

    r = mdf.groupby("a").count()
    result = r.execute().fetch()
    expected = df1.groupby("a").count()
    pd.testing.assert_frame_equal(result, expected)

    r = mdf.groupby("b").count()
    result = r.execute().fetch()
    expected = df1.groupby("b").count()
    pd.testing.assert_frame_equal(result, expected)

    series1 = df1["b"]
    mseries = md.Series(series1)

    r = mseries.groupby(mseries).count()
    result = r.execute().fetch()
    expected = series1.groupby(series1).count()
    pd.testing.assert_series_equal(result, expected)

    series2 = series1.copy()
    series2.index = pd.MultiIndex.from_tuples([(0, 1), (2, 3), (4, 5)])
    mseries = md.Series(series2)

    r = mseries.groupby(mseries).count()
    result = r.execute().fetch()
    expected = series2.groupby(series2).count()
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.skipif(pa is None, reason="pyarrow not installed")
def test_groupby_apply_with_arrow_dtype(setup):
    table = pa.table({"a": [1, 2, 1], "b": ["a", "b", "a"]})
    df1 = table.to_pandas(types_mapper=pd.ArrowDtype)
    mdf = md.DataFrame(df1)

    applied = mdf.groupby("b").apply(lambda df: df.a.sum())
    result = applied.execute().fetch()
    expected = df1.groupby("b").apply(lambda df: df.a.sum())
    pd.testing.assert_series_equal(result, expected)

    series1 = df1["b"]
    mseries = md.Series(series1)

    applied = mseries.groupby(mseries).apply(lambda s: s)
    result = applied.execute().fetch()
    expected = series1.groupby(series1).apply(lambda s: s)
    pd.testing.assert_series_equal(result, expected, check_index_type=False)


def test_groupby_nunique(setup):
    rs = np.random.RandomState(0)
    data_size = 100
    data_dict = {
        "a": rs.randint(0, 10, size=(data_size,)),
        "b": rs.choice(list("abcd"), size=(data_size,)),
        "c": rs.choice(list("abcd"), size=(data_size,)),
    }
    df1 = pd.DataFrame(data_dict)

    # one chunk
    mdf = md.DataFrame(df1)
    pd.testing.assert_frame_equal(
        mdf.groupby("c").nunique().execute().fetch().sort_index(),
        df1.groupby("c").nunique().sort_index(),
    )

    # multiple chunks
    mdf = md.DataFrame(df1, chunk_size=13)
    pd.testing.assert_frame_equal(
        mdf.groupby("b").nunique().execute().fetch().sort_index(),
        df1.groupby("b").nunique().sort_index(),
    )

    # getitem and nunique
    mdf = md.DataFrame(df1, chunk_size=13)
    pd.testing.assert_series_equal(
        mdf.groupby("b")["a"].nunique().execute().fetch().sort_index(),
        df1.groupby("b")["a"].nunique().sort_index(),
    )

    # test with as_index=False
    mdf = md.DataFrame(df1, chunk_size=13)
    if _agg_size_as_frame:
        pd.testing.assert_frame_equal(
            mdf.groupby("b", as_index=False)["a"]
            .nunique()
            .execute()
            .fetch()
            .sort_values(by="b", ignore_index=True),
            df1.groupby("b", as_index=False)["a"]
            .nunique()
            .sort_values(by="b", ignore_index=True),
        )


def _generate_params_for_gpu():
    for data_type in ("df", "series"):
        for chunked in (False, True):
            for as_index in (False, True):
                for sort in (False, True):
                    yield data_type, chunked, as_index, sort


@require_cudf
@pytest.mark.parametrize(
    "data_type,chunked,as_index,sort",
    _generate_params_for_gpu(),
)
def test_gpu_groupby_size(data_type, chunked, as_index, sort, setup_gpu):
    data1 = [i + 1 for i in range(20)]
    data2 = [i * 2 + 1 for i in range(20)]
    data = {"a": data1, "b": data2}

    if data_type == "df":
        df = pd.DataFrame(data)
        expected = df.groupby(["a"], as_index=as_index, sort=sort).size()
    else:
        series = pd.Series(data1 + data2)
        if not as_index and data_type == "series":
            with pytest.raises(Exception):
                series.groupby(level=0, as_index=as_index, sort=sort).size()
            pytest.skip(
                "Skip this since pandas series groupby not support as_index=False"
            )
        expected = series.groupby(level=0, as_index=as_index, sort=sort).size()

    chunk_size = 3 if chunked else None

    if data_type == "df":
        mdf = md.DataFrame(data, chunk_size=chunk_size).to_gpu()
        res = mdf.groupby(["a"], as_index=as_index, sort=sort).size()
    else:
        series = md.Series(data1 + data2, chunk_size=chunk_size).to_gpu()
        res = series.groupby(level=0, as_index=as_index, sort=sort).size()
    actual = res.execute().fetch(to_cpu=False).to_pandas()

    if isinstance(expected, pd.DataFrame):
        # cudf groupby size not ensure order
        actual = actual.sort_values(by="a").reset_index(drop=True)
        pd.testing.assert_frame_equal(expected, actual)
    else:
        actual = actual.sort_index()
        pd.testing.assert_series_equal(expected, actual)


@support_cuda
@pytest.mark.parametrize(
    "as_index",
    [True, False],
)
def test_groupby_agg_on_same_funcs(setup_gpu, as_index, gpu):
    rs = np.random.RandomState(0)
    df = pd.DataFrame(
        {
            "a": rs.choice(["foo", "bar", "baz"], size=100),
            "b": rs.choice(["foo", "bar", "baz"], size=100),
            "c": rs.choice(["foo", "bar", "baz"], size=100),
        },
    )

    mdf = md.DataFrame(df, chunk_size=34, gpu=gpu)

    def g1(x):
        return (x == "foo").sum()

    def g2(x):
        return (x != "bar").sum()

    def g3(x):
        # same as g2
        return (x != "bar").sum()

    pd.testing.assert_frame_equal(
        df.groupby("a", as_index=False).agg((g1, g2, g3)),
        mdf.groupby("a", as_index=False).agg((g1, g2, g3)).execute().fetch(),
    )
    if not gpu:
        # cuDF doesn't support having multiple columns with same names yet.
        pd.testing.assert_frame_equal(
            df.groupby("a", as_index=as_index).agg((g1, g1)),
            mdf.groupby("a", as_index=as_index).agg((g1, g1)).execute().fetch(),
        )

    pd.testing.assert_frame_equal(
        df.groupby("a", as_index=as_index)["b"].agg((g1, g2, g3)),
        mdf.groupby("a", as_index=as_index)["b"].agg((g1, g2, g3)).execute().fetch(),
    )
    if not gpu:
        # cuDF doesn't support having multiple columns with same names yet.
        pd.testing.assert_frame_equal(
            df.groupby("a", as_index=as_index)["b"].agg((g1, g1)),
            mdf.groupby("a", as_index=as_index)["b"].agg((g1, g1)).execute().fetch(),
        )


@support_cuda
def test_groupby_agg_on_custom_funcs(setup_gpu, gpu):
    rs = np.random.RandomState(0)
    df = pd.DataFrame(
        {
            "a": rs.choice(["foo", "bar", "baz"], size=100),
            "b": rs.choice(["foo", "bar", "baz"], size=100),
            "c": rs.choice(["foo", "bar", "baz"], size=100),
        },
    )

    mdf = md.DataFrame(df, chunk_size=34, gpu=gpu)

    def g1(x):
        return ("foo" == x).sum()

    def g2(x):
        return ("foo" != x).sum()

    def g3(x):
        return (x > "bar").sum()

    def g4(x):
        return (x >= "bar").sum()

    def g5(x):
        return (x < "baz").sum()

    def g6(x):
        return (x <= "baz").sum()

    pd.testing.assert_frame_equal(
        df.groupby("a", as_index=False).agg(
            (
                g1,
                g2,
                g3,
                g4,
                g5,
                g6,
            )
        ),
        mdf.groupby("a", as_index=False)
        .agg(
            (
                g1,
                g2,
                g3,
                g4,
                g5,
                g6,
            )
        )
        .execute()
        .fetch(),
    )


@pytest.mark.parametrize(
    "window,min_periods,center,on,closed,agg",
    list(
        product(
            [5, 10],
            [1, 3],
            [False, True],
            [None, "c_1"],
            ["right", "left", "both", "neither"],
            [
                "corr",
                "cov",
                "count",
                "sum",
                "mean",
                "median",
                "var",
                "std",
                "min",
                "max",
                "skew",
                "kurt",
            ],
        )
    ),
)
def test_df_groupby_rolling_agg(setup, window, min_periods, center, on, closed, agg):
    # skipped for now due to https://github.com/pandas-dev/pandas/issues/52299
    if agg in _PAIRWISE_AGG and on is not None:
        return

    pdf = pd.DataFrame(
        data=np.random.randint(low=0, high=10, size=(10, 10)),
        index=pd.date_range(start="2023-01-01", end="2023-01-10", freq="D"),
        columns=[f"c_{i}" for i in range(10)],
    )
    pr = pdf.groupby(by="c_0").rolling(
        window=window,
        min_periods=min_periods,
        center=center,
        on=on,
        axis=0,
        closed=closed,
    )
    presult = getattr(pr, agg)()

    mdf = md.DataFrame(pdf, chunk_size=5)
    mr = mdf.groupby(by="c_0").rolling(
        window=window,
        min_periods=min_periods,
        center=center,
        on=on,
        axis=0,
        closed=closed,
    )
    mresult = getattr(mr, agg)()
    mresult = mresult.execute().fetch()

    pd.testing.assert_frame_equal(presult, mresult.sort_index())


@pytest.mark.parametrize(
    "window,min_periods,center,closed,agg",
    list(
        product(
            [5, 10],
            [1, 3],
            [False, True],
            ["right", "left", "both", "neither"],
            [
                "count",
                "sum",
                "mean",
                "median",
                "var",
                "std",
                "min",
                "max",
                "skew",
                "kurt",
            ],
        )
    ),
)
def test_series_groupby_rolling_agg(setup, window, min_periods, center, closed, agg):
    ps = pd.Series(
        data=np.random.randint(low=0, high=10, size=10),
        index=pd.date_range(start="2023-01-01", end="2023-01-10", freq="D"),
    )
    pr = ps.groupby(ps > 5).rolling(
        window=window, min_periods=min_periods, center=center, axis=0, closed=closed
    )
    presult = getattr(pr, agg)()

    ms = md.Series(ps, chunk_size=5)
    mr = ms.groupby(ms > 5).rolling(
        window=window, min_periods=min_periods, center=center, axis=0, closed=closed
    )
    mresult = getattr(mr, agg)()
    mresult = mresult.execute().fetch()

    pd.testing.assert_series_equal(presult, mresult.sort_index())


@pytest.mark.skipif(pd.__version__ <= "1.5.3", reason="pandas version is too low")
@pytest.mark.parametrize(
    "chunk_size, dropna", list(product([None, 3], [None, "any", "all"]))
)
def test_groupby_nth(setup, chunk_size, dropna):
    df1 = pd.DataFrame(
        {
            "a": np.random.randint(0, 5, size=20),
            "b": np.random.randint(0, 5, size=20),
            "c": np.random.randint(0, 5, size=20),
            "d": np.random.randint(0, 5, size=20),
        }
    )
    mdf = md.DataFrame(df1, chunk_size=chunk_size)

    r = mdf.groupby("b").nth(0)
    pd.testing.assert_frame_equal(
        r.execute().fetch().sort_index(), df1.groupby("b").nth(0)
    )
    r = mdf.groupby("b").nth(-1)
    pd.testing.assert_frame_equal(
        r.execute().fetch().sort_index(), df1.groupby("b").nth(-1)
    )
    r = mdf.groupby("b")[["a", "c"]].nth(0)
    pd.testing.assert_frame_equal(
        r.execute().fetch().sort_index(), df1.groupby("b")[["a", "c"]].nth(0)
    )

    # test nth with list index
    r = mdf.groupby("b").nth([0, 1])
    pd.testing.assert_frame_equal(
        r.execute().fetch().sort_index(), df1.groupby("b").nth([0, 1])
    )

    # test nth with slice
    r = mdf.groupby("b").nth(slice(None, 1))
    pd.testing.assert_frame_equal(
        r.execute().fetch().sort_index(), df1.groupby("b").nth(slice(None, 1))
    )

    # test nth with selection
    r = mdf.groupby("b")[["a", "d"]].nth(0)
    pd.testing.assert_frame_equal(
        r.execute().fetch().sort_index(), df1.groupby("b")[["a", "d"]].nth(0)
    )
    r = mdf.groupby("b")[["c", "a", "d"]].nth(0)
    pd.testing.assert_frame_equal(
        r.execute().fetch().sort_index(), df1.groupby("b")[["c", "a", "d"]].nth(0)
    )
    r = mdf.groupby("b")["c"].nth(0)
    pd.testing.assert_series_equal(
        r.execute().fetch().sort_index(), df1.groupby("b")["c"].nth(0)
    )

    series1 = pd.Series([3, 4, 5, 3, 5, 4, 1, 2, 3])
    ms = md.Series(series1, chunk_size=chunk_size)

    r = ms.groupby(lambda x: x % 2).nth(0)
    pd.testing.assert_series_equal(
        r.execute().fetch().sort_index(), series1.groupby(lambda x: x % 2).nth(0)
    )

    # test with special index
    series1 = pd.Series([3, 4, 5, 3, 5, 4, 1, 2, 3], index=[4, 1, 2, 3, 5, 8, 6, 7, 9])
    ms = md.Series(series1, chunk_size=chunk_size)

    r = ms.groupby(lambda x: x % 2).nth(0)
    pd.testing.assert_series_equal(
        r.execute().fetch().sort_index(),
        series1.groupby(lambda x: x % 2).nth(0).sort_index(),
    )

    df2 = pd.DataFrame(
        {
            "a": [3, 5, 2, np.nan, 1, 2, 4, 6, 2, 4],
            "b": [8, 3, 4, 1, 8, np.nan, 2, 2, 2, 3],
            "c": [1, 8, 8, np.nan, 3, 5, 0, 0, 5, 4],
            "d": [np.nan, 7, 6, 3, 6, 3, 2, 1, 5, 8],
        }
    )

    mdf = md.DataFrame(df2)

    r = mdf.groupby("b").nth(0, dropna=dropna)
    pd.testing.assert_frame_equal(
        r.execute().fetch().sort_index(), df2.groupby("b").nth(0, dropna=dropna)
    )
    r = mdf.groupby("b").nth(-1, dropna=dropna)
    pd.testing.assert_frame_equal(
        r.execute().fetch().sort_index(), df2.groupby("b").nth(-1, dropna=dropna)
    )
    r = mdf.groupby("b")[["a", "c"]].nth(0, dropna=dropna)
    pd.testing.assert_frame_equal(
        r.execute().fetch().sort_index(),
        df2.groupby("b")[["a", "c"]].nth(0, dropna=dropna),
    )

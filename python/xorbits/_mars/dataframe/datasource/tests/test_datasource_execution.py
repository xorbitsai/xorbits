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

import multiprocessing
import os
import tempfile
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from string import printable

import numpy as np
import pandas as pd
import pytest

from ....tensor.core import TENSOR_TYPE

try:
    import pyarrow as pa
except ImportError:  # pragma: no cover
    pa = None
try:
    import fastparquet
except ImportError:  # pragma: no cover
    fastparquet = None
try:
    import sqlalchemy
except ImportError:  # pragma: no cover
    sqlalchemy = None


import itertools

from .... import dataframe as md
from .... import tensor as mt
from ....config import option_context
from ....tests.core import require_cudf, require_cupy
from ....utils import get_next_port, pd_release_version
from ...utils import PD_VERSION_GREATER_THAN_2_10, is_pandas_2
from ..dataframe import from_pandas as from_pandas_df
from ..from_records import from_records
from ..from_tensor import dataframe_from_1d_tileables, dataframe_from_tensor
from ..index import from_pandas as from_pandas_index
from ..index import from_tileable
from ..series import from_pandas as from_pandas_series

_date_range_use_inclusive = pd_release_version[:2] >= (1, 4)


def test_from_pandas_dataframe_execution(setup):
    # test empty DataFrame
    pdf = pd.DataFrame()
    df = from_pandas_df(pdf)

    result = df.execute().fetch()
    pd.testing.assert_frame_equal(pdf, result)

    pdf = pd.DataFrame(columns=list("ab"))
    df = from_pandas_df(pdf)

    result = df.execute().fetch()
    pd.testing.assert_frame_equal(pdf, result)

    pdf = pd.DataFrame(
        np.random.rand(20, 30), index=[np.arange(20), np.arange(20, 0, -1)]
    )
    df = from_pandas_df(pdf, chunk_size=(13, 21))

    result = df.execute().fetch()
    pd.testing.assert_frame_equal(pdf, result)


def test_from_pandas_series_execution(setup):
    # test empty Series
    ps = pd.Series(name="a")
    series = from_pandas_series(ps, chunk_size=13)

    result = series.execute().fetch()
    pd.testing.assert_series_equal(ps, result)

    series = from_pandas_series(ps)

    result = series.execute().fetch()
    pd.testing.assert_series_equal(ps, result)

    ps = pd.Series(
        np.random.rand(20), index=[np.arange(20), np.arange(20, 0, -1)], name="a"
    )
    series = from_pandas_series(ps, chunk_size=13)

    result = series.execute().fetch()
    pd.testing.assert_series_equal(ps, result)


def test_from_pandas_index_execution(setup):
    pd_index = pd.timedelta_range("1 days", periods=10)
    index = from_pandas_index(pd_index, chunk_size=7)

    result = index.execute().fetch()
    pd.testing.assert_index_equal(pd_index, result)


def test_index_execution(setup):
    rs = np.random.RandomState(0)
    pdf = pd.DataFrame(
        rs.rand(20, 10),
        index=np.arange(20, 0, -1),
        columns=["a" + str(i) for i in range(10)],
    )
    df = from_pandas_df(pdf, chunk_size=13)

    # test df.index
    result = df.index.execute().fetch()
    pd.testing.assert_index_equal(result, pdf.index)

    result = df.columns.execute().fetch()
    pd.testing.assert_index_equal(result, pdf.columns)

    # df has unknown chunk shape on axis 0
    df = df[df.a1 < 0.5]

    # test df.index
    result = df.index.execute().fetch()
    pd.testing.assert_index_equal(result, pdf[pdf.a1 < 0.5].index)

    s = pd.Series(pdf["a1"], index=pd.RangeIndex(20))
    series = from_pandas_series(s, chunk_size=13)

    # test series.index which has value
    result = series.index.execute().fetch()
    pd.testing.assert_index_equal(result, s.index)

    s = pdf["a2"]
    series = from_pandas_series(s, chunk_size=13)

    # test series.index
    result = series.index.execute().fetch()
    pd.testing.assert_index_equal(result, s.index)

    # test tensor
    raw = rs.random(20)
    t = mt.tensor(raw, chunk_size=13)

    result = from_tileable(t).execute().fetch()
    pd.testing.assert_index_equal(result, pd.Index(raw))


def test_initializer_execution(setup):
    arr = np.random.rand(20, 30)

    pdf = pd.DataFrame(arr, index=[np.arange(20), np.arange(20, 0, -1)])
    df = md.DataFrame(pdf, chunk_size=(15, 10))
    result = df.execute().fetch()
    pd.testing.assert_frame_equal(pdf, result)

    df = md.DataFrame(arr, index=md.date_range("2020-1-1", periods=20))
    result = df.execute().fetch()
    pd.testing.assert_frame_equal(
        result, pd.DataFrame(arr, index=pd.date_range("2020-1-1", periods=20))
    )

    df = md.DataFrame(
        {"prices": [100, 101, np.nan, 100, 89, 88]},
        index=md.date_range("1/1/2010", periods=6, freq="D"),
    )
    result = df.execute().fetch()
    pd.testing.assert_frame_equal(
        result,
        pd.DataFrame(
            {"prices": [100, 101, np.nan, 100, 89, 88]},
            index=pd.date_range("1/1/2010", periods=6, freq="D"),
        ),
    )

    s = np.random.rand(20)

    ps = pd.Series(s, index=[np.arange(20), np.arange(20, 0, -1)], name="a")
    series = md.Series(ps, chunk_size=7)
    result = series.execute().fetch()
    pd.testing.assert_series_equal(ps, result)

    series = md.Series(s, index=md.date_range("2020-1-1", periods=20))
    result = series.execute().fetch()
    pd.testing.assert_series_equal(
        result, pd.Series(s, index=pd.date_range("2020-1-1", periods=20))
    )

    pi = pd.IntervalIndex.from_tuples([(0, 1), (2, 3), (4, 5)])
    index = md.Index(md.Index(pi))
    result = index.execute().fetch()
    pd.testing.assert_index_equal(pi, result)


def to_object(data_):
    if isinstance(data_, TENSOR_TYPE):
        return data_.execute().fetch(to_cpu=False)
    if isinstance(data_, dict):
        return dict((k, to_object(v)) for k, v in data_.items())
    else:
        return data_


@require_cupy
@require_cudf
@pytest.mark.parametrize(
    "data",
    [
        # TODO: creating GPU dataframe from CPU tensor results in KeyError in teardown stage.
        # mt.arange(1000),
        mt.arange(1000, gpu=True, chunk_size=500),
        # {"foo": mt.arange(1000), "bar": mt.arange(1000)},
        {"foo": mt.arange(500, gpu=True), "bar": mt.arange(500, gpu=True)},
        list(range(1000)),
        {"foo": list(range(500)), "bar": list(range(500))},
        np.arange(1000),
        {"foo": np.arange(500), "bar": np.arange(500)},
    ],
)
def test_dataframe_initializer_gpu(setup_gpu, data):
    import cudf
    import cudf.testing

    expected = cudf.DataFrame(to_object(data))

    df = md.DataFrame(data, gpu=True, chunk_size=500)
    actual = df.execute().fetch(to_cpu=False)
    assert isinstance(actual, cudf.DataFrame)
    cudf.testing.assert_frame_equal(expected, actual)


@require_cupy
@require_cudf
@pytest.mark.parametrize(
    "data",
    [
        # TODO: creating GPU dataframe from CPU tensor results in KeyError in teardown stage.
        # mt.arange(1000),
        mt.arange(1000, gpu=True),
        list(range(1000)),
        np.arange(1000),
    ],
)
def test_series_initializer_gpu(setup_gpu, data):
    import cudf
    import cudf.testing

    expected = cudf.Series(to_object(data))

    s = md.Series(data, gpu=True, chunk_size=500)
    actual = s.execute().fetch(to_cpu=False)
    print(type(actual))
    assert isinstance(actual, cudf.Series)
    cudf.testing.assert_series_equal(expected, actual)


@require_cupy
@require_cudf
@pytest.mark.parametrize(
    "data",
    [
        # TODO: creating GPU index from CPU objects results in KeyError in teardown stage.
        # mt.arange(1000, chunk_size=500),
        mt.arange(1000, gpu=True, chunk_size=500),
        np.arange(1000),
        list(range(1000)),
        list(itertools.product(["foo", "bar"], list(range(500)))),
    ],
)
def test_index_initializer_gpu(setup_gpu, data):
    import cudf
    import cudf.testing

    if isinstance(data, list) and isinstance(data[0], tuple):
        expected = cudf.from_pandas(pd.Index(data))
    else:
        expected = cudf.Index(to_object(data))

    idx = md.Index(data, gpu=True, chunk_size=500)
    actual = idx.execute().fetch(to_cpu=False)
    assert isinstance(actual, cudf.core.index.BaseIndex)
    cudf.testing.assert_index_equal(expected, actual)


def test_index_only(setup):
    df = md.DataFrame(index=[1, 2, 3])
    pd.testing.assert_frame_equal(df.execute().fetch(), pd.DataFrame(index=[1, 2, 3]))

    s = md.Series(index=[1, 2, 3])
    pd.testing.assert_series_equal(s.execute().fetch(), pd.Series(index=[1, 2, 3]))

    df = md.DataFrame(index=md.Index([1, 2, 3]))
    pd.testing.assert_frame_equal(df.execute().fetch(), pd.DataFrame(index=[1, 2, 3]))

    s = md.Series(index=md.Index([1, 2, 3]), dtype=object)
    pd.testing.assert_series_equal(
        s.execute().fetch(), pd.Series(index=[1, 2, 3], dtype=object)
    )


def test_series_from_tensor(setup):
    data = np.random.rand(10)
    series = md.Series(mt.tensor(data), name="a")
    pd.testing.assert_series_equal(series.execute().fetch(), pd.Series(data, name="a"))

    series = md.Series(mt.tensor(data, chunk_size=3))
    pd.testing.assert_series_equal(series.execute().fetch(), pd.Series(data))

    series = md.Series(mt.ones((10,), chunk_size=4))
    pd.testing.assert_series_equal(
        series.execute().fetch(),
        pd.Series(np.ones(10)),
    )

    index_data = np.random.rand(10)
    series = md.Series(
        mt.tensor(data, chunk_size=3),
        name="a",
        index=mt.tensor(index_data, chunk_size=4),
    )
    pd.testing.assert_series_equal(
        series.execute().fetch(), pd.Series(data, name="a", index=index_data)
    )

    series = md.Series(
        mt.tensor(data, chunk_size=3),
        name="a",
        index=md.date_range("2020-1-1", periods=10),
    )
    pd.testing.assert_series_equal(
        series.execute().fetch(),
        pd.Series(data, name="a", index=pd.date_range("2020-1-1", periods=10)),
    )


def test_from_tensor_execution(setup):
    tensor = mt.random.rand(10, 10, chunk_size=5)
    df = dataframe_from_tensor(tensor)
    tensor_res = tensor.execute().fetch()
    pdf_expected = pd.DataFrame(tensor_res)
    df_result = df.execute().fetch()
    pd.testing.assert_index_equal(df_result.index, pd.RangeIndex(0, 10))
    pd.testing.assert_index_equal(df_result.columns, pd.RangeIndex(0, 10))
    pd.testing.assert_frame_equal(df_result, pdf_expected)

    # test from tensor with unknown shape
    tensor2 = tensor[tensor[:, 0] < 0.9]
    df = dataframe_from_tensor(tensor2)
    df_result = df.execute().fetch()
    tensor_res = tensor2.execute().fetch()
    pdf_expected = pd.DataFrame(tensor_res)
    pd.testing.assert_frame_equal(df_result.reset_index(drop=True), pdf_expected)

    # test converted with specified index_value and columns
    tensor2 = mt.random.rand(2, 2, chunk_size=1)
    df2 = dataframe_from_tensor(
        tensor2, index=pd.Index(["a", "b"]), columns=pd.Index([3, 4])
    )
    df_result = df2.execute().fetch()
    pd.testing.assert_index_equal(df_result.index, pd.Index(["a", "b"]))
    pd.testing.assert_index_equal(df_result.columns, pd.Index([3, 4]))

    # test converted from 1-d tensor
    tensor3 = mt.array([1, 2, 3])
    df3 = dataframe_from_tensor(tensor3)
    result3 = df3.execute().fetch()
    pdf_expected = pd.DataFrame(np.array([1, 2, 3]))
    pd.testing.assert_frame_equal(pdf_expected, result3)

    # test converted from identical chunks
    tensor4 = mt.ones((10, 10), chunk_size=3)
    df4 = dataframe_from_tensor(tensor4)
    result4 = df4.execute().fetch()
    pdf_expected = pd.DataFrame(tensor4.execute().fetch())
    pd.testing.assert_frame_equal(pdf_expected, result4)

    # from tensor with given index
    tensor5 = mt.ones((10, 10), chunk_size=3)
    df5 = dataframe_from_tensor(tensor5, index=np.arange(0, 20, 2))
    result5 = df5.execute().fetch()
    pdf_expected = pd.DataFrame(np.ones((10, 10)), index=np.arange(0, 20, 2))
    pd.testing.assert_frame_equal(pdf_expected, result5)

    # from tensor with given index that is a tensor
    raw7 = np.random.rand(10, 10)
    tensor7 = mt.tensor(raw7, chunk_size=3)
    index_raw7 = np.random.rand(10)
    index7 = mt.tensor(index_raw7, chunk_size=4)
    df7 = dataframe_from_tensor(tensor7, index=index7)
    result7 = df7.execute().fetch()
    pdf_expected = pd.DataFrame(raw7, index=index_raw7)
    pd.testing.assert_frame_equal(pdf_expected, result7)

    # from tensor with given index is a md.Index
    raw10 = np.random.rand(10, 10)
    tensor10 = mt.tensor(raw10, chunk_size=3)
    index10 = md.date_range("2020-1-1", periods=10, chunk_size=3)
    df10 = dataframe_from_tensor(tensor10, index=index10)
    result10 = df10.execute().fetch()
    pdf_expected = pd.DataFrame(raw10, index=pd.date_range("2020-1-1", periods=10))
    pd.testing.assert_frame_equal(pdf_expected, result10)

    # from tensor with given columns
    tensor6 = mt.ones((10, 10), chunk_size=3)
    df6 = dataframe_from_tensor(tensor6, columns=list("abcdefghij"))
    result6 = df6.execute().fetch()
    pdf_expected = pd.DataFrame(tensor6.execute().fetch(), columns=list("abcdefghij"))
    pd.testing.assert_frame_equal(pdf_expected, result6)

    # from 1d tensors
    raws8 = [
        ("a", np.random.rand(8)),
        ("b", np.random.randint(10, size=8)),
        ("c", ["".join(np.random.choice(list(printable), size=6)) for _ in range(8)]),
    ]
    tensors8 = OrderedDict((r[0], mt.tensor(r[1], chunk_size=3)) for r in raws8)
    raws8.append(("d", 1))
    raws8.append(("e", pd.date_range("2020-1-1", periods=8)))
    tensors8["d"] = 1
    tensors8["e"] = raws8[-1][1]
    df8 = dataframe_from_1d_tileables(tensors8, columns=[r[0] for r in raws8])
    result = df8.execute().fetch()
    pdf_expected = pd.DataFrame(OrderedDict(raws8))
    pd.testing.assert_frame_equal(result, pdf_expected)

    # from 1d tensors and specify index with a tensor
    index_raw9 = np.random.rand(8)
    index9 = mt.tensor(index_raw9, chunk_size=4)
    df9 = dataframe_from_1d_tileables(
        tensors8, columns=[r[0] for r in raws8], index=index9
    )
    result = df9.execute().fetch()
    pdf_expected = pd.DataFrame(OrderedDict(raws8), index=index_raw9)
    pd.testing.assert_frame_equal(result, pdf_expected)

    # from 1d tensors and specify index
    df11 = dataframe_from_1d_tileables(
        tensors8,
        columns=[r[0] for r in raws8],
        index=md.date_range("2020-1-1", periods=8),
    )
    result = df11.execute().fetch()
    pdf_expected = pd.DataFrame(
        OrderedDict(raws8), index=pd.date_range("2020-1-1", periods=8)
    )
    pd.testing.assert_frame_equal(result, pdf_expected)

    df12 = dataframe_from_1d_tileables({"a": [md.Series([1, 2, 3]).sum() + 1]})
    result = df12.execute().fetch()
    pdf_expected = pd.DataFrame({"a": [pd.Series([1, 2, 3]).sum() + 1]})
    pd.testing.assert_frame_equal(result, pdf_expected)

    # from 1d tensors with unknown shape
    df_raw = pd.DataFrame({"id": list("abc"), "num": [1, 2, 3]})
    df13 = from_pandas_df(df_raw, chunk_size=2)
    s = df13.groupby("id")["num"].count()

    result = dataframe_from_1d_tileables({"t": s.value_counts()}).execute().fetch()
    pdf_expected = pd.DataFrame({"t": s.value_counts().execute().fetch()})
    pd.testing.assert_frame_equal(result, pdf_expected)


def test_from_records_execution(setup):
    dtype = np.dtype([("x", "int"), ("y", "double"), ("z", "<U16")])

    ndarr = np.ones((10,), dtype=dtype)
    pdf_expected = pd.DataFrame.from_records(ndarr, index=pd.RangeIndex(10))

    # from structured array of mars
    tensor = mt.ones((10,), dtype=dtype, chunk_size=3)
    df1 = from_records(tensor)
    df1_result = df1.execute().fetch()
    pd.testing.assert_frame_equal(df1_result, pdf_expected)

    # from structured array of numpy
    df2 = from_records(ndarr)
    df2_result = df2.execute().fetch()
    pd.testing.assert_frame_equal(df2_result, pdf_expected)


def test_read_csv_execution_with_pathlib_Path(setup):
    with tempfile.TemporaryDirectory() as tempdir:
        file_path = os.path.join(tempdir, "test.csv")

        df = pd.DataFrame(
            np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int64),
            columns=["a", "b", "c"],
        )
        df.to_csv(file_path)

        # test the pathlib.Path type for a Single file
        file_path = Path(file_path)
        pdf = pd.read_csv(file_path, index_col=0)
        r = md.read_csv(file_path, index_col=0)
        mdf = r.execute().fetch()
        pd.testing.assert_frame_equal(pdf, mdf)
        # size_res = self.executor.execute_dataframe(r, mock=True)
        # assert sum(s[0] for s in size_res) == os.stat(file_path).st_size

        mdf2 = md.read_csv(file_path, index_col=0, chunk_bytes=10).execute().fetch()
        pd.testing.assert_frame_equal(pdf, mdf2)

        mdf = md.read_csv(file_path, index_col=0, nrows=1).execute().fetch()
        pd.testing.assert_frame_equal(df[:1], mdf)

    # test read directory
    with tempfile.TemporaryDirectory() as tempdir:
        testdir = os.path.join(tempdir, "test_dir")
        os.makedirs(testdir, exist_ok=True)

        df = pd.DataFrame(np.random.rand(300, 3), columns=["a", "b", "c"])

        file_paths = [os.path.join(testdir, f"test{i}.csv") for i in range(3)]
        df[:100].to_csv(file_paths[0])
        df[100:200].to_csv(file_paths[1])
        df[200:].to_csv(file_paths[2])

        # test the pathlib.Path type for a directory
        testdir = Path(testdir)
        # As we can not guarantee the order in which these files are processed,
        # the result may not keep the original order.
        mdf = md.read_csv(testdir, index_col=0).execute().fetch()
        pd.testing.assert_frame_equal(df, mdf.sort_index())

        mdf2 = md.read_csv(testdir, index_col=0, chunk_bytes=50).execute().fetch()
        pd.testing.assert_frame_equal(df, mdf2.sort_index())


def test_read_csv_execution(setup):
    with tempfile.TemporaryDirectory() as tempdir:
        file_path = os.path.join(tempdir, "test.csv")

        df = pd.DataFrame(
            np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int64),
            columns=["a", "b", "c"],
        )
        df.to_csv(file_path)

        pdf = pd.read_csv(file_path, index_col=0)
        r = md.read_csv(file_path, index_col=0)
        mdf = r.execute().fetch()
        pd.testing.assert_frame_equal(pdf, mdf)
        # size_res = self.executor.execute_dataframe(r, mock=True)
        # assert sum(s[0] for s in size_res) == os.stat(file_path).st_size

        mdf2 = md.read_csv(file_path, index_col=0, chunk_bytes=10).execute().fetch()
        pd.testing.assert_frame_equal(pdf, mdf2)

        mdf = md.read_csv(file_path, index_col=0, nrows=1).execute().fetch()
        pd.testing.assert_frame_equal(df[:1], mdf)

    # test names and usecols
    with tempfile.TemporaryDirectory() as tempdir:
        file_path = os.path.join(tempdir, "test.csv")
        df = pd.DataFrame(
            np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int64),
            columns=["a", "b", "c"],
        )
        df.to_csv(file_path, index=False)

        mdf = md.read_csv(file_path, usecols=["c", "b"]).execute().fetch()
        pd.testing.assert_frame_equal(pd.read_csv(file_path, usecols=["c", "b"]), mdf)

        mdf = (
            md.read_csv(file_path, names=["a", "b", "c"], usecols=["c", "b"])
            .execute()
            .fetch()
        )
        pd.testing.assert_frame_equal(
            pd.read_csv(file_path, names=["a", "b", "c"], usecols=["c", "b"]), mdf
        )

        mdf = (
            md.read_csv(file_path, names=["a", "b", "c"], usecols=["a", "c"])
            .execute()
            .fetch()
        )
        pd.testing.assert_frame_equal(
            pd.read_csv(file_path, names=["a", "b", "c"], usecols=["a", "c"]), mdf
        )

        mdf = md.read_csv(file_path, usecols=["a", "c"]).execute().fetch()
        pd.testing.assert_frame_equal(pd.read_csv(file_path, usecols=["a", "c"]), mdf)

    # test sep
    with tempfile.TemporaryDirectory() as tempdir:
        file_path = os.path.join(tempdir, "test.csv")

        df = pd.DataFrame(
            np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=["a", "b", "c"]
        )
        df.to_csv(file_path, sep=";")

        pdf = pd.read_csv(file_path, sep=";", index_col=0)
        mdf = md.read_csv(file_path, sep=";", index_col=0).execute().fetch()
        pd.testing.assert_frame_equal(pdf, mdf)

        mdf2 = (
            md.read_csv(file_path, sep=";", index_col=0, chunk_bytes=10)
            .execute()
            .fetch()
        )
        pd.testing.assert_frame_equal(pdf, mdf2)

    # test missing value
    with tempfile.TemporaryDirectory() as tempdir:
        file_path = os.path.join(tempdir, "test.csv")

        df = pd.DataFrame(
            {
                "c1": [np.nan, "a", "b", "c"],
                "c2": [1, 2, 3, np.nan],
                "c3": [np.nan, np.nan, 3.4, 2.2],
            }
        )
        df.to_csv(file_path)

        pdf = pd.read_csv(file_path, index_col=0)
        mdf = md.read_csv(file_path, index_col=0).execute().fetch()
        pd.testing.assert_frame_equal(pdf, mdf)

        mdf2 = md.read_csv(file_path, index_col=0, chunk_bytes=12).execute().fetch()
        pd.testing.assert_frame_equal(pdf, mdf2)

    with tempfile.TemporaryDirectory() as tempdir:
        file_path = os.path.join(tempdir, "test.csv")

        index = pd.date_range(start="1/1/2018", periods=100)
        df = pd.DataFrame(
            {
                "col1": np.random.rand(100),
                "col2": np.random.choice(["a", "b", "c"], (100,)),
                "col3": np.arange(100),
            },
            index=index,
        )
        df.to_csv(file_path)

        pdf = pd.read_csv(file_path, index_col=0)
        mdf = md.read_csv(file_path, index_col=0).execute().fetch()
        pd.testing.assert_frame_equal(pdf, mdf)

        mdf2 = md.read_csv(file_path, index_col=0, chunk_bytes=100).execute().fetch()
        pd.testing.assert_frame_equal(pdf, mdf2)

        mdf3 = md.read_csv(file_path, index_col=[0, 1]).execute().fetch()
        pdf3 = pd.read_csv(file_path, index_col=[0, 1])
        pd.testing.assert_frame_equal(pdf3, mdf3)

    # test nan
    with tempfile.TemporaryDirectory() as tempdir:
        file_path = os.path.join(tempdir, "test.csv")

        df = pd.DataFrame(
            {
                "col1": np.random.rand(100),
                "col2": np.random.choice(["a", "b", "c"], (100,)),
                "col3": np.arange(100),
            }
        )
        df.iloc[:100, :] = pd.NA
        df.to_csv(file_path)

        pdf = pd.read_csv(file_path, index_col=0)
        mdf = md.read_csv(file_path, index_col=0, head_lines=10, chunk_bytes=200)
        result = mdf.execute().fetch()
        pd.testing.assert_frame_equal(pdf, result)

        # dtypes is inferred as expected
        pd.testing.assert_series_equal(mdf.dtypes, pdf.dtypes)

    # test compression
    with tempfile.TemporaryDirectory() as tempdir:
        file_path = os.path.join(tempdir, "test.gzip")

        index = pd.date_range(start="1/1/2018", periods=100)
        df = pd.DataFrame(
            {
                "col1": np.random.rand(100),
                "col2": np.random.choice(["a", "b", "c"], (100,)),
                "col3": np.arange(100),
            },
            index=index,
        )
        df.to_csv(file_path, compression="gzip")

        pdf = pd.read_csv(file_path, compression="gzip", index_col=0)
        mdf = md.read_csv(file_path, compression="gzip", index_col=0).execute().fetch()
        pd.testing.assert_frame_equal(pdf, mdf)

        mdf2 = (
            md.read_csv(file_path, compression="gzip", index_col=0, chunk_bytes="1k")
            .execute()
            .fetch()
        )
        pd.testing.assert_frame_equal(pdf, mdf2)

    # test multiple files
    for merge_small_file_option in [{"n_sample_file": 1}, None]:
        with tempfile.TemporaryDirectory() as tempdir:
            df = pd.DataFrame(np.random.rand(300, 3), columns=["a", "b", "c"])

            file_paths = [os.path.join(tempdir, f"test{i}.csv") for i in range(3)]
            df[:100].to_csv(file_paths[0])
            df[100:200].to_csv(file_paths[1])
            df[200:].to_csv(file_paths[2])

            mdf = (
                md.read_csv(
                    file_paths,
                    index_col=0,
                    merge_small_file_options=merge_small_file_option,
                )
                .execute()
                .fetch()
            )
            pd.testing.assert_frame_equal(df, mdf)

            mdf2 = (
                md.read_csv(file_paths, index_col=0, chunk_bytes=50).execute().fetch()
            )
            pd.testing.assert_frame_equal(df, mdf2)

    # test wildcards in path
    with tempfile.TemporaryDirectory() as tempdir:
        df = pd.DataFrame(np.random.rand(300, 3), columns=["a", "b", "c"])

        file_paths = [os.path.join(tempdir, f"test{i}.csv") for i in range(3)]
        df[:100].to_csv(file_paths[0])
        df[100:200].to_csv(file_paths[1])
        df[200:].to_csv(file_paths[2])

        # As we can not guarantee the order in which these files are processed,
        # the result may not keep the original order.
        mdf = md.read_csv(f"{tempdir}/*.csv", index_col=0).execute().fetch()
        pd.testing.assert_frame_equal(df, mdf.sort_index())

        mdf2 = (
            md.read_csv(f"{tempdir}/*.csv", index_col=0, chunk_bytes=50)
            .execute()
            .fetch()
        )
        pd.testing.assert_frame_equal(df, mdf2.sort_index())

    # test read directory
    with tempfile.TemporaryDirectory() as tempdir:
        testdir = os.path.join(tempdir, "test_dir")
        os.makedirs(testdir, exist_ok=True)

        df = pd.DataFrame(np.random.rand(300, 3), columns=["a", "b", "c"])

        file_paths = [os.path.join(testdir, f"test{i}.csv") for i in range(3)]
        df[:100].to_csv(file_paths[0])
        df[100:200].to_csv(file_paths[1])
        df[200:].to_csv(file_paths[2])

        # As we can not guarantee the order in which these files are processed,
        # the result may not keep the original order.
        mdf = md.read_csv(testdir, index_col=0).execute().fetch()
        pd.testing.assert_frame_equal(df, mdf.sort_index())

        mdf2 = md.read_csv(testdir, index_col=0, chunk_bytes=50).execute().fetch()
        pd.testing.assert_frame_equal(df, mdf2.sort_index())

    with tempfile.TemporaryDirectory() as tempdir:
        s = "随机生成的中文字符串"
        file_path = os.path.join(tempdir, "test.csv")
        df = pd.DataFrame(
            {"col1": range(len(s)), "col2": np.random.rand(len(s)), "col3": list(s)}
        )
        df.to_csv(file_path, encoding="gbk", index=False)
        pdf = pd.read_csv(file_path, encoding="gbk")
        r = md.read_csv(file_path, encoding="gbk")
        mdf = r.execute().fetch()
        pd.testing.assert_frame_equal(pdf, mdf)

        r = md.read_csv(file_path, encoding="gbk", chunk_bytes=12)
        mdf = r.execute().fetch()
        pd.testing.assert_frame_equal(pdf, mdf)


csv_with_comment = """# comment line
1 2.2
2 4.4
3 6.6
4 8.8
5 11.0
6 13.2
7 15.4
8 17.6
9 19.8
10 22.0
"""
csv_with_comment_and_header = """# comment line
col1 col2
1 2.2
2 4.4
3 6.6
4 8.8
5 11.0
6 13.2
7 15.4
8 17.6
9 19.8
10 22.0
"""


def test_read_csv_execution_with_skiprows(setup):
    # test skiprows(GH-193)
    with tempfile.TemporaryDirectory() as tempdir:
        test_file = os.path.join(tempdir, "with_comments.csv")
        with open(test_file, "w") as f:
            f.write(csv_with_comment)

        pandas_df = pd.read_csv(
            test_file,
            sep=" ",
            index_col=False,
            skiprows=1,
            names=("val1", "val2"),
            dtype={"val1": "int", "val2": "float"},
        )
        mdf = (
            md.read_csv(
                test_file,
                sep=" ",
                index_col=False,
                skiprows=1,
                names=("val1", "val2"),
                dtype={"val1": "int", "val2": "float"},
            )
            .execute()
            .fetch()
        )
        pd.testing.assert_frame_equal(pandas_df, mdf)
        mdf = (
            md.read_csv(
                test_file,
                sep=" ",
                index_col=False,
                skiprows=1,
                names=("val1", "val2"),
                dtype={"val1": "int", "val2": "float"},
                chunk_bytes=20,
            )
            .execute()
            .fetch()
        )
        pd.testing.assert_frame_equal(pandas_df, mdf)

        # comment and header
        test_file = os.path.join(tempdir, "with_comment_and_header.csv")
        with open(test_file, "w") as f:
            f.write(csv_with_comment_and_header)

        pandas_df = pd.read_csv(
            test_file,
            sep=" ",
            index_col=False,
            skiprows=1,
        )
        mdf = (
            md.read_csv(test_file, sep=" ", index_col=False, skiprows=1, chunk_bytes=20)
            .execute()
            .fetch()
        )
        pd.testing.assert_frame_equal(pandas_df, mdf)


@pytest.mark.skipif(pa is None, reason="pyarrow not installed")
def test_read_csv_arrow_backend(setup):
    rs = np.random.RandomState(0)
    df = pd.DataFrame(
        {
            "col1": rs.rand(100),
            "col2": rs.choice(["a" * 2, "b" * 3, "c" * 4], (100,)),
            "col3": np.arange(100),
        }
    )
    with tempfile.TemporaryDirectory() as tempdir:
        file_path = os.path.join(tempdir, "test.csv")
        df.to_csv(file_path, index=False)

        pdf = pd.read_csv(file_path)
        mdf = md.read_csv(file_path, dtype_backend="pyarrow")
        result = mdf.execute().fetch()
        # read_csv with engine="pyarrow" in pandas 1.5 does not use arrow dtype.
        if is_pandas_2():
            assert isinstance(mdf.dtypes.iloc[1], pd.ArrowDtype)
            assert isinstance(result.dtypes.iloc[1], pd.ArrowDtype)
            assert isinstance(result.dtypes.iloc[0], pd.ArrowDtype)
            assert result.to_dict() == pdf.to_dict()
        # There still exists Float64 != float64 dtype check error even if we use
        # convert_dtypes(dtype_backend='numpy_nullable') convert the arrow dtypes
        # back to numpy.
        # pd.testing.assert_frame_equal(arrow_array_to_objects(result), pdf, check_like=True)

    with tempfile.TemporaryDirectory() as tempdir:
        with option_context({"dataframe.dtype_backend": "pyarrow"}):
            file_path = os.path.join(tempdir, "test.csv")
            df.to_csv(file_path, index=False)

            pdf = pd.read_csv(file_path)
            mdf = md.read_csv(file_path)
            result = mdf.execute().fetch()
            # read_csv with engine="pyarrow" in pandas 1.5 does not use arrow dtype.
            if is_pandas_2():
                assert isinstance(mdf.dtypes.iloc[1], pd.ArrowDtype)
                assert isinstance(result.dtypes.iloc[1], pd.ArrowDtype)
                assert isinstance(result.dtypes.iloc[0], pd.ArrowDtype)
                assert result.to_dict() == pdf.to_dict()
            # There still exists Float64 != float64 dtype check error even if we use
            # convert_dtypes(dtype_backend='numpy_nullable') convert the arrow dtypes
            # back to numpy.
            # pd.testing.assert_frame_equal(arrow_array_to_objects(result), pdf)

    # test compression
    with tempfile.TemporaryDirectory() as tempdir:
        file_path = os.path.join(tempdir, "test.gzip")
        df.to_csv(file_path, compression="gzip", index=False)

        pdf = pd.read_csv(file_path, compression="gzip")
        mdf = md.read_csv(file_path, compression="gzip", dtype_backend="pyarrow")
        result = mdf.execute().fetch()
        # read_csv with engine="pyarrow" in pandas 1.5 does not use arrow dtype.
        if is_pandas_2():
            assert isinstance(mdf.dtypes.iloc[1], pd.ArrowDtype)
            assert isinstance(result.dtypes.iloc[1], pd.ArrowDtype)
            assert isinstance(result.dtypes.iloc[0], pd.ArrowDtype)
            assert result.to_dict() == pdf.to_dict()
        # There still exists Float64 != float64 dtype check error even if we use
        # convert_dtypes(dtype_backend='numpy_nullable') convert the arrow dtypes
        # back to numpy.
        # pd.testing.assert_frame_equal(arrow_array_to_objects(result), pdf)


@require_cudf
def test_read_csv_gpu_execution(setup_gpu):
    with tempfile.TemporaryDirectory() as tempdir:
        file_path = os.path.join(tempdir, "test.csv")

        df = pd.DataFrame(
            {
                "col1": np.random.rand(100),
                "col2": np.random.choice(["a", "b", "c"], (100,)),
                "col3": np.arange(100),
            }
        )
        df.to_csv(file_path, index=False)

        pdf = pd.read_csv(file_path)
        mdf = md.read_csv(file_path, gpu=True).execute().fetch(to_cpu=False)
        pd.testing.assert_frame_equal(
            pdf.reset_index(drop=True), mdf.to_pandas().reset_index(drop=True)
        )

        mdf2 = (
            md.read_csv(file_path, gpu=True, chunk_bytes=200)
            .execute()
            .fetch(to_cpu=False)
        )
        pd.testing.assert_frame_equal(
            pdf.reset_index(drop=True), mdf2.to_pandas().reset_index(drop=True)
        )


def test_read_csv_with_specific_names(setup):
    with tempfile.TemporaryDirectory() as tempdir:
        file_path = os.path.join(tempdir, "test_names.csv")
        df = pd.DataFrame(
            np.array(np.random.randint(0, 10, size=(10, 3))), columns=["a", "b", "c"]
        )
        df.to_csv(file_path, index=False)

        pdf = pd.read_csv(file_path, names=["b", "a", "c"], header=0)
        mdf = md.read_csv(file_path, names=["b", "a", "c"], header=0).execute().fetch()
        pd.testing.assert_frame_equal(pdf, mdf)


def test_read_csv_without_index(setup):
    # test csv file without storing index
    with tempfile.TemporaryDirectory() as tempdir:
        file_path = os.path.join(tempdir, "test.csv")

        df = pd.DataFrame(
            np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=["a", "b", "c"]
        )
        df.to_csv(file_path, index=False)

        pdf = pd.read_csv(file_path)
        mdf = md.read_csv(file_path).execute().fetch()
        pd.testing.assert_frame_equal(pdf, mdf)

        mdf2 = md.read_csv(file_path, chunk_bytes=10).execute().fetch()
        pd.testing.assert_frame_equal(pdf, mdf2)

        file_path2 = os.path.join(tempdir, "test.csv")
        df = pd.DataFrame(
            np.random.RandomState(0).rand(100, 10),
            columns=[f"col{i}" for i in range(10)],
        )
        df.to_csv(file_path2, index=False)

        mdf3 = md.read_csv(file_path2, chunk_bytes=os.stat(file_path2).st_size / 5)
        result = mdf3.execute().fetch()
        expected = pd.read_csv(file_path2)
        pd.testing.assert_frame_equal(result, expected)

        # test incremental_index = False
        mdf4 = md.read_csv(
            file_path2,
            chunk_bytes=os.stat(file_path2).st_size / 5,
            incremental_index=False,
        )
        result = mdf4.execute().fetch()
        assert not result.index.is_monotonic_increasing
        expected = pd.read_csv(file_path2)
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)


@pytest.mark.skipif(sqlalchemy is None, reason="sqlalchemy not installed")
def test_read_sql_execution(setup):
    import sqlalchemy as sa

    rs = np.random.RandomState(0)
    test_df = pd.DataFrame(
        {
            "a": np.arange(10).astype(np.int64, copy=False),
            "b": [f"s{i}" for i in range(10)],
            "c": rs.rand(10),
            "d": [
                datetime.fromtimestamp(time.time() + 3600 * (i - 5)) for i in range(10)
            ],
        }
    )

    with tempfile.TemporaryDirectory() as d:
        table_name = "test"
        table_name2 = "test2"
        uri = "sqlite:///" + os.path.join(d, "test.db")

        test_df.to_sql(table_name, uri, index=False)

        # test read with table name
        r = md.read_sql_table("test", uri, chunk_size=4)
        result = r.execute().fetch()
        pd.testing.assert_frame_equal(result, test_df)

        # test read with sql string and offset method
        r = md.read_sql_query(
            "select * from test where c > 0.5", uri, parse_dates=["d"], chunk_size=4
        )
        result = r.execute().fetch()
        pd.testing.assert_frame_equal(
            result, test_df[test_df.c > 0.5].reset_index(drop=True)
        )

        # test read with sql string and partition method with integer cols
        r = md.read_sql(
            "select * from test where b > 's5'",
            uri,
            parse_dates=["d"],
            partition_col="a",
            num_partitions=3,
        )
        result = r.execute().fetch()
        pd.testing.assert_frame_equal(
            result, test_df[test_df.b > "s5"].reset_index(drop=True)
        )

        # test read with sql string and partition method with datetime cols
        r = md.read_sql_query(
            "select * from test where b > 's5'",
            uri,
            parse_dates={"d": "%Y-%m-%d %H:%M:%S.%f"},
            partition_col="d",
            num_partitions=3,
        )
        result = r.execute().fetch()
        pd.testing.assert_frame_equal(
            result, test_df[test_df.b > "s5"].reset_index(drop=True)
        )

        # test read with sql string and partition method with datetime cols
        r = md.read_sql_query(
            "select * from test where b > 's5'",
            uri,
            parse_dates=["d"],
            partition_col="d",
            num_partitions=3,
            index_col="d",
        )
        result = r.execute().fetch()
        pd.testing.assert_frame_equal(result, test_df[test_df.b > "s5"].set_index("d"))

        # test SQL that return no result
        r = md.read_sql_query("select * from test where a > 1000", uri)
        result = r.execute().fetch()
        pd.testing.assert_frame_equal(result, pd.DataFrame(columns=test_df.columns))

        engine = sa.create_engine(uri)
        m = sa.MetaData()
        try:
            # test index_col and columns
            r = md.read_sql_table(
                "test",
                engine.connect(),
                chunk_size=4,
                index_col="a",
                columns=["b", "d"],
            )
            result = r.execute().fetch()
            expected = test_df.copy(deep=True)
            expected.set_index("a", inplace=True)
            del expected["c"]
            pd.testing.assert_frame_equal(result, expected)

            # do not specify chunk_size
            r = md.read_sql_table(
                "test", engine.connect(), index_col="a", columns=["b", "d"]
            )
            result = r.execute().fetch()
            pd.testing.assert_frame_equal(result, expected)

            table = sa.Table(table_name, m, autoload_replace=True, autoload_with=engine)
            r = md.read_sql_table(
                table,
                engine,
                chunk_size=4,
                index_col=[table.columns["a"], table.columns["b"]],
                columns=[table.columns["c"], "d"],
            )
            result = r.execute().fetch()
            expected = test_df.copy(deep=True)
            expected.set_index(["a", "b"], inplace=True)
            pd.testing.assert_frame_equal(result, expected)

            # test table with primary key
            sa.Table(
                table_name2,
                m,
                sa.Column("id", sa.Integer, primary_key=True),
                sa.Column("a", sa.Integer),
                sa.Column("b", sa.String),
                sa.Column("c", sa.Float),
                sa.Column("d", sa.DateTime),
            )
            m.create_all(engine)
            test_df = test_df.copy(deep=True)
            test_df.index.name = "id"
            test_df.to_sql(table_name2, uri, if_exists="append")

            r = md.read_sql_table(table_name2, engine, chunk_size=4, index_col="id")
            result = r.execute().fetch()
            pd.testing.assert_frame_equal(result, test_df)
        finally:
            engine.dispose()


@pytest.mark.skipif(sqlalchemy is None, reason="sqlalchemy not installed")
@pytest.mark.skipif(pa is None, reason="pyarrow not installed")
def test_read_sql_arrow_backend(setup):
    rs = np.random.RandomState(0)
    test_df = pd.DataFrame(
        {
            "a": np.arange(10).astype(np.int64, copy=False),
            "b": [f"s{i}" for i in range(10)],
            "c": rs.rand(10),
            "d": [
                datetime.fromtimestamp(time.time() + 3600 * (i - 5)) for i in range(10)
            ],
        }
    )

    with tempfile.TemporaryDirectory() as d:
        table_name = "test"
        uri = "sqlite:///" + os.path.join(d, "test.db")

        test_df.to_sql(table_name, uri, index=False)

        r = md.read_sql_table("test", uri, chunk_size=4, dtype_backend="pyarrow")
        result = r.execute().fetch()
        if is_pandas_2():
            assert isinstance(r.dtypes.iloc[1], pd.ArrowDtype)
            assert isinstance(result.dtypes.iloc[1], pd.ArrowDtype)
            assert isinstance(result.dtypes.iloc[0], pd.ArrowDtype)
        assert result.to_dict() == test_df.to_dict()
        # There still exists Float64 != float64 dtype check error even if we use
        # convert_dtypes(dtype_backend='numpy_nullable') convert the arrow dtypes
        # back to numpy.
        # pd.testing.assert_frame_equal(arrow_array_to_objects(result), test_df)

        # test read with sql string and offset method
        r = md.read_sql_query(
            "select * from test where c > 0.5",
            uri,
            parse_dates=["d"],
            chunk_size=4,
            dtype_backend="pyarrow",
        )
        result = r.execute().fetch()
        if is_pandas_2():
            assert isinstance(r.dtypes.iloc[1], pd.ArrowDtype)
            assert isinstance(result.dtypes.iloc[1], pd.ArrowDtype)
            assert isinstance(result.dtypes.iloc[0], pd.ArrowDtype)
        assert (
            result.to_dict()
            == test_df[test_df.c > 0.5].reset_index(drop=True).to_dict()
        )
        # There still exists Float64 != float64 dtype check error even if we use
        # convert_dtypes(dtype_backend='numpy_nullable') convert the arrow dtypes
        # back to numpy.
        # pd.testing.assert_frame_equal(
        #     arrow_array_to_objects(result),
        #     test_df[test_df.c > 0.5].reset_index(drop=True),
        # )


@pytest.mark.pd_compat
def test_date_range_execution(setup):
    chunk_sizes = [None, 3]
    inclusives = ["both", "neither", "left", "right"]

    if _date_range_use_inclusive:
        with pytest.warns(FutureWarning, match="closed"):
            md.date_range("2020-1-1", periods=10, closed="right")

    for chunk_size, inclusive in itertools.product(chunk_sizes, inclusives):
        kw = dict()
        if _date_range_use_inclusive:
            kw["inclusive"] = inclusive
        else:
            if inclusive == "neither":
                continue
            elif inclusive == "both":
                inclusive = None
            kw["closed"] = inclusive

        # start, periods, freq
        dr = md.date_range("2020-1-1", periods=10, chunk_size=chunk_size, **kw)

        result = dr.execute().fetch()
        expected = pd.date_range("2020-1-1", periods=10, **kw)
        pd.testing.assert_index_equal(result, expected)

        # end, periods, freq
        dr = md.date_range(end="2020-1-10", periods=10, chunk_size=chunk_size, **kw)

        result = dr.execute().fetch()
        expected = pd.date_range(end="2020-1-10", periods=10, **kw)
        pd.testing.assert_index_equal(result, expected)

        # start, end, freq
        dr = md.date_range("2020-1-1", "2020-1-10", chunk_size=chunk_size, **kw)

        result = dr.execute().fetch()
        expected = pd.date_range("2020-1-1", "2020-1-10", **kw)
        pd.testing.assert_index_equal(result, expected)

        # start, end and periods
        dr = md.date_range(
            "2020-1-1", "2020-1-10", periods=19, chunk_size=chunk_size, **kw
        )

        result = dr.execute().fetch()
        expected = pd.date_range("2020-1-1", "2020-1-10", periods=19, **kw)
        pd.testing.assert_index_equal(result, expected)

        # start, end and freq
        dr = md.date_range(
            "2020-1-1", "2020-1-10", freq="12h", chunk_size=chunk_size, **kw
        )

        result = dr.execute().fetch()
        expected = pd.date_range("2020-1-1", "2020-1-10", freq="12h", **kw)
        pd.testing.assert_index_equal(result, expected)

    # test timezone
    dr = md.date_range("2020-1-1", periods=10, tz="Asia/Shanghai", chunk_size=7)

    result = dr.execute().fetch()
    expected = pd.date_range("2020-1-1", periods=10, tz="Asia/Shanghai")
    pd.testing.assert_index_equal(result, expected)

    # test periods=0
    dr = md.date_range("2020-1-1", periods=0)

    result = dr.execute().fetch()
    expected = pd.date_range("2020-1-1", periods=0)
    pd.testing.assert_index_equal(result, expected)

    # test start == end
    dr = md.date_range("2020-1-1", "2020-1-1", periods=1)

    result = dr.execute().fetch()
    expected = pd.date_range("2020-1-1", "2020-1-1", periods=1)
    pd.testing.assert_index_equal(result, expected)

    # test normalize=True
    dr = md.date_range("2020-1-1", periods=10, normalize=True, chunk_size=4)

    result = dr.execute().fetch()
    expected = pd.date_range("2020-1-1", periods=10, normalize=True)
    pd.testing.assert_index_equal(result, expected)

    # test freq
    dr = md.date_range(start="1/1/2018", periods=5, freq="ME", chunk_size=3)

    result = dr.execute().fetch()
    expected = pd.date_range(start="1/1/2018", periods=5, freq="ME")
    pd.testing.assert_index_equal(result, expected)

    dr = md.date_range(start="2018/01/01", end="2018/07/01", freq="ME")
    result = dr.execute().fetch()
    expected = pd.date_range(start="2018/01/01", end="2018/07/01", freq="ME")
    pd.testing.assert_index_equal(result, expected)


parquet_engines = ["auto"]
if pa is not None:
    parquet_engines.append("pyarrow")
if fastparquet is not None:
    parquet_engines.append("fastparquet")


@pytest.mark.skipif(
    len(parquet_engines) == 1, reason="pyarrow and fastparquet are not installed"
)
@pytest.mark.parametrize("engine", parquet_engines)
def test_read_parquet_arrow(setup, engine):
    test_df = pd.DataFrame(
        {
            "a": np.arange(10).astype(np.int64, copy=False),
            "b": [f"s{i}" for i in range(10)],
            "c": np.random.rand(10),
        }
    )
    if PD_VERSION_GREATER_THAN_2_10 and engine != "fastparquet":
        test_df = test_df.convert_dtypes(dtype_backend="pyarrow")

    with tempfile.TemporaryDirectory() as tempdir:
        file_path = os.path.join(tempdir, "test.parquet")
        test_df.to_parquet(file_path)

        df = md.read_parquet(file_path, engine=engine)
        result = df.execute().fetch()
        pd.testing.assert_frame_equal(result, test_df)
        # size_res = self.executor.execute_dataframe(df, mock=True)
        # assert sum(s[0] for s in size_res) > test_df.memory_usage(deep=True).sum()

    if engine != "fastparquet":
        with tempfile.TemporaryDirectory() as tempdir:
            file_path = os.path.join(tempdir, "test.parquet")
            test_df.to_parquet(file_path, row_group_size=3)

            df = md.read_parquet(
                file_path, groups_as_chunks=True, columns=["a", "b"], engine=engine
            )
            result = df.execute().fetch()
            pd.testing.assert_frame_equal(
                result.reset_index(drop=True), test_df[["a", "b"]]
            )

    if engine != "fastparquet":
        with tempfile.TemporaryDirectory() as tempdir:
            file_path = os.path.join(tempdir, "test.parquet")
            test_df.to_parquet(file_path, row_group_size=5)

            df = md.read_parquet(
                file_path,
                groups_as_chunks=True,
                dtype_backend="pyarrow",
                incremental_index=True,
                engine=engine,
            )
            result = df.execute().fetch()
            assert isinstance(df.dtypes.iloc[1], pd.ArrowDtype)
            assert isinstance(result.dtypes.iloc[1], pd.ArrowDtype)
            assert isinstance(result.dtypes.iloc[0], pd.ArrowDtype)
            assert result.to_dict() == test_df.to_dict()
            # There still exists Float64 != float64 dtype check error even if we use
            # convert_dtypes(dtype_backend='numpy_nullable') convert the arrow dtypes
            # back to numpy.
            # pd.testing.assert_frame_equal(arrow_array_to_objects(result), test_df)

    # test wildcards in path
    for merge_small_file_option in [{"n_sample_file": 1}, None]:
        with tempfile.TemporaryDirectory() as tempdir:
            df = pd.DataFrame(
                {
                    "a": np.arange(300).astype(np.int64, copy=False),
                    "b": [f"s{i}" for i in range(300)],
                    "c": np.random.rand(300),
                }
            )

            if PD_VERSION_GREATER_THAN_2_10 and engine != "fastparquet":
                df = df.convert_dtypes(dtype_backend="pyarrow")

            file_paths = [os.path.join(tempdir, f"test{i}.parquet") for i in range(3)]
            df[:100].to_parquet(file_paths[0], row_group_size=50)
            df[100:200].to_parquet(file_paths[1], row_group_size=30)
            df[200:].to_parquet(file_paths[2])

            mdf = md.read_parquet(f"{tempdir}/*.parquet", engine=engine)
            r = mdf.execute().fetch()
            pd.testing.assert_frame_equal(df, r.sort_values("a").reset_index(drop=True))

            mdf = md.read_parquet(f"{tempdir}", engine=engine)
            r = mdf.execute().fetch()
            pd.testing.assert_frame_equal(df, r.sort_values("a").reset_index(drop=True))

            file_list = [os.path.join(tempdir, name) for name in os.listdir(tempdir)]
            mdf = md.read_parquet(file_list, engine=engine)
            r = mdf.execute().fetch()
            pd.testing.assert_frame_equal(df, r.sort_values("a").reset_index(drop=True))

            # test `dtype_backend="pyarrow"`
            if engine != "fastparquet":
                mdf = md.read_parquet(
                    f"{tempdir}/*.parquet", engine=engine, dtype_backend="pyarrow"
                )
                result = mdf.execute().fetch()
                assert isinstance(mdf.dtypes.iloc[1], pd.ArrowDtype)
                assert isinstance(result.dtypes.iloc[1], pd.ArrowDtype)

                mdf = md.read_parquet(
                    f"{tempdir}/*.parquet",
                    groups_as_chunks=True,
                    engine=engine,
                    merge_small_file_options=merge_small_file_option,
                )
                r = mdf.execute().fetch()
                pd.testing.assert_frame_equal(
                    df, r.sort_values("a").reset_index(drop=True)
                )
            else:
                with pytest.raises(ValueError):
                    mdf = md.read_parquet(
                        f"{tempdir}/*.parquet", engine=engine, dtype_backend="pyarrow"
                    )

    # test partitioned
    with tempfile.TemporaryDirectory() as tempdir:
        df = pd.DataFrame(
            {
                "a": np.random.rand(300),
                "b": [f"s{i}" for i in range(300)],
                "c": np.random.choice(["a", "b", "c"], (300,)),
            }
        )
        df.to_parquet(tempdir, partition_cols=["c"])
        mdf = md.read_parquet(tempdir, engine=engine)
        r = mdf.execute().fetch().astype(df.dtypes)
        pd.testing.assert_frame_equal(
            df.sort_values("a").reset_index(drop=True),
            r.sort_values("a").reset_index(drop=True),
        )


@pytest.mark.skipif(
    len(parquet_engines) == 1, reason="pyarrow and fastparquet are not installed"
)
@pytest.mark.parametrize("engine", parquet_engines)
def test_read_parquet_with_getting_index(setup, engine):
    test_df = pd.DataFrame(
        {
            "a": np.arange(10).astype(np.int64, copy=False),
            "b": [f"s{i}" for i in range(10)],
            "c": np.random.rand(10),
        }
    )
    with tempfile.TemporaryDirectory() as tempdir:
        file = f"{tempdir}/test.pq"
        test_df.to_parquet(file)
        mdf = md.read_parquet(file, engine=engine)
        res = mdf["a"].mean().execute().fetch()
        assert res == test_df["a"].mean()
        pd.testing.assert_index_equal(mdf.keys().execute().fetch(), test_df.keys())


@pytest.mark.skipif(
    len(parquet_engines) == 1, reason="pyarrow and fastparquet are not installed"
)
@pytest.mark.parametrize("engine", parquet_engines)
def test_read_parquet_zip(setup, engine):
    with tempfile.TemporaryDirectory() as tempdir:
        df = pd.DataFrame(
            {
                "a": np.arange(300).astype(np.int64, copy=False),
                "b": [f"s{i}" for i in range(300)],
                "c": np.random.rand(300),
            }
        )
        if PD_VERSION_GREATER_THAN_2_10 and engine != "fastparquet":
            df = df.convert_dtypes(dtype_backend="pyarrow")

        file_paths = [os.path.join(tempdir, f"test{i}.parquet") for i in range(3)]
        df[:100].to_parquet(file_paths[0], row_group_size=50)
        df[100:200].to_parquet(file_paths[1], row_group_size=30)
        df[200:].to_parquet(file_paths[2])
        import zipfile

        zip_file = zipfile.ZipFile(os.path.join(tempdir, "test.zip"), "w")

        zip_file.write(file_paths[0])
        zip_file.write(file_paths[1])
        zip_file.write(file_paths[2])

        zip_file.close()
        mdf = md.read_parquet(f"{tempdir}/test.zip", engine=engine)
        r = mdf.execute().fetch()
        pd.testing.assert_frame_equal(df, r.sort_values("a").reset_index(drop=True))


@require_cudf
def test_read_parquet_zip_gpu(setup):
    with tempfile.TemporaryDirectory() as tempdir:
        df = pd.DataFrame(
            {
                "a": np.arange(300).astype(np.int64, copy=False),
                "b": [f"s{i}" for i in range(300)],
                "c": np.random.rand(300),
            }
        )

        file_paths = [os.path.join(tempdir, f"test{i}.parquet") for i in range(3)]
        df[:100].to_parquet(file_paths[0], row_group_size=50)
        df[100:200].to_parquet(file_paths[1], row_group_size=30)
        df[200:].to_parquet(file_paths[2])
        import zipfile

        zip_file = zipfile.ZipFile(os.path.join(tempdir, "test.zip"), "w")

        zip_file.write(file_paths[0])
        zip_file.write(file_paths[1])
        zip_file.write(file_paths[2])

        zip_file.close()
        mdf = md.read_parquet(os.path.join(tempdir, "test.zip"), gpu=True)
        r = mdf.execute().fetch(to_cpu=False)
        pd.testing.assert_frame_equal(
            df, r.sort_values("a").to_pandas().reset_index(drop=True)
        )


def test_read_parquet_arrow_dtype(setup):
    test_df = pd.DataFrame(
        {
            "a": np.arange(10).astype(np.int64, copy=False),
            "b": [f"s{i}" for i in range(10)],
            "c": np.random.rand(10),
            "d": np.random.rand(10, 4).tolist(),
        }
    )

    with tempfile.TemporaryDirectory() as tempdir:
        file_path = os.path.join(tempdir, "test.parquet")
        test_df.to_parquet(file_path)

        df = md.read_parquet(file_path, dtype_backend="pyarrow")
        result = df.execute().fetch()
        assert isinstance(result.dtypes.iloc[1], pd.ArrowDtype)
        assert isinstance(result.dtypes.iloc[3], pd.ArrowDtype)
        assert isinstance(result.dtypes.iloc[0], pd.ArrowDtype)
        assert result.to_dict() == test_df.to_dict()
        # There still exists Float64 != float64 dtype check error even if we use
        # convert_dtypes(dtype_backend='numpy_nullable') convert the arrow dtypes
        # back to numpy.
        # pd.testing.assert_frame_equal(arrow_array_to_objects(result), test_df)


@pytest.mark.skipif(fastparquet is None, reason="fastparquet not installed")
def test_read_parquet_fast_parquet(setup):
    test_df = pd.DataFrame(
        {
            "a": np.arange(10).astype(np.int64, copy=False),
            "b": [f"s{i}" for i in range(10)],
            "c": np.random.rand(10),
        }
    )

    # test fastparquet engine
    with tempfile.TemporaryDirectory() as tempdir:
        file_path = os.path.join(tempdir, "test.csv")
        test_df.to_parquet(file_path, compression=None)

        df = md.read_parquet(file_path, engine="fastparquet")
        result = df.execute().fetch()
        pd.testing.assert_frame_equal(result, test_df)
        # size_res = self.executor.execute_dataframe(df, mock=True)
        # assert sum(s[0] for s in size_res) > test_df.memory_usage(deep=True).sum()


def _start_tornado(
    port: int, file_path0: str, file_path1: str, csv_path: str, zip_path: str
):
    import tornado.ioloop
    import tornado.web

    class Parquet0Handler(tornado.web.RequestHandler):
        def get(self):
            with open(file_path0, "rb") as f:
                self.write(f.read())

    class Parquet1Handler(tornado.web.RequestHandler):
        def get(self):
            with open(file_path1, "rb") as f:
                self.write(f.read())

    class CSVHandler(tornado.web.RequestHandler):
        def get(self):
            with open(csv_path, "rb") as f:
                self.write(f.read())

    class RangeZipFileHandler(tornado.web.RequestHandler):
        def get(self):
            file_path = zip_path

            file_size = os.path.getsize(file_path)

            range_header = self.request.headers.get("Range")
            if range_header:
                range_start, range_end = self.parse_range_header(range_header)
            else:
                range_start, range_end = 0, file_size - 1

            with open(file_path, "rb") as file:
                file.seek(range_start)
                data = file.read(range_end - range_start + 1)

            self.set_header("Content-Type", "application/zip")
            self.set_header("Content-Disposition", "attachment; filename=test.zip")
            self.set_header(
                "Content-Range", f"bytes {range_start}-{range_end}/{file_size}"
            )
            self.set_header("Accept-Ranges", "bytes")
            self.set_header("Content-Length", len(data))
            self.set_status(206)  # Partial Content
            self.write(data)

        def parse_range_header(self, range_header):
            range_bytes = range_header.replace("bytes=", "").split("-")
            range_start = int(range_bytes[0])
            range_end = int(range_bytes[1]) if range_bytes[1] else None
            return range_start, range_end

    app = tornado.web.Application(
        [
            (r"/read-parquet0", Parquet0Handler),
            (r"/read-parquet1", Parquet1Handler),
            (r"/test.zip", RangeZipFileHandler),
            (r"/read-csv", CSVHandler),
        ]
    )
    app.listen(port)
    tornado.ioloop.IOLoop.current().start()


@pytest.fixture
def start_http_server():
    with tempfile.TemporaryDirectory() as tempdir:
        file_path0 = os.path.join(tempdir, "test0.parquet")
        file_path1 = os.path.join(tempdir, "test1.parquet")
        csv_path = os.path.join(tempdir, "test.csv")

        df = pd.DataFrame(
            {
                "col1": np.random.rand(100),
                "col2": np.random.choice(["a", "b", "c"], (100,)),
                "col3": np.arange(100),
            }
        )
        df.iloc[:50].to_parquet(file_path0)
        df.iloc[50:].to_parquet(file_path1)
        df.to_csv(csv_path)
        import zipfile

        zip_path = os.path.join(tempdir, "test.zip")
        z = zipfile.ZipFile(zip_path, "w")
        z.write(file_path0)
        z.write(file_path1)
        z.close()

        port = get_next_port()
        proc = multiprocessing.Process(
            target=_start_tornado,
            args=(port, file_path0, file_path1, csv_path, zip_path),
        )
        proc.daemon = True
        proc.start()
        time.sleep(5)
        yield df, [
            f"http://127.0.0.1:{port}/read-parquet0",
            f"http://127.0.0.1:{port}/read-parquet1",
        ], f"http://127.0.0.1:{port}/test.zip", f"http://127.0.0.1:{port}/read-csv"
        # Terminate the process
        proc.terminate()


def test_read_parquet_with_http_url(setup, start_http_server):
    df, urls, zip_url, _ = start_http_server
    if PD_VERSION_GREATER_THAN_2_10:
        df = df.convert_dtypes(dtype_backend="pyarrow")
    mdf = md.read_parquet(urls).execute().fetch()
    pd.testing.assert_frame_equal(df, mdf)
    if is_pandas_2():
        arrow_df = df.convert_dtypes(dtype_backend="pyarrow")
        mdf = md.read_parquet(urls, dtype_backend="pyarrow").execute().fetch()
        pd.testing.assert_frame_equal(arrow_df, mdf)
        assert isinstance(mdf.dtypes.iloc[1], pd.ArrowDtype)

    mdf1 = md.read_parquet(urls[:1]).execute().fetch()
    pd.testing.assert_frame_equal(df.iloc[:50], mdf1)

    mdf2 = md.read_parquet(urls[1:]).execute().fetch()
    pd.testing.assert_frame_equal(df[50:], mdf2)

    mdf = md.read_parquet(zip_url)
    pd.testing.assert_frame_equal(df, mdf.execute().fetch())


@require_cudf
def test_read_parquet_gpu_execution(setup_gpu):
    with tempfile.TemporaryDirectory() as tempdir:
        file_path = os.path.join(tempdir, "test.parquet")

        df = pd.DataFrame(
            {
                "col1": np.random.rand(100),
                "col2": np.random.choice(["a", "b", "c"], (100,)),
                "col3": np.arange(100),
            }
        )
        df.to_parquet(file_path, index=False)

        pdf = pd.read_parquet(file_path)
        mdf = md.read_parquet(file_path, gpu=True).execute().fetch(to_cpu=False)
        pd.testing.assert_frame_equal(
            pdf.reset_index(drop=True), mdf.to_pandas().reset_index(drop=True)
        )

        mdf2 = md.read_parquet(file_path, gpu=True).execute().fetch(to_cpu=False)
        pd.testing.assert_frame_equal(
            pdf.reset_index(drop=True), mdf2.to_pandas().reset_index(drop=True)
        )

        mdf3 = (
            md.read_parquet(file_path, gpu=True).head(3).execute().fetch(to_cpu=False)
        )
        pd.testing.assert_frame_equal(
            pdf.reset_index(drop=True).head(3), mdf3.to_pandas().reset_index(drop=True)
        )

    with tempfile.TemporaryDirectory() as tempdir:
        file_path = os.path.join(tempdir, "test.parquet")
        test_df = pd.DataFrame(
            {
                "a": np.arange(10).astype(np.int64, copy=False),
                "b": [f"s{i}" for i in range(10)],
                "c": np.random.rand(10),
            }
        )
        test_df.to_parquet(file_path, row_group_size=3)

        df = md.read_parquet(
            file_path, groups_as_chunks=True, columns=["a", "b"], gpu=True
        )
        result = df.execute().fetch(to_cpu=False).to_pandas()
        pd.testing.assert_frame_equal(
            result.reset_index(drop=True), test_df[["a", "b"]]
        )

    # test partitioned
    with tempfile.TemporaryDirectory() as tempdir:
        df = pd.DataFrame(
            {
                "a": np.random.rand(300),
                "b": [f"s{i}" for i in range(300)],
                "c": np.random.choice(["a", "b", "c"], (300,)),
            }
        )
        df.to_parquet(tempdir, partition_cols=["c"])
        mdf = md.read_parquet(tempdir, gpu=True)
        r = mdf.execute().fetch(to_cpu=False).to_pandas().astype(df.dtypes)
        pd.testing.assert_frame_equal(
            df.sort_values("a").reset_index(drop=True),
            r.sort_values("a").reset_index(drop=True),
        )


@pytest.fixture
def ftp_writable():
    """
    Fixture providing a writable FTP filesystem.
    """
    pytest.importorskip("pyftpdlib")
    import shutil
    import subprocess
    import sys

    from fsspec.implementations.cached import CachingFileSystem
    from fsspec.implementations.ftp import FTPFileSystem

    FTPFileSystem.clear_instance_cache()  # remove lingering connections
    CachingFileSystem.clear_instance_cache()
    d = "temp"
    os.mkdir(d)
    P = subprocess.Popen(
        [sys.executable, "-m", "pyftpdlib", "-d", d, "-u", "user", "-P", "pass", "-w"]
    )
    try:
        time.sleep(1)
        yield "localhost", 2121, "user", "pass"
    finally:
        P.terminate()
        P.wait()
        try:
            shutil.rmtree("temp")
        except Exception:
            pass


def test_read_parquet_ftp(ftp_writable, setup):
    pytest.importorskip("pyftpdlib")
    host, port, user, pw = ftp_writable
    data = {"Column1": [1, 2, 3], "Column2": ["A", "B", "C"]}
    df = pd.DataFrame(data)
    if PD_VERSION_GREATER_THAN_2_10:
        df = df.convert_dtypes(dtype_backend="pyarrow")
    with tempfile.TemporaryDirectory() as tempdir:
        local_file_path = os.path.join(tempdir, "test.parquet")
        df.to_parquet("ftp://{}:{}@{}:{}/test.parquet".format(user, pw, host, port))
        df.to_parquet(local_file_path)
        import zipfile

        from fsspec.implementations.ftp import FTPFileSystem

        fs = FTPFileSystem(host=host, port=port, username=user, password=pw)
        fn = "/test.zip"
        with fs.open(fn, "wb") as f:
            zf = zipfile.ZipFile(f, mode="w")
            zf.write(local_file_path)
            zf.close()
        mdf = md.read_parquet(
            "ftp://{}:{}@{}:{}/test.parquet".format(user, pw, host, port)
        )
        pd.testing.assert_frame_equal(df, mdf.to_pandas())
        mdf_zip = md.read_parquet(
            "ftp://{}:{}@{}:{}/test.zip".format(user, pw, host, port)
        )
        pd.testing.assert_frame_equal(df, mdf_zip.to_pandas())


def test_read_csv_http_url(setup, start_http_server):
    df, _, _, csv_url = start_http_server
    mdf = md.read_csv(csv_url)
    pd.testing.assert_frame_equal(pd.read_csv(csv_url), mdf.execute().fetch())

    mdf = md.read_csv(csv_url, names=["col1", "col2", "col3"])
    pd.testing.assert_frame_equal(
        pd.read_csv(csv_url, names=["col1", "col2", "col3"]), mdf.execute().fetch()
    )

    mdf = md.read_csv(csv_url, header=0)
    pd.testing.assert_frame_equal(pd.read_csv(csv_url, header=0), mdf.execute().fetch())

    mdf = md.read_csv(csv_url, header=None)
    pd.testing.assert_frame_equal(
        pd.read_csv(csv_url, header=None), mdf.execute().fetch()
    )

    if is_pandas_2():
        df = df.convert_dtypes(dtype_backend="pyarrow")
        mdf = md.read_csv(csv_url, dtype_backend="pyarrow").execute().fetch()
        pd.testing.assert_frame_equal(
            pd.read_csv(csv_url, dtype_backend="pyarrow"), mdf
        )
        assert isinstance(mdf.dtypes.iloc[1], pd.ArrowDtype)

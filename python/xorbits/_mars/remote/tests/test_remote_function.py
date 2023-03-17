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

from collections import defaultdict

import numpy as np
import pandas as pd
import pytest
import xoscar as mo

from ... import dataframe as md
from ... import get_context
from ... import tensor as mt
from ...core import tile
from ...dataframe.core import DATAFRAME_OR_SERIES_TYPE, DATAFRAME_TYPE, SERIES_TYPE
from ...deploy.oscar.session import get_default_session
from ...learn.utils import shuffle
from ...lib.mmh3 import hash as mmh3_hash
from ...tensor.core import TENSOR_TYPE
from .. import ExecutableTuple, spawn


def test_params():
    def f(x):
        return x + 1

    r = spawn(f, args=(1,))
    c = tile(r).chunks[0]
    assert isinstance(c.params, dict)
    c.params = c.get_params_from_data(2)
    assert isinstance(c.params, dict)

    params = c.params
    params.pop("index", None)
    r.params = params
    r.refresh_params()


def test_remote_function(setup):
    session = setup

    def f1(x):
        return x + 1

    def f2(x, y, z=None):
        return x * y * (z[0] + z[1])

    rs = np.random.RandomState(0)
    raw1 = rs.rand(10, 10)
    raw2 = rs.rand(10, 10)

    r1 = spawn(f1, raw1)
    r2 = spawn(f1, raw2)
    r3 = spawn(f2, (r1, r2), {"z": [r1, r2]})

    result = r3.execute().fetch()
    expected = (raw1 + 1) * (raw2 + 1) * (raw1 + 1 + raw2 + 1)
    np.testing.assert_almost_equal(result, expected)

    with pytest.raises(TypeError):
        spawn(f2, (r1, r2), kwargs=())

    with pytest.raises(ValueError, match="Unexpected kw: k"):
        spawn(f2, (r1, r2), k=1)

    session_id = session.session_id

    def f():
        assert get_default_session().session_id == session_id
        return mt.ones((2, 3)).sum().to_numpy()

    assert spawn(f).execute().fetch() == 6


def test_specific_output_types(setup):
    pd_df = pd.DataFrame(np.ones((10, 3)), columns=["a", "b", "c"])

    def f1():
        return pd_df

    r = spawn(f1, output_types="dataframe").execute()

    assert isinstance(r, DATAFRAME_TYPE)
    assert r.index_value is not None
    pd.testing.assert_frame_equal(r.fetch(), pd_df)
    pd.testing.assert_index_equal(r.columns.to_pandas(), pd_df.columns)

    pd_series = pd.Series(np.ones((10,)), name="a")

    def f2():
        return pd_series

    r = spawn(f2, output_types="series").execute()

    assert isinstance(r, SERIES_TYPE)
    assert r.index_value is not None
    assert r.name == pd_series.name
    pd.testing.assert_series_equal(r.fetch(), pd_series)

    def f3(v):
        if v > 0:
            return pd_series
        else:
            return pd_df

    r = spawn(f3, args=(1,), output_types="df_or_series").execute()

    assert isinstance(r, DATAFRAME_OR_SERIES_TYPE)
    assert r.index_value is not None
    assert r.name == pd_series.name
    assert r.shape == pd_series.shape
    assert getattr(r, "dtypes", None) is None
    s = r.ensure_data()
    assert isinstance(s, SERIES_TYPE)
    pd.testing.assert_series_equal(s.fetch(), pd_series)

    r = spawn(f3, args=(0,), output_types="df_or_series").execute()

    assert isinstance(r, DATAFRAME_OR_SERIES_TYPE)
    assert r.index_value is not None
    pd.testing.assert_series_equal(r.dtypes, pd_df.dtypes)
    assert r.shape == pd_df.shape
    assert getattr(r, "dtype", None) is None
    s = r.ensure_data()
    assert isinstance(s, DATAFRAME_TYPE)
    pd.testing.assert_frame_equal(s.fetch(), pd_df)

    np_array = np.random.rand(10, 10)

    def f2():
        return np_array

    r = spawn(f2, output_types="tensor").execute()

    assert isinstance(r, TENSOR_TYPE)
    assert r.dtype == np_array.dtype
    np.testing.assert_array_equal(r.fetch(), np_array)


def test_context(setup_cluster):
    def get_workers():
        ctx = get_context()
        return ctx.get_worker_addresses()

    def f1(worker: str):
        ctx = get_context()
        assert worker == ctx.worker_address
        return np.random.rand(3, 3)

    def f2(data_key: str, worker: str):
        ctx = get_context()
        assert worker == ctx.worker_address
        meta = ctx.get_chunks_meta([data_key], fields=["bands"])[0]
        assert len(meta) == 1
        ctx.get_chunks_result([data_key], fetch_only=True)
        # fetched, two workers have the data
        meta = ctx.get_chunks_meta([data_key], fields=["bands"])[0]
        assert len(meta["bands"]) == 2

    workers = spawn(get_workers).execute().fetch()
    assert len(workers) == len(set(workers)) > 1

    r1 = spawn(f1, args=(workers[0],), expect_worker=workers[0]).execute()
    data_key = r1._fetch_infos(fields=["data_key"])["data_key"][0]
    r2 = spawn(f2, args=(data_key, workers[1]), expect_worker=workers[1])
    r2.execute()

    def get_bands():
        ctx = get_context()
        return [b for b in ctx.get_worker_bands() if b[1].startswith("numa-")]

    def f3(band: tuple):
        ctx = get_context()
        assert band == ctx.band
        return np.random.rand(3, 3)

    def f4(data_key: str, band: tuple):
        ctx = get_context()
        assert band == ctx.band
        meta = ctx.get_chunks_meta([data_key], fields=["bands"])[0]
        assert len(meta) == 1
        ctx.get_chunks_result([data_key], fetch_only=True)
        # fetched, two bands have the data
        meta = ctx.get_chunks_meta([data_key], fields=["bands"])[0]
        assert len(meta["bands"]) == 2

    bands = spawn(get_bands).execute().fetch()
    assert len(bands) == len(set(bands)) > 1

    r3 = spawn(f3, args=(bands[0],), expect_band=bands[0]).execute()
    data_key = r3._fetch_infos(fields=["data_key"])["data_key"][0]
    r4 = spawn(f4, args=(data_key, bands[1]), expect_band=bands[1])
    r4.execute()


def test_multi_output(setup):
    sentences = ["word1 word2", "word2 word3", "word3 word2 word1"]

    def mapper(s):
        word_to_count = defaultdict(lambda: 0)
        for word in s.split():
            word_to_count[word] += 1

        downsides = [defaultdict(lambda: 0), defaultdict(lambda: 0)]
        for word, count in word_to_count.items():
            downsides[mmh3_hash(word) % 2][word] += count

        return downsides

    def reducer(word_to_count_list):
        d = defaultdict(lambda: 0)
        for word_to_count in word_to_count_list:
            for word, count in word_to_count.items():
                d[word] += count

        return dict(d)

    outs = [], []
    for sentence in sentences:
        out1, out2 = spawn(mapper, sentence, n_output=2)
        outs[0].append(out1)
        outs[1].append(out2)

    rs = []
    for out in outs:
        r = spawn(reducer, out)
        rs.append(r)

    result = dict()
    for wc in ExecutableTuple(rs).to_object():
        result.update(wc)

    assert result == {"word1": 2, "word2": 3, "word3": 2}


def test_chained_remote(setup):
    def f(x):
        return x + 1

    def g(x):
        return x * 2

    s = spawn(g, spawn(f, 2))

    result = s.execute().fetch()
    assert result == 6


def test_input_tileable(setup):
    def f(t, x):
        return (t * x).sum().to_numpy()

    rs = np.random.RandomState(0)
    raw = rs.rand(5, 4)

    t1 = mt.tensor(raw, chunk_size=3)
    t2 = t1.sum(axis=0)
    s = spawn(f, args=(t2, 3))

    result = s.execute().fetch()
    expected = (raw.sum(axis=0) * 3).sum()
    assert pytest.approx(result) == expected

    df1 = md.DataFrame(raw, chunk_size=3)
    df1.execute()
    df2 = shuffle(df1)
    df2.execute()

    def f2(input_df):
        bonus = input_df.iloc[:, 0].fetch().sum()
        return input_df.sum().to_pandas() + bonus

    for df in [df1, df2]:
        s = spawn(f2, args=(df,))

        result = s.execute().fetch()
        expected = pd.DataFrame(raw).sum() + raw[:, 0].sum()
        pd.testing.assert_series_equal(result, expected)


def test_unknown_shape_inputs(setup):
    def f(t, x):
        assert all(not np.isnan(s) for s in t.shape)
        return (t * x).sum().to_numpy(extra_config={"check_nsplits": False})

    rs = np.random.RandomState(0)
    raw = rs.rand(5, 4)

    t1 = mt.tensor(raw, chunk_size=3)
    t2 = t1[t1 > 0]
    s = spawn(f, args=(t2, 3))

    result = s.execute().fetch()
    expected = (raw[raw > 0] * 3).sum()
    assert pytest.approx(result) == expected


def test_none_outputs(setup):
    def f(*_args):
        pass

    r1 = spawn(f, args=(0,))
    r2 = spawn(f, args=(r1, 1))
    r3 = spawn(f, args=(r1, 2))
    r4 = spawn(f, args=(r2, r3))

    assert r4.execute().fetch() is None


def test_remote_with_unpickable(setup_cluster):
    def f(*_):
        class Unpickleable:
            def __reduce__(self):
                raise ValueError

        raise KeyError(Unpickleable())

    with pytest.raises(mo.SendMessageFailed):
        d = spawn(f, retry_when_fail=False)
        d.execute()

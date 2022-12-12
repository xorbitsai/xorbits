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

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from .... import pandas as xpd
from ....core.data import DataRef


def test_pandas_dataframe_methods(setup):
    raw = pd.DataFrame(
        {
            "foo": ["one", "one", "one", "two", "two", "two"],
            "bar": ["A", "B", "C", "A", "B", "C"],
            "baz": [1, 2, 3, 4, 5, 6],
            "zoo": ["x", "y", "z", "q", "w", "t"],
        }
    )
    df = xpd.DataFrame(raw)
    with pytest.warns(Warning) as w:
        r = df.pivot(index="foo", columns="bar", values="baz")
        assert len(w) == 1
        assert "DataFrame.pivot will fallback to Pandas" == str(w[0].message)

    assert len(getattr(r.data._mars_entity, "_executed_sessions")) == 1
    expected = raw.pivot(index="foo", columns="bar", values="baz")
    assert expected.shape == r.shape
    assert str(expected) == str(r)
    assert isinstance(r, DataRef)
    pd.testing.assert_frame_equal(expected, r.to_pandas())

    # multi chunk and follow other operations
    df = xpd.DataFrame(raw, chunk_size=3)
    with pytest.warns(Warning) as w:
        r = df.pivot(index="foo", columns="bar", values="baz").max(axis=0)
        assert "DataFrame.pivot will fallback to Pandas" == str(w[0].message)

    expected = raw.pivot(index="foo", columns="bar", values="baz").max(axis=0)
    assert expected.shape == r.shape
    assert str(expected) == str(r)
    pd.testing.assert_series_equal(expected, r.to_pandas())

    # can be inferred
    df = xpd.DataFrame(raw)
    with pytest.warns(Warning) as w:
        r = df.median(numeric_only=True)
        assert "DataFrame.median will fallback to Pandas" == str(w[0].message)

    # assert if r is executed
    assert len(getattr(r.data._mars_entity, "_executed_sessions")) == 0
    r2 = r.sum()
    expected = raw.median(numeric_only=True).sum()
    assert str(r2) == str(expected)
    assert isinstance(r2, DataRef)
    np.testing.assert_approx_equal(r2.fetch(), expected)

    # output is not series or dataframe
    df = xpd.DataFrame(raw)
    with pytest.warns(Warning) as w:
        r = df.to_json()
        assert "DataFrame.to_json will fallback to Pandas" == str(w[0].message)

    # assert if r is executed
    assert len(getattr(r.data._mars_entity, "_executed_sessions")) == 1
    expected = raw.to_json()
    assert str(r.to_object()) == str(expected)
    assert isinstance(r, DataRef)


def test_pandas_module_methods(setup):
    raw = pd.DataFrame(
        [["a", "b"], ["c", "d"]], index=["row 1", "row 2"], columns=["col 1", "col 2"]
    )
    # output could be series or dataframe
    with pytest.warns(Warning) as w:
        r = xpd.read_json(raw.to_json())
        assert "xorbits.pandas.read_json will fallback to Pandas" == str(w[0].message)

    assert str(r) == str(raw)
    assert isinstance(r, DataRef)
    pd.testing.assert_frame_equal(r.to_pandas(), raw)

    # output is dataframe
    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "tmp.xlsx")
        raw.to_excel(path)
        with pytest.warns(Warning) as w:
            r = xpd.read_excel(path)
            assert "xorbits.pandas.read_excel will fallback to Pandas" == str(
                w[0].message
            )

        expected = pd.read_excel(path)
        assert str(r) == str(expected)
        assert isinstance(r, DataRef)
        pd.testing.assert_frame_equal(r.to_pandas(), expected)

    # input has xorbit data
    raw = pd.DataFrame(
        {
            "foo": ["one", "one", "one", "two", "two", "two"],
            "bar": ["A", "B", "C", "A", "B", "C"],
            "baz": [1, 2, 3, 4, 5, 6],
            "zoo": ["x", "y", "z", "q", "w", "t"],
        }
    )
    with pytest.warns(Warning) as w:
        r = xpd.pivot(xpd.DataFrame(raw), index="foo", columns="bar", values="baz")
        assert "xorbits.pandas.pivot will fallback to Pandas" == str(w[0].message)

    expected = pd.pivot(raw, index="foo", columns="bar", values="baz")
    assert str(r) == str(expected)
    assert isinstance(r, DataRef)
    pd.testing.assert_frame_equal(r.to_pandas(), expected)

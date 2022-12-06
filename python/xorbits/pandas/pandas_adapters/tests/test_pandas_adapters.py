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

import warnings

import pandas as pd

from .... import pandas as xpd


def test_dataframe_pivot(setup):
    raw = pd.DataFrame(
        {
            "foo": ["one", "one", "one", "two", "two", "two"],
            "bar": ["A", "B", "C", "A", "B", "C"],
            "baz": [1, 2, 3, 4, 5, 6],
            "zoo": ["x", "y", "z", "q", "w", "t"],
        }
    )
    df = xpd.DataFrame(raw)
    with warnings.catch_warnings(record=True) as w:
        r = df.pivot(index="foo", columns="bar", values="baz")
        assert len(w) == 1
        assert "fallback to Pandas" in str(w[0].message)

    expected = raw.pivot(index="foo", columns="bar", values="baz")
    assert expected.shape == r.shape
    assert str(expected) == str(r)
    pd.testing.assert_frame_equal(expected, r.to_pandas())

    # multi chunk and follow other operations
    df = xpd.DataFrame(raw, chunk_size=3)
    with warnings.catch_warnings(record=True) as w:
        r = df.pivot(index="foo", columns="bar", values="baz").max(axis=0)
        assert "fallback to Pandas" in str(w[0].message)

    expected = raw.pivot("foo", columns="bar", values="baz").max(axis=0)
    assert expected.shape == r.shape
    assert str(expected) == str(r)
    pd.testing.assert_series_equal(expected, r.to_pandas())

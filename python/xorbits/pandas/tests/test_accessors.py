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

from ... import pandas as xpd
from ...core import DataRef
from .. import Series


def test_str_accessor(setup):
    # xs being a xorbits.pandas.core.Series instance
    s = pd.Series(["A_Str"])
    xs = xpd.Series(s)
    assert isinstance(xs, Series)
    pd.testing.assert_series_equal(s.str.split("_"), xs.str.split("_").to_pandas())

    # xs being a xorbits.core.data.DataRef instance
    s = s + "_Series"
    xs = xs + "_Series"
    assert not isinstance(xs, Series)
    assert isinstance(xs, DataRef)
    pd.testing.assert_series_equal(s.str.split("_"), xs.str.split("_").to_pandas())


def test_dt_accessor(setup):
    import numpy as np

    # xs being a xorbits.pandas.core.Series instance
    a = np.arange(
        np.datetime64("2000-01-01"), np.datetime64("2000-01-03"), dtype=np.datetime64
    )
    s = pd.Series(a)
    xs = xpd.Series(a)
    assert isinstance(xs, Series)
    pd.testing.assert_series_equal(s.dt.year, xs.dt.year.to_pandas())

    # xs being a xorbits.core.data.DataRef instance
    s = pd.concat((s, s))
    xs = xpd.concat((xs, xs))
    assert not isinstance(xs, Series)
    assert isinstance(xs, DataRef)
    pd.testing.assert_series_equal(s.dt.year, xs.dt.year.to_pandas())


def test_class_docstring():
    docstring = xpd.Series.str.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from pandas."
    )

    docstring = xpd.Series.str.split.__doc__
    assert docstring is not None and docstring.endswith(
        "pandas.core.strings.accessor.StringMethods."
    )


def test_obj_docstring(setup, dummy_str_series):
    assert isinstance(dummy_str_series, xpd.Series)

    docstring = dummy_str_series.str.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from pandas."
    )

    docstring = dummy_str_series.str.split.__doc__
    assert docstring is not None and docstring.endswith(
        "pandas.core.strings.accessor.StringMethods."
    )

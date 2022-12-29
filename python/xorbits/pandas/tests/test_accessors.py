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
import pytest

from ... import pandas as xpd
from ..accessors import StringAccessor


def test_str_accessor(setup):
    # xs being a xorbits.pandas.core.Series instance
    s = pd.Series(["A_Str"])
    xs = xpd.Series(s)
    pd.testing.assert_series_equal(s.str.split("_"), xs.str.split("_").to_pandas())

    # xs being a xorbits.core.data.DataRef instance
    s = s + "_Series"
    xs = xs + "_Series"
    pd.testing.assert_series_equal(s.str.split("_"), xs.str.split("_").to_pandas())


def test_dt_accessor(setup):
    import numpy as np

    # xs being a xorbits.pandas.core.Series instance
    a = np.arange(
        np.datetime64("2000-01-01"), np.datetime64("2000-01-03"), dtype=np.datetime64
    )
    s = pd.Series(a)
    xs = xpd.Series(a)
    # function.
    pd.testing.assert_series_equal(s.dt.month_name(), xs.dt.month_name().to_pandas())
    # property.
    pd.testing.assert_series_equal(s.dt.year, xs.dt.year.to_pandas())

    # xs being a xorbits.core.data.DataRef instance.
    s = pd.concat((s, s))
    xs = xpd.concat((xs, xs))
    pd.testing.assert_series_equal(s.dt.month_name(), xs.dt.month_name().to_pandas())
    pd.testing.assert_series_equal(s.dt.year, xs.dt.year.to_pandas())


def test_getting_nonexistent_attr(setup):
    accessor = xpd.Series(["A_Str"]).str
    assert isinstance(accessor, StringAccessor)

    with pytest.raises(AttributeError, match="no attribute 'foo'"):
        getattr(accessor, "foo")


def test_cls_docstring():
    docstring = xpd.Series.str.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from pandas."
    )

    docstring = xpd.Series.str.split.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from pandas.core.strings.accessor.StringMethods."
    )

    docstring = xpd.Series.dt.__doc__
    assert docstring == ""

    docstring = xpd.Series.dt.month_name.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from "
        "pandas.core.indexes.accessors.CombinedDatetimelikeProperties."
    )

    docstring = xpd.Series.dt.year.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from "
        "pandas.core.indexes.accessors.CombinedDatetimelikeProperties."
    )


def test_obj_docstring(setup, dummy_str_series, dummy_dt_series, dummy_df):
    assert isinstance(dummy_str_series, xpd.Series)

    docstring = dummy_str_series.str.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from pandas."
    )

    docstring = dummy_str_series.str.split.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from pandas.core.strings.accessor.StringMethods."
    )

    docstring = dummy_dt_series.dt.__doc__
    assert docstring == ""

    # skip dummy_dt_series.dt.year.__doc__ since it is a property.

    docstring = dummy_dt_series.dt.month_name.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from "
        "pandas.core.indexes.accessors.CombinedDatetimelikeProperties."
    )

    assert dummy_df.loc.__doc__ is not None and dummy_df.loc.__doc__.endswith(
        "This docstring was copied from pandas."
    )
    assert dummy_df.iloc.__doc__ is not None and dummy_df.iloc.__doc__.endswith(
        "This docstring was copied from pandas."
    )
    assert dummy_df.at.__doc__ is not None and dummy_df.at.__doc__.endswith(
        "This docstring was copied from pandas."
    )
    assert dummy_df.iat.__doc__ is not None and dummy_df.iat.__doc__.endswith(
        "This docstring was copied from pandas."
    )

    assert xpd.DataFrame.loc.__doc__ is not None and xpd.DataFrame.loc.__doc__.endswith(
        "This docstring was copied from pandas."
    )
    assert (
        xpd.DataFrame.iloc.__doc__ is not None
        and xpd.DataFrame.iloc.__doc__.endswith(
            "This docstring was copied from pandas."
        )
    )
    assert xpd.DataFrame.at.__doc__ is not None and xpd.DataFrame.at.__doc__.endswith(
        "This docstring was copied from pandas."
    )
    assert xpd.DataFrame.iat.__doc__ is not None and xpd.DataFrame.iat.__doc__.endswith(
        "This docstring was copied from pandas."
    )

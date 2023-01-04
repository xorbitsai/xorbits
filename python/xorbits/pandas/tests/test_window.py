# Copyright 2022-2023 XProbe Inc.
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
import pytest

from ... import pandas as xpd


def test_rolling(setup):
    s = pd.Series([1, 2, 3, 4, 5])
    xs = xpd.Series(s)
    expected = s.rolling(3).sum()
    actual = xs.rolling(3).sum().to_pandas()
    pd.testing.assert_series_equal(expected, actual)


@pytest.mark.skip("Incompatible behavior")
def test_ewm(setup):
    df = pd.DataFrame({"B": [0, 1, 2, np.nan, 4]})
    xdf = xpd.DataFrame(df)
    #           B
    # 0  0.000000
    # 1  0.750000
    # 2  1.615385
    # 3  1.615385
    # 4  3.670213
    expected = df.ewm(com=0.5).mean()
    #           B
    #        mean
    # 0  0.000000
    # 1  0.750000
    # 2  1.615385
    # 3  1.615385
    # 4  3.670213
    actual = xdf.ewm(com=0.5).mean().to_pandas()
    pd.testing.assert_frame_equal(expected, actual)


@pytest.mark.skip("Incompatible behavior")
def test_expanding(setup):
    df = pd.DataFrame({"B": [0, 1, 2, np.nan, 4]})
    xdf = xpd.DataFrame(df)
    expected = df.expanding(1).sum()
    actual = xdf.expanding(1).sum()
    pd.testing.assert_frame_equal(expected, actual)


def test_cls_docstring():
    docstring = xpd.window.Rolling.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from pandas."
    )

    docstring = xpd.window.Rolling.sum.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from pandas.core.window.rolling.Rolling."
    )

    docstring = xpd.window.Expanding.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from pandas."
    )

    docstring = xpd.window.Expanding.sum.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from pandas.core.window.expanding.Expanding."
    )

    docstring = xpd.window.ExponentialMovingWindow.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from pandas."
    )

    docstring = xpd.window.ExponentialMovingWindow.mean.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from pandas.core.window.ewm.ExponentialMovingWindow."
    )


def test_obj_docstring(setup, dummy_str_series):
    assert isinstance(dummy_str_series, xpd.Series)

    docstring = dummy_str_series.rolling.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from pandas.core.series.Series."
    )
    docstring = dummy_str_series.ewm.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from pandas.core.series.Series."
    )
    docstring = dummy_str_series.expanding.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from pandas.core.series.Series."
    )

    docstring = dummy_str_series.rolling(3).sum.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from pandas.core.window.rolling.Rolling."
    )
    docstring = dummy_str_series.expanding(3).sum.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from pandas.core.window.expanding.Expanding."
    )
    docstring = dummy_str_series.ewm(com=0.5).mean.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from pandas.core.window.ewm.ExponentialMovingWindow."
    )

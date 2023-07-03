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

import pandas as pd
import pytest

from .... import pandas as xpd

df_raw = pd.DataFrame(
    {
        "A": [1.0] * 100,
        "B": [2.0] * 100,
        "C": pd.date_range(start="2023-06-19", periods=100),
    }
)
df = xpd.DataFrame(df_raw)


def test_options_execute(setup):
    # test pandas options
    pd.set_option("display.max_rows", 100)
    xpd.set_option("display.max_rows", 100)
    assert (
        pd.get_option("display.max_rows") == xpd.get_option("display.max_rows") == 100
    )
    assert df_raw.__str__() == df.__str__()

    # test xorbits options
    xpd.set_option("chunk_store_limit", 100000)
    assert xpd.get_option("chunk_store_limit") == 100000

    # test reset options
    xpd.reset_option("chunk_store_limit")
    assert xpd.get_option("chunk_store_limit") == 134217728
    xpd.reset_option("display.max_rows")
    assert xpd.get_option("display.max_rows") == 60

    # test option_context
    with xpd.option_context("display.max_rows", 100, "chunk_store_limit", 100000):
        assert xpd.get_option("display.max_rows") == 100
        assert xpd.get_option("chunk_store_limit") == 100000

    # test set_eng_float_format
    pd.set_eng_float_format(accuracy=3)
    xpd.set_eng_float_format(accuracy=3)
    assert df_raw.__str__() == df.__str__()

    # test error value
    with pytest.raises(ValueError):
        xpd.set_option("display.max_rows", "100")

    # test error option
    with pytest.raises(pd._config.config.OptionError):
        xpd.set_option("non-exist", None)

    with pytest.raises(pd._config.config.OptionError):
        xpd.get_option("non-exist")

    with pytest.raises(pd._config.config.OptionError):
        xpd.reset_option("non-exist")

    with pytest.raises(ValueError):
        xpd.option_context("non-exist", 100)

    # test invalid type
    with pytest.raises(ValueError):
        xpd.option_context({"display.max_rows": 100})


def test_docstring(setup):
    docstring = xpd.set_option.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from pandas."
    )

    docstring = xpd.get_option.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from pandas."
    )

    docstring = xpd.reset_option.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from pandas."
    )

    docstring = xpd.option_context.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from pandas."
    )

    docstring = xpd.set_eng_float_format.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from pandas."
    )

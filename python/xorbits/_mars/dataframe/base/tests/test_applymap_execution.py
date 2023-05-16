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

from .... import dataframe as md
from ....dataframe.core import DATAFRAME_OR_SERIES_TYPE, DataFrame
from ....dataframe.fetch.core import DataFrameFetch


def test_frame_applymap_execution(setup):
    df = pd.DataFrame([[1, 2.12], [3.356, 4.567]])
    mdf = md.DataFrame(df)

    apply_func = lambda x: len(str(x))

    res = mdf.applymap(apply_func).execute()
    pd.testing.assert_frame_equal(res.fetch(), df.applymap(apply_func))

    apply_func = np.square
    res = mdf.applymap(apply_func).execute()
    pd.testing.assert_frame_equal(res.fetch(), df.applymap(apply_func))


def test_frame_applymap_nan(setup):
    df = pd.DataFrame([[pd.NA, 2.12], [3.356, 4.567]])
    mdf = md.DataFrame(df)

    apply_func = lambda x: len(str(x))
    res = mdf.applymap(apply_func, na_action="ignore").execute()
    pd.testing.assert_frame_equal(
        res.fetch(), df.applymap(apply_func, na_action="ignore")
    )

    apply_func = np.sqrt

    with pytest.raises(TypeError):
        mdf.apply(apply_func, na_action="ignore")

    res = mdf.applymap(apply_func, na_action="ignore", skip_infer=True).execute()
    pd.testing.assert_frame_equal(
        res.fetch(), df.applymap(apply_func, na_action="ignore")
    )

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
import pytest

from ...core.data import DataRef
from ...numpy import ndarray
from ...pandas import DataFrame, Index, Series
from ...pandas.groupby import DataFrameGroupBy, SeriesGroupBy


def test_instance_check(dummy_df, dummy_int_series, dummy_int_2d_array):
    # DataFrame
    assert dummy_df.__class__ is DataFrame
    assert isinstance(dummy_df, DataFrame)
    assert isinstance(dummy_df, DataRef)

    computed = dummy_df + dummy_df
    assert computed.__class__ is DataRef
    assert isinstance(computed, DataFrame)

    # Series
    assert dummy_int_series.__class__ is Series
    assert isinstance(dummy_int_series, Series)
    assert isinstance(dummy_df, DataRef)

    computed = dummy_int_series + dummy_int_series
    assert computed.__class__ is DataRef
    assert isinstance(computed, Series)

    # Index
    index = Index([1, 2, 3])
    assert index.__class__ is Index
    assert isinstance(index, Index)

    # DataFrameGroupBy
    grouped = dummy_df.groupby("foo")
    assert grouped.__class__ is DataRef
    assert isinstance(grouped, DataFrameGroupBy)

    # SeriesGroupBy
    grouped = dummy_int_series.groupby()
    assert grouped.__class__ is DataRef
    assert isinstance(grouped, SeriesGroupBy)

    # tensor
    assert dummy_int_2d_array.__class__ is DataRef
    assert isinstance(dummy_int_2d_array, ndarray)

    computed = dummy_int_2d_array + 1
    assert computed.__class__ is DataRef
    assert isinstance(computed, ndarray)


def test_instance_check_on_illegal_subclass():
    class MyDataFrame(DataFrame):
        pass

    my_df = MyDataFrame({"foo": (1, 2, 3)})
    isinstance(my_df, MyDataFrame)

    assert isinstance(my_df + 1, DataFrame)
    with pytest.raises(TypeError):
        isinstance(my_df + 1, MyDataFrame)

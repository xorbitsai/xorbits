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

from ...pandas import get_dummies
from ..execution import need_to_execute, run


def test_deferred_execution_repr(setup, dummy_df):
    # own data, no execution
    assert need_to_execute(dummy_df)
    s = repr(dummy_df)
    assert need_to_execute(dummy_df)
    assert s == repr(dummy_df.to_pandas())

    # trigger execution
    r = dummy_df.sum(axis=0)
    s = repr(r)
    assert not need_to_execute(r)
    assert s == repr(r.to_pandas())


def test_deferred_execution_print(setup, dummy_df):
    # own data, no execution
    assert need_to_execute(dummy_df)
    s = str(dummy_df)
    assert need_to_execute(dummy_df)
    assert s == repr(dummy_df.to_pandas())

    # trigger execution
    r = dummy_df.sum(axis=0)
    s = str(r)
    assert not need_to_execute(r)
    assert s == repr(r.to_pandas())


def test_deferred_execution_iterrows(setup, dummy_df):
    assert need_to_execute(dummy_df)
    # own data, no execution
    for _ in dummy_df.iterrows():
        pass
    assert need_to_execute(dummy_df)

    # trigger execution
    sorted_df = dummy_df.sort_values(by="foo")
    for _ in sorted_df.iterrows():
        pass
    assert not need_to_execute(sorted_df)


def test_deferred_execution_itertuples(setup, dummy_df):
    assert need_to_execute(dummy_df)
    # own data, no execution
    for _ in dummy_df.itertuples():
        pass
    assert need_to_execute(dummy_df)

    # trigger execution
    sorted_df = dummy_df.sort_values(by="foo")
    for _ in sorted_df.itertuples():
        pass
    assert not need_to_execute(sorted_df)


def test_deferred_execution_transpose_1(setup, dummy_df):
    # transpose.
    transposed = dummy_df.transpose()
    assert need_to_execute(transposed)
    s = str(transposed)
    assert not need_to_execute(transposed)
    assert s == str(transposed.to_pandas())


def test_deferred_execution_transpose_2(setup, dummy_df):
    transposed = dummy_df.T
    assert need_to_execute(transposed)
    s = str(transposed)
    assert not need_to_execute(transposed)
    assert s == str(transposed.to_pandas())


def test_deferred_execution_get_dummies(setup, dummy_df):
    dummy_encoded = get_dummies(dummy_df)
    assert need_to_execute(dummy_encoded)
    s = str(dummy_encoded)
    assert not need_to_execute(dummy_encoded)
    assert s == str(dummy_encoded.to_pandas())


def test_deferred_execution_groupby_apply(setup, dummy_df):
    groupby_applied = dummy_df.groupby("foo").apply(lambda df: df.sum())
    assert need_to_execute(groupby_applied)
    s = str(groupby_applied)
    assert not need_to_execute(groupby_applied)
    assert s == str(groupby_applied.to_pandas())


def test_manual_execution(setup, dummy_int_series):
    assert need_to_execute(dummy_int_series)
    run(dummy_int_series)
    assert not need_to_execute(dummy_int_series)

    series_to_execute = [dummy_int_series + i for i in range(3)]
    assert all([need_to_execute(ref) for ref in series_to_execute])
    run(series_to_execute)
    assert not any([need_to_execute(ref) for ref in series_to_execute])

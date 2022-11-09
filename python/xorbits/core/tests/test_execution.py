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

from ...pandas import get_dummies
from ..execution import need_to_execute


def test_deferred_execution_repr(setup, dummy_df):
    assert need_to_execute(dummy_df)
    repr(dummy_df)
    assert not need_to_execute(dummy_df)


def test_deferred_execution_print(setup, dummy_df):
    assert need_to_execute(dummy_df)
    print(dummy_df)
    assert not need_to_execute(dummy_df)


def test_deferred_execution_iterrows(setup, dummy_df):
    assert need_to_execute(dummy_df)
    for (_, _) in dummy_df.iterrows():
        pass
    assert not need_to_execute(dummy_df)


def test_deferred_execution_itertuples(setup, dummy_df):
    assert need_to_execute(dummy_df)
    for _ in dummy_df.itertuples():
        pass
    assert not need_to_execute(dummy_df)


def test_deferred_execution_transpose_1(setup, dummy_df):
    # transpose.
    transposed = dummy_df.transpose()
    assert need_to_execute(transposed)
    print(transposed)
    assert not need_to_execute(transposed)


def test_deferred_execution_transpose_2(setup, dummy_df):
    transposed = dummy_df.T
    assert need_to_execute(transposed)
    print(transposed)
    assert not need_to_execute(transposed)


def test_deferred_execution_get_dummies(setup, dummy_df):
    dummy_encoded = get_dummies(dummy_df)
    assert need_to_execute(dummy_encoded)
    print(dummy_encoded)
    assert not need_to_execute(dummy_encoded)


def test_deferred_execution_groupby_apply(setup, dummy_df):
    groupby_applied = dummy_df.groupby("foo").apply(lambda df: df.sum())
    assert need_to_execute(groupby_applied)
    print(groupby_applied)
    assert not need_to_execute(groupby_applied)

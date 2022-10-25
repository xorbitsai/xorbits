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

from ...dataframe import get_dummies
from ..execution import need_to_execute


def test_deferred_execution_repr(dummy_xdf):
    assert need_to_execute(dummy_xdf)
    repr(dummy_xdf)
    assert not need_to_execute(dummy_xdf)


def test_deferred_execution_print(dummy_xdf):
    assert need_to_execute(dummy_xdf)
    print(dummy_xdf)
    assert not need_to_execute(dummy_xdf)


def test_deferred_execution_iterrows(dummy_xdf):
    assert need_to_execute(dummy_xdf)
    for (_, _) in dummy_xdf.iterrows():
        pass
    assert not need_to_execute(dummy_xdf)


def test_deferred_execution_itertuples(dummy_xdf):
    assert need_to_execute(dummy_xdf)
    for _ in dummy_xdf.itertuples():
        pass
    assert not need_to_execute(dummy_xdf)


def test_deferred_execution_transpose_1(dummy_xdf):
    # transpose.
    transposed = dummy_xdf.transpose()
    assert need_to_execute(transposed)
    print(transposed)
    assert not need_to_execute(transposed)


def test_deferred_execution_transpose_2(dummy_xdf):
    transposed = dummy_xdf.T
    assert need_to_execute(transposed)
    print(transposed)
    assert not need_to_execute(transposed)


def test_deferred_execution_get_dummies(dummy_xdf):
    dummy_encoded = get_dummies(dummy_xdf)
    assert need_to_execute(dummy_encoded)
    print(dummy_encoded)
    assert not need_to_execute(dummy_encoded)


def test_deferred_execution_groupby_apply(dummy_xdf):
    groupby_applied = dummy_xdf.groupby("foo").apply(lambda df: df.sum())
    assert need_to_execute(groupby_applied)
    print(groupby_applied)
    assert not need_to_execute(groupby_applied)

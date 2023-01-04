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

import unittest.mock as mock

from ..core.execution import need_to_execute
from .mock_pydevd_xml import var_to_xml


@mock.patch("xorbits.utils.is_pydev_evaluating_value", side_effect=lambda: True)
def test_safe_repr_str(setup, dummy_df):
    df = dummy_df + 1
    assert "xorbits.core.data.DataRef object" in repr(df)
    assert "xorbits.core.data.DataRef object" in str(df)
    assert need_to_execute(df)


def test_safe_repr_str_for_mock_pydevd_xml(setup, dummy_df):
    df = dummy_df + 1
    assert "xorbits.core.data.DataRef object" in var_to_xml(df)

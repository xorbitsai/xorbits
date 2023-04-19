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


@pytest.mark.parametrize(
    "test_dict",
    [
        "dummy_dict_int",
        "dummy_dict_float",
        "dummy_dict_str",
        "dummy_dict_bool",
        "dummy_dict_mixed",
    ],
)
def test_array_conversion(test_dict, request):
    test_dict = request.getfixturevalue(test_dict)
    df = pd.DataFrame(test_dict)
    xdf = xpd.DataFrame(test_dict)

    expected = df.__array__()
    np.testing.assert_array_equal(xdf.__array__(), expected)
    np.testing.assert_array_equal(np.array(xdf), expected)

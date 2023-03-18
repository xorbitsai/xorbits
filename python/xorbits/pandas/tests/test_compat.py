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

from ... import numpy as xnp
from ... import pandas as xpd

def test_array_conversion(setup, dummy_dict_num, dummy_dict_str, dummy_dict_bool, dummy_dict_mixed):
    dicts = [dummy_dict_num, dummy_dict_str, dummy_dict_bool, dummy_dict_mixed]

    for dict in dicts:
        df = pd.DataFrame(dict)
        xdf = xpd.DataFrame(dict)

        assert np.array_equal(df.__array__(), xdf.__array__())
        assert np.array_equal(np.array(df), np.array(xdf))


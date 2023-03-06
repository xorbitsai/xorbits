# Copyright 2022-2023 XProbe Inc.
# derived from copyright 1999-2021 Alibaba Group Holding Ltd.
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

import re

import numpy as np
import pandas as pd
import pytest

from ....core.adapter import MARS_DATAFRAME_TYPE, MARS_OBJECT_TYPE, MarsOutputType
from ..fallback import unimplemented_func, wrap_fallback_module_method


def test_unimplemented_func():
    with pytest.raises(NotImplementedError):
        unimplemented_func()


def test_wrap_fallback_module_method(setup):
    np_wrap = wrap_fallback_module_method(
        np.random, "default_rng", MarsOutputType.object, "Test Numpy"
    )
    assert callable(np_wrap)

    raw = pd.DataFrame(
        [["a", "b"], ["c", "d"]], index=["row 1", "row 2"], columns=["col 1", "col 2"]
    )
    pd_wrap = wrap_fallback_module_method(
        pd, "read_json", MarsOutputType.dataframe, "Test Pandas"
    )
    assert callable(pd_wrap)

    with pytest.warns(Warning) as w:
        np_remote = np_wrap()
        assert "Test Numpy" == str(w[0].message)
        assert re.match("RemoteFunction <key=.*>", str(np_remote.op))
        assert type(np_remote) in MARS_OBJECT_TYPE
        assert type(np_remote.fetch()) == type(np.random.default_rng())

    with pytest.warns(Warning) as w:
        pd_remote = pd_wrap(raw.to_json())
        assert "Test Pandas" == str(w[0].message)
        assert re.match("RemoteFunction <key=.*>", str(pd_remote.op))
        assert type(pd_remote) in MARS_DATAFRAME_TYPE
        assert type(pd_remote.fetch()) == type(pd.read_json(raw.to_json()))

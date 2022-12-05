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

import re

from xorbits.core import DataRef

from .... import pandas as xpd
from .... import remote


def test_spawn(setup):
    def to_str(df):
        return str(df)

    # scaler arguments
    ret = remote.spawn(to_str, (1,))
    assert isinstance(ret, DataRef)

    assert re.match("Object <op=RemoteFunction, key=.*>", str(ret))
    assert ret.to_object() == "1"

    xdf = xpd.DataFrame((1, 2, 3))
    ret = remote.spawn(to_str, (xdf,))
    assert isinstance(ret, DataRef)
    assert re.match("Object <op=RemoteFunction, key=.*>", str(ret))
    assert ret.to_object() == "   0\n0  1\n1  2\n2  3"

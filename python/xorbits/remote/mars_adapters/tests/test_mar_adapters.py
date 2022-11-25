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

from xorbits.core import DataRef

from .... import remote


def test_spawn(setup):
    def inc(i: int):
        return i + 1

    ret = remote.spawn(inc, (1,))
    assert isinstance(ret, DataRef)
    import re

    assert re.match("Object <op=RemoteFunction, key=.*>", str(ret))
    assert ret.fetch() == 2


def test_run_script(setup):
    import io

    script = io.StringIO('print("hello world!")')

    ret = remote.run_script(script)
    assert isinstance(ret, DataRef)
    import re

    assert re.match("Object <op=RunScript, key=.*>", str(ret))
    assert ret.fetch() == {"status": "ok"}

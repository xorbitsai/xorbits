# Copyright 1999-2022 Alibaba Group Holding Ltd.
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

import xorbits.remote as xr
from xorbits import init, shutdown, run


class EmptyRemotesExecutionSuite:
    """
    Benchmark that times running a number of empty subtasks
    """

    def setup(self):
        init()

    def teardown(self):
        shutdown()

    def time_remotes(self):
        def empty_fun(_i):
            pass

        remotes = [xr.spawn(empty_fun, args=(i,)) for i in range(1000)]
        run(remotes)

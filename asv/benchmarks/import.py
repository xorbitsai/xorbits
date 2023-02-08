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

import subprocess
import sys

# make sure necessary pyc files generated
import xorbits.pandas as xpd
import xorbits.numpy as xnp

del xpd, xnp


class ImportPackageSuite:
    """
    Benchmark that times performance of chunk graph builder
    """

    def time_import_xorbits(self):
        proc = subprocess.Popen([sys.executable, "-c", "import xorbits"])
        proc.wait(120)

    def time_import_xorbits_numpy(self):
        proc = subprocess.Popen([sys.executable, "-c", "import xorbits.numpy"])
        proc.wait(120)

    def time_import_xorbits_pandas(self):
        proc = subprocess.Popen([sys.executable, "-c", "import xorbits.tensor"])
        proc.wait(120)

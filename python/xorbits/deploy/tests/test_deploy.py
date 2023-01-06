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

import concurrent.futures
import sys

import psutil
import pytest

from ... import pandas as pd
from ..._mars.oscar.backends.router import Router
from .. import init, shutdown


def _safe_shutdown():
    try:
        shutdown(isolation=False)
    except concurrent.futures.TimeoutError:
        Router.set_instance(None)
        subprocesses = psutil.Process().children(recursive=True)
        for proc in subprocesses:
            proc.terminate()
        for proc in subprocesses:
            try:
                proc.wait(1)
            except (psutil.TimeoutExpired, psutil.NoSuchProcess):
                pass
            try:
                proc.kill()
            except psutil.NoSuchProcess:
                pass


@pytest.mark.skipif(
    sys.platform == "win32" and sys.version_info[:2] < (3, 8),
    reason="Skip for windows & Python < 3.8",
)
def test_init():
    init(n_cpu=2)
    try:
        assert repr(pd.Series([1, 2, 3]).sum()) == "6"
        init()
        with pytest.raises(ValueError):
            init(init_local=True)
    finally:
        _safe_shutdown()

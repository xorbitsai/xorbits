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

import os
import subprocess
import sys
import tempfile
import time
from typing import List

import psutil

from .. import init
from .._mars.utils import get_next_port
from ..core.api import API

CONFIG_CONTENT = """\
"@inherits": "@default"
scheduling:
  mem_hard_limit: null"""
# 100 sec to timeout
TIMEOUT = 100


def _stop_processes(procs: List[subprocess.Popen]):
    sub_ps_procs = []
    for proc in procs:
        if not proc:
            continue

        try:
            sub_ps_procs.extend(psutil.Process(proc.pid).children(recursive=True))
        except psutil.NoSuchProcess:
            continue
        proc.terminate()

    for proc in procs:
        try:
            proc.wait(5)
        except subprocess.TimeoutExpired:
            pass

    for ps_proc in sub_ps_procs + procs:
        try:
            ps_proc.kill()
        except psutil.NoSuchProcess:
            pass


def test_cluster(dummy_df):
    port = get_next_port()
    web_port = get_next_port()
    supervisor_addr = f"127.0.0.1:{port}"
    web_addr = f"http://127.0.0.1:{web_port}"

    # gen config file
    fd, path = tempfile.mkstemp()
    with os.fdopen(fd, mode="w") as f:
        f.write(CONFIG_CONTENT)

    supervisor_process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "xorbits.supervisor",
            "-H",
            "127.0.0.1",
            "-p",
            str(port),
            "-w",
            str(web_port),
            "-f",
            path,
        ]
    )
    worker_process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "xorbits.worker",
            "-s",
            supervisor_addr,
            "-f",
            path,
        ]
    )

    try:
        for p in [supervisor_process, worker_process]:
            try:
                retcode = p.wait(1)
            except subprocess.TimeoutExpired:
                # supervisor & worker will run forever,
                # timeout means everything goes well, at least looks well.
                continue
            else:
                if retcode:
                    std_err = p.communicate()[1].decode()
                    raise RuntimeError("Start cluster failed, stderr: \n" + std_err)

        start = time.time()
        is_timeout = True
        while time.time() - start <= TIMEOUT:
            try:
                init(web_addr)
                api = API.create()
                n_worker = len(api.list_workers())
                if n_worker == 0:
                    raise RuntimeError("Cluster not ready")
            except:  # noqa: E722  # nosec  # pylint: disable=bare-except
                time.sleep(0.5)
                continue

            assert repr(dummy_df.foo.sum()) == "3"
            is_timeout = False
            break
        if is_timeout:
            raise TimeoutError
    finally:
        _stop_processes([supervisor_process, worker_process])

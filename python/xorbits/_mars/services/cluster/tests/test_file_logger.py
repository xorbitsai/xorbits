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
import shutil
import tempfile

import pytest
import xoscar as mo

from ....constants import (
    DEFAULT_MARS_LOG_FILE_NAME,
    MARS_LOG_DIR_KEY,
    MARS_TMP_DIR_PREFIX,
)
from ....oscar import create_actor_pool
from ..file_logger import FileLoggerActor

full_content = "qwert\nasdfg\nzxcvb\nyuiop\nhjkl;\nnm,./"


@pytest.fixture
async def actor_pool():
    # prepare
    mars_tmp_dir = tempfile.mkdtemp(prefix=MARS_TMP_DIR_PREFIX)
    os.environ[MARS_LOG_DIR_KEY] = mars_tmp_dir
    pool = await create_actor_pool("127.0.0.1", n_process=0)
    async with pool:
        yield pool

    shutil.rmtree(mars_tmp_dir)


@pytest.mark.asyncio
async def test_file_logger(actor_pool):
    pool_addr = actor_pool.external_address
    logger_ref = await mo.create_actor(
        FileLoggerActor,
        uid=FileLoggerActor.default_uid(),
        address=pool_addr,
    )

    log_dir = os.environ.get(MARS_LOG_DIR_KEY)
    assert log_dir is not None

    log_file = os.path.join(log_dir, DEFAULT_MARS_LOG_FILE_NAME)
    with open(log_file, "w", newline="\n") as f:
        f.write(full_content)

    byte_num = 5
    expected_data = ""
    content = await logger_ref.fetch_logs(byte_num, 0)
    assert content == expected_data

    byte_num = 6
    expected_data = "nm,./"
    content = await logger_ref.fetch_logs(byte_num, 0)
    assert content == expected_data

    byte_num = 11
    expected_data = "nm,./"
    content = await logger_ref.fetch_logs(byte_num, 0)
    assert content == expected_data

    byte_num = 12
    expected_data = "hjkl;\nnm,./"
    content = await logger_ref.fetch_logs(byte_num, 0)
    assert content == expected_data

    byte_num = 50
    expected_data = "qwert\nasdfg\nzxcvb\nyuiop\nhjkl;\nnm,./"
    content = await logger_ref.fetch_logs(byte_num, 0)
    assert content == expected_data

    byte_num = -1
    expected_data = "qwert\nasdfg\nzxcvb\nyuiop\nhjkl;\nnm,./"
    content = await logger_ref.fetch_logs(byte_num, 0)
    assert content == expected_data

    byte_num = -1
    offset = 1
    expected_data = "wert\nasdfg\nzxcvb\nyuiop\nhjkl;\nnm,./"
    content = await logger_ref.fetch_logs(byte_num, offset)
    assert content == expected_data

    offset = 35
    expected_data = ""
    content = await logger_ref.fetch_logs(byte_num, offset)
    assert content == expected_data

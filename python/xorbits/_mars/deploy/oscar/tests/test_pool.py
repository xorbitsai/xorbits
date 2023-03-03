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

import logging
import os
import tempfile

import pytest

from ....constants import MARS_LOG_DIR_KEY, MARS_TMP_DIR_PREFIX
from ..pool import (
    _config_logging,
    _get_root_logger_level_and_format,
    _parse_file_logging_config,
)


@pytest.fixture
def init():
    root_level, _ = _get_root_logger_level_and_format()
    file_logging_config = os.path.join(
        os.path.dirname(__file__), "..", "file-logging.conf"
    )
    logger_sections = [
        "logger_main",
        "logger_deploy",
        "logger_oscar",
        "logger_services",
        "logger_dataframe",
        "logger_learn",
        "logger_tensor",
        "handler_file_handler",
    ]
    yield file_logging_config, logger_sections, root_level


def test_parse_file_logging_config(init):
    fp, sections, root_level = init
    log_path = "mock_path"
    config = _parse_file_logging_config(fp, log_path, "FATAL")
    assert config["handler_stream_handler"]["level"] == root_level
    assert config["handler_stream_handler"].get("formatter") is not None
    assert config["handler_stream_handler"]["formatter"] == "console"
    for sec in sections:
        if sec != "handler_file_handler":
            assert config[sec]["level"] == "FATAL"
        else:
            assert config[sec]["level"] == root_level

    formatter = "foo"
    config = _parse_file_logging_config(fp, log_path, "FATAL", formatter=formatter)
    assert config["formatter_formatter"]["format"] == formatter

    config = _parse_file_logging_config(fp, log_path, level="", formatter=formatter)
    assert config["logger_dataframe"]["level"] == "DEBUG"

    config = _parse_file_logging_config(
        fp, log_path, level="", formatter=formatter, from_cmd=True
    )
    assert config["logger_tensor"]["level"] == "DEBUG"

    assert config["handler_stream_handler"]["level"] == "DEBUG"
    assert config["formatter_formatter"]["format"] == formatter


def test_config_logging(init, caplog):
    _, _, root_level = init
    kwargs = {"logging_conf": {}}
    with caplog.at_level(logging.DEBUG):
        _config_logging(**kwargs)
    log_path = os.environ.get(MARS_LOG_DIR_KEY)
    assert log_path is not None
    assert os.path.basename(os.path.dirname(log_path)).startswith(MARS_TMP_DIR_PREFIX)

    with tempfile.TemporaryDirectory() as folder:
        kwargs = {"logging_conf": {"log_dir": folder, "from_cmd": True}}
        _config_logging(**kwargs)
        log_path = os.environ.get(MARS_LOG_DIR_KEY)
        assert log_path is not None
        assert os.path.dirname(os.path.dirname(log_path)) == folder

        cnt = 0
        file_handler = None
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.FileHandler):
                cnt += 1
                file_handler = handler
        assert cnt == 1
        assert file_handler is not None
        assert file_handler.level == logging.getLevelName("DEBUG")
        assert file_handler.baseFilename == os.environ.get(MARS_LOG_DIR_KEY)


def test_pool_with_no_web_config(init):
    kwargs = {"web": False}
    _config_logging(**kwargs)
    log_path = os.environ.get(MARS_LOG_DIR_KEY)
    assert log_path is None


# TODO: test non existent logging config file
# TODO: test non existent log dir
# TODO: test default log dir

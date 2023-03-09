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
import sys
import tempfile

import pytest

from ....constants import (
    DEFAULT_MARS_LOG_BACKUP_COUNT,
    DEFAULT_MARS_LOG_DIR,
    DEFAULT_MARS_LOG_DIR_WIN,
    DEFAULT_MARS_LOG_MAX_BYTES,
    MARS_LOG_DIR_KEY,
)
from ..pool import (
    _config_logging,
    _get_default_logging_config_path,
    _get_log_config_path,
    _get_log_dir,
    _get_or_create_default_log_dir,
    _get_root_logger_level_and_format,
    _parse_file_logging_config,
)


def test_parse_file_logging_config():
    with tempfile.TemporaryDirectory(prefix="mars_test_") as tempdir:
        logging_conf = {
            "from_cmd": True,
            "level": "ERROR",
            "format": "%(asctime)s %(clientip)-15s %(user)-8s %(message)s",
            "log_dir": tempdir,
        }

        parsed_logging_conf = _parse_file_logging_config(logging_conf)

        # check if log level has been applied.
        loggers = parsed_logging_conf["loggers"]["keys"].split(",")
        for lg in loggers:
            lg = lg.strip()
            assert parsed_logging_conf[f"logger_{lg}"]["level"] == logging_conf["level"]

        # check if log format has been applied.
        formatters = parsed_logging_conf["formatters"]["keys"].split(",")
        for formatter in formatters:
            formatter = formatter.strip()
            assert (
                parsed_logging_conf[f"formatter_{formatter}"]["format"]
                == logging_conf["format"]
            )

        # check if log dir has been applied.
        assert "handler_file_handler" in parsed_logging_conf
        assert tempdir in parsed_logging_conf["handler_file_handler"]["args"]


def test_interactive_env_parse_file_logging_config(caplog):
    with tempfile.TemporaryDirectory(prefix="mars_test_") as tempdir:
        # check if config by logging has the highest priority.
        with caplog.at_level(logging.ERROR):
            logging_conf = {
                "from_cmd": False,
                "level": "DEBUG",
                "log_dir": tempdir,
            }
            parsed_logging_conf = _parse_file_logging_config(logging_conf)
            assert "handler_stream_handler" in parsed_logging_conf
            assert parsed_logging_conf["handler_stream_handler"]["level"] == "ERROR"


def test_user_defined_log_dir_and_file_logging_config():
    with tempfile.TemporaryDirectory(prefix="mars_test_") as tempdir:
        logging_config_file = os.path.join(tempdir, "test-file-logging.conf")
        with open(logging_config_file, "w") as fd:
            content = """[loggers]
keys=root

[handlers]
keys=stream_handler

[formatters]
keys=formatter

[logger_root]
level=WARN
handlers=stream_handler

[handler_stream_handler]
class=StreamHandler
formatter=formatter
level=DEBUG
args=(sys.stderr,)

[formatter_formatter]
format=%(asctime)s %(name)-12s %(process)d %(levelname)-8s %(message)s
            """
            fd.write(content)
        logging_conf = {"log_dir": tempdir, "log_config": logging_config_file}

        with pytest.raises(ValueError, match="Unable to change the log directory"):
            _parse_file_logging_config(logging_conf)


def test_get_root_logger_level_and_format(caplog):
    with caplog.at_level(logging.DEBUG):
        level, fmt = _get_root_logger_level_and_format()
        assert level == "DEBUG"


def test_config_logging(caplog):
    with tempfile.TemporaryDirectory(prefix="mars_test_") as tempdir:
        # non-interactive mode.
        logging_conf = {
            "from_cmd": True,
            "level": "INFO",
            "log_dir": tempdir,
        }
        kwargs = {"logging_conf": logging_conf}
        _config_logging(**kwargs)

        log_path = os.environ.get(MARS_LOG_DIR_KEY)
        assert log_path is not None
        assert log_path.startswith(tempdir)

        # check if root logger's level is DEBUG.
        logger = logging.getLogger()
        assert logger.level == logging.INFO
        # check if root logger's handlers are as expected.
        assert len(logger.handlers) == 2
        assert isinstance(logger.handlers[0], logging.StreamHandler)
        assert logger.handlers[0].level == logging.WARN
        assert isinstance(logger.handlers[1], logging.handlers.RotatingFileHandler)
        assert logger.handlers[1].level == logging.DEBUG
        assert logger.handlers[1].baseFilename.startswith(tempdir)
        assert logger.handlers[1].mode == "a"
        assert logger.handlers[1].maxBytes == DEFAULT_MARS_LOG_MAX_BYTES
        assert logger.handlers[1].backupCount == DEFAULT_MARS_LOG_BACKUP_COUNT

        # interactive mode.
        logging_conf = {
            "from_cmd": False,
            "level": "INFO",
            "log_dir": tempdir,
        }
        kwargs = {"logging_conf": logging_conf}
        with caplog.at_level(logging.ERROR):
            _config_logging(**kwargs)

            # check if root logger's level is DEBUG.
            logger = logging.getLogger()
            assert logger.level == logging.INFO
            # check if root logger's handlers are as expected.
            assert len(logger.handlers) == 2
            assert isinstance(logger.handlers[0], logging.StreamHandler)
            # the level of stream handler should be changed.
            assert logger.handlers[0].level == logging.ERROR
            assert isinstance(logger.handlers[1], logging.handlers.RotatingFileHandler)
            # the level of file handler should not be changed.
            assert logger.handlers[1].level == logging.DEBUG
            assert logger.handlers[1].baseFilename.startswith(tempdir)
            assert logger.handlers[1].mode == "a"
            assert logger.handlers[1].maxBytes == DEFAULT_MARS_LOG_MAX_BYTES
            assert logger.handlers[1].backupCount == DEFAULT_MARS_LOG_BACKUP_COUNT


def test_non_existent_file_logging_config():
    with pytest.raises(RuntimeError, match="Logging configuration file does not exist"):
        _get_log_config_path("/path/to/non_existent.conf")


def test_non_existent_log_dir():
    with pytest.raises(RuntimeError, match="Log directory does not exist"):
        _get_log_dir("/path/to/non_existent/")


def test_default_log_dir():
    default_log_dir = _get_or_create_default_log_dir()
    if sys.platform.startswith("win"):
        assert default_log_dir == DEFAULT_MARS_LOG_DIR_WIN
    else:
        assert default_log_dir == DEFAULT_MARS_LOG_DIR
    assert os.path.exists(default_log_dir) and os.path.isdir(default_log_dir)


def test_default_file_logging_config():
    default_logging_config_path = _get_default_logging_config_path()
    assert os.path.exists(default_logging_config_path)

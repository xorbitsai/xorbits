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

import unittest.mock as mock

from ..core.execution import need_to_execute
from ..utils import (
    get_default_logging_config_file_absolute,
    get_local_package_version,
    get_local_py_version,
    get_xorbits_root_absolute,
)
from .mock_pydevd_xml import var_to_xml


@mock.patch("xorbits.utils.is_pydev_evaluating_value", side_effect=lambda: True)
def test_safe_repr_str(setup, dummy_df):
    df = dummy_df + 1
    assert "xorbits.core.data.DataRef object" in repr(df)
    assert "xorbits.core.data.DataRef object" in str(df)
    assert need_to_execute(df)


def test_safe_repr_str_for_mock_pydevd_xml(setup, dummy_df):
    df = dummy_df + 1
    assert "xorbits.core.data.DataRef object" in var_to_xml(df)


def test_get_local_py_version():
    ret = get_local_py_version()
    assert ret.startswith("3")


def test_get_local_package_version():
    pytest_version = get_local_package_version("pytest")
    assert pytest_version is not None

    ret = get_local_package_version("yptset")
    assert ret is None


def test_get_xorbits_root_absolute():
    root = get_xorbits_root_absolute()
    assert root.is_absolute()
    assert root.name == "xorbits"


def test_get_default_logging_config_file():
    default_logging_config_file = get_default_logging_config_file_absolute()
    assert default_logging_config_file.exists()
    assert default_logging_config_file.is_absolute()
    assert default_logging_config_file.name == "file-logging.conf"
    assert default_logging_config_file.parent.name == "oscar"
    assert default_logging_config_file.parent.parent.name == "deploy"
    assert default_logging_config_file.parent.parent.parent.name == "xorbits"

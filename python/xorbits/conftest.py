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

import pytest
from xoscar.backends.router import Router

from . import numpy as np
from . import pandas as pd
from ._mars.config import option_context
from ._mars.deploy.oscar.session import clear_default_session
from .tests.core import init_test


def pytest_addoption(parser):
    parser.addoption(
        "--gpu", action="store_true", default=False, help="run test cases on GPU."
    )


@pytest.fixture
def gpu(request):
    return request.config.option.gpu


@pytest.fixture
def doctest_namespace():
    return {"pd": pd, "np": np}


@pytest.fixture
def empty_df():
    return pd.DataFrame()


@pytest.fixture
def dummy_df():
    return pd.DataFrame({"foo": (0, 1, 2), "bar": ("a", "b", "c")})


@pytest.fixture
def dummy_int_series():
    return pd.Series([1, 2, 3, 4, 5])


@pytest.fixture
def dummy_str_series():
    return pd.Series(["foo", "bar", "baz"])


@pytest.fixture
def dummy_dt_series():
    return pd.Series(pd.date_range("2000-01-01", periods=3, freq="s"))


@pytest.fixture
def dummy_int_1d_array():
    return np.array([0, 1, 2])


@pytest.fixture
def dummy_int_2d_array():
    return np.arange(9).reshape(3, 3)


@pytest.fixture(scope="package")
def _setup_test_session():
    sess = init_test(
        address="test://127.0.0.1",
        backend="mars",
        init_local=True,
        default=True,
        web=False,
        timeout=300,
    )
    with option_context({"show_progress": False}):
        try:
            yield sess
        finally:
            sess.stop_server(isolation=False)
            Router.set_instance(None)


@pytest.fixture
def setup(_setup_test_session):
    _setup_test_session.as_default()
    yield _setup_test_session
    clear_default_session()


@pytest.fixture
def _setup_with_web():
    sess = init_test(web=True)
    try:
        yield sess
    finally:
        sess.stop_server(isolation=False)
        Router.set_instance(None)


@pytest.fixture
def setup_with_web(_setup_with_web):
    _setup_with_web.as_default()
    yield _setup_with_web
    clear_default_session()

# Copyright 2022-2023 XProbe Inc.
# derived from copyright 1999-2021 Alibaba Group Holding Ltd.
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
import os
import time

import psutil
import pytest
from xoscar.aio.lru import clear_all_alru_caches
from xoscar.backends.router import Router

from .config import option_context
from .core.mode import is_build_mode, is_kernel_mode
from .utils import lazy_import

ray = lazy_import("ray")
MARS_CI_BACKEND = os.environ.get("MARS_CI_BACKEND", "mars")


@pytest.fixture(autouse=True)
def auto_cleanup(request):
    request.addfinalizer(clear_all_alru_caches)


@pytest.fixture(scope="module", autouse=True)
def check_router_cleaned(request):
    def route_checker():
        if Router.get_instance() is not None:
            assert len(Router.get_instance()._mapping) == 0
            assert len(Router.get_instance()._local_mapping) == 0

    request.addfinalizer(route_checker)


@pytest.fixture
def stop_mars():
    try:
        yield
    finally:
        import xorbits._mars

        xorbits._mars.stop_server()


@pytest.fixture(scope="module")
def _new_test_session(check_router_cleaned):
    from .deploy.oscar.tests.session import new_test_session

    sess = new_test_session(
        address="test://127.0.0.1",
        backend=MARS_CI_BACKEND,
        init_local=True,
        default=True,
        timeout=300,
    )
    with option_context({"show_progress": False}):
        try:
            yield sess
        finally:
            sess.stop_server(isolation=False)
            Router.set_instance(None)


@pytest.fixture(scope="module")
def _new_integrated_test_session(check_router_cleaned):
    from .deploy.oscar.tests.session import new_test_session

    sess = None
    for i in range(3):
        try:
            sess = new_test_session(
                address="127.0.0.1",
                backend=MARS_CI_BACKEND,
                init_local=True,
                n_worker=2,
                default=True,
                timeout=300,
            )
        except ChildProcessError:
            time.sleep(1)
            if i == 2:
                raise
            else:
                continue
        else:
            break
    with option_context({"show_progress": False}):
        try:
            yield sess
        finally:
            try:
                sess.stop_server(isolation=False)
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


@pytest.fixture(scope="module")
def _new_gpu_test_session(check_router_cleaned):  # pragma: no cover
    from .deploy.oscar.tests.session import new_test_session
    from .resource import cuda_count

    cuda_devices = list(range(min(cuda_count(), 2)))

    sess = new_test_session(
        address="127.0.0.1",
        backend=MARS_CI_BACKEND,
        init_local=True,
        n_worker=1,
        n_cpu=1,
        cuda_devices=cuda_devices,
        default=True,
        timeout=300,
    )
    with option_context({"show_progress": False}):
        try:
            yield sess
        finally:
            sess.stop_server(isolation=False)
            Router.set_instance(None)


@pytest.fixture
def setup(_new_test_session):
    _new_test_session.as_default()
    yield _new_test_session
    assert not (is_build_mode() or is_kernel_mode())


@pytest.fixture
def setup_cluster(_new_integrated_test_session):
    _new_integrated_test_session.as_default()
    yield _new_integrated_test_session


@pytest.fixture
def setup_gpu(_new_gpu_test_session):  # pragma: no cover
    _new_gpu_test_session.as_default()
    yield _new_test_session

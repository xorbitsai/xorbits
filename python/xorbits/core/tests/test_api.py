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

from ... import remote as mr
from ... import run
from ..api import API, _ClientAPI, _MarsContextBasedAPI


def test_ctx_api(setup):
    def check_api_in_func():
        api = API.create()
        assert isinstance(api, _MarsContextBasedAPI)
        assert len(api.list_workers()) > 0

    # should have no error
    run(mr.spawn(check_api_in_func))


def test_client_api(setup):
    api = API.create()
    assert isinstance(api, _ClientAPI)
    assert len(api.list_workers()) > 0
    # call twice
    assert len(api.list_workers()) > 0


def test_not_inited():
    with pytest.raises(ValueError, match="Xorbits is not inited"):
        API.create()

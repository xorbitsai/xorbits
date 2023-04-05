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

from . import _version
from .config import options
from .core.context import get_context
from .core.entrypoints import init_extension_entrypoints
from .session import execute, fetch, fetch_log, new_session, stop_server

# load third party extensions.
init_extension_entrypoints()
del init_extension_entrypoints

__version__ = _version.get_versions()["version"]

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

import asyncio
import contextlib
import sys

from xoscar.aio.file import AioFileObject
from xoscar.aio.lru import alru_cache
from xoscar.aio.parallelism import AioEvent

from .file import AioFilesystem
from .isolation import Isolation, get_isolation, new_isolation, stop_isolation

if sys.version_info[:2] < (3, 9):
    from ._threads import to_thread

    asyncio.to_thread = to_thread

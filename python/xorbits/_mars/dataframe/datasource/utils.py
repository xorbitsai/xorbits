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

from __future__ import annotations

import os

from ...lib.filesystem import LocalFileSystem, get_fs


def convert_to_abspath(
    path: str | tuple[str] | list[str], storage_options: dict | None = None
) -> str | list[str]:
    # convert path to abs_path
    if isinstance(path, (list, tuple)):
        abs_path = [
            os.path.abspath(p)
            if isinstance(get_fs(p, storage_options), LocalFileSystem)
            else p
            for p in path
        ]
    elif isinstance(get_fs(path, storage_options), LocalFileSystem):
        abs_path = os.path.abspath(path)
    else:
        abs_path = path
    return abs_path

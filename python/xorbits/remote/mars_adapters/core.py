# Copyright 2022 XProbe Inc.
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

from typing import Callable, Dict

from ...core.adapter import mars_remote, wrap_mars_callable


def _collect_module_callables() -> Dict[str, Callable]:
    module_callables: Dict[str, Callable] = dict()

    module_callables["spawn"] = wrap_mars_callable(
        mars_remote.spawn, attach_docstring=False, is_method=False
    )
    module_callables["run_script"] = wrap_mars_callable(
        mars_remote.run_script, attach_docstring=False, is_method=False
    )
    return module_callables


MARS_REMOTE_CALLABLES = _collect_module_callables()

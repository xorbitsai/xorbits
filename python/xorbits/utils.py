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

import sys
import traceback
from typing import Callable


def is_debug() -> bool:
    """Return if the debugger is currently active"""
    return hasattr(sys, "gettrace") and sys.gettrace() is not None


def is_pydev_evaluating_value() -> bool:
    for frame in traceback.extract_stack():
        if "pydev" in frame.filename and frame.name == "var_to_xml":
            return True
    return False


def safe_repr_str(f: Callable):
    def inn(self, *args, **kwargs):
        if is_pydev_evaluating_value():
            # if is evaluating value from pydev, pycharm, etc
            # skip repr or str
            return getattr(object, f.__name__)(self)
        else:
            return f(self, *args, **kwargs)

    return inn

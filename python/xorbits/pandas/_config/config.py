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

from functools import reduce
from typing import Any

import pandas as pd

from ..._mars.config import option_context as mars_option_context
from ..._mars.config import options
from ...core.utils.docstring import attach_module_callable_docstring


def set_option(pat: Any, value: Any) -> None:
    try:
        attr_list = pat.split(".")
        if len(attr_list) == 1:
            getattr(options, attr_list[0])
            setattr(options, attr_list[0], value)
        else:
            setattr(reduce(getattr, attr_list[:-1], options), attr_list[-1], value)
    except:
        pd.set_option(pat, value)


def get_option(pat: Any) -> Any:
    try:
        attr_list = pat.split(".")
        return reduce(getattr, attr_list, options)
    except:
        return pd.get_option(pat)


def reset_option(pat) -> None:
    try:
        options.reset_option(pat)
    except:
        pd.reset_option(pat)


def describe_option(*args, **kwargs) -> None:
    pd.describe_option(*args, **kwargs)


class option_context:
    def __init__(self, *args):
        # convert tuple to dict
        context_dict = dict(args[i : i + 2] for i in range(0, len(args), 2))
        mars_dict = {}
        pd_dict = {}
        for key, value in context_dict.items():
            try:
                pd.get_option(key)
                pd_dict[key] = value
            except:
                mars_dict[key] = value

        self.option_contexts = mars_option_context(mars_dict)
        self.pandas_option_context = pd.option_context(
            *tuple(item for sublist in pd_dict.items() for item in sublist)
        )

    def __enter__(self):
        self.option_contexts.__enter__()
        self.pandas_option_context.__enter__()

    def __exit__(self, type, value, traceback):
        self.option_contexts.__exit__(type, value, traceback)
        self.pandas_option_context.__exit__(type, value, traceback)


def set_eng_float_format(accuracy: int = 3, use_eng_prefix: bool = False) -> None:
    pd.set_eng_float_format(accuracy, use_eng_prefix)


attach_module_callable_docstring(set_option, pd, pd.set_option)

attach_module_callable_docstring(get_option, pd, pd.get_option)

attach_module_callable_docstring(reset_option, pd, pd.reset_option)

attach_module_callable_docstring(describe_option, pd, pd.describe_option)

attach_module_callable_docstring(option_context, pd, pd.option_context)

attach_module_callable_docstring(set_eng_float_format, pd, pd.set_eng_float_format)

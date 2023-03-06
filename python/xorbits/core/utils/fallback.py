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

import functools
import warnings
from types import ModuleType

from ...core.adapter import MarsOutputType, mars_remote


def unimplemented_func():
    """
    Not implemented yet.
    """
    raise NotImplementedError(f"This function is not implemented yet.")


def wrap_fallback_module_method(
    mod: ModuleType, func_name: str, output_type: MarsOutputType, warning_str: str
):
    # wrap member function
    @functools.wraps(getattr(mod, func_name))
    def _wrapped(*args, **kwargs):
        warnings.warn(warning_str, RuntimeWarning)

        # use mars remote to execute functions
        def execute_func(f_name: str, *args, **kwargs):
            from xorbits.core.adapter import MarsEntity

            def _replace_data(nested):
                if isinstance(nested, dict):
                    vals = list(nested.values())
                else:
                    vals = list(nested)

                new_vals = []
                for val in vals:
                    if isinstance(val, (dict, list, tuple, set)):
                        new_val = _replace_data(val)
                    else:
                        if isinstance(val, MarsEntity):
                            new_val = val.fetch()
                        else:
                            new_val = val
                    new_vals.append(new_val)
                if isinstance(nested, dict):
                    return type(nested)((k, v) for k, v in zip(nested.keys(), new_vals))
                else:
                    return type(nested)(new_vals)

            return getattr(mod, f_name)(*_replace_data(args), **_replace_data(kwargs))

        new_args = (func_name,) + args
        ret = mars_remote.spawn(
            execute_func, args=new_args, kwargs=kwargs, output_types=output_type
        )
        if output_type == MarsOutputType.df_or_series:
            ret = ret.ensure_data()
        else:
            ret = ret.execute()
        return ret

    return _wrapped

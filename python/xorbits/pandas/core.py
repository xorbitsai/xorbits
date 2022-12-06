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

from ..core import DataRef


class DataFrame(DataRef):
    def __init__(self, *args, **kwargs):
        from .mars_adapters import MARS_DATAFRAME_CALLABLES

        ref = MARS_DATAFRAME_CALLABLES["DataFrame"](*args, **kwargs)
        super().__init__(ref.data)


def _install_dataframe_docstring():
    import pandas

    from ..core.utils.docstring import attach_module_callable_docstring

    attach_module_callable_docstring(DataFrame, pandas, pandas.DataFrame)


_install_dataframe_docstring()

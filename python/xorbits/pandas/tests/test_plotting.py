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


from ... import pandas as xpd


def test_class_docstring():
    docstring = xpd.DataFrame.plot.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from pandas."
    )

    docstring = xpd.DataFrame.plot.pie.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from pandas.plotting._core.PlotAccessor."
    )


def test_obj_docstring(setup, dummy_str_series, dummy_df):
    objs = [dummy_str_series, dummy_df]

    for obj in objs:
        docstring = obj.plot.__doc__
        assert docstring is not None and docstring.endswith(
            "This docstring was copied from pandas."
        )

        docstring = obj.plot.pie.__doc__
        assert docstring is not None and docstring.endswith(
            "This docstring was copied from pandas.plotting._core.PlotAccessor."
        )

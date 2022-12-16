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

from ....core.utils.docstring import (
    add_arg_disclaimer,
    add_docstring_disclaimer,
    skip_doctest,
)


def test_add_docstring_disclaimer():
    import pandas as pd

    assert "foo" == add_docstring_disclaimer(None, None, "foo")
    assert add_docstring_disclaimer(None, pd.DataFrame, None) is None
    assert add_docstring_disclaimer(None, pd.DataFrame, "\n\n\n").endswith(
        "This docstring was copied from pandas.core.frame.DataFrame."
    )

    assert (
        add_docstring_disclaimer(pd, pd.DataFrame, "test\n", True)
        .split("\n")[-2]
        .endswith(
            f".. warning:: This method has not been implemented yet. Xorbits will try to "
            f"execute it with {pd.__name__}."
        )
    )
    assert (
        add_docstring_disclaimer(pd, pd.DataFrame, "test\n", False)
        == "test\n\n\nThis docstring was copied from pandas.core.frame.DataFrame."
    )


def test_skip_doctest():
    doc = ">>> a = 0"
    assert doc + "  # doctest: +SKIP" == skip_doctest(doc)
    doc = ">>> a = 0  # doctest: +FOO"
    assert doc + ", +SKIP" == skip_doctest(doc)
    doc = ">>> a = 0  # doctest: +SKIP"
    assert doc == skip_doctest(doc)


def test_add_arg_disclaimer():
    def src(a: str, b: str, e: str):
        """
        Parameters
        ----------
        a : str
            foo
        b : str
            bar
        e : str

        Returns
        -------
        """
        pass

    def dest(b: str, c: str, d: str):
        """
        Parameters
        ----------
        b : str
            bar
        c : str
            baz
        d : str

        Returns
        -------
        """
        pass

    expected = (
        """
        Parameters
        ----------
        a : str  (Not supported yet)
            foo
        b : str
            bar
        e : str  (Not supported yet)

        Returns
        -------
        """
        + """
        Extra Parameters
        ----------------
        c : str
            baz
        d : str
        """
    )

    assert add_arg_disclaimer(src, dest, src.__doc__) == expected

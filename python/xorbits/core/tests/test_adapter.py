# -*- coding: utf-8 -*-
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

import pytest

from ..adapter import (
    MarsEntity,
    add_arg_disclaimer,
    add_docstring_disclaimer,
    from_mars,
    mars_dataframe,
    skip_doctest,
    to_mars,
)
from ..data import DataRef


def test_to_mars(setup, dummy_df):
    assert isinstance(to_mars(dummy_df), MarsEntity)

    # tuple.
    ins = (dummy_df, "foo")
    assert isinstance(to_mars(ins)[0], MarsEntity)
    assert not isinstance(to_mars(ins)[1], MarsEntity)
    # nested tuple.
    ins = ((dummy_df, "foo"), "bar")
    assert isinstance(to_mars(ins)[0][0], MarsEntity)
    assert not isinstance(to_mars(ins)[0][1], MarsEntity)
    assert not isinstance(to_mars(ins)[1], MarsEntity)
    ins = ([dummy_df, "foo"], "bar")
    assert isinstance(to_mars(ins)[0][0], MarsEntity)
    assert not isinstance(to_mars(ins)[0][1], MarsEntity)
    assert not isinstance(to_mars(ins)[1], MarsEntity)
    ins = ({"foo": dummy_df, "bar": "baz"}, "bar")
    assert isinstance(to_mars(ins)[0]["foo"], MarsEntity)
    assert not isinstance(to_mars(ins)[0]["bar"], MarsEntity)
    assert not isinstance(to_mars(ins)[1], MarsEntity)

    # list.
    ins = [dummy_df, "foo"]
    assert isinstance(to_mars(ins)[0], MarsEntity)
    assert not isinstance(to_mars(ins)[1], MarsEntity)
    # nested list.
    ins = [(dummy_df, "foo"), "bar"]
    assert isinstance(to_mars(ins)[0][0], MarsEntity)
    assert not isinstance(to_mars(ins)[0][1], MarsEntity)
    assert not isinstance(to_mars(ins)[1], MarsEntity)
    ins = [[dummy_df, "foo"], "bar"]
    assert isinstance(to_mars(ins)[0][0], MarsEntity)
    assert not isinstance(to_mars(ins)[0][1], MarsEntity)
    assert not isinstance(to_mars(ins)[1], MarsEntity)
    ins = [{"foo": dummy_df, "bar": "baz"}, "bar"]
    assert isinstance(to_mars(ins)[0]["foo"], MarsEntity)
    assert not isinstance(to_mars(ins)[0]["bar"], MarsEntity)
    assert not isinstance(to_mars(ins)[1], MarsEntity)

    # dict.
    ins = {"foo": dummy_df, "bar": "baz"}
    assert isinstance(to_mars(ins)["foo"], MarsEntity)
    assert not isinstance(to_mars(ins)["bar"], MarsEntity)
    # nested dict.
    ins = {"foo": (dummy_df, "foo"), "bar": "baz"}
    assert isinstance(to_mars(ins)["foo"][0], MarsEntity)
    assert not isinstance(to_mars(ins)["foo"][1], MarsEntity)
    assert not isinstance(to_mars(ins)["bar"], MarsEntity)
    ins = {"foo": [dummy_df, "foo"], "bar": "baz"}
    assert isinstance(to_mars(ins)["foo"][0], MarsEntity)
    assert not isinstance(to_mars(ins)["foo"][1], MarsEntity)
    assert not isinstance(to_mars(ins)["bar"], MarsEntity)
    ins = {"foo": {"bar": dummy_df}, "bar": "baz"}
    assert isinstance(to_mars(ins)["foo"]["bar"], MarsEntity)
    assert not isinstance(to_mars(ins)["bar"], MarsEntity)


def test_from_mars():
    mdf = mars_dataframe.DataFrame({"foo": (1, 2, 3), "bar": (4, 5, 6)})
    assert isinstance(from_mars(mdf), DataRef)

    # tuple.
    ins = (mdf, "foo")
    assert isinstance(from_mars(ins)[0], DataRef)
    assert not isinstance(from_mars(ins)[1], DataRef)
    # nested tuple.
    ins = ((mdf, "foo"), "bar")
    assert isinstance(from_mars(ins)[0][0], DataRef)
    assert not isinstance(from_mars(ins)[0][1], DataRef)
    assert not isinstance(from_mars(ins)[1], DataRef)
    ins = ([mdf, "foo"], "bar")
    assert isinstance(from_mars(ins)[0][0], DataRef)
    assert not isinstance(from_mars(ins)[0][1], DataRef)
    assert not isinstance(from_mars(ins)[1], DataRef)
    ins = ({"foo": mdf, "bar": "baz"}, "bar")
    assert isinstance(from_mars(ins)[0]["foo"], DataRef)
    assert not isinstance(from_mars(ins)[0]["bar"], DataRef)
    assert not isinstance(from_mars(ins)[1], DataRef)

    # list.
    ins = [mdf, "foo"]
    assert isinstance(from_mars(ins)[0], DataRef)
    assert not isinstance(from_mars(ins)[1], DataRef)
    # nested list.
    ins = [(mdf, "foo"), "bar"]
    assert isinstance(from_mars(ins)[0][0], DataRef)
    assert not isinstance(from_mars(ins)[0][1], DataRef)
    assert not isinstance(from_mars(ins)[1], DataRef)
    ins = [[mdf, "foo"], "bar"]
    assert isinstance(from_mars(ins)[0][0], DataRef)
    assert not isinstance(from_mars(ins)[0][1], DataRef)
    assert not isinstance(from_mars(ins)[1], DataRef)
    ins = [{"foo": mdf, "bar": "baz"}, "bar"]
    assert isinstance(from_mars(ins)[0]["foo"], DataRef)
    assert not isinstance(from_mars(ins)[0]["bar"], DataRef)
    assert not isinstance(from_mars(ins)[1], DataRef)

    # dict.
    ins = {"foo": mdf, "bar": "baz"}
    assert isinstance(from_mars(ins)["foo"], DataRef)
    assert not isinstance(from_mars(ins)["bar"], DataRef)
    # nested dict.
    ins = {"foo": (mdf, "foo"), "bar": "baz"}
    assert isinstance(from_mars(ins)["foo"][0], DataRef)
    assert not isinstance(from_mars(ins)["foo"][1], DataRef)
    assert not isinstance(from_mars(ins)["bar"], DataRef)
    ins = {"foo": [mdf, "foo"], "bar": "baz"}
    assert isinstance(from_mars(ins)["foo"][0], DataRef)
    assert not isinstance(from_mars(ins)["foo"][1], DataRef)
    assert not isinstance(from_mars(ins)["bar"], DataRef)
    ins = {"foo": {"bar": mdf}, "bar": "baz"}
    assert isinstance(from_mars(ins)["foo"]["bar"], DataRef)
    assert not isinstance(from_mars(ins)["bar"], DataRef)


def test_on_nonexistent_attr(dummy_df):
    with pytest.raises(
        AttributeError, match="'dataframe' object has no attribute 'nonexistent_attr'"
    ):
        dummy_df.nonexistent_attr


def test_on_nonexistent_magic_method(dummy_df):
    with pytest.raises(
        AttributeError, match="'index' object has no attribute '__add__'"
    ):
        dummy_df.index + dummy_df.index


def test_add_docstring_disclaimer():
    import pandas as pd

    assert "foo" == add_docstring_disclaimer(None, None, "foo")
    assert add_docstring_disclaimer(None, pd.DataFrame, None) is None
    assert add_docstring_disclaimer(None, pd.DataFrame, "").endswith(
        "This docstring was copied from pandas.core.frame.DataFrame"
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

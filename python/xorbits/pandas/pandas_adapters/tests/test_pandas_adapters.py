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

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from .... import pandas as xpd
from ....core.data import DataRef


def test_pandas_dataframe_methods(setup):
    """
    All the fallbacks:

    applymap
    asfreq
    asof
    at_time
    between_time
    bool
    boxplot
    clip
    combine
    combine_first
    compare
    convert_dtypes
    cov
    divide
    droplevel
    equals
    filter
    first
    first_valid_index
    get
    hist
    idxmax
    idxmin
    infer_objects
    info
    interpolate
    isetitem
    items
    iteritems
    last
    last_valid_index
    lookup
    mad
    median
    mode
    nlargest
    nsmallest
    pipe
    pivot
    pivot_table
    rank
    reorder_levels
    resample
    set_flags
    slice_shift
    squeeze
    subtract
    swapaxes
    swaplevel
    take
    to_clipboard
    to_dict
    to_excel
    to_feather
    to_gbq
    to_hdf
    to_html
    to_json
    to_latex
    to_markdown
    to_numpy
    to_orc
    to_period
    to_pickle
    to_records
    to_stata
    to_string
    to_timestamp
    to_xarray
    to_xml
    truncate
    tz_convert
    tz_localize
    unstack
    update
    value_counts
    xs
    """
    raw = pd.DataFrame(
        {
            "foo": ["one", "one", "one", "two", "two", "two"],
            "bar": ["A", "B", "C", "A", "B", "C"],
            "baz": [1, 2, 3, 4, 5, 6],
            "zoo": ["x", "y", "z", "q", "w", "t"],
        }
    )
    df = xpd.DataFrame(raw)
    with pytest.warns(Warning) as w:
        r = df.pivot(index="foo", columns="bar", values="baz")
        assert len(w) == 1
        assert "DataFrame.pivot will fallback to Pandas" == str(w[0].message)

    assert len(getattr(r.data._mars_entity, "_executed_sessions")) == 1
    expected = raw.pivot(index="foo", columns="bar", values="baz")
    assert expected.shape == r.shape
    assert str(expected) == str(r)
    assert isinstance(r, DataRef)
    pd.testing.assert_frame_equal(expected, r.to_pandas())

    # multi chunk and follow other operations
    df = xpd.DataFrame(raw, chunk_size=3)
    with pytest.warns(Warning) as w:
        r = df.pivot(index="foo", columns="bar", values="baz") + 1
        assert "DataFrame.pivot will fallback to Pandas" == str(w[0].message)

    expected = raw.pivot(index="foo", columns="bar", values="baz") + 1
    assert expected.shape == r.shape
    assert str(expected) == str(r)
    pd.testing.assert_frame_equal(expected, r.to_pandas())

    # can be inferred
    df = xpd.DataFrame(raw)
    with pytest.warns(Warning) as w:
        r = df.median(numeric_only=True)
        assert "DataFrame.median will fallback to Pandas" == str(w[0].message)

    # assert if r is executed
    assert len(getattr(r.data._mars_entity, "_executed_sessions")) == 0
    r2 = r.sum()
    expected = raw.median(numeric_only=True).sum()
    assert str(r2) == str(expected)
    assert isinstance(r2, DataRef)
    np.testing.assert_approx_equal(r2.fetch(), expected)

    # output is not series or dataframe
    df = xpd.DataFrame(raw)
    with pytest.warns(Warning) as w:
        r = df.to_json()
        assert "DataFrame.to_json will fallback to Pandas" == str(w[0].message)

    # assert if r is executed
    assert len(getattr(r.data._mars_entity, "_executed_sessions")) == 1
    expected = raw.to_json()
    assert str(r.to_object()) == str(expected)
    assert isinstance(r, DataRef)


def test_pandas_series_methods(setup):
    """
    All the fallbacks:

    argmax
    argmin
    argsort
    asfreq
    asof
    at_time
    between_time
    bool
    clip
    combine
    combine_first
    compare
    convert_dtypes
    cov
    divide
    divmod
    droplevel
    equals
    factorize
    filter
    first
    first_valid_index
    get
    hist
    idxmax
    idxmin
    infer_objects
    info
    interpolate
    item
    last
    last_valid_index
    mad
    mode
    nlargest
    nsmallest
    pipe
    pop
    rank
    ravel
    rdivmod
    reorder_levels
    repeat
    resample
    searchsorted
    set_flags
    slice_shift
    squeeze
    subtract
    swapaxes
    swaplevel
    take
    to_clipboard
    to_excel
    to_hdf
    to_json
    to_latex
    to_list
    to_markdown
    to_numpy
    to_period
    to_pickle
    to_string
    to_timestamp
    to_xarray
    tolist
    transpose
    truncate
    tz_convert
    tz_localize
    unstack
    update
    view
    xs
    """

    # output is a scalar and type inference failed.
    s = pd.Series(
        {
            "Corn Flakes": 100.0,
            "Almond Delight": 110.0,
            "Cinnamon Toast Crunch": 120.0,
            "Cocoa Puff": 110.0,
        }
    )
    xs = xpd.Series(s)
    with pytest.warns(Warning) as w:
        r = xs.argmax()
        assert "Series.argmax will fallback to Pandas" == str(w[0].message)
    assert len(getattr(r.data._mars_entity, "_executed_sessions")) == 1
    assert s.argmax() == r.to_object()

    # output is a series and type inference succeed.
    a = pd.Series([1, 1, 1, np.nan], index=["a", "b", "c", "d"])
    xa = xpd.Series(a)
    with pytest.warns(Warning) as w:
        r = xa.divide(10, fill_value=0)
        assert "Series.divide will fallback to Pandas" == str(w[0].message)
    assert len(getattr(r.data._mars_entity, "_executed_sessions")) == 0
    pd.testing.assert_series_equal(a.divide(10, fill_value=0), r.to_pandas())

    # multi chunks.
    s = pd.Series(np.random.randn(12))
    xs = xpd.Series(s, chunk_size=4)
    with pytest.warns(Warning) as w:
        r = xs.divide(10)
        assert "Series.divide will fallback to Pandas" == str(w[0].message)
    assert len(getattr(r.data._mars_entity, "_executed_sessions")) == 0
    pd.testing.assert_series_equal(s.divide(10), r.to_pandas())

    # divide by another series.
    # TODO: TypeError: cannot pickle 'weakref' object
    # b = pd.Series([1, np.nan, 1, np.nan], index=['a', 'b', 'd', 'e'])
    # xb = xpd.Series(b)
    # with pytest.warns(Warning) as w:
    #     r = xa.divide(xb, fill_value=0)
    #     assert "Series.divide will fallback to Pandas" == str(w[0].message)
    # pd.testing.assert_series_equal(a.divide(b, fill_value=0), r.to_pandas())

    # combine.
    # TODO: AttributeError: 'series' object has no attribute '__array__'.
    # s1 = pd.Series({'falcon': 330.0, 'eagle': 160.0})
    # s2 = pd.Series({'falcon': 345.0, 'eagle': 200.0, 'duck': 30.0})
    # xs1 = xpd.Series(s1)
    # xs2 = xpd.Series(s2)
    # with pytest.warns(Warning) as w:
    #     r = xs1.combine(xs2, max)
    #     assert "DataFrame.combine will fallback to Pandas" == str(w[0].message)
    # assert len(getattr(r.data._mars_entity, "_executed_sessions")) == 1
    # pd.testing.assert_series_equal(s1.combine(s2, max), r.to_pandas())

    # asof.
    # TODO: AssertionError: shape in metadata (4,) is not consistent with real shape (2,).
    # s = pd.Series([1, 2, np.nan, 4], index=[10, 20, 30, 40])
    # xs = xpd.Series(s)
    # assert s.asof(20) == xs.asof(20).to_object()
    # pd.testing.assert_series_equal(s.asof([5, 20]), xs.asof([5, 20]).to_pandas())


def test_pandas_index_methods(setup):
    """
    All the fallbacks:

    append
    argmax
    argmin
    argsort
    asof
    asof_locs
    delete
    difference
    droplevel
    equals
    factorize
    format
    get_indexer
    get_indexer_for
    get_indexer_non_unique
    get_level_values
    get_loc
    get_slice_bound
    get_value
    groupby
    holds_integer
    identical
    insert
    intersection
    is_
    is_boolean
    is_categorical
    is_floating
    is_integer
    is_interval
    is_mixed
    is_numeric
    is_object
    is_type_compatible
    isin
    isnull
    item
    join
    notnull
    nunique
    putmask
    ravel
    reindex
    repeat
    searchsorted
    set_value
    shift
    slice_indexer
    slice_locs
    sort
    sort_values
    sortlevel
    symmetric_difference
    take
    to_flat_index
    to_list
    to_native_types
    to_numpy
    tolist
    transpose
    union
    unique
    view
    where
    """

    # output is a scalar.
    s = pd.Series([1, 2, 3])
    i = s.index
    xs = xpd.Series(s)
    xi = xs.index
    with pytest.warns(Warning) as w:
        r = xi.argmax()
        assert "RangeIndex.argmax will fallback to Pandas" == str(w[0].message)
    assert len(getattr(r.data._mars_entity, "_executed_sessions")) == 1
    assert i.argmax() == r.to_object()

    # output is an array.
    np.testing.assert_array_equal(i.notnull(), xi.notnull().to_object())

    # TODO: TypeError: Input must be Index or array-like
    # i1 = pd.Index([2, 1, 3, 4])
    # xi1 = xpd.Index(i1)
    # i2 = pd.Index([3, 4, 5, 6])
    # xi2 = xpd.Index(i2)
    # pandas.testing.assert_index_equal(i1.difference(i2), xi1.difference(xi2))


def test_pandas_module_methods(setup):
    raw = pd.DataFrame(
        [["a", "b"], ["c", "d"]], index=["row 1", "row 2"], columns=["col 1", "col 2"]
    )
    # output could be series or dataframe
    with pytest.warns(Warning) as w:
        r = xpd.read_json(raw.to_json())
        assert "xorbits.pandas.read_json will fallback to Pandas" == str(w[0].message)

    assert str(r) == str(raw)
    assert isinstance(r, DataRef)
    pd.testing.assert_frame_equal(r.to_pandas(), raw)

    # output is dataframe
    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "tmp.xlsx")
        raw.to_excel(path)
        with pytest.warns(Warning) as w:
            r = xpd.read_excel(path)
            assert "xorbits.pandas.read_excel will fallback to Pandas" == str(
                w[0].message
            )

        expected = pd.read_excel(path)
        assert str(r) == str(expected)
        assert isinstance(r, DataRef)
        pd.testing.assert_frame_equal(r.to_pandas(), expected)

    # input has xorbit data
    raw = pd.DataFrame(
        {
            "foo": ["one", "one", "one", "two", "two", "two"],
            "bar": ["A", "B", "C", "A", "B", "C"],
            "baz": [1, 2, 3, 4, 5, 6],
            "zoo": ["x", "y", "z", "q", "w", "t"],
        }
    )
    with pytest.warns(Warning) as w:
        r = xpd.pivot(xpd.DataFrame(raw), index="foo", columns="bar", values="baz")
        assert "xorbits.pandas.pivot will fallback to Pandas" == str(w[0].message)

    expected = pd.pivot(raw, index="foo", columns="bar", values="baz")
    assert str(r) == str(expected)
    assert isinstance(r, DataRef)
    pd.testing.assert_frame_equal(r.to_pandas(), expected)


def test_to_numpy(setup):
    df = pd.DataFrame((1, 2, 3))
    xdf = xpd.DataFrame((1, 2, 3))
    np.testing.assert_array_equal(df.to_numpy(), xdf.to_numpy().to_numpy())

    s = pd.Series((1, 2, 3))
    xs = xpd.Series(s)
    np.testing.assert_array_equal(s.to_numpy(), xs.to_numpy().to_numpy())

    i = s.index
    xi = xs.index
    np.testing.assert_array_equal(i.to_numpy(), xi.to_numpy().to_numpy())


def test_docstring():
    docstring = xpd.DataFrame.asof.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from pandas.core.frame.DataFrame."
    )

    docstring = xpd.Series.asof.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from pandas.core.series.Series."
    )

    docstring = xpd.Index.asof.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from pandas.core.indexes.base.Index."
    )


def test_dir(setup):
    assert pd.__dir__().sort() == xpd.__dir__().sort()

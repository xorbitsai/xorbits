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

import pytest

from ...pandas import get_dummies
from ...remote import spawn
from ..execution import _is_in_final_results, need_to_execute, run


def test_deferred_execution_repr(setup, dummy_df):
    # own data, no execution
    assert need_to_execute(dummy_df)
    s = repr(dummy_df)
    assert need_to_execute(dummy_df)
    assert s == repr(dummy_df.to_pandas())

    # trigger execution
    r = dummy_df.sum(axis=0)
    s = repr(r)
    assert not need_to_execute(r)
    assert s == repr(r.to_pandas())


def test_deferred_execution_print(setup, dummy_df):
    # own data, no execution
    assert need_to_execute(dummy_df)
    s = str(dummy_df)
    assert need_to_execute(dummy_df)
    assert s == repr(dummy_df.to_pandas())

    # trigger execution
    r = dummy_df.sum(axis=0)
    s = str(r)
    assert not need_to_execute(r)
    assert s == repr(r.to_pandas())


def test_deferred_execution_iterrows(setup, dummy_df):
    assert need_to_execute(dummy_df)
    # own data, no execution
    for _ in dummy_df.iterrows():
        pass
    assert need_to_execute(dummy_df)

    # trigger execution
    sorted_df = dummy_df.sort_values(by="foo")
    for _ in sorted_df.iterrows():
        pass
    assert not need_to_execute(sorted_df)


def test_deferred_execution_itertuples(setup, dummy_df):
    assert need_to_execute(dummy_df)
    # own data, no execution
    for _ in dummy_df.itertuples():
        pass
    assert need_to_execute(dummy_df)

    # trigger execution
    sorted_df = dummy_df.sort_values(by="foo")
    for _ in sorted_df.itertuples():
        pass
    assert not need_to_execute(sorted_df)


def test_deferred_execution_transpose_1(setup, dummy_df):
    # transpose.
    transposed = dummy_df.transpose()
    assert need_to_execute(transposed)
    s = str(transposed)
    assert not need_to_execute(transposed)
    assert s == str(transposed.to_pandas())


def test_deferred_execution_transpose_2(setup, dummy_df):
    transposed = dummy_df.T
    assert need_to_execute(transposed)
    s = str(transposed)
    assert not need_to_execute(transposed)
    assert s == str(transposed.to_pandas())


def test_deferred_execution_get_dummies(setup, dummy_df):
    dummy_encoded = get_dummies(dummy_df)
    assert need_to_execute(dummy_encoded)
    s = str(dummy_encoded)
    assert not need_to_execute(dummy_encoded)
    assert s == str(dummy_encoded.to_pandas())


def test_deferred_execution_groupby_apply(setup, dummy_df):
    groupby_applied = dummy_df.groupby("foo").apply(lambda df: df.sum())
    assert need_to_execute(groupby_applied)
    s = str(groupby_applied)
    assert not need_to_execute(groupby_applied)
    assert s == str(groupby_applied.to_pandas())


def test_manual_execution(setup, dummy_int_series):
    assert need_to_execute(dummy_int_series)
    run(dummy_int_series)
    assert not need_to_execute(dummy_int_series)

    series_to_execute = [dummy_int_series + i for i in range(3)]
    assert all([need_to_execute(ref) for ref in series_to_execute])
    run(series_to_execute)
    assert not any([need_to_execute(ref) for ref in series_to_execute])


def tests_int_conversion(setup, dummy_int_2d_array):
    import xorbits.numpy as np
    import xorbits.remote as xr

    assert int(dummy_int_2d_array[0][2]) == 2
    assert int(xr.spawn(lambda: "2")) == 2
    with pytest.raises(TypeError):
        int(np.atleast_2d(3.0))


def tests_float_conversion(setup, dummy_int_2d_array, dummy_df):
    import xorbits.numpy as np
    import xorbits.remote as xr

    assert float(dummy_int_2d_array[0][2]) == 2.0
    assert float(xr.spawn(lambda: "1.2")) == 1.2
    with pytest.raises(TypeError):
        float(np.atleast_2d(3.0))
    with pytest.raises(TypeError):
        float(dummy_df["foo"])


def tests_index_conversion(setup, dummy_int_2d_array, dummy_str_series):
    test = 0
    for i in range(dummy_int_2d_array[0][2]):
        test += 1
    assert test == 2
    import xorbits.remote as xr

    for i in range(xr.spawn(lambda: 2)):
        test += 1
    assert test == 4
    with pytest.raises(TypeError):
        range(dummy_str_series[0])
    with pytest.raises(TypeError):
        range(xr.spawn(lambda: "foo"))


def test_bool_conversion(setup, dummy_df, dummy_int_2d_array, dummy_str_series):
    test = 0
    if dummy_int_2d_array[0][0]:
        test += 1
    assert test == 0
    if dummy_int_2d_array[0][1]:
        test += 1
    assert test == 1
    if dummy_str_series[0]:
        test += 1
    assert test == 2
    import xorbits.remote as xr

    if xr.spawn(lambda: ""):
        test += 1
    assert test == 2
    import xorbits.numpy as np

    if np.zeros(0).size > 0:
        test += 1
    assert test == 2
    import xorbits.pandas as pd

    with pytest.raises(ValueError):
        if np.zeros(2):
            pass
    with pytest.raises(ValueError):
        if pd.DataFrame({}):
            pass
    with pytest.raises(ValueError):
        if dummy_df.groupby(["foo"]):
            pass


def test_len(setup, dummy_df, dummy_int_series, dummy_int_2d_array):
    assert need_to_execute(dummy_df)
    assert need_to_execute(dummy_int_series)
    assert need_to_execute(dummy_int_2d_array)

    assert len(dummy_df) == 3
    assert len(dummy_int_series) == 5
    assert len(dummy_int_2d_array) == 3

    filtered = dummy_df[dummy_df["foo"] < 2]
    assert len(filtered) == 2
    assert not need_to_execute(filtered)

    filtered = dummy_int_series[dummy_int_series < 2]
    assert len(filtered) == 1
    assert not need_to_execute(filtered)

    filtered = dummy_int_2d_array[dummy_int_2d_array < 2]
    assert len(filtered) == 2
    assert not need_to_execute(filtered)

    with pytest.raises(TypeError):
        len(dummy_int_2d_array.sum())

    obj = spawn(lambda _: 1)
    with pytest.raises(TypeError):
        len(obj)
    assert need_to_execute(obj)


def test_shape(setup, dummy_df, dummy_int_series, empty_df):
    assert need_to_execute(dummy_df)
    assert need_to_execute(dummy_int_series)
    assert need_to_execute(empty_df)

    assert dummy_df.shape == (3, 2)
    assert dummy_int_series.shape == (5,)
    assert empty_df.shape == (0, 0)

    filtered = dummy_df[dummy_df["foo"] < 2]
    assert filtered.shape == (2, 2)
    assert not need_to_execute(filtered)

    filtered = dummy_int_series[dummy_int_series < 2]
    assert filtered.shape == (1,)
    assert not need_to_execute(filtered)

    obj = spawn(lambda _: 1)
    with pytest.raises(TypeError):
        obj.shape
    assert need_to_execute(obj)


def test_is_in_final_results(setup, dummy_df, dummy_str_series):
    assert not _is_in_final_results(dummy_df, [])
    assert _is_in_final_results(dummy_df, [dummy_df])
    assert _is_in_final_results(dummy_df, [dummy_df + 1])
    assert not _is_in_final_results(dummy_df, [dummy_str_series])

    assert _is_in_final_results(dummy_df, [dummy_df.groupby(dummy_str_series).size()])
    assert _is_in_final_results(
        dummy_str_series, [dummy_df.groupby(dummy_str_series).size()]
    )
    assert not _is_in_final_results(
        dummy_df + 1, [dummy_df.groupby(dummy_str_series).size()]
    )
    assert not _is_in_final_results(
        dummy_df.groupby(dummy_str_series), [dummy_df.groupby(dummy_str_series).size()]
    )


def test_array_conversion(setup):
    import numpy as np
    import pandas as pd

    from ... import pandas as xpd

    data = {
        "foo": [1, 2, 3],
        "bar": [1.0, 2.0, 3.0],
        "baz": ["a", "b", "c"],
        "qux": [True, False, True],
        "mixed": [1, 2.0, "a"],
    }

    df = pd.DataFrame(data)
    xdf = xpd.DataFrame(data)

    expected = df.__array__()
    np.testing.assert_array_equal(xdf.__array__(), expected)
    np.testing.assert_array_equal(np.array(xdf), expected)

    # ensure the dataframe needs to be executed.
    df = pd.concat([df, df], axis=0)
    xdf = xpd.concat([xdf, xdf], axis=0)

    expected = df.__array__()
    np.testing.assert_array_equal(xdf.__array__(), expected)
    np.testing.assert_array_equal(np.array(xdf), expected)


@pytest.fixture
def ip():
    from IPython.testing.globalipapp import start_ipython

    yield start_ipython()


def test_interactive_execution(setup, ip):
    ip.run_cell(raw_cell="import xorbits")
    ip.run_cell(raw_cell="import xorbits.pandas as pd")

    r = ip.run_cell(raw_cell="xorbits.core.execution._is_interactive()")
    assert r.result
    r = ip.run_cell(raw_cell="xorbits.core.execution._is_ipython_available()")
    assert r.result

    ip.run_cell(raw_cell="df = pd.DataFrame({'foo': [1, 2, 3]})")
    ip.run_cell(raw_cell="df + 1")
    r = ip.run_cell(raw_cell="xorbits.core.execution.need_to_execute(df)")
    assert not r.result

    # test if unrelated data refs are skipped.
    ip.run_cell(raw_cell="df2 = pd.DataFrame({'bar': [1, 2, 3]})")
    ip.run_cell(raw_cell="df + 2")
    r = ip.run_cell(raw_cell="xorbits.core.execution.need_to_execute(df2)")
    assert r.result

    # test if IPython cached variables are skipped.
    ip.run_cell(raw_cell="_ = pd.DataFrame({'baz': [1, 2, 3]})")
    ip.run_cell(raw_cell="df + 3")
    r = ip.run_cell(raw_cell="xorbits.core.execution.need_to_execute(_)")
    assert r.result


def test_getitem(setup):
    import pandas as pd

    from ... import pandas as xpd

    data = {"a": ["abc", "def"]}
    df = pd.DataFrame(data)
    xdf = xpd.DataFrame(data)
    result = xdf[xdf["a"].str[0] == "a"]
    expected = df[df["a"].str[0] == "a"]
    pd.testing.assert_frame_equal(result.to_pandas(), expected)


# TODO: process exit cause hang
# def test_execution_with_process_exit_message(mocker):
#     import numpy as np
#     from xoscar.errors import ServerClosed

#     import xorbits
#     import xorbits.remote as xr

#     mocker.patch(
#         "xorbits._mars.services.subtask.api.SubtaskAPI.run_subtask_in_slot",
#         side_effect=ServerClosed,
#     )

#     with pytest.raises(
#         ServerClosed,
#         match=r".*?\(.*?\) with address .*? Out-of-Memory \(OOM\) problem",
#     ):
#         xorbits.run(xr.spawn(lambda *_: np.random.rand(10**4, 10**4)))

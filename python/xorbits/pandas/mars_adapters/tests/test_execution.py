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

import pandas as pd

from .... import pandas as xpd
from ....core.execution import need_to_execute


def test_on_dtypes_being_none(setup):
    # magic method.
    df = xpd.DataFrame({"foo": (1, 2, 3), "bar": (4, 5, 6)})
    t = df.T
    assert need_to_execute(t)
    _ = t + 1
    assert not need_to_execute(t)

    # regular method.
    t = df.T
    assert need_to_execute(t)
    _ = t.sum()
    assert not need_to_execute(t)

    # attribute accessing.
    t = df.T
    assert need_to_execute(t)
    str(t.dtypes)
    assert not need_to_execute(t)


def test_from_mars_execution_condition(setup, dummy_df, dummy_str_series):
    with tempfile.TemporaryDirectory() as tempdir:
        # dataframe to_csv.
        csv_path = os.path.join(tempdir, "frame.csv")
        assert not need_to_execute(dummy_df.to_csv(csv_path))
        assert os.path.exists(csv_path)
        # dataframe to_parquet.
        parquet_path = os.path.join(tempdir, "frame.pq")
        os.mkdir(parquet_path)
        assert not need_to_execute(dummy_df.to_parquet(os.path.join(parquet_path)))
        print(os.listdir(parquet_path))
        assert os.path.exists(os.path.join(parquet_path, "0.parquet"))
        # dataframe to_pickle.
        pickle_path = os.path.join(tempdir, "frame.pkl")
        assert not need_to_execute(
            dummy_df.to_pickle(os.path.join(tempdir, "frame.pkl"))
        )
        assert os.path.exists(pickle_path)
        # dataframe to_json.
        assert not need_to_execute(dummy_df.to_json())

        # series to_csv.
        csv_path = os.path.join(tempdir, "frame.csv")
        assert not need_to_execute(
            dummy_str_series.to_csv(os.path.join(tempdir, "series.csv"))
        )
        assert os.path.exists(csv_path)

        # TODO: series falling back methods not implemented yet.
        # assert not need_to_execute(dummy_str_series.to_pickle(os.path.join(tempdir, "series.pkl")))
        # assert not need_to_execute(dummy_str_series.to_json())


def test_own_data(setup, dummy_df, dummy_str_series):
    pd_df = pd.DataFrame({"foo": (0, 1, 2), "bar": ("a", "b", "c")})
    for row, expected_row in zip(
        dummy_df.itertuples(index=False), pd_df.itertuples(index=False)
    ):
        assert row == expected_row
    assert dummy_df.itertuples.__doc__
    assert need_to_execute(dummy_df)

    for (index, row), (expected_index, expected_row) in zip(
        dummy_df.iterrows(), pd_df.iterrows()
    ):
        assert index == expected_index
        pd.testing.assert_series_equal(row, expected_row)
    assert dummy_df.iterrows.__doc__
    assert need_to_execute(dummy_df)

    columns = dummy_df.columns
    assert list(columns) == ["foo", "bar"]
    assert need_to_execute(columns)

    pd_series = pd.Series(["foo", "bar", "baz"])
    for (index, value), (expected_index, expected_value) in zip(
        dummy_str_series.items(), pd_series.items()
    ):
        assert value == expected_value
        assert index == expected_index
    assert dummy_str_series.items.__doc__
    assert need_to_execute(dummy_str_series)

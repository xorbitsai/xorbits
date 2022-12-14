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
import os
import tempfile

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

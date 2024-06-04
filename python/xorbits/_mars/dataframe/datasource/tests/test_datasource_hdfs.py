# Copyright 2022-2023 XProbe Inc.
# derived from copyright 1999-2021 Alibaba Group Holding Ltd.
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

import random
from io import BytesIO, StringIO

import numpy as np
import pandas as pd
import pytest

from .... import dataframe as md
from ....dataframe.utils import PD_VERSION_GREATER_THAN_2_10
from ....tests.core import require_hadoop

TEST_DIR = "/tmp/test"


@require_hadoop
@pytest.fixture(scope="module")
def setup_hdfs():
    import pyarrow

    hdfs = pyarrow.fs.HadoopFileSystem(host="localhost", port=8020)
    file = hdfs.get_file_info(TEST_DIR)
    if file.type == pyarrow.fs.FileType.Directory:
        hdfs.delete_dir(TEST_DIR)
    if file.type == pyarrow.fs.FileType.File:
        hdfs.delete_file(TEST_DIR)
    try:
        yield hdfs
    finally:
        file = hdfs.get_file_info(TEST_DIR)
        if file.type == pyarrow.fs.FileType.Directory:
            hdfs.delete_dir(TEST_DIR)
        if file.type == pyarrow.fs.FileType.File:
            hdfs.delete_file(TEST_DIR)


@require_hadoop
def test_read_csv_execution(setup, setup_hdfs):
    hdfs = setup_hdfs

    with hdfs.open_output_stream(f"{TEST_DIR}/simple_test.csv") as f:
        f.write(b"name,amount,id\nAlice,100,1\nBob,200,2")

    df = md.read_csv(f"hdfs://localhost:8020{TEST_DIR}/simple_test.csv")
    expected = pd.read_csv(BytesIO(b"name,amount,id\nAlice,100,1\nBob,200,2"))
    res = df.to_pandas()
    pd.testing.assert_frame_equal(expected, res)

    test_df = pd.DataFrame(
        {
            "A": np.random.rand(20),
            "B": [
                pd.Timestamp("2020-01-01") + pd.Timedelta(days=random.randint(0, 31))
                for _ in range(20)
            ],
            "C": np.random.rand(20),
            "D": np.random.randint(0, 100, size=(20,)),
            "E": ["foo" + str(random.randint(0, 999999)) for _ in range(20)],
        }
    )
    buf = StringIO()
    test_df[:10].to_csv(buf)
    csv_content = buf.getvalue().encode()

    buf = StringIO()
    test_df[10:].to_csv(buf)
    csv_content2 = buf.getvalue().encode()

    with hdfs.open_output_stream(f"{TEST_DIR}/chunk_test.csv") as f:
        f.write(csv_content)

    df = md.read_csv(f"hdfs://localhost:8020{TEST_DIR}/chunk_test.csv", chunk_bytes=50)
    expected = pd.read_csv(BytesIO(csv_content))
    res = df.to_pandas()
    pd.testing.assert_frame_equal(
        expected.reset_index(drop=True), res.reset_index(drop=True)
    )

    test_read_dir = f"{TEST_DIR}/test_read_csv_directory"
    hdfs.create_dir(test_read_dir)
    with hdfs.open_output_stream(f"{test_read_dir}/part.csv") as f:
        f.write(csv_content)
    with hdfs.open_output_stream(f"{test_read_dir}/part2.csv") as f:
        f.write(csv_content2)

    df = md.read_csv(f"hdfs://localhost:8020{test_read_dir}", chunk_bytes=50)
    expected = pd.concat(
        [pd.read_csv(BytesIO(csv_content)), pd.read_csv(BytesIO(csv_content2))]
    )
    res = df.to_pandas()
    pd.testing.assert_frame_equal(
        expected.reset_index(drop=True), res.reset_index(drop=True)
    )


@require_hadoop
def test_read_parquet_execution(setup, setup_hdfs):
    hdfs = setup_hdfs

    test_df = pd.DataFrame(
        {
            "a": np.arange(10).astype(np.int64, copy=False),
            "b": [f"s{i}" for i in range(10)],
            "c": np.random.rand(10),
        }
    )
    test_df2 = pd.DataFrame(
        {
            "a": np.arange(10).astype(np.int64, copy=False),
            "b": [f"s{i}" for i in range(10)],
            "c": np.random.rand(10),
        }
    )

    with hdfs.open_output_stream(f"{TEST_DIR}/test.parquet") as f:
        f.write(b"name,amount,id\nAlice,100,1\nBob,200,2")
    with hdfs.open_output_stream(f"{TEST_DIR}/test.parquet") as f:
        test_df.to_parquet(f, row_group_size=3)

    df = md.read_parquet(f"hdfs://localhost:8020{TEST_DIR}/test.parquet")
    res = df.to_pandas()
    if PD_VERSION_GREATER_THAN_2_10:
        expected = test_df.convert_dtypes(dtype_backend="pyarrow")
    pd.testing.assert_frame_equal(res, expected)

    hdfs.create_dir(f"{TEST_DIR}/test_partitioned")

    with hdfs.open_output_stream(f"{TEST_DIR}/test_partitioned/file1.parquet") as f:
        test_df.to_parquet(f, row_group_size=3)
    with hdfs.open_output_stream(f"{TEST_DIR}/test_partitioned/file2.parquet") as f:
        test_df2.to_parquet(f, row_group_size=3)

    df = md.read_parquet(f"hdfs://localhost:8020{TEST_DIR}/test_partitioned")
    res = df.to_pandas()
    if PD_VERSION_GREATER_THAN_2_10:
        test_df = test_df.convert_dtypes(dtype_backend="pyarrow")
        test_df2 = test_df2.convert_dtypes(dtype_backend="pyarrow")
    pd.testing.assert_frame_equal(res, pd.concat([test_df, test_df2]))

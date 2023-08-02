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

import datasets
import pandas as pd
import pytest

from ....._mars.tests.core import mock
from ..core import from_huggingface

SAMPLE_DATASET_IDENTIFIER = "lhoestq/test"  # has dataset script
SAMPLE_DATASET_IDENTIFIER2 = "lhoestq/test2"  # only has data files
SAMPLE_DATASET_IDENTIFIER3 = (
    "mariosasko/test_multi_dir_dataset"  # has multiple data directories
)
SAMPLE_DATASET_IDENTIFIER4 = "mariosasko/test_imagefolder_with_metadata"  # imagefolder with a metadata file outside of the train/test directories


def test_split_arg_required():
    with pytest.raises(Exception, match="split"):
        from_huggingface(SAMPLE_DATASET_IDENTIFIER)


@pytest.mark.parametrize(
    "path",
    [
        SAMPLE_DATASET_IDENTIFIER,
        SAMPLE_DATASET_IDENTIFIER2,
        SAMPLE_DATASET_IDENTIFIER3,
        SAMPLE_DATASET_IDENTIFIER4,
    ],
)
def test_from_huggingface_execute(setup, path):
    xds = from_huggingface(path, split="train")
    xds.execute()
    ds = xds.fetch()
    ds_expected = datasets.load_dataset(path, split="train")
    assert ds.to_dict() == ds_expected.to_dict()
    # Trigger datasets issue: https://github.com/huggingface/datasets/issues/6066
    # assert str(ds) == str(ds_expected)


def test_from_huggingface_file_lock(setup):
    real_lock_init = datasets.utils.filelock.BaseFileLock.__init__

    with mock.patch(
        "datasets.utils.filelock.BaseFileLock.__init__",
        autospec=True,
        side_effect=real_lock_init,
    ) as mock_lock:
        xds = from_huggingface(SAMPLE_DATASET_IDENTIFIER3, split="train")
        xds.execute()
        xds.fetch()
        assert mock_lock.call_count > 2
        lock_files = [call_arg.args[1] for call_arg in mock_lock.call_args_list]
        assert all(isinstance(f, str) for f in lock_files)
        assert len(lock_files) == len(set(lock_files))


@pytest.mark.parametrize(
    "path",
    [
        SAMPLE_DATASET_IDENTIFIER,
        SAMPLE_DATASET_IDENTIFIER2,
    ],
)
def test_to_dataframe_execute(setup, path):
    xds = from_huggingface(path, split="train")
    mdf = xds.to_dataframe()
    mdf.execute()
    df = mdf.fetch()
    assert isinstance(df, pd.DataFrame)
    assert len(df["text"]) > 0


@pytest.mark.parametrize(
    "path",
    [
        SAMPLE_DATASET_IDENTIFIER,
        SAMPLE_DATASET_IDENTIFIER2,
    ],
)
def test_map_execute(setup, path):
    def add_prefix(x):
        x["text"] = "xorbits: " + x["text"]
        return x

    xds = from_huggingface(path, split="train")
    xds = xds.map(add_prefix)
    xds.execute()
    ds = xds.fetch()
    assert isinstance(ds, datasets.Dataset)
    assert ds[0]["text"].startswith("xorbits:")


@pytest.mark.parametrize(
    "path",
    [
        SAMPLE_DATASET_IDENTIFIER,
        SAMPLE_DATASET_IDENTIFIER2,
    ],
)
def test_rechunk_execute(setup, path):
    xds = from_huggingface(path, split="train")
    xds = xds.rechunk(2)
    xds.execute()
    ds = xds.fetch()
    assert isinstance(ds, datasets.Dataset)
    assert len(ds.data.blocks) == 2


def test_getitem_execute(setup):
    xds = from_huggingface(SAMPLE_DATASET_IDENTIFIER2, split="train")
    xds = xds.rechunk(2)
    with pytest.raises(NotImplementedError):
        _ = xds[1:3:2]
    with pytest.raises(NotImplementedError):
        _ = xds[1, 2]
    a = xds["text"]
    assert a == ["foo"] * 10
    a = xds[5]
    assert a == {"text": "foo"}
    a = xds[:]
    assert type(a) == dict
    assert a["text"] == ["foo"] * 10
    a = xds[4:6]
    assert len(a["text"]) == 2
    a = xds[8:12]
    assert len(a["text"]) == 2
    # Check empty result.
    a = xds[10:12]
    assert a == {"text": []}
    a = xds[5:5]
    assert a == {"text": []}
    a = xds[5:4]
    assert a == {"text": []}


def test_export(setup):
    pass
    # db = from_huggingface("cifar10", split="train")
    # r = db.export("/Users/codingl2k1/Work/xorbits/python/out2", max_chunk_rows=1)
    # print(r)

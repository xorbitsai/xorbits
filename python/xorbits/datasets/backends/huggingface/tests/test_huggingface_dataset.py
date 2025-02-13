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
import glob
import json
import os
import shutil
import tempfile
from pathlib import Path

import datasets
import filelock
import pandas as pd
import pytest

from ....._mars.tests.core import mock
from ..from_huggingface import from_huggingface

SAMPLE_DATASET_IDENTIFIER = (
    "lhoestq/test"  # has dataset script. WARNING: now has been deleted
)
SAMPLE_DATASET_IDENTIFIER2 = "lhoestq/test2"  # only has data files
SAMPLE_DATASET_IDENTIFIER3 = (
    "hf-internal-testing/multi_dir_dataset"  # has multiple data directories
)
SAMPLE_DATASET_IDENTIFIER4 = "hf-internal-testing/imagefolder_with_metadata"  # imagefolder with a metadata file outside of the train/test directories


@pytest.mark.skip(reason="lhoestq/test repository has been deleted")
def test_split_arg_required():
    with pytest.raises(Exception, match="split"):
        from_huggingface(SAMPLE_DATASET_IDENTIFIER)


@pytest.mark.parametrize(
    "path",
    [
        # SAMPLE_DATASET_IDENTIFIER,
        SAMPLE_DATASET_IDENTIFIER2,
        SAMPLE_DATASET_IDENTIFIER3,
        SAMPLE_DATASET_IDENTIFIER4,
    ],
)
def test_from_huggingface_execute(setup, path):
    xds = from_huggingface(path, split="train", trust_remote_code=True)
    xds.execute()
    ds = xds.fetch()
    ds_expected = datasets.load_dataset(path, split="train")
    assert ds.to_dict() == ds_expected.to_dict()
    # Trigger datasets issue: https://github.com/huggingface/datasets/issues/6066
    # assert str(ds) == str(ds_expected)


def test_from_huggingface_file_lock(setup):
    real_lock_init = filelock.BaseFileLock.__init__

    with mock.patch(
        "filelock.BaseFileLock.__init__",
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
        # SAMPLE_DATASET_IDENTIFIER,
        SAMPLE_DATASET_IDENTIFIER2,
    ],
)
def test_to_dataframe_execute(setup, path):
    xds = from_huggingface(path, split="train", trust_remote_code=True)
    mdf = xds.to_dataframe()
    mdf.execute()
    df = mdf.fetch()
    assert isinstance(df, pd.DataFrame)
    assert len(df["text"]) > 0


@pytest.mark.parametrize(
    "path",
    [
        # SAMPLE_DATASET_IDENTIFIER,
        SAMPLE_DATASET_IDENTIFIER2,
    ],
)
def test_map_execute(setup, path):
    def add_prefix(x):
        x["text"] = "xorbits: " + x["text"]
        return x

    xds = from_huggingface(path, split="train", trust_remote_code=True)
    xds = xds.map(add_prefix)
    xds.execute()
    ds = xds.fetch()
    assert isinstance(ds, datasets.Dataset)
    assert ds[0]["text"].startswith("xorbits:")


@pytest.mark.parametrize(
    "path",
    [
        # SAMPLE_DATASET_IDENTIFIER,
        SAMPLE_DATASET_IDENTIFIER2,
    ],
)
def test_rechunk_execute(setup, path):
    xds = from_huggingface(path, split="train", trust_remote_code=True)
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
    tmp_dir = Path(tempfile.gettempdir())
    export_dir = tmp_dir.joinpath("test_export")
    shutil.rmtree(export_dir, ignore_errors=True)
    db = from_huggingface("cifar10", split="train", trust_remote_code=True)
    # Test invalid export dir
    Path(export_dir).touch()
    with pytest.raises(Exception, match="dir"):
        db.export(export_dir)
    os.remove(export_dir)
    # Test check version
    version_dir = export_dir.joinpath("0.0.0")
    os.makedirs(version_dir, exist_ok=True)
    with pytest.raises(Exception, match="exist"):
        db.export(export_dir, overwrite=False)
    # Test export
    shutil.rmtree(export_dir)
    try:
        db.export(export_dir, max_chunk_rows=100, create_if_not_exists=True)
        with open(version_dir.joinpath("info.json"), "r") as f:
            info = json.load(f)
        assert info["num_rows"] == 50000
        data_dir = version_dir.joinpath("data")
        with open(data_dir.joinpath(".meta", "info.json"), "r") as f:
            data_meta_info = json.load(f)
        data_arrow_files = glob.glob(data_dir.joinpath("*.arrow").as_posix())
        assert len(data_arrow_files) == 50000 / 100
        assert len(data_arrow_files) == data_meta_info["num_files"]
        assert info["num_rows"] == data_meta_info["num_rows"]

        db = from_huggingface("imdb", split="train", trust_remote_code=True)
        db.export(
            export_dir,
            column_groups={"my_text": ["text"], "my_label": ["label"]},
            max_chunk_rows=1000,
        )
        with open(version_dir.joinpath("info.json"), "r") as f:
            info = json.load(f)
        assert info["num_rows"] == 25000
        assert info["groups"] == ["my_text", "my_label"]
        my_text_dir = version_dir.joinpath("my_text")
        with open(my_text_dir.joinpath(".meta", "info.json"), "r") as f:
            my_text_meta_info = json.load(f)
        assert my_text_meta_info["num_columns"] == 1
    finally:
        shutil.rmtree(export_dir)

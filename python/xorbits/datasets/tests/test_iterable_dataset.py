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

import collections
import shutil
import tempfile
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from ..._mars.tests.core import mock
from ..backends.huggingface.from_huggingface import from_huggingface
from ..iterable_dataset import IterableDataset, map_retry


class MyObject(object):
    def my_method(self):
        pass


def make_dummy_object(_):
    return MyObject()


def test_map_retry():
    class MyException(Exception):
        pass

    def may_raise(i):
        if i == 1:
            raise MyException("test raise")
        return i

    call_count = [0, 0, 0]

    def raise_retry(i):
        call_count[i] += 1
        if call_count[i] == 3:
            return i
        raise MyException("test raise")

    call_count2 = [0, 0, 0]

    def raise_retry2(i):
        call_count2[i] += 1
        if call_count2[i] == 3:
            return i
        if i == 1:
            raise MyException("test raise")
        return i

    with ThreadPoolExecutor() as executor:
        r = map_retry(executor, may_raise, [0, 1, 2])
        with pytest.raises(MyException):
            list(r)

        r = map_retry(executor, lambda x: x * x, [0, 1, 2])
        assert list(r) == [0, 1, 4]

        r = map_retry(executor, raise_retry, [0, 1, 2], retry=2)
        list(r)
        assert call_count == [3, 3, 3]

        r = map_retry(executor, raise_retry2, [0, 1, 2], retry=2)
        list(r)
        assert call_count2 == [1, 3, 1]

        # Issue #14406: Result iterator should not keep an internal
        # reference to result objects.
        for obj in map_retry(executor, make_dummy_object, range(10)):
            wr = weakref.ref(obj)
            del obj
            assert wr() is None


def test_exception_handler():
    tmp_dir = Path(tempfile.gettempdir())

    export_dir = tmp_dir.joinpath("test_iterable_dataset")
    shutil.rmtree(export_dir, ignore_errors=True)
    db = from_huggingface("imdb", split="train")
    db.export(
        export_dir,
        column_groups={"my_text": ["text"], "my_label": ["label"]},
        max_chunk_rows=1000,
    )

    try:
        ds = IterableDataset(export_dir, shuffle=True)
        with mock.patch(
            "fsspec.implementations.local.LocalFileSystem.open"
        ) as mock_open:
            mock_open.side_effect = Exception("test raise when fetch")
            with pytest.raises(Exception, match="fetch"):
                len(list(ds))
        with mock.patch(
            "xorbits.datasets.iterable_dataset.Formatter.format_batch"
        ) as mock_format:
            mock_format.side_effect = Exception("test raise when format")
            with pytest.raises(Exception, match="format"):
                len(list(ds))
        with mock.patch(
            "xorbits.datasets.iterable_dataset.map_retry"
        ) as mock_map_retry:
            mock_map_retry.side_effect = Exception("test raise in _prefetcher")
            with pytest.raises(Exception, match="_prefetcher"):
                len(list(ds))
        with mock.patch("numpy.random.default_rng") as mock_default_rng:
            mock_default_rng.side_effect = Exception("test raise in _formatter")
            with pytest.raises(Exception, match="_formatter"):
                len(list(ds))
    finally:
        shutil.rmtree(export_dir, ignore_errors=True)


def test_iterable_dataset():
    import PIL.Image
    import torch

    tmp_dir = Path(tempfile.gettempdir())

    export_dir = tmp_dir.joinpath("test_iterable_dataset")
    shutil.rmtree(export_dir, ignore_errors=True)
    db = from_huggingface("imdb", split="train")
    db.export(
        export_dir,
        column_groups={"my_text": ["text"], "my_label": ["label"]},
        max_chunk_rows=1000,
    )

    try:
        ds = IterableDataset(
            export_dir, shuffle=False, distributed_rank=0, distributed_world_size=2
        )
        len1 = len(ds)
        len_list1 = len(list(ds))
        assert len1 == len_list1
        ds = IterableDataset(
            export_dir, shuffle=False, distributed_rank=1, distributed_world_size=2
        )
        len2 = len(ds)
        len_list2 = len(list(ds))
        assert len2 == len_list2
        assert len1 + len2 == len(IterableDataset(export_dir))
        # Check len does not affect shuffle.
        ds = IterableDataset(
            export_dir, shuffle=True, distributed_rank=2, distributed_world_size=5
        )
        before_len_list = list(ds)
        len(ds)
        after_len_list = list(ds)
        assert before_len_list == after_len_list

        ds = IterableDataset(
            export_dir, shuffle=True, distributed_rank=2, distributed_world_size=5
        )
        torch_result = list(torch.utils.data.DataLoader(ds, num_workers=2))
        assert len(torch_result) == len(after_len_list)
        assert {i["text"][0] for i in torch_result} == {
            i["text"] for i in after_len_list
        }
    finally:
        shutil.rmtree(export_dir, ignore_errors=True)

    export_dir = tmp_dir.joinpath("test_iterable_dataset")
    shutil.rmtree(export_dir, ignore_errors=True)
    db = from_huggingface("cifar10", split="train")
    db.export(export_dir)

    try:
        ds = IterableDataset(export_dir, shuffle=True)
        assert ds.schema.names == ["img", "label"]
        assert ds.column_groups == ["mdata", "data"]
        assert len(ds) == 50000
        idx = 0
        s = None
        for idx, s in enumerate(ds):
            pass
        assert idx == 50000 - 1
        assert s.keys() == {"img", "label"}
        assert isinstance(s["img"], PIL.Image.Image)
        assert type(s["label"]) is int
        assert ds.epoch() == 0
        labels1 = [s["label"] for s in ds]
        labels2 = [s["label"] for s in ds]
        assert labels1 == labels2
        ds.set_epoch(1)
        assert ds.epoch() == 1
        labels3 = [s["label"] for s in ds]
        labels4 = [s["label"] for s in ds]
        assert labels3 == labels4
        assert labels1 != labels3
        assert len(labels1) == len(labels3)
        counter1 = collections.Counter(labels1)
        counter3 = collections.Counter(labels3)
        assert counter1 == counter3
        dss1 = IterableDataset(
            export_dir, distributed_rank=0, distributed_world_size=5, shuffle=True
        )
        labels1 = [s["label"] for s in dss1]
        dss2 = IterableDataset(
            export_dir, distributed_rank=1, distributed_world_size=5, shuffle=True
        )
        labels2 = [s["label"] for s in dss2]
        dss3 = IterableDataset(
            export_dir, distributed_rank=2, distributed_world_size=5, shuffle=True
        )
        labels3 = [s["label"] for s in dss3]
        dss4 = IterableDataset(
            export_dir, distributed_rank=3, distributed_world_size=5, shuffle=True
        )
        labels4 = [s["label"] for s in dss4]
        dss5 = IterableDataset(
            export_dir, distributed_rank=4, distributed_world_size=5, shuffle=True
        )
        labels5 = [s["label"] for s in dss5]
        assert (
            len(labels1) + len(labels2) + len(labels3) + len(labels4) + len(labels5)
            == 50000
        )
        assert (
            collections.Counter(labels1)
            + collections.Counter(labels2)
            + collections.Counter(labels3)
            + collections.Counter(labels4)
            + collections.Counter(labels5)
            == counter1
        )
    finally:
        shutil.rmtree(export_dir, ignore_errors=True)

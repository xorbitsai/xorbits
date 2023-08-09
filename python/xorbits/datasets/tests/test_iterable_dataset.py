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

import collections
import shutil
import tempfile
from pathlib import Path

from ..backends.huggingface.from_huggingface import from_huggingface
from ..iterable_dataset import IterableDataset


def test_iterable_dataset():
    import PIL.Image

    tmp_dir = Path(tempfile.gettempdir())
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
            print(idx)
        assert idx == 50000 - 1
        assert s.keys() == {"img", "label"}
        assert isinstance(s["img"], PIL.Image.Image)
        assert type(s["label"]) is int
        assert ds.epoch() == 0
        labels1 = [s["label"] for s in ds]
        labels2 = [s["label"] for s in ds]
        assert labels1 == labels2
        ds.set_epoch(1)
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
        shutil.rmtree(export_dir)

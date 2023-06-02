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

import os
import pkgutil
import subprocess as sp
import sys
import tempfile
from typing import Any

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sps
from xoscar.serialization import AioDeserializer, AioSerializer

from ...lib.filesystem import LocalFileSystem
from ...lib.sparse import SparseMatrix, SparseNDArray
from ...tests.core import require_cudf, require_cupy
from ..base import StorageLevel
from ..cuda import CudaStorage
from ..filesystem import AlluxioStorage, DiskStorage, JuiceFSStorage
from ..plasma import PlasmaStorage
from ..shared_memory import SharedMemoryStorage
from ..vineyard import VineyardStorage

try:
    import vineyard
except ImportError:
    vineyard = None

require_lib = lambda x: x
params = [
    "filesystem",
    "shared_memory",
]
if (
    not sys.platform.startswith("win")
    and pkgutil.find_loader("pyarrow.plasma") is not None
):
    params.append("plasma")
alluxio = sp.getoutput("echo $ALLUXIO_HOME")
juicefs = sp.getoutput("echo $JUICEFS_HOME")
if "alluxio" in alluxio:
    params.append("alluxio")
if "juicefs" in juicefs:
    params.append("juicefs")
if vineyard is not None:
    params.append("vineyard")


@pytest.fixture(params=params)
async def storage_context(request):
    if request.param == "filesystem":
        tempdir = tempfile.mkdtemp()
        params, teardown_params = await DiskStorage.setup(
            fs=LocalFileSystem(), root_dirs=[tempdir]
        )
        storage = DiskStorage(**params)
        assert storage.level == StorageLevel.DISK

        yield storage

        await storage.teardown(**teardown_params)
    elif request.param == "alluxio":
        tempdir = tempfile.mkdtemp()
        params, teardown_params = await AlluxioStorage.setup(
            root_dir=tempdir, local_environ=True
        )
        storage = AlluxioStorage(**params)
        assert storage.level == StorageLevel.MEMORY

        yield storage
        await storage.teardown(**teardown_params)
    elif request.param == "juicefs":
        tempdir = tempfile.mkdtemp()
        params, teardown_params = await JuiceFSStorage.setup(
            root_dirs=[tempdir],
            local_environ=True,
            metadata_url="redis://127.0.0.1:6379/1",
        )
        storage = JuiceFSStorage(**params)
        assert storage.level == StorageLevel.MEMORY

        yield storage
        await storage.teardown(**teardown_params)

        with pytest.raises(
            ValueError,
            match="For external storage JuiceFS, you must specify the metadata url for its metadata storage, for example 'redis://172.17.0.5:6379/1'.",
        ):
            await JuiceFSStorage.setup(root_dirs=[tempdir], local_environ=True)
    elif request.param == "plasma":
        plasma_storage_size = 10 * 1024 * 1024
        if sys.platform == "darwin":
            plasma_dir = "/tmp"
        else:
            plasma_dir = "/dev/shm"
        params, teardown_params = await PlasmaStorage.setup(
            store_memory=plasma_storage_size,
            plasma_directory=plasma_dir,
            check_dir_size=False,
        )
        storage = PlasmaStorage(**params)
        assert storage.level == StorageLevel.MEMORY

        yield storage

        await PlasmaStorage.teardown(**teardown_params)
    elif request.param == "vineyard":
        vineyard_size = "256M"
        params, teardown_params = await VineyardStorage.setup(
            vineyard_size=vineyard_size
        )
        storage = VineyardStorage(**params)
        assert storage.level == StorageLevel.MEMORY

        yield storage

        await VineyardStorage.teardown(**teardown_params)
    elif request.param == "shared_memory":
        params, teardown_params = await SharedMemoryStorage.setup()
        storage = SharedMemoryStorage(**params)
        assert storage.level == StorageLevel.MEMORY

        yield storage

        teardown_params["object_ids"] = storage._object_ids
        await SharedMemoryStorage.teardown(**teardown_params)


def test_storage_level():
    level = StorageLevel.DISK | StorageLevel.MEMORY
    assert level == StorageLevel.DISK.value | StorageLevel.MEMORY.value

    assert (StorageLevel.DISK | StorageLevel.MEMORY) & StorageLevel.DISK
    assert not (StorageLevel.DISK | StorageLevel.MEMORY) & StorageLevel.GPU

    assert StorageLevel.GPU < StorageLevel.MEMORY < StorageLevel.DISK
    assert StorageLevel.DISK > StorageLevel.MEMORY > StorageLevel.GPU


@pytest.mark.asyncio
@require_lib
async def test_base_operations(storage_context):
    storage = storage_context

    data1 = np.random.rand(10, 10)
    put_info1 = await storage.put(data1)
    get_data1 = await storage.get(put_info1.object_id)
    np.testing.assert_array_equal(data1, get_data1)

    info1 = await storage.object_info(put_info1.object_id)
    # FIXME: remove os check when size issue fixed
    assert info1.size == put_info1.size

    data2 = pd.DataFrame(
        {
            "col1": np.arange(10),
            "col2": [f"str{i}" for i in range(10)],
            "col3": np.random.rand(10),
        },
    )
    put_info2 = await storage.put(data2)
    get_data2 = await storage.get(put_info2.object_id)
    pd.testing.assert_frame_equal(data2, get_data2)

    info2 = await storage.object_info(put_info2.object_id)
    # FIXME: remove os check when size issue fixed
    assert info2.size == put_info2.size

    # FIXME: remove when list functionality is ready for vineyard.
    if not isinstance(storage, (VineyardStorage, SharedMemoryStorage)):
        num = len(await storage.list())
        # juicefs automatically generates 4 files accesslog, config, stats and trash so the num should be 6 for juicefs
        if isinstance(storage, JuiceFSStorage):
            assert num == 6
        else:
            assert num == 2
        await storage.delete(info2.object_id)

    # test SparseMatrix
    s1 = sps.csr_matrix([[1, 0, 1], [0, 0, 1]])
    s = SparseNDArray(s1)
    put_info3 = await storage.put(s)
    get_data3 = await storage.get(put_info3.object_id)
    assert isinstance(get_data3, SparseMatrix)
    np.testing.assert_array_equal(get_data3.toarray(), s1.A)
    np.testing.assert_array_equal(get_data3.todense(), s1.A)


@pytest.mark.asyncio
@require_lib
async def test_reader_and_writer(storage_context):
    storage = storage_context

    if isinstance(storage, VineyardStorage):
        pytest.skip(
            "open_{reader,writer} in vineyard doesn't use the DEFAULT_SERIALIZATION"
        )

    # test writer and reader
    t = np.random.random(10)
    buffers = await AioSerializer(t).run()
    size = sum(getattr(buf, "nbytes", len(buf)) for buf in buffers)
    async with await storage.open_writer(size=size) as writer:
        for buf in buffers:
            await writer.write(buf)

    async with await storage.open_reader(writer.object_id) as reader:
        r = await AioDeserializer(reader).run()

    np.testing.assert_array_equal(t, r)

    # test writer and reader with seek offset
    t = np.random.random(10)
    buffers = await AioSerializer(t).run()
    size = sum(getattr(buf, "nbytes", len(buf)) for buf in buffers)
    async with await storage.open_writer(size=20 + size) as writer:
        await writer.write(b" " * 10)
        for buf in buffers:
            await writer.write(buf)
        await writer.write(b" " * 10)

    async with await storage.open_reader(writer.object_id) as reader:
        with pytest.raises((OSError, ValueError)):
            await reader.seek(-1)

        assert 5 == await reader.seek(5)
        assert 10 == await reader.seek(5, os.SEEK_CUR)
        assert 10 == await reader.seek(-10 - size, os.SEEK_END)
        assert 10 == await reader.tell()
        r = await AioDeserializer(reader).run()

    np.testing.assert_array_equal(t, r)


@pytest.mark.asyncio
@require_lib
async def test_reader_and_writer_vineyard(storage_context):
    storage = storage_context

    if not isinstance(storage, VineyardStorage):
        pytest.skip(
            "open_{reader,writer} in vineyard doesn't use the DEFAULT_SERIALIZATION"
        )

    # test writer and reader
    t = np.random.random(10)
    tinfo = await storage.put(t)

    # testing the roundtrip of `open_{reader,writer}`.

    buffers = []
    async with await storage.open_reader(tinfo.object_id) as reader:
        while True:
            buf = await reader.read()
            if buf:
                buffers.append(buf)
            else:
                break

    writer_object_id = None
    async with await storage.open_writer() as writer:
        for buf in buffers:
            await writer.write(buf)

        # The `object_id` of `StorageFileObject` returned by `open_writer` in vineyard
        # storage only available after `close` and before `__exit__` of `AioFileObject`.
        #
        # As `StorageFileObject.object_id` is only used for testing here, I think its
        # fine to have such a hack.
        await writer.close()
        writer_object_id = writer._file._object_id

    t2 = await storage.get(writer_object_id)
    np.testing.assert_array_equal(t, t2)


def _gen_data_for_cuda_backend():
    import datetime

    import cudf
    import cupy

    for data in [
        # cupy array
        cupy.asarray(np.random.rand(10, 10)),
        # cudf df
        cudf.DataFrame(
            pd.DataFrame(
                {
                    "col1": np.arange(10),
                    "col2": [f"str{i}" for i in range(10)],
                    "col3": np.random.rand(10),
                },
            )
        ),
        # empty df
        cudf.DataFrame(),
        # just a ``None`` obj
        None,
        # python object
        [
            datetime.datetime.now(),
            "Los Angeles Lakers",
            6,
            pd.DataFrame(np.random.rand(20, 10)),
            np.ones((5, 5)),
        ],
        # both python obj and cupy/cudf obj
        (
            ("xxx-yyy", "data"),
            cudf.DataFrame(np.arange(50)),
            cupy.ones((8, 8)) + 1,
            pd.DataFrame(),
        ),
    ]:
        yield data


def _compare_single_obj(obj1: Any, obj2: Any):
    import cudf
    import cupy

    if isinstance(obj1, cudf.DataFrame):
        cudf.testing.assert_frame_equal(obj1, obj2)
    elif isinstance(obj1, cupy.ndarray):
        cupy.testing.assert_array_equal(obj1, obj2)
    elif isinstance(obj1, pd.DataFrame):
        pd.testing.assert_frame_equal(obj1, obj2)
    elif isinstance(obj1, np.ndarray):
        np.testing.assert_array_equal(obj1, obj2)
    else:
        assert obj1 == obj2


def _compare_objs(obj1: Any, obj2: Any):
    if isinstance(obj1, (list, tuple)):
        for o1, o2 in zip(obj1, obj2):
            _compare_single_obj(o1, o2)
    else:
        _compare_single_obj(obj1, obj2)


@require_cupy
@require_cudf
@pytest.mark.asyncio
@pytest.mark.parametrize("data", _gen_data_for_cuda_backend())
async def test_cuda_backend(data):
    params, teardown_params = await CudaStorage.setup()
    storage = CudaStorage(**params)
    assert storage.level == StorageLevel.GPU

    put_info = await storage.put(data)
    get_data = await storage.get(put_info.object_id)
    _compare_objs(data, get_data)

    with pytest.raises(NotImplementedError):
        await storage.get(put_info.object_id, conditions=[])

    info1 = await storage.object_info(put_info.object_id)
    assert info1.size == put_info.size

    read_chunk = 100
    writer = await storage.open_writer(put_info.size)
    async with await storage.open_reader(put_info.object_id) as reader:
        while True:
            content = await reader.read(read_chunk)
            if not (isinstance(content, str) and content == ""):
                await writer.write(content)
            else:
                break
    writer._file._write_close()
    write_data = await storage.get(writer._file._object_id)
    _compare_objs(write_data, get_data)
    await storage.delete(put_info.object_id)

    # data1 = cupy.asarray(np.random.rand(10, 10))
    # put_info1 = await storage.put(data1)
    # get_data1 = await storage.get(put_info1.object_id)
    # cupy.testing.assert_array_equal(data1, get_data1)
    #
    # with pytest.raises(NotImplementedError):
    #     await storage.get(put_info1.object_id, conditions=[])
    #
    # info1 = await storage.object_info(put_info1.object_id)
    # assert info1.size == put_info1.size
    #
    # data2 = cudf.DataFrame(
    #     pd.DataFrame(
    #         {
    #             "col1": np.arange(10),
    #             "col2": [f"str{i}" for i in range(10)],
    #             "col3": np.random.rand(10),
    #         },
    #     )
    # )
    # put_info2 = await storage.put(data2)
    # get_data2 = await storage.get(put_info2.object_id)
    # cudf.testing.assert_frame_equal(data2, get_data2)
    #
    # with pytest.raises(NotImplementedError):
    #     await storage.get(put_info2.object_id, conditions=[])
    #
    # info2 = await storage.object_info(put_info2.object_id)
    # assert info2.size == put_info2.size
    #
    # await CudaStorage.teardown(**teardown_params)
    #
    # # test writer and reader
    # for put_info, get_data in zip([put_info1, put_info2], [get_data1, get_data2]):
    #     read_chunk = 100
    #     writer = await storage.open_writer(put_info.size)
    #     async with await storage.open_reader(put_info.object_id) as reader:
    #         while True:
    #             content = await reader.read(read_chunk)
    #             if not (isinstance(content, str) and content == ""):
    #                 await writer.write(content)
    #             else:
    #                 break
    #     writer._file._write_close()
    #     write_data = await storage.get(writer._file._object_id)
    #
    #     if isinstance(get_data, cudf.DataFrame):
    #         cudf.testing.assert_frame_equal(write_data, get_data)
    #     else:
    #         cupy.testing.assert_array_equal(write_data, get_data)
    #
    # await storage.delete(put_info1.object_id)
    # await storage.delete(put_info2.object_id)

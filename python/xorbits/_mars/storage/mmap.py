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
import asyncio
import mmap
import os
import warnings
from typing import Any, Dict, List, Tuple, Union

import cloudpickle
from xoscar.serialization import deserialize
from xoscar.serialization.aio import BUFFER_SIZES_NAME, get_header_length

from ..utils import implements
from .base import ObjectInfo, StorageBackend, StorageLevel, register_storage_backend
from .core import BufferWrappedFileObject, StorageFileObject
from .filesystem import DiskStorage


class MMAPFileObject(BufferWrappedFileObject):
    def __init__(self, object_id: Any, view: memoryview, size: int):
        super().__init__(object_id, "r", size=size)
        self._mv = view
        self._buffer = view

    @property
    def object_id(self):
        return self._object_id

    @property
    def buffer(self):
        return self._buffer

    def _write_init(self):  # pragma: no cover
        raise NotImplementedError("MMAPFileObject only supports read operations.")

    def write(self, content: Union[bytes, memoryview]):  # pragma: no cover
        raise NotImplementedError("MMAPFileObject only supports read operations.")

    def _read_init(self):  # pragma: no cover
        pass

    def read(self, size=-1):
        offset = self._offset
        size = self._size if size < 0 else size
        end = min(self._size, offset + size)
        result = self._mv[offset:end]
        self._offset = end
        return result

    def _read_close(self):  # pragma: no cover
        pass

    def _write_close(self):  # pragma: no cover
        pass

    def seek(self, offset: int, whence: int = os.SEEK_SET):
        return self._get_new_offset(offset, whence=whence)


class MMAPStorageFileObject(StorageFileObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@register_storage_backend
class MMAPStorage(DiskStorage):
    name = "mmap"
    is_seekable = True

    def __init__(self, **kw):
        DiskStorage.__init__(self, **kw)

    @classmethod
    @implements(StorageBackend.setup)
    async def setup(cls, **kwargs) -> Tuple[Dict, Dict]:
        disk_init, disk_teardown = await DiskStorage.setup(**kwargs)

        return disk_init, disk_teardown

    @staticmethod
    @implements(StorageBackend.teardown)
    async def teardown(**kwargs):
        fs = kwargs.get("fs")
        root_dirs = kwargs.get("root_dirs")
        try:
            for d in root_dirs:
                fs.delete(d, recursive=True)
        except:  # pragma: no cover
            warnings.warn(
                f"Unexpected exceptions occur when deleting directories, "
                f"please delete these directories manually: {root_dirs}"
            )

    @property
    @implements(StorageBackend.level)
    def level(self) -> StorageLevel:
        return StorageLevel.MEMORY

    @property
    @implements(StorageBackend.size)
    def size(self) -> Union[int, None]:
        return self._size

    @staticmethod
    def _get_file_via_mmap(object_id: str):
        with open(object_id, "r+b") as f:
            mm = mmap.mmap(f.fileno(), 0)
            v = memoryview(mm)
            header_bytes = v[:11]
            header_length = get_header_length(header_bytes)
            header = cloudpickle.loads(v[11 : 11 + header_length])
            buffer_sizes = header[0].pop(BUFFER_SIZES_NAME)
            # get buffers
            buffers = []
            start = 11 + header_length
            for size in buffer_sizes:
                buffers.append(v[start : start + size])
                start = start + size
            return deserialize(header, buffers)

    @implements(StorageBackend.get)
    async def get(self, object_id, **kwargs) -> object:
        return await asyncio.to_thread(self._get_file_via_mmap, object_id)

    @implements(StorageBackend.put)
    async def put(self, obj, importance=0) -> ObjectInfo:
        return await super().put(obj, importance=importance)

    @implements(StorageBackend.object_info)
    async def object_info(self, object_id) -> ObjectInfo:
        return await super().object_info(object_id)

    @implements(StorageBackend.delete)
    async def delete(self, object_id):
        await super().delete(object_id)

    @implements(StorageBackend.open_writer)
    async def open_writer(self, size=None) -> StorageFileObject:
        return await super().open_writer(size=size)

    @implements(StorageBackend.open_reader)
    async def open_reader(self, object_id) -> StorageFileObject:
        with open(object_id, "r+b") as f:
            mm = mmap.mmap(f.fileno(), 0)
            file = MMAPFileObject(object_id, memoryview(mm), mm.size())
            return MMAPStorageFileObject(file, object_id=object_id)

    @implements(StorageBackend.list)
    async def list(self) -> List:  # pragma: no cover
        raise NotImplementedError("MMAP storage does not support list")

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
import asyncio
import logging
import os
import uuid
from typing import Dict, List, Optional, Tuple

from xoscar.serialization import AioDeserializer, AioSerializer

from ..lib.aio import AioFilesystem
from ..lib.filesystem import FileSystem, LocalFileSystem, get_fs
from ..utils import implements, mod_hash
from .base import ObjectInfo, StorageBackend, StorageLevel, register_storage_backend
from .core import StorageFileObject

logger = logging.getLogger(__name__)


@register_storage_backend
class FileSystemStorage(StorageBackend):
    name = "filesystem"

    def __init__(
        self, fs: FileSystem, root_dirs: List[str], level: StorageLevel, size: int
    ):
        self._fs = AioFilesystem(fs)
        self._root_dirs = root_dirs
        self._level = level
        self._size = size

    @classmethod
    @implements(StorageBackend.setup)
    async def setup(cls, **kwargs) -> Tuple[Dict, Dict]:
        root_dirs = kwargs.pop("root_dirs")
        level = kwargs.pop("level")
        size = kwargs.pop("size", None)
        fs = kwargs.pop("fs", None)
        if kwargs:  # pragma: no cover
            raise TypeError(
                f'FileSystemStorage got unexpected config: {",".join(kwargs)}'
            )

        # `root_dirs` is actually a single directory here
        # but the interface is designed to accept a list of directories
        # so we need to split it if it's a single directory
        if isinstance(root_dirs, str):
            root_dirs = root_dirs.split(":")
        if isinstance(level, str):
            level = StorageLevel.from_str(level)

        if fs is None:
            fs = get_fs(root_dirs[0])

        for d in root_dirs:
            if not fs.exists(d):
                fs.mkdir(d)
        params = dict(fs=fs, root_dirs=root_dirs, level=level, size=size)
        return params, params

    @staticmethod
    @implements(StorageBackend.teardown)
    async def teardown(**kwargs):
        fs = kwargs.get("fs")
        root_dirs = kwargs.get("root_dirs")
        for d in root_dirs:
            fs.delete(d, recursive=True)

    @property
    @implements(StorageBackend.level)
    def level(self) -> StorageLevel:
        return self._level

    @property
    @implements(StorageBackend.size)
    def size(self) -> Optional[int]:
        return self._size

    def _generate_path(self):
        file_name = str(uuid.uuid4())
        selected_index = mod_hash(file_name, len(self._root_dirs))
        selected_dir = self._root_dirs[selected_index]
        return os.path.join(selected_dir, file_name)

    @implements(StorageBackend.get)
    async def get(self, object_id, **kwargs) -> object:
        if kwargs:  # pragma: no cover
            raise NotImplementedError(f'Got unsupported args: {",".join(kwargs)}')

        file = await self._fs.open(object_id, "rb")
        async with file as f:
            deserializer = AioDeserializer(f)
            return await deserializer.run()

    @implements(StorageBackend.put)
    async def put(self, obj, importance: int = 0) -> ObjectInfo:
        serializer = AioSerializer(obj)
        buffers = await serializer.run()
        buffer_size = sum(getattr(buf, "nbytes", len(buf)) for buf in buffers)

        path = self._generate_path()
        file = await self._fs.open(path, "wb")
        async with file as f:
            for buffer in buffers:
                await f.write(buffer)

        return ObjectInfo(size=buffer_size, object_id=path)

    @implements(StorageBackend.delete)
    async def delete(self, object_id):
        await self._fs.delete(object_id)

    @implements(StorageBackend.list)
    async def list(self) -> List:
        file_list = []
        for d in self._root_dirs:
            file_list.extend(list(await self._fs.ls(d)))
        return file_list

    @implements(StorageBackend.object_info)
    async def object_info(self, object_id) -> ObjectInfo:
        stat = await self._fs.stat(object_id)
        return ObjectInfo(size=stat["size"], object_id=object_id)

    @implements(StorageBackend.open_writer)
    async def open_writer(self, size=None) -> StorageFileObject:
        path = self._generate_path()
        file = await self._fs.open(path, "wb")
        return StorageFileObject(file, file.name)

    @implements(StorageBackend.open_reader)
    async def open_reader(self, object_id) -> StorageFileObject:
        file = await self._fs.open(object_id, "rb")
        return StorageFileObject(file, file.name)


@register_storage_backend
class DiskStorage(FileSystemStorage):
    name = "disk"

    @classmethod
    @implements(StorageBackend.setup)
    async def setup(cls, **kwargs) -> Tuple[Dict, Dict]:
        kwargs["level"] = StorageLevel.DISK
        return await super().setup(**kwargs)


@register_storage_backend
class AlluxioStorage(FileSystemStorage):
    name = "alluxio"

    def __init__(
        self,
        root_dirs: List[str],
        local_environ: bool,  # local_environ means standalone mode
        level: StorageLevel = None,
        size: int = None,
    ):
        self._fs = AioFilesystem(LocalFileSystem())
        self._root_dirs = root_dirs
        self._level = level
        self._size = size
        self._local_environ = local_environ

    @classmethod
    @implements(StorageBackend.setup)
    async def setup(cls, **kwargs) -> Tuple[Dict, Dict]:
        kwargs["level"] = StorageLevel.MEMORY
        root_dirs = kwargs.get("root_dirs")
        # `root_dirs` is actually a single directory here
        # but the interface is designed to accept a list of directories
        # so we need to split it if it's a single directory
        if isinstance(root_dirs, str):
            root_dirs = root_dirs.split(":")
        local_environ = kwargs.get("local_environ")
        if local_environ:
            proc = await asyncio.create_subprocess_shell(
                f"""$ALLUXIO_HOME/bin/alluxio fs mkdir /alluxio-storage
                $ALLUXIO_HOME/integration/fuse/bin/alluxio-fuse mount {root_dirs[0]} /alluxio-storage
                """
            )
            await proc.wait()
        params = dict(
            root_dirs=root_dirs,
            level=StorageLevel.MEMORY,
            size=None,
            local_environ=local_environ,
        )
        return params, dict(root_dirs=root_dirs)

    @staticmethod
    @implements(StorageBackend.teardown)
    async def teardown(**kwargs):
        root_dirs = kwargs.get("root_dirs")
        proc = await asyncio.create_subprocess_shell(
            f"""$ALLUXIO_HOME/integration/fuse/bin/alluxio-fuse unmount {root_dirs[0]} /alluxio-storage
            $ALLUXIO_HOME/bin/alluxio fs rm -R /alluxio-storage
            """
        )
        await proc.wait()


@register_storage_backend
class JuiceFSStorage(FileSystemStorage):
    name = "juicefs"

    def __init__(
        self,
        root_dirs: List[str],
        in_k8s: bool = False,
        local_environ: bool = False,  # local_environ means standalone mode
        level: StorageLevel = None,
        size: int = None,
    ):
        self._root_dirs = root_dirs
        self._in_k8s = in_k8s
        if not self._in_k8s:
            self._fs = AioFilesystem(LocalFileSystem())
            self._level = level
            self._size = size
            self._local_environ = local_environ

    @classmethod
    @implements(StorageBackend.setup)
    async def setup(cls, **kwargs) -> Tuple[Dict, Dict]:
        kwargs["level"] = StorageLevel.MEMORY
        in_k8s = kwargs.get("in_k8s")
        root_dirs = kwargs.get("root_dirs")
        # `root_dirs` is actually a single directory here
        # but the interface is designed to accept a list of directories
        # so we need to split it if it's a single directory
        if isinstance(root_dirs, str):
            root_dirs = root_dirs.split(":")
        params = dict(
            root_dirs=root_dirs,
            level=StorageLevel.MEMORY,
            size=None,
        )
        if not in_k8s:
            local_environ = kwargs.get("local_environ")
            metadata_url = kwargs.get("metadata_url", None)
            if metadata_url is None:
                raise ValueError(
                    "For external storage JuiceFS, you must specify the metadata url for its metadata storage, for example 'redis://172.17.0.5:6379/1'."
                )
            if local_environ:
                proc = await asyncio.create_subprocess_shell(
                    f"""juicefs format {metadata_url} jfs
                    juicefs mount {metadata_url} {root_dirs[0]} -d
                    """
                )
                await proc.wait()
            params["local_environ"] = local_environ
        return params, dict(root_dirs=root_dirs)

    @staticmethod
    @implements(StorageBackend.teardown)
    async def teardown(**kwargs):
        in_k8s = kwargs.get("in_k8s")
        if not in_k8s:
            root_dirs = kwargs.get("root_dirs")
            proc = await asyncio.create_subprocess_shell(
                f"""juicefs umount {root_dirs[0]}
                """
            )
            await proc.wait()

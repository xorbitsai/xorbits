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
import os
from abc import ABC, abstractmethod
from concurrent.futures import Executor
from typing import Any, Optional, Union

from ..lib.aio import AioFileObject


class StorageFileObject(AioFileObject):
    def __init__(
        self,
        file: Any,
        object_id: Any,
        loop: asyncio.BaseEventLoop = None,
        executor: Executor = None,
    ):
        self._object_id = object_id
        super().__init__(file, loop=loop, executor=executor)

    @property
    def object_id(self):
        return self._object_id

    @property
    def buffer(self):
        if hasattr(self._file, "buffer"):
            return self._file.buffer

    @property
    def header(self):
        if hasattr(self._file, "header"):
            return self._file.header

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await super().__aexit__(exc_type, exc_val, exc_tb)
        if self._executor:
            self._executor.shutdown(wait=False)


class BufferWrappedFileObject(ABC):
    def __init__(self, object_id: Any, mode: str, size: Optional[int] = None):
        # check arguments
        assert mode in ("w", "r"), 'mode must be "w" or "r"'
        if mode == "w" and size is None:  # pragma: no cover
            raise ValueError("size must be provided to write")

        self._object_id = object_id
        self._size = size
        self._mode = mode

        self._offset = 0
        self._initialized = False
        self._closed = False

        self._mv = None
        self._buffer = None

    @abstractmethod
    def _read_init(self):
        """
        Initialization for read purpose.
        """

    @abstractmethod
    def _write_init(self):
        """
        Initialization for write purpose.
        """

    @property
    def object_id(self):
        return self._object_id

    @property
    def buffer(self):
        self.init()
        return self._buffer

    @property
    def mode(self):
        return self._mode

    def init(self):
        if not self._initialized:
            if self.mode == "w":
                self._write_init()
            else:
                self._read_init()
            self._initialized = True

    def read(self, size=-1):
        if not self._initialized:
            self._read_init()
            self._initialized = True

        offset = self._offset
        size = self._size if size < 0 else size
        end = min(self._size, offset + size)
        result = self._mv[offset:end]
        self._offset = end
        return result

    def write(self, content: Union[bytes, memoryview]):
        if not self._initialized:
            self._write_init()
            self._initialized = True

        offset = self._offset
        content_length = getattr(content, "nbytes", len(content))
        new_offset = offset + content_length
        self._mv[offset:new_offset] = content
        self._offset = new_offset

    def _get_new_offset(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_END:
            new_offset = self._size + offset
        elif whence == os.SEEK_CUR:
            new_offset = self._offset + offset
        else:
            assert whence == os.SEEK_SET
            new_offset = offset
        if new_offset < 0 or new_offset >= self._size:
            raise ValueError(
                f"File offset should be limited to (0, {self._size}), "
                f"now is {new_offset}"
            )
        self._offset = new_offset
        return self._offset

    def seek(self, offset: int, whence: int = os.SEEK_SET):
        if not self._initialized:
            self._read_init()
            self._initialized = True

        return self._get_new_offset(offset, whence=whence)

    def tell(self):
        return self._offset

    @abstractmethod
    def _read_close(self):
        """
        Close for read.
        """

    @abstractmethod
    def _write_close(self):
        """
        Close for write.
        """

    def close(self):
        if self._closed:
            return

        self._closed = True
        if self._mode == "w":
            self._write_close()
        else:
            self._read_close()
        self._mv = None
        self._buffer = None

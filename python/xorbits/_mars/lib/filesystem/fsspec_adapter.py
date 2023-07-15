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

from typing import BinaryIO, Dict, Iterator, List, TextIO, Tuple, Union
from urllib.parse import ParseResult, urlparse, urlunparse

from fsspec import filesystem
from fsspec.core import stringify_path

from ...utils import implements
from .core import FileSystem, path_type


class FsSpecAdapter(FileSystem):
    def __init__(self, scheme: str, **kwargs):
        self._fs = filesystem(scheme, **kwargs)
        self._scheme = scheme

    @implements(FileSystem.cat)
    def cat(self, path: path_type) -> bytes:
        return self._fs.cat_file(self._normalize_path(path))

    @implements(FileSystem.ls)
    def ls(self, path: path_type) -> List[path_type]:
        if self.isfile(path):  # pragma: no cover
            return self._append_scheme(
                self._fs.ls(self._normalize_path(path), detail=False)
            )
        entries = []
        for entry in self._fs.ls(self._normalize_path(path), detail=False):
            if entry.strip("/") in path.strip("/"):  # pragma: no cover
                continue
            if isinstance(entry, Dict):
                entries.append(entry.get("name"))
            elif isinstance(entry, str):
                entries.append(entry)
            else:  # pragma: no cover
                raise TypeError(f"Expect str or dict, but got {type(entry)}")
        return self._append_scheme(entries)

    @implements(FileSystem.delete)
    def delete(self, path: path_type, recursive: bool = False):
        raise NotImplementedError

    @implements(FileSystem.stat)
    def stat(self, path: path_type) -> Dict:
        return self._fs.info(self._normalize_path(path))

    @implements(FileSystem.rename)
    def rename(self, path: path_type, new_path: path_type):
        raise NotImplementedError

    @implements(FileSystem.mkdir)
    def mkdir(self, path: path_type, create_parents: bool = True):
        raise NotImplementedError

    @implements(FileSystem.exists)
    def exists(self, path: path_type):
        return self._fs.exists(self._normalize_path(path))

    @implements(FileSystem.isdir)
    def isdir(self, path: path_type) -> bool:
        return self._fs.isdir(self._normalize_path(path))

    @implements(FileSystem.isfile)
    def isfile(self, path: path_type) -> bool:
        return self._fs.isfile(self._normalize_path(path))

    @implements(FileSystem._isfilestore)
    def _isfilestore(self) -> bool:
        raise NotImplementedError

    @implements(FileSystem.open)
    def open(self, path: path_type, mode: str = "rb") -> Union[BinaryIO, TextIO]:
        return self._fs.open(self._normalize_path(path), mode=mode)

    @implements(FileSystem.walk)
    def walk(self, path: path_type) -> Iterator[Tuple[str, List[str], List[str]]]:
        for root, dirs, files in self._fs.walk(path):
            yield self._append_scheme([root])[0], self._append_scheme(
                dirs
            ), self._append_scheme(files)

    @implements(FileSystem.glob)
    def glob(self, path: path_type, recursive: bool = False) -> List[path_type]:
        from ._glob import FileSystemGlob

        return self._append_scheme(
            FileSystemGlob(self).glob(self._normalize_path(path), recursive=recursive)
        )

    @staticmethod
    def _normalize_path(path: path_type) -> str:
        """
        Stringify path and remove its scheme.
        """
        path_str = stringify_path(path)
        parsed = urlparse(path_str)
        if parsed.scheme:
            return urlunparse(
                ParseResult(
                    scheme="",
                    netloc=parsed.netloc,
                    path=parsed.path,
                    params="",
                    query="",
                    fragment="",
                )
            )
        else:
            return path_str

    def _append_scheme(self, paths: List[path_type]) -> List[path_type]:
        return [
            urlunparse(
                ParseResult(
                    scheme=self._scheme,
                    netloc="",
                    path=path,
                    params="",
                    query="",
                    fragment="",
                )
            )
            for path in paths
        ]

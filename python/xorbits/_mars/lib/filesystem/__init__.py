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

from .azure import AzureBlobFileSystem
from .base import FileSystem
from .core import file_size, get_fs, get_scheme, glob, open_file, register_filesystem
from .fsmap import FSMap

# noinspection PyUnresolvedReferences
from .hdfs import HadoopFileSystem
from .local import LocalFileSystem
from .s3 import S3FileSystem

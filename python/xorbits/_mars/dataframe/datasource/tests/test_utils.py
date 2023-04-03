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
import os.path

from ..utils import convert_to_abspath


def test_convert_to_abspath():
    s3_path = "s3://"
    local_path = "./test.csv"
    local_path_2 = "./test_2.csv"
    try:
        # test non-local path convert
        s3_abspath = convert_to_abspath(s3_path)
        assert s3_abspath == s3_path
        # test local path convert
        local_relpath = convert_to_abspath(local_path)
        assert local_relpath == os.path.abspath(local_path)
        # test local path list convert
        local_relpath_lst = [local_path, local_path_2]
        local_abspath_lst = convert_to_abspath(local_relpath_lst)
        assert local_abspath_lst[0] == os.path.abspath(local_relpath_lst[0])
        assert local_abspath_lst[1] == os.path.abspath(local_relpath_lst[1])
        # test non-local/local path list convert
        mix_relpath_lst = [s3_path, local_path]
        mix_abspath_lst = convert_to_abspath(mix_relpath_lst)
        assert mix_abspath_lst[0] == mix_relpath_lst[0]
        assert mix_abspath_lst[1] == os.path.abspath(mix_relpath_lst[1])
    finally:
        pass

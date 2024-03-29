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

# noinspection PyUnresolvedReferences
from sklearn.utils import gen_batches

from .collect_ports import collect_ports
from .core import (
    concat_chunks,
    convert_to_tensor_or_dataframe,
    copy_learned_attributes,
    get_chunk_n_rows,
)
from .shuffle import shuffle
from .validation import (
    assert_all_finite,
    check_array,
    check_consistent_length,
    check_X_y,
    column_or_1d,
)

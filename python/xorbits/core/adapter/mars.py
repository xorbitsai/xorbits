# Copyright 2022 XProbe Inc.
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

from core.data.core import DataType

from ..._mars.mars.core import OutputType, register_output_types
from ..._mars.mars.dataframe.core import DataFrameData
from ..._mars.mars.dataframe.core import DataFrame


_DATATYPE_TO_DATA_CLS = {DataType.dataframe: DataFrameData}
_DATATYPE_TO_DATA_REF_CLS = {DataType.dataframe: DataFrame}


def get_data_cls(data_type: DataType):
    return _DATATYPE_TO_DATA_CLS[data_type]


def get_data_ref_cls(data_type: DataType):
    return _DATATYPE_TO_DATA_REF_CLS[data_type]


def register(data_cls, data_ref_cls):
    register_output_types(
        OutputType.dataframe, (data_ref_cls, data_cls), (data_ref_cls, data_cls)
    )

# -*- coding: utf-8 -*-
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

# To avoid possible naming conflict, mars functions and classes should be renamed.
# Functions should be renamed by adding a prefix 'mars_', and classes should be renamed
# by adding a prefix 'Mars'.

from .._mars import dataframe as mars_dataframe
from .._mars import execute as mars_execute
from .._mars import new_session as mars_new_session
from .._mars import stop_server as mars_stop_server
from .._mars.core import Entity as MarsEntity
from .._mars.dataframe.core import DataFrame as MarsDataFrame
from .._mars.dataframe.core import DataFrameGroupBy as MarsDataFrameGroupBy
from .._mars.dataframe.core import Index as MarsIndex
from .._mars.dataframe.core import Series as MarsSeries
from .._mars.dataframe.datasource.dataframe import (
    DataFrameDataSource as MarsDataFrameDataSource,
)
from .._mars.dataframe.indexing.loc import DataFrameLoc as MarsDataFrameLoc

__all__ = [
    "mars_dataframe",
    "mars_execute",
    "mars_new_session",
    "mars_stop_server",
    "MarsDataFrame",
    "MarsDataFrameDataSource",
    "MarsDataFrameGroupBy",
    "MarsDataFrameLoc",
    "MarsEntity",
    "MarsIndex",
    "MarsSeries",
]

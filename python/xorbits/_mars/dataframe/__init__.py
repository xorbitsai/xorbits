#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

from . import arithmetic, base, datastore, groupby, indexing
from . import merge as merge_
from . import missing, plotting, reduction, sort, statistics, ufunc, window

# do imports to register operands
from .base.cut import cut
from .base.eval import mars_eval as eval  # pylint: disable=redefined-builtin
from .base.get_dummies import get_dummies
from .base.melt import melt
from .base.qcut import qcut
from .base.to_numeric import to_numeric
from .base.value_counts import value_counts
from .contrib.raydataset import to_ray_dataset, to_ray_mldataset
from .datasource.date_range import date_range
from .datasource.from_index import series_from_index
from .datasource.from_records import from_records
from .datasource.from_tensor import dataframe_from_tensor, series_from_tensor
from .datasource.from_vineyard import from_vineyard
from .datasource.read_csv import read_csv
from .datasource.read_parquet import read_parquet
from .datasource.read_raydataset import (
    read_ray_dataset,
    read_ray_mldataset,
    read_raydataset,
)
from .datasource.read_sql import read_sql, read_sql_query, read_sql_table
from .fetch import DataFrameFetch, DataFrameFetchShuffle
from .initializer import DataFrame, Index, Series
from .merge import concat, merge
from .missing.checkna import isna, isnull, notna, notnull
from .reduction import CustomReduction, unique
from .tseries.to_datetime import to_datetime

del (
    reduction,
    statistics,
    arithmetic,
    indexing,
    merge_,
    base,
    groupby,
    missing,
    ufunc,
    datastore,
    sort,
    window,
    plotting,
)
del DataFrameFetch, DataFrameFetchShuffle

# noinspection PyUnresolvedReferences
from pandas import (
    BooleanDtype,
    CategoricalDtype,
    DateOffset,
    DatetimeTZDtype,
    Int8Dtype,
    Int16Dtype,
    Int32Dtype,
    Int64Dtype,
    Interval,
    IntervalDtype,
    NaT,
    PeriodDtype,
    SparseDtype,
    StringDtype,
    Timedelta,
    Timestamp,
    UInt8Dtype,
    UInt16Dtype,
    UInt32Dtype,
    UInt64Dtype,
    offsets,
)

# noinspection PyUnresolvedReferences
from .arrays import ArrowListArray, ArrowListDtype, ArrowStringArray, ArrowStringDtype
from .core import (
    CategoricalIndex,
    DatetimeIndex,
    Float64Index,
    IntervalIndex,
    MultiIndex,
    PeriodIndex,
    RangeIndex,
    TimedeltaIndex,
    UInt64Index,
)

try:
    from pandas import NA, NamedAgg
except ImportError:  # pragma: no cover
    pass

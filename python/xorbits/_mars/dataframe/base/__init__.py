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
from .apply import df_apply, series_apply
from .applymap import df_applymap
from .astype import astype, index_astype
from .cartesian_chunk import cartesian_chunk
from .check_monotonic import (
    check_monotonic,
    is_monotonic,
    is_monotonic_decreasing,
    is_monotonic_increasing,
)
from .cut import cut
from .describe import describe
from .diff import df_diff, series_diff
from .drop import df_drop, df_pop, index_drop, series_drop
from .drop_duplicates import (
    df_drop_duplicates,
    index_drop_duplicates,
    series_drop_duplicates,
)
from .duplicated import df_duplicated, index_duplicated, series_duplicated
from .eval import df_eval, df_query
from .explode import df_explode, series_explode
from .isin import df_isin, series_isin
from .map import index_map, series_map
from .map_chunk import map_chunk
from .melt import melt
from .memory_usage import df_memory_usage, index_memory_usage, series_memory_usage
from .nlargest import dataframe_nlargest, series_nlargest
from .nsmallest import dataframe_nsmallest, series_nsmallest
from .nunique import nunique_dataframe, nunique_series
from .pct_change import pct_change
from .pivot import df_pivot
from .pivot_table import df_pivot_table
from .qcut import qcut
from .rebalance import rebalance
from .rechunk import rechunk
from .select_dtypes import select_dtypes
from .shift import shift, tshift
from .stack import stack
from .to_cpu import to_cpu
from .to_gpu import to_gpu
from .transform import df_transform, series_transform
from .transpose import transpose
from .value_counts import value_counts


def _install():
    from ..core import DATAFRAME_TYPE, INDEX_TYPE, SERIES_TYPE
    from .accessor import CachedAccessor, DatetimeAccessor, StringAccessor
    from .datetimes import _datetime_method_to_handlers
    from .standardize_range_index import ChunkStandardizeRangeIndex
    from .string_ import _string_method_to_handlers

    for t in DATAFRAME_TYPE:
        setattr(t, "to_gpu", to_gpu)
        setattr(t, "to_cpu", to_cpu)
        setattr(t, "rechunk", rechunk)
        setattr(t, "describe", describe)
        setattr(t, "apply", df_apply)
        setattr(t, "applymap", df_applymap)
        setattr(t, "transform", df_transform)
        setattr(t, "isin", df_isin)
        setattr(t, "shift", shift)
        setattr(t, "tshift", tshift)
        setattr(t, "diff", df_diff)
        setattr(t, "astype", astype)
        setattr(t, "drop", df_drop)
        setattr(t, "pop", df_pop)
        setattr(t, "pivot", df_pivot)
        setattr(t, "pivot_table", df_pivot_table)
        setattr(
            t, "__delitem__", lambda df, items: df_drop(df, items, axis=1, inplace=True)
        )
        setattr(t, "drop_duplicates", df_drop_duplicates)
        setattr(t, "duplicated", df_duplicated)
        setattr(t, "melt", melt)
        setattr(t, "memory_usage", df_memory_usage)
        setattr(t, "select_dtypes", select_dtypes)
        setattr(t, "map_chunk", map_chunk)
        setattr(t, "cartesian_chunk", cartesian_chunk)
        setattr(t, "rebalance", rebalance)
        setattr(t, "stack", stack)
        setattr(t, "explode", df_explode)
        setattr(t, "eval", df_eval)
        setattr(t, "query", df_query)
        setattr(t, "pct_change", pct_change)
        setattr(t, "transpose", transpose)
        setattr(t, "nlargest", dataframe_nlargest)
        setattr(t, "nsmallest", dataframe_nsmallest)
        setattr(t, "nunique", nunique_dataframe)

    for t in SERIES_TYPE:
        setattr(t, "to_gpu", to_gpu)
        setattr(t, "to_cpu", to_cpu)
        setattr(t, "rechunk", rechunk)
        setattr(t, "map", series_map)
        setattr(t, "describe", describe)
        setattr(t, "apply", series_apply)
        setattr(t, "transform", series_transform)
        setattr(t, "isin", series_isin)
        setattr(t, "shift", shift)
        setattr(t, "tshift", tshift)
        setattr(t, "diff", series_diff)
        setattr(t, "value_counts", value_counts)
        setattr(t, "astype", astype)
        setattr(t, "drop", series_drop)
        setattr(t, "drop_duplicates", series_drop_duplicates)
        setattr(t, "duplicated", series_duplicated)
        setattr(t, "memory_usage", series_memory_usage)
        setattr(t, "map_chunk", map_chunk)
        setattr(t, "cartesian_chunk", cartesian_chunk)
        setattr(t, "rebalance", rebalance)
        setattr(t, "explode", series_explode)
        setattr(t, "check_monotonic", check_monotonic)
        setattr(t, "is_monotonic", property(fget=is_monotonic))
        setattr(t, "is_monotonic_increasing", property(fget=is_monotonic_increasing))
        setattr(t, "is_monotonic_decreasing", property(fget=is_monotonic_decreasing))
        setattr(t, "pct_change", pct_change)
        setattr(t, "nlargest", series_nlargest)
        setattr(t, "nsmallest", series_nsmallest)
        setattr(t, "nunique", nunique_series)

    for t in INDEX_TYPE:
        setattr(t, "map", index_map)
        setattr(t, "rechunk", rechunk)
        setattr(t, "rebalance", rebalance)
        setattr(t, "drop", index_drop)
        setattr(t, "drop_duplicates", index_drop_duplicates)
        setattr(t, "duplicated", index_duplicated)
        setattr(t, "memory_usage", index_memory_usage)
        setattr(t, "astype", index_astype)
        setattr(t, "value_counts", value_counts)
        setattr(t, "check_monotonic", check_monotonic)
        setattr(t, "is_monotonic", property(fget=is_monotonic))
        setattr(t, "is_monotonic_increasing", property(fget=is_monotonic_increasing))
        setattr(t, "is_monotonic_decreasing", property(fget=is_monotonic_decreasing))

    for method in _string_method_to_handlers:
        if not hasattr(StringAccessor, method):
            StringAccessor._register(method)

    for method in _datetime_method_to_handlers:
        if not hasattr(DatetimeAccessor, method):
            DatetimeAccessor._register(method)

    for series in SERIES_TYPE:
        series.str = CachedAccessor("str", StringAccessor)
        series.dt = CachedAccessor("dt", DatetimeAccessor)

    for indexes in INDEX_TYPE:
        indexes.str = CachedAccessor("str", StringAccessor)


_install()
del _install

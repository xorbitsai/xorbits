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

from .aggregation import DataFrameAggregate
from .all import DataFrameAll
from .any import DataFrameAny
from .core import CustomReduction
from .count import DataFrameCount
from .cummax import DataFrameCummax
from .cummin import DataFrameCummin
from .cumprod import DataFrameCumprod
from .cumsum import DataFrameCumsum
from .custom_reduction import DataFrameCustomReduction
from .kurtosis import DataFrameKurtosis
from .max import DataFrameMax
from .mean import DataFrameMean
from .min import DataFrameMin
from .nunique import DataFrameNunique
from .prod import DataFrameProd
from .reduction_size import DataFrameSize
from .sem import DataFrameSem
from .skew import DataFrameSkew
from .str_concat import DataFrameStrConcat, build_str_concat_object
from .sum import DataFrameSum
from .unique import DataFrameUnique, unique
from .var import DataFrameVar


def _install():
    from ..core import DATAFRAME_TYPE, INDEX_TYPE, SERIES_TYPE
    from .aggregation import aggregate
    from .all import all_dataframe, all_index, all_series
    from .any import any_dataframe, any_index, any_series
    from .count import count_dataframe, count_series
    from .cummax import cummax
    from .cummin import cummin
    from .cumprod import cumprod
    from .cumsum import cumsum
    from .kurtosis import kurt_dataframe, kurt_series
    from .max import max_dataframe, max_index, max_series
    from .mean import mean_dataframe, mean_series
    from .min import min_dataframe, min_index, min_series
    from .nunique import nunique_dataframe, nunique_series
    from .prod import prod_dataframe, prod_series
    from .reduction_size import size_dataframe, size_series
    from .sem import sem_dataframe, sem_series
    from .skew import skew_dataframe, skew_series
    from .std import std_dataframe, std_series
    from .sum import sum_dataframe, sum_series
    from .var import var_dataframe, var_series

    funcs = [
        ("sum", sum_series, sum_dataframe),
        ("prod", prod_series, prod_dataframe),
        ("product", prod_series, prod_dataframe),
        ("max", max_series, max_dataframe),
        ("min", min_series, min_dataframe),
        ("count", count_series, count_dataframe),
        ("mean", mean_series, mean_dataframe),
        ("var", var_series, var_dataframe),
        ("std", std_series, std_dataframe),
        ("all", all_series, all_dataframe),
        ("any", any_series, any_dataframe),
        ("cummax", cummax, cummax),
        ("cummin", cummin, cummin),
        ("cumprod", cumprod, cumprod),
        ("cumsum", cumsum, cumsum),
        ("agg", aggregate, aggregate),
        ("aggregate", aggregate, aggregate),
        ("nunique", nunique_series, nunique_dataframe),
        ("sem", sem_series, sem_dataframe),
        ("skew", skew_series, skew_dataframe),
        ("kurt", kurt_series, kurt_dataframe),
        ("kurtosis", kurt_series, kurt_dataframe),
        ("unique", unique, None),
        ("_reduction_size", size_dataframe, size_series),
    ]
    for func_name, series_func, df_func in funcs:
        if df_func is not None:  # pragma: no branch
            for t in DATAFRAME_TYPE:
                setattr(t, func_name, df_func)
        if series_func is not None:  # pragma: no branch
            for t in SERIES_TYPE:
                setattr(t, func_name, series_func)

    for t in INDEX_TYPE:
        setattr(t, "agg", aggregate)
        setattr(t, "aggregate", aggregate)
        setattr(t, "all", all_index)
        setattr(t, "any", any_index)
        setattr(t, "min", min_index)
        setattr(t, "max", max_index)


_install()
del _install

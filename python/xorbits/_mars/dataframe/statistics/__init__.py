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

from .quantile import DataFrameQuantile


def _install():
    from ..core import DATAFRAME_TYPE, SERIES_TYPE
    from .corr import df_corr, df_corrwith, series_autocorr, series_corr
    from .quantile import quantile_dataframe, quantile_series

    for t in SERIES_TYPE:
        t.quantile = quantile_series
        t.corr = series_corr
        t.autocorr = series_autocorr

    for t in DATAFRAME_TYPE:
        t.quantile = quantile_dataframe
        t.corr = df_corr
        t.corrwith = df_corrwith


_install()
del _install

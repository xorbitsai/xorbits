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
import functools
import inspect
import warnings
from functools import lru_cache
from typing import Any, Callable, Dict, Type

import pandas as pd

from ..._mars.core import Entity as MarsEntity
from ...core.adapter import (
    ClsMethodWrapper,
    MarsOutputType,
    get_cls_members,
    wrap_mars_callable,
)
from ...core.data import DataType
from ...core.utils.fallback import wrap_fallback_module_method

_NO_ANNOTATION_FUNCS: Dict[Callable, MarsOutputType] = {
    pd.read_pickle: MarsOutputType.object,
    pd.crosstab: MarsOutputType.dataframe,
    pd.infer_freq: MarsOutputType.object,
    pd.interval_range: MarsOutputType.index,
    pd.json_normalize: MarsOutputType.dataframe,
    pd.lreshape: MarsOutputType.dataframe,
    pd.merge_asof: MarsOutputType.dataframe,
    pd.merge_ordered: MarsOutputType.dataframe,
    pd.period_range: MarsOutputType.index,
    pd.pivot: MarsOutputType.dataframe,
    pd.pivot_table: MarsOutputType.dataframe,
    pd.read_excel: MarsOutputType.dataframe,
    pd.read_fwf: MarsOutputType.dataframe,
    pd.read_gbq: MarsOutputType.dataframe,
    pd.read_hdf: MarsOutputType.object,
    pd.read_html: MarsOutputType.object,
    pd.read_json: MarsOutputType.df_or_series,
    pd.read_orc: MarsOutputType.dataframe,
    pd.read_sas: MarsOutputType.dataframe,
    pd.read_spss: MarsOutputType.dataframe,
    pd.read_stata: MarsOutputType.dataframe,
    pd.read_table: MarsOutputType.dataframe,
    pd.wide_to_long: MarsOutputType.dataframe,
}

if pd.__version__ >= "1.3.0":  # pragma: no branch
    _NO_ANNOTATION_FUNCS[pd.read_xml] = MarsOutputType.dataframe


class PandasClsMethodWrapper(ClsMethodWrapper):
    def _generate_fallback_data(self, mars_entity: MarsEntity):
        return mars_entity.to_pandas()

    def _generate_warning_msg(self, entity: MarsEntity, func_name: str):
        return f"{type(entity).__name__}.{func_name} will fallback to Pandas"

    def _get_output_type(self, func: Callable) -> MarsOutputType:
        return_annotation = inspect.signature(func).return_annotation
        # since pandas v2.1.0, the return_annotation of `pd.read_pickle` becomes `DataFrame | Series`,
        # see https://github.com/pandas-dev/pandas/blob/v2.1.0/pandas/io/pickle.py#L116-L214 .
        # However, the output_type should be `object` according to its document
        # (https://pandas.pydata.org/docs/reference/api/pandas.read_pickle.html).
        # So here's an extra check.
        if return_annotation is inspect.Signature.empty or func == pd.read_pickle:
            # mostly for python3.7 whose return_annotation is always empty
            return _NO_ANNOTATION_FUNCS.get(func, MarsOutputType.object)
        all_types = [t.strip() for t in return_annotation.split("|")]
        has_df = "DataFrame" in all_types
        has_series = "Series" in all_types
        if has_df and has_series:
            # output could be df or series, use OutputType.df_or_series
            output_type = MarsOutputType.df_or_series
        elif has_df:
            # output is df, use OutputType.dataframe
            output_type = MarsOutputType.dataframe
        elif has_series:
            # output is series, use OutputType.series
            output_type = MarsOutputType.series
        else:
            # otherwise, use object
            output_type = MarsOutputType.object
        return output_type

    def _get_docstring_src_module(self):
        return pd


class PandasFetchDataMethodWrapper(PandasClsMethodWrapper):
    def get_wrapped(self):
        wrapped = super().get_wrapped()

        @functools.wraps(getattr(self.library_cls, self.func_name))
        def _wrapped(entity: MarsEntity, *args, **kwargs):
            def fetch_wrapped(func):
                ret = func(entity, *args, **kwargs)
                return ret.fetch()

            return fetch_wrapped(wrapped)

        return _wrapped


def _collect_pandas_cls_members(pd_cls: Type, data_type: DataType):
    members = get_cls_members(data_type)
    for name, pd_cls_member in inspect.getmembers(pd_cls, inspect.isfunction):
        if name == "tolist" and pd_cls == pd.Series:
            pandas_series_tolist_method_wrapper = PandasFetchDataMethodWrapper(
                library_cls=pd_cls, func_name=name, fallback_warning=True
            )
            members[name] = pandas_series_tolist_method_wrapper.get_wrapped()
        elif name not in members and not name.startswith("_"):
            pandas_cls_method_wrapper = PandasClsMethodWrapper(
                library_cls=pd_cls, func_name=name, fallback_warning=True
            )
            members[name] = pandas_cls_method_wrapper.get_wrapped()
    # make to_numpy an alias of to_tensor
    members["to_numpy"] = members["to_tensor"]


def _collect_pandas_dataframe_members():
    _collect_pandas_cls_members(pd.DataFrame, DataType.dataframe)


def _collect_pandas_series_members():
    _collect_pandas_cls_members(pd.Series, DataType.series)


def _collect_pandas_index_members():
    _collect_pandas_cls_members(pd.Index, DataType.index)


@lru_cache(maxsize=1)
def collect_pandas_module_members() -> Dict[str, Any]:
    from ..mars_adapters.core import MARS_DATAFRAME_CALLABLES

    module_methods: Dict[str, Any] = dict()
    with warnings.catch_warnings():
        # suppress warnings raised by pandas when import xorbits.pandas
        warning_members = [
            "pandas.Float64Index",
            "pandas.Int64Index",
            "pandas.UInt64Index",
        ]
        for m in warning_members:
            warning_message = (
                f"{m} is deprecated and will be removed from pandas in a future version. "
                "Use pandas.Index with the appropriate dtype instead."
            )
            warnings.filterwarnings(
                "ignore", category=FutureWarning, message=warning_message
            )
        for name, cls_member in inspect.getmembers(pd):
            if (
                name not in MARS_DATAFRAME_CALLABLES
                and inspect.isfunction(cls_member)
                and not name.startswith("_")
            ):
                warning_str = f"xorbits.pandas.{name} will fallback to Pandas"
                pandas_cls_method_wrapper = PandasClsMethodWrapper()
                output_type = pandas_cls_method_wrapper._get_output_type(
                    func=getattr(pd, name)
                )

                module_methods[name] = wrap_mars_callable(
                    wrap_fallback_module_method(pd, name, output_type, warning_str),
                    attach_docstring=True,
                    is_cls_member=False,
                    docstring_src_module=pd,
                    docstring_src=getattr(pd, name, None),
                    fallback_warning=True,
                )
    return module_methods

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
from typing import Any, Callable, Dict, Type

import pandas as pd

from ...core.adapter import (
    MarsEntity,
    MarsOutputType,
    from_mars,
    get_cls_members,
    mars_remote,
    wrap_mars_callable,
)
from ...core.data import DataType
from ...core.utils.docstring import attach_cls_member_docstring
from ...core.utils.fallback import wrap_fallback_module_method

_NO_ANNOTATION_FUNCS: Dict[Callable, MarsOutputType] = {
    pd.read_pickle: MarsOutputType.dataframe,
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


def _get_output_type(func: Callable) -> MarsOutputType:
    return_annotation = inspect.signature(func).return_annotation
    if return_annotation is inspect.Signature.empty:
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


def wrap_pandas_cls_method(cls: Type, func_name: str, fallback_warning: bool = False):
    # wrap pd.DataFrame member functions
    @functools.wraps(getattr(cls, func_name))
    def _wrapped(entity: MarsEntity, *args, **kwargs):
        def _spawn(entity: MarsEntity) -> MarsEntity:
            """
            Execute pandas fallback with mars remote.
            """

            def execute_func(
                mars_entity: MarsEntity, f_name: str, *args, **kwargs
            ) -> Any:
                pd_data = mars_entity.to_pandas()
                return getattr(pd_data, f_name)(*args, **kwargs)

            new_args = (entity, func_name) + args
            ret = mars_remote.spawn(
                execute_func, args=new_args, kwargs=kwargs, output_types="object"
            )
            return from_mars(ret.execute())

        def _map_chunk(entity: MarsEntity, skip_infer: bool = False) -> MarsEntity:
            """
            Execute pandas fallback with map_chunk.
            """
            ret = entity.map_chunk(
                lambda x, *args, **kwargs: getattr(x, func_name)(*args, **kwargs),
                args=args,
                kwargs=kwargs,
                skip_infer=skip_infer,
            )
            if skip_infer:
                ret = ret.ensure_data()
            return from_mars(ret)

        warnings.warn(
            f"{type(entity).__name__}.{func_name} will fallback to Pandas",
            RuntimeWarning,
        )

        # rechunk mars tileable as one chunk
        one_chunk_entity = entity.rechunk(max(entity.shape))

        if hasattr(one_chunk_entity, "map_chunk"):
            try:
                return _map_chunk(one_chunk_entity, skip_infer=False)
            except TypeError:
                # when infer failed in map_chunk, we would use remote to execute
                # or skip inferring
                output_type = _get_output_type(getattr(cls, func_name))
                if output_type == MarsOutputType.object:
                    return _spawn(one_chunk_entity)
                else:
                    # skip_infer = True to avoid TypeError raised by inferring
                    return _map_chunk(one_chunk_entity, skip_infer=True)
        else:
            return _spawn(one_chunk_entity)

    attach_cls_member_docstring(
        _wrapped,
        func_name,
        docstring_src_module=pd,
        docstring_src_cls=cls,
        fallback_warning=fallback_warning,
    )
    return _wrapped


def _collect_pandas_cls_members(pd_cls: Type, data_type: DataType):
    members = get_cls_members(data_type)
    for name, pd_cls_member in inspect.getmembers(pd_cls, inspect.isfunction):
        if name not in members and not name.startswith("_"):
            members[name] = wrap_pandas_cls_method(pd_cls, name, fallback_warning=True)
    # make to_numpy an alias of to_tensor
    members["to_numpy"] = members["to_tensor"]


def _collect_pandas_dataframe_members():
    _collect_pandas_cls_members(pd.DataFrame, DataType.dataframe)


def _collect_pandas_series_members():
    _collect_pandas_cls_members(pd.Series, DataType.series)


def _collect_pandas_index_members():
    _collect_pandas_cls_members(pd.Index, DataType.index)


def _collect_pandas_module_members() -> Dict[str, Any]:
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
                output_type = _get_output_type(getattr(pd, name))

                module_methods[name] = wrap_mars_callable(
                    wrap_fallback_module_method(pd, name, output_type, warning_str),
                    attach_docstring=True,
                    is_cls_member=False,
                    docstring_src_module=pd,
                    docstring_src=getattr(pd, name, None),
                    fallback_warning=True,
                )
    return module_methods


PANDAS_MODULE_METHODS = _collect_pandas_module_members()

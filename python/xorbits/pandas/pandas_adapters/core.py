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

import functools
import inspect
import warnings
from typing import Any, Callable, Dict

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
    pd.read_xml: MarsOutputType.dataframe,
    pd.wide_to_long: MarsOutputType.dataframe,
}


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


def wrap_pandas_dataframe_method(func_name):
    # wrap pd.DataFrame member functions
    @functools.wraps(getattr(pd.DataFrame, func_name))
    def _wrapped(entity: MarsEntity, *args, **kwargs):
        warnings.warn(
            f"{type(entity).__name__}.{func_name} will fallback to Pandas",
            RuntimeWarning,
        )
        # rechunk mars tileable as one chunk
        one_chunk_data = entity.rechunk(max(entity.shape))

        # use map_chunk to execute pandas function
        try:
            ret = one_chunk_data.map_chunk(
                lambda x, *args, **kwargs: getattr(x, func_name)(*args, **kwargs),
                args=args,
                kwargs=kwargs,
            )
        except TypeError:
            # when infer failed in map_chunk, we would use remote to execute
            # or skip inferring
            output_type = _get_output_type(getattr(pd.DataFrame, func_name))
            if output_type == MarsOutputType.object:
                # for object type, use remote to execute
                def execute_func(mars_entity, f_name: str, *args, **kwargs):
                    pd_data = mars_entity.to_pandas()
                    return getattr(pd_data, f_name)(*args, **kwargs)

                new_args = (entity, func_name) + args
                ret = mars_remote.spawn(
                    execute_func, args=new_args, kwargs=kwargs, output_types=output_type
                )
                ret = ret.execute()
            else:
                # skip_infer = True to avoid TypeError raised by inferring
                ret = one_chunk_data.map_chunk(
                    lambda x, *args, **kwargs: getattr(x, func_name)(*args, **kwargs),
                    args=args,
                    kwargs=kwargs,
                    skip_infer=True,
                )
                ret = ret.ensure_data()
        return from_mars(ret)

    attach_cls_member_docstring(_wrapped, func_name, DataType.dataframe)
    return _wrapped


def wrap_pandas_module_method(func_name):
    # wrap pd member function
    @functools.wraps(getattr(pd, func_name))
    def _wrapped(*args, **kwargs):
        warnings.warn(
            f"xorbits.pandas.{func_name} will fallback to Pandas", RuntimeWarning
        )
        output_type = _get_output_type(getattr(pd, func_name))

        # use mars remote to execute pandas functions
        def execute_func(f_name: str, *args, **kwargs):
            import pandas as pd

            from xorbits.core.adapter import MarsEntity

            def _replace_data(nested):
                if isinstance(nested, dict):
                    vals = list(nested.values())
                else:
                    vals = list(nested)

                new_vals = []
                for val in vals:
                    if isinstance(val, (dict, list, tuple, set)):
                        new_val = _replace_data(val)
                    else:
                        if isinstance(val, MarsEntity):
                            new_val = val.fetch()
                        else:
                            new_val = val
                    new_vals.append(new_val)
                if isinstance(nested, dict):
                    return type(nested)((k, v) for k, v in zip(nested.keys(), new_vals))
                else:
                    return type(nested)(new_vals)

            return getattr(pd, f_name)(*_replace_data(args), **_replace_data(kwargs))

        new_args = (func_name,) + args
        ret = mars_remote.spawn(
            execute_func, args=new_args, kwargs=kwargs, output_types=output_type
        )
        if output_type == MarsOutputType.df_or_series:
            ret = ret.ensure_data()
        else:
            ret = ret.execute()
        return ret

    return _wrapped


def _collect_pandas_dataframe_members():
    dataframe_members = get_cls_members(DataType.dataframe)
    for name, cls_member in inspect.getmembers(pd.DataFrame):
        if (
            name not in dataframe_members
            and inspect.isfunction(cls_member)
            and not name.startswith("_")
        ):
            dataframe_members[name] = wrap_pandas_dataframe_method(name)

    # make to_numpy an alias of to_tensor
    dataframe_members["to_numpy"] = dataframe_members["to_tensor"]


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
                module_methods[name] = wrap_mars_callable(
                    wrap_pandas_module_method(name),
                    attach_docstring=True,
                    is_cls_member=False,
                    docstring_src_module=pd,
                    docstring_src=getattr(pd, name, None),
                    fallback_warning=True,
                )
    return module_methods


PANDAS_MODULE_METHODS = _collect_pandas_module_members()

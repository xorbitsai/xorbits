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

import pandas as pd

from ...core.adapter import MarsEntity, from_mars, get_cls_members
from ...core.data import DataType
from ...core.utils.docstring import attach_class_member_docstring


def wrap_pandas_dataframe_method(func_name):
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
            # skip_infer = True to avoid TypeError raised by inferring
            ret = one_chunk_data.map_chunk(
                lambda x, *args, **kwargs: getattr(x, func_name)(*args, **kwargs),
                args=args,
                kwargs=kwargs,
                skip_infer=True,
            )
            ret = ret.ensure_data()
        return from_mars(ret)

    attach_class_member_docstring(_wrapped, func_name, DataType.dataframe)
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

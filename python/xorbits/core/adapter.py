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

import functools
import inspect
from collections import defaultdict
from types import ModuleType
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Type, Union

# For maintenance, any module wants to import from mars, it should import from here.
from .._mars import dataframe as mars_dataframe
from .._mars import execute as mars_execute
from .._mars import new_session as mars_new_session
from .._mars import remote as mars_remote
from .._mars import stop_server as mars_stop_server
from .._mars import tensor as mars_tensor
from .._mars.core import Entity as MarsEntity
from .._mars.core import OutputType as MarsOutputType
from .._mars.core.entity.objects import OBJECT_TYPE as MARS_OBJECT_TYPE
from .._mars.dataframe import DataFrame as MarsDataFrame
from .._mars.dataframe import Index as MarsIndex
from .._mars.dataframe import Series as MarsSeries
from .._mars.dataframe.base.accessor import CachedAccessor as MarsCachedAccessor
from .._mars.dataframe.base.accessor import DatetimeAccessor as MarsDatetimeAccessor
from .._mars.dataframe.base.accessor import StringAccessor as MarsStringAccessor
from .._mars.dataframe.core import CATEGORICAL_TYPE as MARS_CATEGORICAL_TYPE
from .._mars.dataframe.core import DATAFRAME_GROUPBY_TYPE as MARS_DATAFRAME_GROUPBY_TYPE
from .._mars.dataframe.core import (
    DATAFRAME_OR_SERIES_TYPE as MARS_DATAFRAME_OR_SERIES_TYPE,
)
from .._mars.dataframe.core import DATAFRAME_TYPE as MARS_DATAFRAME_TYPE
from .._mars.dataframe.core import INDEX_TYPE as MARS_INDEX_TYPE
from .._mars.dataframe.core import SERIES_GROUPBY_TYPE as MARS_SERIES_GROUPBY_TYPE
from .._mars.dataframe.core import SERIES_TYPE as MARS_SERIES_TYPE
from .._mars.dataframe.core import DataFrameGroupBy as MarsDataFrameGroupBy
from .._mars.dataframe.core import SeriesGroupBy as MarsSeriesGroupBy
from .._mars.dataframe.datastore.to_csv import DataFrameToCSV as MarsDataFrameToCSV
from .._mars.dataframe.datastore.to_parquet import (
    DataFrameToParquet as MarsDataFrameToParquet,
)
from .._mars.dataframe.datastore.to_sql import (
    DataFrameToSQLTable as MarsDataFrameToSQLTable,
)
from .._mars.dataframe.datastore.to_vineyard import (
    DataFrameToVineyardChunk as MarsDataFrameToVineyardChunk,
)
from .._mars.dataframe.indexing.loc import DataFrameLoc as MarsDataFrameLoc
from .._mars.dataframe.plotting.core import PlotAccessor as MarsPlotAccessor
from .._mars.dataframe.window.ewm.core import EWM as MarsEWM
from .._mars.dataframe.window.expanding.core import Expanding as MarsExpanding
from .._mars.dataframe.window.rolling.core import Rolling as MarsRolling
from .._mars.deploy.oscar import session
from .._mars.tensor.core import TENSOR_TYPE as MARS_TENSOR_TYPE
from .._mars.tensor.core import Tensor as MarsTensor
from .._mars.tensor.core import flatiter as mars_flatiter
from .._mars.tensor.lib.index_tricks import CClass as MarsCClass
from .._mars.tensor.lib.index_tricks import MGridClass as MarsMGridClass
from .._mars.tensor.lib.index_tricks import OGridClass as MarsOGridClass
from .._mars.tensor.lib.index_tricks import RClass as MarsRClass
from .data import DATA_MEMBERS, Data, DataRef, DataType

# mars class name -> execution conditions
_TO_MARS_EXECUTION_CONDITION: Dict[
    str, List[Callable[["MarsEntity"], bool]]
] = defaultdict(list)
_FROM_MARS_EXECUTION_CONDITION: Dict[
    str, List[Callable[["MarsEntity"], bool]]
] = defaultdict(list)


def register_to_mars_execution_condition(
    mars_entity_type: str, condition: Callable[["MarsEntity"], bool]
):
    _TO_MARS_EXECUTION_CONDITION[mars_entity_type].append(condition)


def register_from_mars_execution_condition(
    mars_entity_type: str, condition: Callable[["MarsEntity"], bool]
):
    _FROM_MARS_EXECUTION_CONDITION[mars_entity_type].append(condition)


_MARS_CLS_TO_CONVERTER: Dict[Type, Callable] = {}


def register_converter(from_cls_list: List[Type]):
    """
    A decorator for convenience of registering a class converter.
    """

    def decorate(cls: Type):
        for from_cls in from_cls_list:
            assert from_cls not in _MARS_CLS_TO_CONVERTER
            _MARS_CLS_TO_CONVERTER[from_cls] = cls
        return cls

    return decorate


def wrap_magic_method(method_name: str) -> Callable[[Any], Any]:
    def wrapped(self: DataRef, *args, **kwargs):
        # trigger on condition execution.
        mars_entity = to_mars(self)
        if (mars_entity is None) or (
            not hasattr(mars_entity, method_name)
        ):  # pragma: no cover
            raise AttributeError(
                f"'{self.data.data_type.name}' object has no attribute '{method_name}'"
            )
        else:
            return wrap_mars_callable(
                getattr(mars_entity, method_name),
                attach_docstring=False,
                is_cls_member=True,
            )(*args, **kwargs)

    return wrapped


def wrap_generator(wrapped: Generator):
    for item in wrapped:
        yield from_mars(item)


def wrap_member_func(member_func: Callable, mars_entity: MarsEntity):
    @functools.wraps(member_func)
    def _wrapped(*args, **kwargs):
        return member_func(mars_entity, *args, **kwargs)

    return _wrapped


class MemberProxy:
    @classmethod
    def getattr(cls, ref: DataRef, item: str):
        # trigger on condition execution.
        mars_entity = to_mars(ref)
        data_type = ref.data.data_type
        member = DATA_MEMBERS[data_type].get(item, None)
        if member is not None and callable(member):
            ret = wrap_member_func(member, mars_entity)
            ret.__doc__ = member.__doc__
            return ret

        if not hasattr(mars_entity, item):
            raise AttributeError(f"'{data_type.name}' object has no attribute '{item}'")

        attr = getattr(mars_entity, item, None)
        if callable(attr):
            return wrap_mars_callable(
                attr,
                attach_docstring=True,
                is_cls_member=True,
                member_name=item,
                data_type=data_type,
            )
        else:
            # e.g. string accessor
            return from_mars(attr)

    @classmethod
    def setattr(cls, ref: DataRef, key: str, value: Any):
        # trigger on condition execution.
        mars_entity = to_mars(ref)
        if isinstance(getattr(type(mars_entity), key, None), property):
            # call the setter of the specified property.
            getattr(type(mars_entity), key).fset(mars_entity, to_mars(value))
        else:
            mars_entity.__setattr__(key, value)


def to_mars(inp: Union[DataRef, Tuple, List, Dict]):
    """
    Convert xorbits data references to mars entities and execute them if needed.
    """
    from ..numpy.mars_adapters.core import MarsGetItemProxy
    from ..pandas.mars_adapters.core import MarsGetAttrProxy

    if isinstance(inp, DataRef):
        mars_entity = getattr(inp.data, "_mars_entity", None)
        if mars_entity is None:  # pragma: no cover
            raise TypeError(f"Can't covert {inp} to mars entity")
        conditions = _TO_MARS_EXECUTION_CONDITION[type(mars_entity).__name__]
        for cond in conditions:
            if cond(mars_entity):
                from .execution import run

                run(inp)
        return mars_entity
    elif isinstance(inp, (MarsGetItemProxy, MarsGetAttrProxy)):
        # converters.
        return getattr(inp, "_mars_obj")
    elif isinstance(inp, tuple):
        return tuple(to_mars(i) for i in inp)
    elif isinstance(inp, list):
        return list(to_mars(i) for i in inp)
    elif isinstance(inp, dict):
        return dict((k, to_mars(v)) for k, v in inp.items())
    else:
        return inp


def from_mars(inp: Union[MarsEntity, Tuple, List, Dict, None]):
    """
    Convert mars entities to xorbits data references.
    """
    if isinstance(inp, MarsEntity):
        conditions = _FROM_MARS_EXECUTION_CONDITION[type(inp).__name__]
        ret = DataRef(Data.from_mars(inp))
        for cond in conditions:
            if cond(inp):
                from .execution import run

                run(ret)
        return ret
    elif type(inp) in _MARS_CLS_TO_CONVERTER:
        return _MARS_CLS_TO_CONVERTER[type(inp)](inp)
    elif isinstance(inp, tuple):
        return tuple(from_mars(i) for i in inp)
    elif isinstance(inp, list):
        return list(from_mars(i) for i in inp)
    elif isinstance(inp, dict):
        return dict((k, from_mars(v)) for k, v in inp.items())
    elif isinstance(inp, Generator):
        return wrap_generator(inp)
    else:
        return inp


def wrap_mars_callable(
    c: Callable, attach_docstring: bool, is_cls_member: bool, **kwargs
) -> Callable:
    """
    A function wrapper that makes arguments of the wrapped callable be mars compatible types and
    return value be xorbits compatible types.
    """

    @functools.wraps(c)
    def wrapped(*args, **kwargs):
        return from_mars(c(*to_mars(args), **to_mars(kwargs)))

    if attach_docstring:
        if is_cls_member:
            from .utils.docstring import attach_cls_member_docstring

            return attach_cls_member_docstring(wrapped, **kwargs)
        else:
            from .utils.docstring import attach_module_callable_docstring

            return attach_module_callable_docstring(wrapped, **kwargs)
    else:
        # for methods that do not need a docstring, like methods from mars, we need to reset the
        # docstring to prevent users from seeing a mars docstring.
        wrapped.__doc__ = ""
        return wrapped


def collect_cls_members(
    cls: Type,
    data_type: Optional[DataType] = None,
    docstring_src_module: Optional[ModuleType] = None,
    docstring_src_cls: Optional[Type] = None,
) -> Dict[str, Any]:
    cls_members: Dict[str, Any] = {}
    for name, cls_member in inspect.getmembers(cls):
        if inspect.isfunction(cls_member) and not name.startswith("_"):
            cls_members[name] = wrap_mars_callable(
                cls_member,
                attach_docstring=True,
                is_cls_member=True,
                member_name=name,
                data_type=data_type,
                docstring_src_module=docstring_src_module,
                docstring_src_cls=docstring_src_cls,
            )
        elif isinstance(cls_member, property):
            fget = cls_member.fget
            if fget is None:  # pragma: no cover
                raise ValueError(f"property {name} does not have a valid fget method.")
            c = wrap_mars_callable(
                fget,
                attach_docstring=True,
                is_cls_member=True,
                member_name=name,
                data_type=data_type,
                docstring_src_module=docstring_src_module,
                docstring_src_cls=docstring_src_cls,
            )
            cls_members[name] = property(c)
            cls_members[name].__doc__ = c.__doc__

    return cls_members


def register_data_members(data_type: DataType, members: Dict[str, Any]):
    DATA_MEMBERS[data_type].update(members)


def get_cls_members(data_type: DataType) -> Dict[str, Any]:
    if data_type not in DATA_MEMBERS:  # pragma: no cover
        raise ValueError(f"{data_type} do not have any bound class member.")

    return DATA_MEMBERS[data_type]


def replace_warning_msg_on_no_session():
    session.warning_msg = (
        """No existing session found, creating a new local session now."""
    )


replace_warning_msg_on_no_session()

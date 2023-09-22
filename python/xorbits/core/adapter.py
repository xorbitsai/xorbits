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

# To avoid possible naming conflict, mars functions and classes should be renamed.
# Functions should be renamed by adding a prefix 'mars_', and classes should be renamed
# by adding a prefix 'Mars'.

import functools
import inspect
import warnings
from abc import ABC, abstractmethod
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
from .._mars.core.entity.executable import ExecutableTuple
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
from .._mars.dataframe.indexing.at import DataFrameAt as MarsDataFrameAt
from .._mars.dataframe.indexing.iat import DataFrameIat as MarsDataFrameIat
from .._mars.dataframe.indexing.iloc import DataFrameIloc as MarsDataFrameIloc
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
from .utils.docstring import attach_cls_member_docstring


def own_data(mars_entity: MarsEntity) -> bool:
    # There are several mars operands which holds data directly. For example,
    # `DataFrameDataSource`, it is an operand represents creating a DataFrame
    # from Pandas DataFrame and its member `data` is the Pandas DataFrame. For
    # those mars entities that created by operands like these can skip executions
    # when users iterate or print entities. This function is to check if
    # mars entity's operand owns data.
    from .._mars.dataframe.datasource.dataframe import DataFrameDataSource
    from .._mars.dataframe.datasource.index import IndexDataSource
    from .._mars.dataframe.datasource.series import SeriesDataSource
    from .._mars.tensor.datasource import ArrayDataSource

    if (
        isinstance(
            mars_entity.op,
            (ArrayDataSource, DataFrameDataSource, SeriesDataSource, IndexDataSource),
        )
        and mars_entity.op.data is not None
    ):
        return True
    return False


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


class ClsMethodWrapper(ABC):
    def __init__(
        self,
        func_name: str = "",
        library_cls: Type = object,
        fallback_warning: bool = False,
    ):
        self.library_cls = library_cls
        self.func_name = func_name
        self.fallback_warning = fallback_warning

    @abstractmethod
    def _generate_fallback_data(self, mars_entity: MarsEntity) -> Any:
        """
        let mars entity fallback to data according to the library

        Parameters
        ----------
        mars_entity: MarsEntity

        Returns
        -------

        """

    @abstractmethod
    def _generate_warning_msg(self, mars_entity: MarsEntity, func_name: str) -> str:
        """
        generate fallback warning message according to the library

        Parameters
        ----------
        mars_entity: MarsEntity
        func_name: str

        Returns
        -------
        warning_msg: str
        """

    @abstractmethod
    def _get_output_type(self, func: Callable) -> MarsOutputType:
        """
        get output type according to the library

        Parameters
        ----------
        func: Callable

        Returns
        -------
        output_type: MarsOutputType
        """

    @abstractmethod
    def _get_docstring_src_module(self) -> ModuleType:
        """
        get docstring src module according to the library
        """

    def get_wrapped(self) -> Callable:
        """
        wrap pd.DataFrame member functions, np.ndarray methods, and other methods

        returns a callable
        """

        @functools.wraps(getattr(self.library_cls, self.func_name))
        def _wrapped(entity: MarsEntity, *args, **kwargs):
            def _spawn(entity: MarsEntity) -> MarsEntity:
                """
                Execute pandas/numpy fallback with mars remote.
                """

                def execute_func(
                    mars_entity: MarsEntity, f_name: str, *args, **kwargs
                ) -> Any:
                    ret = self._generate_fallback_data(mars_entity)
                    return getattr(ret, f_name)(*args, **kwargs)

                new_args = (entity, self.func_name) + args
                ret = mars_remote.spawn(
                    execute_func, args=new_args, kwargs=kwargs, output_types="object"
                )
                return from_mars(ret.execute())

            def _map_chunk(entity: MarsEntity, skip_infer: bool = False) -> MarsEntity:
                """
                Execute pandas fallback with map_chunk.
                """
                ret = entity.map_chunk(
                    lambda x, *args, **kwargs: getattr(x, self.func_name)(
                        *args, **kwargs
                    ),
                    args=args,
                    kwargs=kwargs,
                    skip_infer=skip_infer,
                )
                if skip_infer:
                    ret = ret.ensure_data()
                return from_mars(ret)

            warnings.warn(
                self._generate_warning_msg(entity, self.func_name),
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
                    output_type = self._get_output_type(
                        getattr(self.library_cls, self.func_name)
                    )
                    if output_type == MarsOutputType.object:
                        return _spawn(one_chunk_entity)
                    else:
                        # skip_infer = True to avoid TypeError raised by inferring
                        return _map_chunk(one_chunk_entity, skip_infer=True)
            else:
                return _spawn(one_chunk_entity)

        attach_cls_member_docstring(
            _wrapped,
            self.func_name,
            docstring_src_module=self._get_docstring_src_module(),
            docstring_src_cls=self.library_cls,
            fallback_warning=self.fallback_warning,
        )
        return _wrapped


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


class MarsGetItemProxy:
    def __init__(self, mars_obj):
        self._mars_obj = mars_obj

    def __getitem__(self, item):
        return from_mars(self._mars_obj[to_mars(item)])


class MarsGetAttrProxy:
    def __init__(self, obj: Any):
        self._mars_obj = to_mars(obj)

    def __getattr__(self, item):
        mars_obj = object.__getattribute__(self, "_mars_obj")
        attr = getattr(mars_obj, item, None)
        if attr is None:
            raise AttributeError(f"no attribute '{item}'")
        elif callable(attr):  # pragma: no cover
            return wrap_mars_callable(attr, attach_docstring=False, is_cls_member=True)
        else:  # pragma: no cover
            # class variable
            return from_mars(attr)

    def __getitem__(self, item):
        return from_mars(self._mars_obj[to_mars(item)])


def to_mars(inp: Union[DataRef, Tuple, List, Dict]):
    """
    Convert xorbits data references to mars entities and execute them if needed.
    """

    if isinstance(inp, DataRef):
        mars_entity = getattr(inp.data, "_mars_entity", None)
        if mars_entity is None:  # pragma: no cover
            raise TypeError(f"Can't convert {inp} to mars entity")
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
        if type(inp) is tuple or isinstance(inp, ExecutableTuple):
            return tuple(to_mars(i) for i in inp)
        else:
            # named tuple
            return type(inp)(*map(to_mars, inp))
    elif isinstance(inp, list):
        # in-place modification of list
        # preserve weak references to list, avoiding access issues
        for i, item in enumerate(inp):
            inp[i] = to_mars(item)
        return inp
    elif isinstance(inp, dict):
        # in-place modification of dict
        # preserve weak references to dict, avoiding access issues
        for k, v in inp.items():
            inp[k] = to_mars(v)
        return inp
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
        if type(inp) is tuple or isinstance(inp, ExecutableTuple):
            return tuple(from_mars(i) for i in inp)
        else:
            # named tuple
            return type(inp)(*map(from_mars, inp))
    elif isinstance(inp, list):
        # in-place modification of list
        # preserve weak references to list, avoiding access issues
        for i, item in enumerate(inp):
            inp[i] = from_mars(item)
        return inp
    elif isinstance(inp, dict):
        # in-place modification of dict
        # preserve weak references to dict, avoiding access issues
        for k, v in inp.items():
            inp[k] = from_mars(v)
        return inp
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
        # Tileable and TileableData object may have functions that have the same names.
        # For example, Index and IndexData both have `copy` function, but they have completely different semantics.
        # Therefore, when the Index's `copy` method has been collected,
        # the method of the same name on IndexData cannot be collected again.
        if cls.__name__.endswith("Data") and name in DATA_MEMBERS[data_type]:  # type: ignore
            continue
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

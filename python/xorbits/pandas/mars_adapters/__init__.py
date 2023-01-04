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

from .core import MARS_DATAFRAME_CALLABLES, MARS_DATAFRAME_MAGIC_METHODS


def _install():
    from ...core.adapter import (
        MARS_DATAFRAME_GROUPBY_TYPE,
        MARS_DATAFRAME_TYPE,
        MARS_INDEX_TYPE,
        MARS_SERIES_GROUPBY_TYPE,
        MARS_SERIES_TYPE,
        MarsDataFrame,
        MarsDataFrameGroupBy,
        MarsSeries,
        collect_cls_members,
        register_data_members,
        wrap_magic_method,
    )
    from ...core.data import DataRef, DataType
    from .core import (
        _register_from_mars_execution_conditions,
        _register_to_mars_execution_conditions,
        wrap_iteration_functions,
        wrap_user_defined_functions,
    )

    for method in MARS_DATAFRAME_MAGIC_METHODS:
        setattr(DataRef, method, wrap_magic_method(method))

    _register_to_mars_execution_conditions()
    _register_from_mars_execution_conditions()

    for cls in MARS_DATAFRAME_TYPE:
        register_data_members(
            DataType.dataframe, collect_cls_members(cls, DataType.dataframe)
        )
    for cls in MARS_SERIES_TYPE:
        register_data_members(
            DataType.series, collect_cls_members(cls, DataType.series)
        )
    for cls in MARS_INDEX_TYPE:
        register_data_members(DataType.index, collect_cls_members(cls, DataType.index))
    for cls in MARS_DATAFRAME_GROUPBY_TYPE:
        register_data_members(
            DataType.dataframe_groupby,
            collect_cls_members(cls, DataType.dataframe_groupby),
        )
    for cls in MARS_SERIES_GROUPBY_TYPE:
        register_data_members(
            DataType.series_groupby, collect_cls_members(cls, DataType.series_groupby)
        )

    # install DataFrame user defined functions:
    # DataFrame.apply
    # DataFrame.transform
    # DataFrame.map_chunk
    # DataFrame.cartesian_chunk
    dataframe_udfs = dict()
    dataframe_udfs["apply"] = wrap_user_defined_functions(
        MarsDataFrame.apply, "apply", DataType.dataframe
    )
    dataframe_udfs["transform"] = wrap_user_defined_functions(
        MarsDataFrame.apply, "transform", DataType.dataframe
    )
    dataframe_udfs["map_chunk"] = wrap_user_defined_functions(
        MarsDataFrame.map_chunk, "map_chunk", DataType.dataframe
    )
    dataframe_udfs["cartesian_chunk"] = wrap_user_defined_functions(
        MarsDataFrame.cartesian_chunk, "cartesian_chunk", DataType.dataframe
    )
    register_data_members(DataType.dataframe, dataframe_udfs)

    # install Serise user defined functions
    # Sereis.map
    register_data_members(
        DataType.series,
        dict(map=wrap_user_defined_functions(MarsSeries.map, "map", DataType.series)),
    )

    # install DataFrameGroupBy user defined functions:
    # GroupBy.apply
    # GroupBy.transform
    dataframe_groupby_udfs = dict()
    dataframe_groupby_udfs["apply"] = wrap_user_defined_functions(
        MarsDataFrameGroupBy.apply, "apply", DataType.dataframe_groupby
    )
    dataframe_groupby_udfs["transform"] = wrap_user_defined_functions(
        MarsDataFrameGroupBy.transform, "transform", DataType.dataframe_groupby
    )
    register_data_members(DataType.dataframe_groupby, dataframe_groupby_udfs)

    # install iteration functions
    # DataFrame.itertuples, DataFrame.iterrows
    dataframe_iteration_functions = dict()
    dataframe_iteration_functions["itertuples"] = wrap_iteration_functions(
        MarsDataFrame.itertuples, "itertuples", DataType.dataframe, True
    )
    dataframe_iteration_functions["iterrows"] = wrap_iteration_functions(
        MarsDataFrame.iterrows, "iterrows", DataType.dataframe, True
    )
    register_data_members(DataType.dataframe, dataframe_iteration_functions)

    # Series.items
    register_data_members(
        DataType.series,
        dict(
            items=wrap_iteration_functions(
                MarsSeries.items, "items", DataType.series, True
            )
        ),
    )

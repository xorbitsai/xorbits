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

import numpy as np
import pandas as pd

from .... import opcodes
from ....core import recursive_tile
from ....serialization.serializables import (
    AnyField,
    BoolField,
    DictField,
    FieldTypes,
    Int32Field,
    Int64Field,
    KeyField,
    ListField,
    StringField,
    TupleField,
)
from ....utils import calc_nsplits, has_unknown_shape, lazy_import
from ...core import DATAFRAME_TYPE
from ...operands import DataFrameOperand, DataFrameOperandMixin
from ...utils import build_empty_df, build_empty_series, parse_index

cudf = lazy_import("cudf")
N = 5


class DataFrameRollingCorr(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = opcodes.ROLLING_AGG

    _input = KeyField("input")
    _window = AnyField("window")
    _min_periods = Int64Field("min_periods")
    _center = BoolField("center")
    _win_type = StringField("win_type")
    _on = StringField("on")
    _axis = Int32Field("axis")
    _closed = StringField("closed")
    _func = AnyField("func")
    _func_args = TupleField("func_args")
    _func_kwargs = DictField("func_kwargs")
    # for chunks
    _preds = ListField("preds", FieldTypes.key)
    _succs = ListField("succs", FieldTypes.key)
    # for parallel
    _column_indices = ListField("column_indices", FieldTypes.key)

    def __init__(
        self,
        input=None,
        window=None,
        min_periods=None,
        center=None,  # pylint: disable=redefined-builtin
        win_type=None,
        on=None,
        axis=None,
        closed=None,
        func=None,
        func_args=None,
        func_kwargs=None,
        output_types=None,
        preds=None,
        succs=None,
        **kw
    ):
        super().__init__(
            _input=input,
            _window=window,
            _min_periods=min_periods,
            _center=center,
            _win_type=win_type,
            _on=on,
            _axis=axis,
            _closed=closed,
            _func=func,
            _func_args=func_args,
            _func_kwargs=func_kwargs,
            _output_types=output_types,
            _preds=preds,
            _succs=succs,
            **kw
        )

    @property
    def input(self):
        return self._input

    @property
    def window(self):
        return self._window

    @property
    def min_periods(self):
        return self._min_periods

    @property
    def center(self):
        return self._center

    @property
    def win_type(self):
        return self._win_type

    @property
    def on(self):
        return self._on

    @property
    def axis(self):
        return self._axis

    @property
    def closed(self):
        return self._closed

    @property
    def func(self):
        return self._func

    @property
    def func_args(self):
        return self._func_args

    @property
    def func_kwargs(self):
        return self._func_kwargs

    @property
    def preds(self):
        return self._preds if self._preds is not None else []

    @property
    def succs(self):
        return self._succs if self._succs is not None else []

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        input_iter = iter(self._inputs)
        self._input = next(input_iter)
        if self._preds is not None:
            self._preds = [next(input_iter) for _ in self._preds]
        if self._succs is not None:
            self._succs = [next(input_iter) for _ in self._succs]

    def __call__(self, rolling):
        inp = rolling.input

        if isinstance(inp, DATAFRAME_TYPE):
            pd_index = inp.index_value.to_pandas()
            empty_df = build_empty_df(inp.dtypes, index=pd_index[:0])
            params = rolling.params.copy()
            if params["win_type"] == "freq":
                params["win_type"] = None
            if self._func != "count":
                empty_df = empty_df._get_numeric_data()
            if self._axis == 0:
                index_value = inp.index_value
            else:
                # index_value = parse_index(
                #     test_df.index, rolling.params, inp, store_data=False
                # )
                raise NotImplementedError
            dtypes = pd.Series(
                index=inp.dtypes.index, data=[np.dtype(np.float64)] * inp.shape[1]
            )
            return self.new_dataframe(
                [inp],
                shape=(inp.shape[0] * inp.shape[1], inp.shape[1]),
                dtypes=dtypes,
                index_value=index_value,
                columns_value=parse_index(empty_df.columns, store_data=True),
            )
        else:
            pd_index = inp.index_value.to_pandas()
            empty_series = build_empty_series(
                inp.dtype, index=pd_index[:0], name=inp.name
            )
            test_obj = empty_series.rolling(**rolling.params).agg(self._func)
            if isinstance(test_obj, pd.DataFrame):
                return self.new_dataframe(
                    [inp],
                    shape=(inp.shape[0], test_obj.shape[1]),
                    dtypes=test_obj.dtypes,
                    index_value=inp.index_value,
                    columns_value=parse_index(test_obj.dtypes.index, store_data=True),
                )
            else:
                return self.new_series(
                    [inp],
                    shape=inp.shape,
                    dtype=test_obj.dtype,
                    index_value=inp.index_value,
                    name=test_obj.name,
                )

    @classmethod
    def _check_can_be_tiled(cls, op, is_window_int):
        inp = op.input
        axis = op.axis

        if axis == 0 and inp.ndim == 2:
            if has_unknown_shape(inp):
                yield
            inp = yield from recursive_tile(inp.rechunk({1: inp.shape[1]}))

        if is_window_int:
            # if window is integer
            if any(np.isnan(ns) for ns in inp.nsplits[op.axis]):
                yield
        else:
            # if window is offset
            # must be aware of index's meta including min and max
            for i in range(inp.chunk_shape[axis]):
                chunk_index = [0, 0]
                chunk_index[axis] = i
                chunk = inp.cix[tuple(chunk_index)]

                if axis == 0:
                    index_value = chunk.index_value
                else:
                    index_value = chunk.columns_value
                if pd.isnull(index_value.min_val) or pd.isnull(index_value.max_val):
                    yield

        return inp

    @classmethod
    def tile(cls, op: "DataFrameRollingCorr"):
        inp = op.inputs[0]
        degree_parallel = len(inp.columns) // N
        out = op.outputs[0]
        axis = op.axis
        # input_ndim = inp.ndim
        output_ndim = out.ndim
        # check if can be tiled
        # inp = yield from cls._check_can_be_tiled(op, is_window_int)

        out_chunks = []
        for j in range(N):
            chunk_op = op.copy().reset_key()

            out_chunk_index = [None] * output_ndim
            out_chunk_index[axis] = j
            if output_ndim == 2:
                out_chunk_index[1 - axis] = 0
            out_chunk_index = tuple(out_chunk_index)

            chunk_params = {"index": out_chunk_index}
            # consider the last chunk
            out_shape = list(out.shape)
            if j == N - 1:
                column_indices = (j * degree_parallel, inp.shape[1])
                out_shape[0] = out_shape[0] * (inp.shape[1] - j * degree_parallel)

            else:
                column_indices = (j * degree_parallel, (j + 1) * degree_parallel)
                out_shape[0] = out_shape[0] * degree_parallel
            chunk_params["shape"] = tuple(out_shape)
            chunk_params["column_indices"] = column_indices
            # set other chunk_op parameters
            start_column_index, end_column_index = column_indices
            column_names = inp.columns_value.value._data[
                start_column_index:end_column_index
            ]
            if np.isnan(inp.shape[0]):
                index = None
            else:
                index = pd.MultiIndex.from_product(
                    [range(inp.shape[0]), column_names.tolist()]
                )
            index_value = parse_index(index, inp, store_data=False)
            chunk_params["index_value"] = index_value if axis == 0 else out.index_value
            chunk_params["dtypes"] = out.dtypes if axis == 0 else inp.dtypes
            chunk_params["columns_value"] = (
                out.columns_value if axis == 0 else inp.columns_value
            )
            chunk_op._column_indices = column_indices
            out_chunk = chunk_op.new_chunk(inp.chunks, kws=[chunk_params])
            out_chunks.append(out_chunk)

        params = out.params
        params["chunks"] = out_chunks
        if out.ndim == 1:
            params["shape"] = (inp.shape[0],)
        else:
            params["shape"] = (
                inp.shape[0] * params["shape"][1],
                params["shape"][1],
            )

        params["nsplits"] = calc_nsplits({c.index: c.shape for c in out_chunks})
        new_op = op.copy()
        return new_op.new_tileables([inp], kws=[params])

    @classmethod
    def execute(cls, ctx, op: "DataFrameRollingCorr"):
        inp = ctx[op.input.key]
        axis = op.axis
        win_type = op.win_type
        window = op.window
        if win_type == "freq":
            win_type = None
            window = pd.Timedelta(window)
        # op.inputs cat
        if len(inp.shape) == 2:
            max_raw = 0
            max_column = 0
            for op_inp in op.inputs:
                max_raw = max(max_raw, op_inp.index[0])
                max_column = max(max_column, op_inp.index[1])
            df_matrix = [[None] * (max_column + 1) for _ in range(max_raw + 1)]
            for op_inp in op.inputs:
                df_matrix[op_inp.index[0]][op_inp.index[1]] = ctx[op_inp.key]
            data = pd.concat([pd.concat(row, axis=1) for row in df_matrix], axis=0)
        else:
            data = pd.concat([ctx[row.key] for row in op.inputs], axis=0)

        start_column_index, end_column_index = op._column_indices
        column_names = data.columns[start_column_index:end_column_index].tolist()

        index = pd.MultiIndex.from_product([column_names, data.index])
        data_value = []
        for column in column_names:
            _value = (
                data[column]
                .rolling(
                    window=window,
                    min_periods=op.min_periods,
                    center=op.center,
                    win_type=win_type,
                    on=op.on,
                    axis=axis,
                    closed=op.closed,
                )
                .corr(data)
                .values
            )
            data_value.append(_value)
        data_value = np.concatenate(data_value, axis=0)
        result = pd.DataFrame(data_value, index=index, columns=data.columns)
        result = result.swaplevel(1, 0)

        ctx[op.outputs[0].key] = result

import numpy as np
import pandas as pd

from ... import opcodes
from ...core import OutputType
from ...core.operand import MapReduceOperand, OperandStage
from ..operands import DataFrameOperandMixin, DataFrameShuffleProxy


class GroupByLen(DataFrameOperandMixin, MapReduceOperand):
    _op_type_ = opcodes.GROUPBY_LEN

    def __call__(self, groupby):
        return self.new_scalar([groupby])

    @classmethod
    def tile(cls, op: "GroupByLen"):
        in_groupby = op.inputs[0]
        output_type = op.output_types

        # generate map chunks
        map_chunks = []
        for chunk in in_groupby.chunks:
            map_op = op.copy().reset_key()
            map_op.stage = OperandStage.map
            map_op._shuffle_size = in_groupby.chunk_shape[0]
            map_op._output_types = output_type
            chunk_inputs = [chunk]

            map_chunks.append(map_op.new_chunk(chunk_inputs, index=chunk.index))

        proxy_chunk = DataFrameShuffleProxy(output_types=[output_type]).new_chunk(
            map_chunks, shape=()
        )

        # generate reduce chunks, we only need one reducer here.
        reduce_chunks = []
        reduce_op = op.copy().reset_key()
        reduce_op._output_types = [OutputType.scalar]
        reduce_op.stage = OperandStage.reduce
        reduce_op.n_reducers = 1
        reduce_chunks.append(reduce_op.new_chunk([proxy_chunk], index=1))

        # generate the result chunk:
        out_chunks = []
        for chunk in reduce_chunks:
            groupby_len_op = op.copy().reset_key()
            out_chunks.append(groupby_len_op.new_chunk([chunk]))

        # final wrap up:
        new_op = op.copy()
        params = op.outputs[0].params.copy()
        params["nsplits"] = ((np.nan,) * len(out_chunks),)
        params["chunks"] = out_chunks
        return new_op.new_tileables([in_groupby], **params)

    @classmethod
    def execute_map(cls, ctx, op):
        chunk = op.outputs[0]
        in_df_grouped = ctx[op.inputs[0].key]

        # grouped object .size() method ensure every unique keys
        summary = in_df_grouped.size()
        sum_indexes = summary.index

        res = []
        for index in sum_indexes:
            res.append(index)

        # use series to convey every index store in this level
        ctx[chunk.key, 1] = pd.Series(res)

    @classmethod
    def execute_reduce(cls, ctx, op: "GroupByLen"):
        chunk = op.outputs[0]
        input_idx_to_series = dict(op.iter_mapper_data(ctx))
        row_idxes = sorted(input_idx_to_series.keys())

        res = set()
        for row_index in row_idxes:
            row_series = input_idx_to_series.get(row_index, None)
            res.update(row_series)

        res_len = len(res)
        ctx[chunk.key] = res_len

    @classmethod
    def execute(cls, ctx, op: "GroupByLen"):
        if op.stage == OperandStage.map:
            cls.execute_map(ctx, op)
        elif op.stage == OperandStage.reduce:
            cls.execute_reduce(ctx, op)
        else:
            ctx[op.output[0].key] = ctx[op.inputs[0].key]


def groupby_len(groupby):
    op = GroupByLen()
    return op(groupby).execute().fetch()

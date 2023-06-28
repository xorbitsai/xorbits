import pandas as pd

from ... import opcodes
from ...core import OutputType
from ...core.operand import Operand, OperandStage
from ..operands import DataFrameOperandMixin


class GroupByLen(DataFrameOperandMixin, Operand):
    _op_type_ = opcodes.GROUPBY_LEN

    def __call__(self, groupby):
        return self.new_scalar([groupby])

    @classmethod
    def tile(cls, op: "GroupByLen"):
        in_groupby = op.inputs[0]

        # generate map chunks
        map_chunks = []
        for chunk in in_groupby.chunks:
            map_op = op.copy().reset_key()
            map_op.stage = OperandStage.map
            map_op.output_types = [OutputType.series]
            chunk_inputs = [chunk]

            map_chunks.append(map_op.new_chunk(chunk_inputs))

        # generate reduce chunks, we only need one reducer here.
        out_chunks = []
        reduce_op = op.copy().reset_key()
        reduce_op.output_types = [OutputType.scalar]
        reduce_op.stage = OperandStage.reduce
        params = dict(dtype=int)
        out_chunks.append(
            reduce_op.new_chunk(map_chunks, shape=(), index=(0,), dtype=int)
        )

        # final wrap up:
        new_op = op.copy()
        params = op.outputs[0].params.copy()

        params.pop("shape")

        params["chunks"] = out_chunks
        return new_op.new_scalars(op.inputs, **params)

    @classmethod
    def execute_map(cls, ctx, op: "GroupByLen"):
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
        key = op.inputs[0].key

        res = set()
        input_series = ctx[key, 1]
        res.update(input_series)

        res_len = len(res)
        ctx[chunk.key] = res_len

    @classmethod
    def execute(cls, ctx, op: "GroupByLen"):
        if op.stage == OperandStage.map:
            cls.execute_map(ctx, op)
        elif op.stage == OperandStage.reduce:
            cls.execute_reduce(ctx, op)


def groupby_len(groupby):
    op = GroupByLen()
    return op(groupby).execute().fetch()

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

try:
    import jax
    from jax import config

    config.update("jax_enable_x64", True)
    JAX_INSTALLED = True
except ImportError:
    JAX_INSTALLED = False

from .. import arithmetic, reduction
from ..operands import TensorFuse
from .core import TensorFuseChunkMixin


class TensorJAXFuseChunk(TensorFuse, TensorFuseChunkMixin):
    _op_type_ = None  # no opcode, cannot be serialized

    @classmethod
    def execute(cls, ctx, op):
        chunk = op.outputs[0]
        inputs = [ctx[c.key] for c in op.inputs]
        input_dict = {c.key: i for c, i in zip(op.inputs, inputs)}
        res = np.array(_evaluate(chunk, input_dict))
        res = _maybe_keepdims(chunk, res)
        if chunk.ndim == 0 and res.ndim == 1 and res.size == 0:
            res = res.dtype.type(0)
        ctx[chunk.key] = res


# recursively compute the inputs
def _evaluate(chunk, input_dict):
    op_type = type(chunk.op)

    if chunk.key in input_dict:
        return input_dict[chunk.key]
    elif op_type is TensorJAXFuseChunk:
        return _evaluate(chunk.composed[-1], input_dict)
    elif op_type in ARITHMETIC_SUPPORT:
        if hasattr(chunk.op, "inputs"):
            if hasattr(chunk.op, "input"):
                return _get_jax_function(chunk.op)(
                    _evaluate(chunk.op.input, input_dict)
                )
            elif np.isscalar(chunk.op.lhs):
                rhs = _evaluate(chunk.op.rhs, input_dict)
                return _get_jax_function(chunk.op)(chunk.op.lhs, rhs)
            elif np.isscalar(chunk.op.rhs):
                lhs = _evaluate(chunk.op.lhs, input_dict)
                return _get_jax_function(chunk.op)(lhs, chunk.op.rhs)
            else:
                lhs = _evaluate(chunk.op.lhs, input_dict)
                rhs = _evaluate(chunk.op.rhs, input_dict)
                return _get_jax_function(chunk.op)(lhs, rhs)
        return _get_jax_function(chunk.op)
    elif op_type in TREE_SUPPORT:
        func = _get_jax_function(chunk.op)
        inputs = [_evaluate(input, input_dict) for input in chunk.inputs]

        # passing func and inputs as parameters seems to cause error with jax.jit
        def _fusion():
            output = func(inputs[0], inputs[1])
            for input in inputs[2:]:
                output = func(output, input)
            return output

        return jax.jit(_fusion)()
    elif op_type in REDUCTION_SUPPORT:
        ax = chunk.op.axis
        data = _evaluate(chunk.inputs[0], input_dict)
        if len(ax) == data.ndim:
            return _get_jax_function(chunk.op)(data)
        else:
            return _get_jax_function(chunk.op)(data, axis=ax)
    else:
        raise TypeError(f"unsupported operator in jax: {op_type.__name__}")


def _get_jax_function(operand):
    func_name = getattr(operand, "_func_name")
    return getattr(jax.numpy, func_name)


def _maybe_keepdims(chunk, res):
    out_chunk = chunk.composed[-1] if type(chunk.op) == TensorJAXFuseChunk else chunk
    if type(out_chunk.op) in REDUCTION_SUPPORT and out_chunk.op.keepdims:
        res = np.reshape(res, out_chunk.shape)
    return res


ARITHMETIC_SUPPORT = {
    arithmetic.TensorNegative,
    arithmetic.TensorAbs,
    arithmetic.TensorConj,
    arithmetic.TensorExp,
    arithmetic.TensorLog,
    arithmetic.TensorLog10,
    arithmetic.TensorExpm1,
    arithmetic.TensorLog1p,
    arithmetic.TensorSqrt,
    arithmetic.TensorSin,
    arithmetic.TensorCos,
    arithmetic.TensorTan,
    arithmetic.TensorArcsin,
    arithmetic.TensorArccos,
    arithmetic.TensorArctan,
    arithmetic.TensorSinh,
    arithmetic.TensorCosh,
    arithmetic.TensorTanh,
    arithmetic.TensorArcsinh,
    arithmetic.TensorArccosh,
    arithmetic.TensorArctanh,
    arithmetic.TensorFloor,
    arithmetic.TensorCeil,
    arithmetic.TensorNot,
    arithmetic.TensorAdd,
    arithmetic.TensorSubtract,
    arithmetic.TensorMultiply,
    arithmetic.TensorDivide,
    arithmetic.TensorMod,
    arithmetic.TensorPower,
    arithmetic.TensorLshift,
    arithmetic.TensorRshift,
    arithmetic.TensorEqual,
    arithmetic.TensorNotEqual,
    arithmetic.TensorLessThan,
    arithmetic.TensorLessEqual,
    arithmetic.TensorGreaterThan,
    arithmetic.TensorGreaterEqual,
    arithmetic.TensorAnd,
    arithmetic.TensorOr,
}

TREE_SUPPORT = {
    arithmetic.TensorTreeAdd,
    arithmetic.TensorTreeMultiply,
}

REDUCTION_SUPPORT = {
    reduction.TensorSum,
    reduction.TensorProd,
    reduction.TensorMax,
    reduction.TensorMin,
}

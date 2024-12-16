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

import functools as ft

import numpy as np

try:
    import jax
    import jax.numpy as jnp
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
        inputs = [jnp.asarray(ctx[c.key]) for c in op.inputs]
        input_dict = {c.key: i for c, i in zip(op.inputs, inputs)}
        res = np.asarray(_eval(input_dict, chunk))
        res = _maybe_keepdims(chunk, res)
        if chunk.ndim == 0 and res.ndim == 1 and res.size == 0:
            res = res.dtype.type(0)
        ctx[chunk.key] = res


if JAX_INSTALLED:

    @ft.partial(jax.jit, static_argnums=(1,))
    def _eval(input_dict, chunk):
        op_composed = chunk.composed
        env = input_dict

        def read(key):
            return env[key]

        def write(key, value):
            env[key] = value

        for comp in op_composed:
            op_type = type(comp.op)
            # eval unary
            if op_type in UNARY_ARITHMETIC_SUPPORT:
                # get input
                data = read(comp.op.input.key)
                res = _get_jax_function(comp.op)(data)
                write(comp.op.outputs[0].key, res)

            # eval binary
            elif op_type in BINARY_ARITHMETIC_SUPPORT:
                if hasattr(comp.op, "lhs"):
                    if hasattr(comp.op.lhs, "key"):
                        lhs = read(comp.op.lhs.key)
                    else:
                        lhs = comp.op.lhs
                if hasattr(comp.op, "rhs"):
                    if hasattr(comp.op.rhs, "key"):
                        rhs = read(comp.op.rhs.key)
                    else:
                        rhs = comp.op.rhs
                res = _get_jax_function(comp.op)(lhs, rhs)
                write(comp.op.outputs[0].key, res)

            # eval reduction
            elif op_type in REDUCTION_SUPPORT:
                # get input
                ax = comp.op.axis
                data = read(comp.op.input.key)
                if len(ax) == data.ndim:
                    res = _get_jax_function(comp.op)(data)
                else:
                    res = _get_jax_function(comp.op)(data, axis=ax)
                write(comp.op.outputs[0].key, res)
            # eval tree
            elif op_type in TREE_SUPPORT:
                func = _get_jax_function(comp.op)
                inputs = [read(input.key) for input in comp.inputs]
                res = func(inputs[0], inputs[1])
                for input in inputs[2:]:
                    res = func(res, input)
                write(comp.op.outputs[0].key, res)
            else:
                raise TypeError(f"unsupported operator in jax: {op_type.__name__}")

        return read(chunk.op.outputs[0].key)

else:

    def _eval(input_dict, chunk):
        raise ImportError("jax is not installed")


def _get_jax_function(operand):
    func_name = getattr(operand, "_func_name")
    return getattr(jax.numpy, func_name)


def _maybe_keepdims(chunk, res):
    out_chunk = chunk.composed[-1] if type(chunk.op) == TensorJAXFuseChunk else chunk
    if type(out_chunk.op) in REDUCTION_SUPPORT and out_chunk.op.keepdims:
        res = jnp.reshape(res, out_chunk.shape)
    return res


UNARY_ARITHMETIC_SUPPORT = {
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
}

BINARY_ARITHMETIC_SUPPORT = {
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

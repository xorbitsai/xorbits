# Copyright 2022-2023 XProbe Inc.
# derived from copyright 1999-2021 Alibaba Group Holding Ltd.
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

from ...core import is_build_mode
from .abs import TensorAbs, abs
from .absolute import TensorAbsolute, absolute
from .add import TensorAdd, TensorTreeAdd, add, tree_add
from .angle import TensorAngle, angle
from .arccos import TensorArccos, arccos
from .arccosh import TensorArccosh, arccosh
from .arcsin import TensorArcsin, arcsin
from .arcsinh import TensorArcsinh, arcsinh
from .arctan import TensorArctan, arctan
from .arctan2 import TensorArctan2, arctan2
from .arctanh import TensorArctanh, arctanh
from .around import TensorAround
from .around import around
from .around import around as round_
from .bitand import TensorBitand, bitand
from .bitor import TensorBitor, bitor
from .bitxor import TensorBitxor, bitxor
from .cbrt import TensorCbrt, cbrt
from .ceil import TensorCeil, ceil
from .clip import TensorClip, clip
from .conj import TensorConj
from .conj import conj
from .conj import conj as conjugate
from .copysign import TensorCopysign, copysign
from .cos import TensorCos, cos
from .cosh import TensorCosh, cosh
from .deg2rad import TensorDeg2rad, deg2rad
from .degrees import TensorDegrees, degrees
from .divide import TensorDivide, divide
from .equal import TensorEqual, equal
from .exp import TensorExp, exp
from .exp2 import TensorExp2, exp2
from .expm1 import TensorExpm1, expm1
from .fabs import TensorFabs, fabs
from .fix import TensorFix, fix
from .float_power import TensorFloatPower, float_power
from .floor import TensorFloor, floor
from .floordiv import TensorFloorDiv, floordiv
from .fmax import TensorFMax, fmax
from .fmin import TensorFMin, fmin
from .fmod import TensorFMod, fmod
from .frexp import TensorFrexp, frexp
from .greater import TensorGreaterThan, greater
from .greater_equal import TensorGreaterEqual, greater_equal
from .hypot import TensorHypot, hypot
from .i0 import TensorI0, i0
from .imag import TensorImag, imag
from .invert import TensorInvert, invert
from .isclose import TensorIsclose, isclose
from .iscomplex import TensorIsComplex, iscomplex
from .isfinite import TensorIsFinite, isfinite
from .isinf import TensorIsInf, isinf
from .isnan import TensorIsNan, isnan
from .isreal import TensorIsReal, isreal
from .ldexp import TensorLdexp, ldexp
from .less import TensorLessThan, less
from .less_equal import TensorLessEqual, less_equal
from .log import TensorLog, log
from .log1p import TensorLog1p, log1p
from .log2 import TensorLog2, log2
from .log10 import TensorLog10, log10
from .logaddexp import TensorLogAddExp, logaddexp
from .logaddexp2 import TensorLogAddExp2, logaddexp2
from .logical_and import TensorAnd, logical_and
from .logical_not import TensorNot, logical_not
from .logical_or import TensorOr, logical_or
from .logical_xor import TensorXor, logical_xor
from .lshift import TensorLshift, lshift
from .maximum import TensorMaximum, maximum
from .minimum import TensorMinimum, minimum
from .mod import TensorMod
from .mod import mod
from .mod import mod as remainder
from .modf import TensorModf, modf
from .multiply import TensorMultiply, TensorTreeMultiply, multiply, tree_multiply
from .nan_to_num import TensorNanToNum, nan_to_num
from .negative import TensorNegative, negative
from .nextafter import TensorNextafter, nextafter
from .not_equal import TensorNotEqual, not_equal
from .positive import TensorPositive, positive
from .power import TensorPower, power
from .rad2deg import TensorRad2deg, rad2deg
from .radians import TensorRadians, radians
from .real import TensorReal, real
from .reciprocal import TensorReciprocal, reciprocal
from .rint import TensorRint, rint
from .rshift import TensorRshift, rshift
from .setimag import TensorSetImag
from .setreal import TensorSetReal
from .sign import TensorSign, sign
from .signbit import TensorSignbit, signbit
from .sin import TensorSin, sin
from .sinc import TensorSinc, sinc
from .sinh import TensorSinh, sinh
from .spacing import TensorSpacing, spacing
from .sqrt import TensorSqrt, sqrt
from .square import TensorSquare, square
from .subtract import TensorSubtract, subtract
from .tan import TensorTan, tan
from .tanh import TensorTanh, tanh
from .truediv import TensorTrueDiv, truediv
from .trunc import TensorTrunc, trunc


def _wrap_iop(func):
    def inner(self, *args, **kwargs):
        kwargs["out"] = self
        return func(self, *args, **kwargs)

    return inner


def _install():
    from ..core import TENSOR_TYPE, Tensor, TensorData
    from ..datasource import tensor as astensor
    from .add import add, radd
    from .bitand import bitand, rbitand
    from .bitor import bitor, rbitor
    from .bitxor import bitxor, rbitxor
    from .divide import divide, rdivide
    from .floordiv import floordiv, rfloordiv
    from .lshift import lshift, rlshift
    from .mod import mod, rmod
    from .multiply import multiply, rmultiply
    from .power import power, rpower
    from .rshift import rrshift, rshift
    from .subtract import rsubtract, subtract
    from .truediv import rtruediv, truediv

    def _wrap_equal(func):
        def eq(x1, x2, **kwargs):
            if is_build_mode():
                return astensor(x1)._equals(x2)
            return func(x1, x2, **kwargs)

        return eq

    for cls in TENSOR_TYPE:
        setattr(cls, "__add__", add)
        setattr(cls, "__iadd__", _wrap_iop(add))
        setattr(cls, "__radd__", radd)
        setattr(cls, "__sub__", subtract)
        setattr(cls, "__isub__", _wrap_iop(subtract))
        setattr(cls, "__rsub__", rsubtract)
        setattr(cls, "__mul__", multiply)
        setattr(cls, "__imul__", _wrap_iop(multiply))
        setattr(cls, "__rmul__", rmultiply)
        setattr(cls, "__div__", divide)
        setattr(cls, "__idiv__", _wrap_iop(divide))
        setattr(cls, "__rdiv__", rdivide)
        setattr(cls, "__truediv__", truediv)
        setattr(cls, "__itruediv__", _wrap_iop(truediv))
        setattr(cls, "__rtruediv__", rtruediv)
        setattr(cls, "__floordiv__", floordiv)
        setattr(cls, "__ifloordiv__", _wrap_iop(floordiv))
        setattr(cls, "__rfloordiv__", rfloordiv)
        setattr(cls, "__pow__", power)
        setattr(cls, "__ipow__", _wrap_iop(power))
        setattr(cls, "__rpow__", rpower)
        setattr(cls, "__mod__", mod)
        setattr(cls, "__imod__", _wrap_iop(mod))
        setattr(cls, "__rmod__", rmod)
        setattr(cls, "__lshift__", lshift)
        setattr(cls, "__ilshift__", _wrap_iop(lshift))
        setattr(cls, "__rlshift__", rlshift)
        setattr(cls, "__rshift__", rshift)
        setattr(cls, "__irshift__", _wrap_iop(rshift))
        setattr(cls, "__rrshift__", rrshift)

        setattr(cls, "__eq__", _wrap_equal(equal))
        setattr(cls, "__ne__", not_equal)
        setattr(cls, "__lt__", less)
        setattr(cls, "__le__", less_equal)
        setattr(cls, "__gt__", greater)
        setattr(cls, "__ge__", greater_equal)
        setattr(cls, "__and__", bitand)
        setattr(cls, "__iand__", _wrap_iop(bitand))
        setattr(cls, "__rand__", rbitand)
        setattr(cls, "__or__", bitor)
        setattr(cls, "__ior__", _wrap_iop(bitor))
        setattr(cls, "__ror__", rbitor)
        setattr(cls, "__xor__", bitxor)
        setattr(cls, "__ixor__", _wrap_iop(bitxor))
        setattr(cls, "__rxor__", rbitxor)

        setattr(cls, "__neg__", negative)
        setattr(cls, "__pos__", positive)
        setattr(cls, "__abs__", abs)
        setattr(cls, "__invert__", invert)

    setattr(Tensor, "round", round_)
    setattr(Tensor, "conj", conj)
    setattr(Tensor, "conjugate", conjugate)
    setattr(TensorData, "round", round_)
    setattr(TensorData, "conj", conj)
    setattr(TensorData, "conjugate", conjugate)


_install()
del _install


BIN_UFUNC = {
    add,
    subtract,
    multiply,
    divide,
    truediv,
    floordiv,
    power,
    mod,
    fmod,
    logaddexp,
    logaddexp2,
    equal,
    not_equal,
    less,
    less_equal,
    greater,
    greater_equal,
    arctan2,
    hypot,
    bitand,
    bitor,
    bitxor,
    lshift,
    rshift,
    logical_and,
    logical_or,
    logical_xor,
    maximum,
    minimum,
    float_power,
    remainder,
    fmax,
    fmin,
    copysign,
    nextafter,
    ldexp,
}

UNARY_UFUNC = {
    square,
    arcsinh,
    rint,
    sign,
    conj,
    tan,
    absolute,
    deg2rad,
    log,
    fabs,
    exp2,
    invert,
    negative,
    sqrt,
    arctan,
    positive,
    cbrt,
    log10,
    sin,
    rad2deg,
    log2,
    arcsin,
    expm1,
    arctanh,
    cosh,
    sinh,
    cos,
    reciprocal,
    tanh,
    log1p,
    exp,
    arccos,
    arccosh,
    around,
    logical_not,
    conjugate,
    isfinite,
    isinf,
    isnan,
    signbit,
    spacing,
    floor,
    ceil,
    trunc,
    degrees,
    radians,
    angle,
    isreal,
    iscomplex,
    real,
    imag,
    fix,
    i0,
    sinc,
    nan_to_num,
}

#!/usr/bin/env python
# -*- coding: utf-8 -*-
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


# noinspection PyUnresolvedReferences
# import numpy types
# noinspection PyUnresolvedReferences
# noinspection PyUnresolvedReferences
from numpy import NAN, NINF, AxisError, Inf, NaN
from numpy import bool_ as bool
from numpy import (
    bytes_,
    cfloat,
    character,
    complex64,
    complex128,
    complexfloating,
    datetime64,
    double,
    dtype,
    e,
    errstate,
    finfo,
    flexible,
    float16,
    float32,
    float64,
)
from numpy import float_ as float
from numpy import floating, generic, geterr, inexact, inf, int8, int16, int32, int64
from numpy import int_ as int
from numpy import intc, integer, intp, nan, newaxis, number
from numpy import object_ as object
from numpy import (
    pi,
    seterr,
    signedinteger,
    timedelta64,
    uint,
    uint8,
    uint16,
    uint32,
    uint64,
    unicode_,
    unsignedinteger,
    void,
)

# noinspection PyUnresolvedReferences
from ..core import ExecutableTuple
from . import fft, lib, linalg, random, special, stats, ufunc
from .arithmetic import absolute
from .arithmetic import absolute as abs
from .arithmetic import (
    add,
    angle,
    arccos,
    arccosh,
    arcsin,
    arcsinh,
    arctan,
    arctan2,
    arctanh,
    around,
)
from .arithmetic import bitand as bitwise_and
from .arithmetic import bitor as bitwise_or
from .arithmetic import bitxor as bitwise_xor
from .arithmetic import (
    cbrt,
    ceil,
    clip,
    conj,
    conjugate,
    copysign,
    cos,
    cosh,
    deg2rad,
    degrees,
    divide,
    equal,
    exp,
    exp2,
    expm1,
    fabs,
    fix,
    float_power,
    floor,
)
from .arithmetic import floordiv as floor_divide
from .arithmetic import fmax, fmin, fmod, frexp, greater, greater_equal, hypot, i0, imag
from .arithmetic import invert
from .arithmetic import invert as bitwise_not
from .arithmetic import (
    isclose,
    iscomplex,
    isfinite,
    isinf,
    isnan,
    isreal,
    ldexp,
    less,
    less_equal,
    log,
    log1p,
    log2,
    log10,
    logaddexp,
    logaddexp2,
    logical_and,
    logical_not,
    logical_or,
    logical_xor,
)
from .arithmetic import lshift as left_shift
from .arithmetic import (
    maximum,
    minimum,
    mod,
    modf,
    multiply,
    nan_to_num,
    negative,
    nextafter,
    not_equal,
    positive,
    power,
    rad2deg,
    radians,
    real,
    reciprocal,
    remainder,
    rint,
)
from .arithmetic import round_
from .arithmetic import round_ as round
from .arithmetic import rshift as right_shift
from .arithmetic import (
    sign,
    signbit,
    sin,
    sinc,
    sinh,
    spacing,
    sqrt,
    square,
    subtract,
    tan,
    tanh,
    tree_add,
    tree_multiply,
)
from .arithmetic import truediv as true_divide
from .arithmetic import trunc
from .base import (
    argpartition,
    argsort,
    argtopk,
    argwhere,
    array_split,
    atleast_1d,
    atleast_2d,
    atleast_3d,
    broadcast_arrays,
    broadcast_to,
    copy,
    copyto,
    delete,
    diff,
    dsplit,
    ediff1d,
    expand_dims,
    flip,
    fliplr,
    flipud,
    hsplit,
    in1d,
    insert,
    isin,
    moveaxis,
    ndim,
    partition,
    ravel,
    repeat,
    result_type,
    roll,
    rollaxis,
    searchsorted,
    setdiff1d,
    shape,
    sort,
    split,
    squeeze,
    swapaxes,
    tile,
    topk,
    transpose,
    trapz,
    unique,
    vsplit,
    where,
)

# types
from .core import Tensor
from .datasource import (
    arange,
    array,
    asarray,
    ascontiguousarray,
    asfortranarray,
    diag,
    diagflat,
    empty,
    empty_like,
    eye,
    from_dataframe,
)
from .datasource import fromhdf5
from .datasource import fromhdf5 as from_hdf5
from .datasource import fromtiledb
from .datasource import fromtiledb as from_tiledb
from .datasource import fromvineyard
from .datasource import fromvineyard as from_vineyard
from .datasource import fromzarr
from .datasource import fromzarr as from_zarr
from .datasource import (
    full,
    full_like,
    identity,
    indices,
    linspace,
    meshgrid,
    ones,
    ones_like,
    scalar,
    tensor,
    tril,
    triu,
    zeros,
    zeros_like,
)
from .datastore import tohdf5
from .datastore import tohdf5 as to_hdf5
from .datastore import totiledb  # pylint: disable=reimported
from .datastore import totiledb as to_tiledb
from .datastore import tovineyard
from .datastore import tovineyard as to_vineyard
from .datastore import tozarr
from .datastore import tozarr as to_zarr
from .einsum import einsum
from .fetch import TensorFetch, TensorFetchShuffle

# register fuse op and fetch op
from .fuse import TensorCpFuseChunk, TensorFuseChunk, TensorNeFuseChunk
from .images import imread
from .indexing import (
    choose,
    compress,
    extract,
    fill_diagonal,
    flatnonzero,
    nonzero,
    take,
    unravel_index,
)

# noinspection PyUnresolvedReferences
from .lib.index_tricks import c_, mgrid, ndindex, ogrid, r_
from .linalg.dot import dot
from .linalg.inner import inner, innerproduct
from .linalg.matmul import matmul
from .linalg.tensordot import tensordot
from .linalg.vdot import vdot
from .merge import (
    append,
    block,
    column_stack,
    concatenate,
    dstack,
    hstack,
    stack,
    union1d,
    vstack,
)
from .rechunk import rechunk
from .reduction import (
    all,
    allclose,
    any,
    argmax,
    argmin,
    array_equal,
    count_nonzero,
    cumprod,
    cumsum,
)
from .reduction import max
from .reduction import max as amax
from .reduction import mean
from .reduction import min
from .reduction import min as amin
from .reduction import (
    nanargmax,
    nanargmin,
    nancumprod,
    nancumsum,
    nanmax,
    nanmean,
    nanmin,
    nanprod,
    nanstd,
    nansum,
    nanvar,
)
from .reduction import prod
from .reduction import prod as product
from .reduction import std, sum, var
from .reshape import reshape
from .statistics import (
    average,
    bincount,
    corrcoef,
    cov,
    digitize,
    histogram,
    histogram_bin_edges,
    median,
    percentile,
    ptp,
    quantile,
)

del (
    TensorFuseChunk,
    TensorCpFuseChunk,
    TensorNeFuseChunk,
    TensorFetch,
    TensorFetchShuffle,
    ufunc,
)

# distutils: language = c++
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

import collections
import importlib
import importlib.util as importlib_utils
import itertools
import os
import pickle
import time
import types
import uuid
from datetime import date, datetime, timedelta, tzinfo
from enum import Enum
from functools import lru_cache, partial

cimport cython

import cloudpickle
import numpy as np
import pandas as pd

from cpython cimport PyBytes_FromStringAndSize
from libc.stdint cimport uint8_t, uint32_t
from xoscar._utils cimport TypeDispatcher, to_binary, to_str
from xoscar._utils import NamedType

try:
    from pandas.tseries.offsets import Tick as PDTick
except ImportError:
    PDTick = None

from .lib.mmh3 import hash as mmh_hash
from .lib.mmh3 import hash_bytes as mmh_hash_bytes
from .lib.mmh3 import hash_from_buffer as mmh3_hash_from_buffer


cdef bint _has_cupy = bool(importlib_utils.find_spec('cupy'))
cdef bint _has_cudf = bool(importlib_utils.find_spec('cudf'))
cdef bint _has_sqlalchemy = bool(importlib_utils.find_spec('sqlalchemy'))
cdef bint _has_interval_array_inclusive = hasattr(
    pd.arrays.IntervalArray, "inclusive"
)


cdef extern from "MurmurHash3.h":
    void MurmurHash3_x64_128(const void * key, Py_ssize_t len, uint32_t seed, void * out)


cdef bytes _get_mars_key(const uint8_t[:] bufferview):
    cdef const uint8_t *data = &bufferview[0]
    cdef uint8_t out[16]
    MurmurHash3_x64_128(data, len(bufferview), 0, out)
    out[0] |= 0xC0
    return PyBytes_FromStringAndSize(<char*>out, 16)


cpdef unicode to_text(s, encoding='utf-8'):
    if type(s) is unicode:
        return <unicode>s
    elif isinstance(s, bytes):
        return (<bytes>s).decode('utf-8')
    elif isinstance(s, unicode):
        return unicode(s)
    elif s is None:
        return None
    else:
        raise TypeError(f"Could not convert from {s} to unicode.")

cdef inline build_canonical_bytes(tuple args, kwargs):
    if kwargs:
        args = args + (kwargs,)
    return pickle.dumps(tokenize_handler(args))


def tokenize(*args, **kwargs):
    return _get_mars_key(build_canonical_bytes(args, kwargs)).hex()


def tokenize_int(*args, **kwargs):
    return mmh_hash(build_canonical_bytes(args, kwargs))


cdef class Tokenizer(TypeDispatcher):
    def __call__(self, object obj, *args, **kwargs):
        try:
            return self.get_handler(type(obj))(obj, *args, **kwargs)
        except KeyError:
            if hasattr(obj, '__mars_tokenize__') and not isinstance(obj, type):
                if len(args) == 0 and len(kwargs) == 0:
                    return obj.__mars_tokenize__()
                else:
                    obj = obj.__mars_tokenize__()
                    return self.get_handler(type(obj))(obj, *args, **kwargs)
            if callable(obj):
                if PDTick is not None and not isinstance(obj, PDTick):
                    return tokenize_function(obj)

            try:
                return cloudpickle.dumps(obj)
            except:
                raise TypeError(f'Cannot generate token for {obj}, type: {type(obj)}') from None


cdef inline list iterative_tokenize(object ob):
    cdef list dq = [ob]
    cdef int dq_pos = 0
    cdef list h_list = []
    while dq_pos < len(dq):
        x = dq[dq_pos]
        dq_pos += 1
        if type(x) in _primitive_types:
            h_list.append(x)
        elif isinstance(x, (list, tuple)):
            dq.extend(x)
        elif isinstance(x, set):
            dq.extend(sorted(x))
        elif isinstance(x, dict):
            dq.extend(sorted(x.items()))
        else:
            h_list.append(tokenize_handler(x))

        if dq_pos >= 64 and len(dq) < dq_pos * 2:  # pragma: no cover
            dq = dq[dq_pos:]
            dq_pos = 0
    return h_list


cdef inline tuple tokenize_numpy(ob):
    cdef int offset

    # Return random bytes and metadata for objects > 64MB
    if ob.nbytes > 64 * 1024 ** 2:
        return os.urandom(16), ob.dtype, ob.shape, ob.strides

    if not ob.shape:
        return str(ob), ob.dtype
    if hasattr(ob, 'mode') and getattr(ob, 'filename', None):
        if hasattr(ob.base, 'ctypes'):
            offset = (ob.ctypes.get_as_parameter().value -
                      ob.base.ctypes.get_as_parameter().value)
        else:
            offset = 0  # root memmap's have mmap object as base
        return (ob.filename, os.path.getmtime(ob.filename), ob.dtype,
                ob.shape, ob.strides, offset)
    if ob.dtype.hasobject:
        try:
            data = mmh_hash_bytes('-'.join(ob.flat).encode('utf-8', errors='surrogatepass'))
        except UnicodeDecodeError:
            data = mmh_hash_bytes(b'-'.join([to_binary(x) for x in ob.flat]))
        except TypeError:
            try:
                data = mmh_hash_bytes(pickle.dumps(ob, pickle.HIGHEST_PROTOCOL))
            except:
                # nothing can do, generate uuid
                data = uuid.uuid4().hex
    else:
        try:
            data = mmh_hash_bytes(ob.ravel().view('i1').data)
        except (BufferError, AttributeError, ValueError):
            data = mmh_hash_bytes(ob.copy().ravel().view('i1').data)
    return data, ob.dtype, ob.shape, ob.strides


cdef inline _extract_range_index_attr(object range_index, str attr):
    try:
        return getattr(range_index, attr)
    except AttributeError:  # pragma: no cover
        return getattr(range_index, '_' + attr)


cdef list tokenize_pandas_index(ob):
    cdef long long start
    cdef long long stop
    cdef long long end
    if isinstance(ob, pd.RangeIndex):
        start = _extract_range_index_attr(ob, 'start')
        stop = _extract_range_index_attr(ob, 'stop')
        step = _extract_range_index_attr(ob, 'step')
        # for range index, there is no need to get the values
        return iterative_tokenize([ob.name, getattr(ob, 'names', None), slice(start, stop, step)])
    else:
        return iterative_tokenize([ob.name, getattr(ob, 'names', None), ob.values])


cdef list tokenize_pandas_series(ob):
    return iterative_tokenize([ob.name, ob.dtype, ob.values, ob.index])


cdef list tokenize_pandas_dataframe(ob):
    l = [block.values for block in ob._mgr.blocks]
    l.extend([ob.columns, ob.index])
    return iterative_tokenize(l)


cdef list tokenize_pandas_categorical(ob):
    l = ob.tolist()
    l.append(ob.shape)
    return iterative_tokenize(l)


cdef list tokenize_pd_extension_dtype(ob):
    return iterative_tokenize([ob.name])


cdef list tokenize_categories_dtype(ob):
    return iterative_tokenize([ob.categories, ob.ordered])


cdef list tokenize_interval_dtype(ob):
    return iterative_tokenize([type(ob).__name__, ob.subtype])


cdef list tokenize_pandas_time_arrays(ob):
    return iterative_tokenize([ob.asi8, ob.dtype])


cdef list tokenize_pandas_tick(ob):
    return iterative_tokenize([ob.freqstr])


cdef list tokenize_pandas_interval_arrays(ob):  # pragma: no cover
    if _has_interval_array_inclusive:
        return iterative_tokenize([ob.left, ob.right, ob.inclusive])
    else:
        return iterative_tokenize([ob.left, ob.right, ob.closed])


cdef list tokenize_sqlalchemy_data_type(ob):
    return iterative_tokenize([repr(ob)])


cdef list tokenize_sqlalchemy_selectable(ob):
    return iterative_tokenize([str(ob)])


cdef list tokenize_enum(ob):
    cls = type(ob)
    return iterative_tokenize([id(cls), cls.__name__, ob.name])


@lru_cache(500)
def tokenize_function(ob):
    if isinstance(ob, partial):
        args = iterative_tokenize(ob.args)
        keywords = iterative_tokenize(ob.keywords.items()) if ob.keywords else None
        return tokenize_function(ob.func), args, keywords
    else:
        try:
            if isinstance(ob, types.FunctionType):
                return iterative_tokenize([pickle.dumps(ob, protocol=0), id(ob)])
            else:
                return pickle.dumps(ob, protocol=0)
        except:
            pass
        try:
            return cloudpickle.dumps(ob, protocol=0)
        except:
            return str(ob)


@lru_cache(500)
def tokenize_pickled_with_cache(ob):
    return pickle.dumps(ob)


def tokenize_cupy(ob):
    from xoscar.serialization import serialize
    header, _buffers = serialize(ob)
    return iterative_tokenize([header, ob.data.ptr])


def tokenize_cudf(ob):
    from xoscar.serialization import serialize
    header, buffers = serialize(ob)
    return iterative_tokenize([header] + [(buf._owner._ptr, buf.size, buf._offset) for buf in buffers])


cdef Tokenizer tokenize_handler = Tokenizer()

cdef set _primitive_types = {
    int, float, str, unicode, bytes, complex, type(None), type, slice, date, datetime, timedelta
}
for t in _primitive_types:
    tokenize_handler.register(t, lambda ob: ob)

for t in (np.dtype, np.generic):
    tokenize_handler.register(t, lambda ob: ob)

for t in (list, tuple, dict, set):
    tokenize_handler.register(t, iterative_tokenize)

tokenize_handler.register(np.ndarray, tokenize_numpy)
tokenize_handler.register(np.random.RandomState, lambda ob: iterative_tokenize(ob.get_state()))
tokenize_handler.register(memoryview, lambda ob: mmh3_hash_from_buffer(ob))
tokenize_handler.register(Enum, tokenize_enum)
tokenize_handler.register(pd.Index, tokenize_pandas_index)
tokenize_handler.register(pd.Series, tokenize_pandas_series)
tokenize_handler.register(pd.DataFrame, tokenize_pandas_dataframe)
tokenize_handler.register(pd.Categorical, tokenize_pandas_categorical)
tokenize_handler.register(pd.CategoricalDtype, tokenize_categories_dtype)
tokenize_handler.register(pd.IntervalDtype, tokenize_interval_dtype)
tokenize_handler.register(tzinfo, tokenize_pickled_with_cache)
tokenize_handler.register(pd.arrays.DatetimeArray, tokenize_pandas_time_arrays)
tokenize_handler.register(pd.arrays.TimedeltaArray, tokenize_pandas_time_arrays)
tokenize_handler.register(pd.arrays.PeriodArray, tokenize_pandas_time_arrays)
tokenize_handler.register(pd.arrays.IntervalArray, tokenize_pandas_interval_arrays)
tokenize_handler.register(pd.api.extensions.ExtensionDtype, tokenize_pd_extension_dtype)
if _has_cupy:
    tokenize_handler.register('cupy.ndarray', tokenize_cupy)
if _has_cudf:
    tokenize_handler.register('cudf.DataFrame', tokenize_cudf)
    tokenize_handler.register('cudf.Series', tokenize_cudf)
    tokenize_handler.register('cudf.Index', tokenize_cudf)

if PDTick is not None:
    tokenize_handler.register(PDTick, tokenize_pandas_tick)
if _has_sqlalchemy:
    tokenize_handler.register(
        "sqlalchemy.sql.sqltypes.TypeEngine", tokenize_sqlalchemy_data_type
    )
    tokenize_handler.register(
        "sqlalchemy.sql.Selectable", tokenize_sqlalchemy_selectable
    )

cpdef register_tokenizer(cls, handler):
    tokenize_handler.register(cls, handler)


@cython.nonecheck(False)
@cython.cdivision(True)
cpdef long long ceildiv(long long x, long long y) nogil:
    return x // y + (x % y != 0)


cdef class Timer:
    cdef readonly object start
    cdef readonly object duration

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *_):
        self.duration = time.time() - self.start


cdef class CUnionFind:
    cdef dict parent
    def __init__(self):
        self.parent = {}

    cpdef find(self, bytes x):
        cdef bytes px
        if x not in self.parent:
            return x

        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])

        return self.parent[x]

    cpdef union_(self, bytes x, bytes y):
        cdef bytes px, py
        px = self.find(x)
        py = self.find(y)
        min_val = min(px, py)
        self.parent[px] = self.parent[py] = min(px, py)
        return min_val

    cpdef union_uf(self, CUnionFind uf):
        cdef bytes x
        for x in uf.parent:
            if x not in self.parent:
                self.parent[x] = uf.parent[x]
            else:
                self.parent[x] = self.union_(self.find(x), uf.find(x))



__all__ = ['to_str', 'to_binary', 'to_text', 'tokenize', 'tokenize_int',
           'register_tokenizer', 'ceildiv', 'Timer', 'CUnionFind']

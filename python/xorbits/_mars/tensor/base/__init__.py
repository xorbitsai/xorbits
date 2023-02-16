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

from .argpartition import argpartition
from .argsort import argsort
from .argtopk import argtopk
from .argwhere import TensorArgwhere, argwhere
from .array_split import array_split
from .astype import TensorAstype
from .atleast_1d import atleast_1d
from .atleast_2d import atleast_2d
from .atleast_3d import atleast_3d
from .broadcast_arrays import broadcast_arrays
from .broadcast_to import TensorBroadcastTo, broadcast_to
from .copy import copy
from .copyto import TensorCopyTo, copyto
from .delete import delete
from .diff import diff
from .dsplit import dsplit
from .ediff1d import ediff1d
from .expand_dims import expand_dims
from .flatten import flatten
from .flip import flip
from .fliplr import fliplr
from .flipud import flipud
from .hsplit import hsplit
from .in1d import in1d
from .insert import insert
from .isin import TensorIsIn, isin
from .map_chunk import TensorMapChunk, map_chunk
from .moveaxis import moveaxis
from .ndim import ndim
from .partition import partition
from .ravel import ravel
from .rebalance import rebalance
from .repeat import TensorRepeat, repeat
from .result_type import result_type
from .roll import roll
from .rollaxis import rollaxis
from .searchsorted import TensorSearchsorted, searchsorted
from .setdiff1d import setdiff1d
from .shape import shape
from .sort import sort
from .split import TensorSplit, split
from .squeeze import TensorSqueeze, squeeze
from .swapaxes import TensorSwapAxes, swapaxes
from .tile import tile
from .to_cpu import to_cpu
from .to_gpu import to_gpu
from .topk import topk
from .transpose import TensorTranspose, transpose
from .trapz import trapz
from .unique import unique
from .vsplit import vsplit
from .where import TensorWhere, where


def _install():
    from ..core import Tensor, TensorData
    from .astype import _astype

    for cls in (Tensor, TensorData):
        setattr(cls, "astype", _astype)
        setattr(cls, "swapaxes", swapaxes)
        setattr(cls, "squeeze", squeeze)
        setattr(cls, "repeat", repeat)
        setattr(cls, "ravel", ravel)
        setattr(cls, "flatten", flatten)
        setattr(cls, "to_gpu", to_gpu)
        setattr(cls, "to_cpu", to_cpu)
        setattr(cls, "rebalance", rebalance)
        setattr(cls, "map_chunk", map_chunk)


_install()
del _install

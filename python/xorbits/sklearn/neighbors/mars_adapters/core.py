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

import sklearn.neighbors as sk_neighbors

from ...._mars.learn import neighbors as mars_neighbors
from ...._mars.learn.neighbors import NearestNeighbors as MarsNearestNeighbors
from ....core.utils.docstring import attach_module_callable_docstring
from ...utils import SKLearnBase, _collect_module_callables, _install_cls_members


class NearestNeighbors(SKLearnBase):
    _marscls = MarsNearestNeighbors


SKLEARN_NEIGHBORS_CLS_MAP = {
    NearestNeighbors: MarsNearestNeighbors,
}

MARS_SKLEARN_NEIGHBORS_CALLABLES = _collect_module_callables(
    mars_neighbors, sk_neighbors, skip_members=["register_op"]
)
_install_cls_members(
    SKLEARN_NEIGHBORS_CLS_MAP, MARS_SKLEARN_NEIGHBORS_CALLABLES, sk_neighbors
)
attach_module_callable_docstring(
    NearestNeighbors, sk_neighbors, sk_neighbors.NearestNeighbors
)

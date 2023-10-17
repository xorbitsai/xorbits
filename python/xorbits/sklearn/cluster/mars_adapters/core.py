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

import sklearn.cluster as sk_cluster

from ...._mars.learn import cluster as mars_cluster
from ...._mars.learn.cluster import KMeans as MarsKMeans
from ....core.utils.docstring import attach_module_callable_docstring
from ...utils import SKLearnBase, _collect_module_callables, _install_cls_members


class KMeans(SKLearnBase):
    _marscls = MarsKMeans


SKLEARN_CLUSTER_CLS_MAP = {KMeans: MarsKMeans}

MARS_SKLEARN_CLUSTER_CALLABLES = _collect_module_callables(
    mars_cluster, sk_cluster, skip_members=["register_op"]
)
_install_cls_members(
    SKLEARN_CLUSTER_CLS_MAP, MARS_SKLEARN_CLUSTER_CALLABLES, sk_cluster
)
attach_module_callable_docstring(KMeans, sk_cluster, sk_cluster.KMeans)

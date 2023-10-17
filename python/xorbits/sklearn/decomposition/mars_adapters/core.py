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

import sklearn.decomposition as sk_decomposition

from ...._mars.learn import decomposition as mars_decomposition
from ...._mars.learn.decomposition import PCA as MarsPCA
from ...._mars.learn.decomposition import TruncatedSVD as MarsTruncatedSVD
from ....core.utils.docstring import attach_module_callable_docstring
from ...utils import SKLearnBase, _collect_module_callables, _install_cls_members


class PCA(SKLearnBase):
    _marscls = MarsPCA


class TruncatedSVD(SKLearnBase):
    _marscls = MarsTruncatedSVD


SKLEARN_DECOMP_CLS_MAP = {PCA: MarsPCA, TruncatedSVD: MarsTruncatedSVD}

MARS_SKLEARN_DECOMP_CALLABLES = _collect_module_callables(
    mars_decomposition, sk_decomposition, skip_members=["register_op"]
)
_install_cls_members(
    SKLEARN_DECOMP_CLS_MAP, MARS_SKLEARN_DECOMP_CALLABLES, sk_decomposition
)
attach_module_callable_docstring(PCA, sk_decomposition, sk_decomposition.PCA)
attach_module_callable_docstring(
    TruncatedSVD, sk_decomposition, sk_decomposition.TruncatedSVD
)

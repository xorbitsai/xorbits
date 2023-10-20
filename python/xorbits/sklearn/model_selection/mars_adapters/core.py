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

import sklearn.model_selection as sk_ml

from ...._mars.learn import model_selection as mars_ml
from ...._mars.learn.model_selection import KFold as MarsKFold
from ...._mars.learn.model_selection import ParameterGrid as MarsParameterGrid
from ....core.utils.docstring import attach_module_callable_docstring
from ...utils import SKLearnBase, _collect_module_callables, _install_cls_members


class KFold(SKLearnBase):
    _marscls = MarsKFold


class ParameterGrid(SKLearnBase):
    _marscls = MarsParameterGrid

    def __len__(self):
        return len(self.mars_instance)

    def __iter__(self):
        return iter(self.mars_instance)

    def __getitem__(self, index):
        return self.mars_instance[index]


SKLEARN_ML_CLS_MAP = {
    KFold: MarsKFold,
    ParameterGrid: MarsParameterGrid,
}

MARS_SKLEARN_ML_CALLABLES = _collect_module_callables(
    mars_ml, sk_ml, skip_members=["register_op"]
)
_install_cls_members(SKLEARN_ML_CLS_MAP, MARS_SKLEARN_ML_CALLABLES, sk_ml)
attach_module_callable_docstring(KFold, sk_ml, sk_ml.KFold)
attach_module_callable_docstring(ParameterGrid, sk_ml, sk_ml.ParameterGrid)

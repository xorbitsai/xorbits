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

import sklearn.semi_supervised as sk_ss

from ...._mars.learn import semi_supervised as mars_ss
from ...._mars.learn.semi_supervised import LabelPropagation as MarsLabelPropagation
from ....core.utils.docstring import attach_module_callable_docstring
from ...utils import SKLearnBase, _collect_module_callables, _install_cls_members


class LabelPropagation(SKLearnBase):
    _marscls = MarsLabelPropagation


SKLEARN_SS_CLS_MAP = {
    LabelPropagation: MarsLabelPropagation,
}

MARS_SKLEARN_SS_CALLABLES = _collect_module_callables(
    mars_ss, sk_ss, skip_members=["register_op"]
)
_install_cls_members(SKLEARN_SS_CLS_MAP, MARS_SKLEARN_SS_CALLABLES, sk_ss)
attach_module_callable_docstring(LabelPropagation, sk_ss, sk_ss.LabelPropagation)

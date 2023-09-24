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

import sklearn.linear_model as sk_lm

from ...._mars.learn import linear_model as mars_lm
from ...._mars.learn.glm import LogisticRegression as MarsLogisticRegression
from ...._mars.learn.linear_model import LinearRegression as MarsLinearRegression
from ....core.utils.docstring import attach_module_callable_docstring
from ...utils import SKLearnBase, _collect_module_callables, _install_cls_members


class LinearRegression(SKLearnBase):
    _marscls = MarsLinearRegression


class LogisticRegression(SKLearnBase):
    _marscls = MarsLogisticRegression


SKLEARN_LM_CLS_MAP = {
    LinearRegression: MarsLinearRegression,
    LogisticRegression: MarsLogisticRegression,
}

MARS_SKLEARN_LM_CALLABLES = _collect_module_callables(
    mars_lm, sk_lm, skip_members=["register_op"]
)
_install_cls_members(SKLEARN_LM_CLS_MAP, MARS_SKLEARN_LM_CALLABLES, sk_lm)
attach_module_callable_docstring(LinearRegression, sk_lm, sk_lm.LinearRegression)
attach_module_callable_docstring(LogisticRegression, sk_lm, sk_lm.LogisticRegression)

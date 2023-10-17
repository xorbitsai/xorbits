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

import sklearn.ensemble as sk_en

from ...._mars.learn import ensemble as mars_en
from ...._mars.learn.ensemble import BaggingClassifier as MarsBaggingClassifier
from ...._mars.learn.ensemble import BaggingRegressor as MarsBaggingRegressor
from ...._mars.learn.ensemble import IsolationForest as MarsIsolationForest
from ....core.utils.docstring import attach_module_callable_docstring
from ...utils import SKLearnBase, _collect_module_callables, _install_cls_members


class BaggingClassifier(SKLearnBase):
    _marscls = MarsBaggingClassifier


class BaggingRegressor(SKLearnBase):
    _marscls = MarsBaggingRegressor


class IsolationForest(SKLearnBase):
    _marscls = MarsIsolationForest


SKLEARN_EN_CLS_MAP = {
    BaggingClassifier: MarsBaggingClassifier,
    IsolationForest: MarsIsolationForest,
    BaggingRegressor: MarsBaggingRegressor,
}

MARS_SKLEARN_EN_CALLABLES = _collect_module_callables(
    mars_en, sk_en, skip_members=["register_op"]
)
_install_cls_members(SKLEARN_EN_CLS_MAP, MARS_SKLEARN_EN_CALLABLES, sk_en)
attach_module_callable_docstring(BaggingClassifier, sk_en, sk_en.BaggingClassifier)
attach_module_callable_docstring(BaggingRegressor, sk_en, sk_en.BaggingRegressor)
attach_module_callable_docstring(IsolationForest, sk_en, sk_en.IsolationForest)

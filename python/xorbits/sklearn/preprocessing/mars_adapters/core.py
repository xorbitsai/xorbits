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

import sklearn.preprocessing as sk_preproc

from ...._mars.learn import preprocessing as mars_preproc
from ...._mars.learn.preprocessing import LabelBinarizer as MarsLabelBinarizer
from ...._mars.learn.preprocessing import LabelEncoder as MarsLabelEncoder
from ...._mars.learn.preprocessing import MinMaxScaler as MarsMinMaxScaler
from ....core.utils.docstring import attach_module_callable_docstring
from ...utils import SKLearnBase, _collect_module_callables, _install_cls_members


class MinMaxScaler(SKLearnBase):
    _marscls = MarsMinMaxScaler


class LabelBinarizer(SKLearnBase):
    _marscls = MarsLabelBinarizer


class LabelEncoder(SKLearnBase):
    _marscls = MarsLabelEncoder


SKLEARN_PREPROC_CLS_MAP = {
    MinMaxScaler: MarsMinMaxScaler,
    LabelEncoder: MarsLabelEncoder,
    LabelBinarizer: MarsLabelBinarizer,
}

MARS_SKLEARN_PREPROC_CALLABLES = _collect_module_callables(
    mars_preproc, sk_preproc, skip_members=["register_op"]
)
_install_cls_members(
    SKLEARN_PREPROC_CLS_MAP, MARS_SKLEARN_PREPROC_CALLABLES, sk_preproc
)
attach_module_callable_docstring(MinMaxScaler, sk_preproc, sk_preproc.MinMaxScaler)
attach_module_callable_docstring(LabelBinarizer, sk_preproc, sk_preproc.LabelBinarizer)
attach_module_callable_docstring(LabelEncoder, sk_preproc, sk_preproc.LabelEncoder)

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

import sklearn.metrics as sk_metrics

from ...._mars.learn import metrics as mars_metrics
from ...utils import _collect_module_callables

MARS_SKLEARN_METRICS_CALLABLES = _collect_module_callables(
    mars_metrics, sk_metrics, skip_members=["register_op"]
)

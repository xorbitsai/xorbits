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

# import xgboost as xgb

from .... import xgboost as xxgb


def test_XGBClassifier(setup, dummy_xgb_data):
    X, y_classifier, _ = dummy_xgb_data
    clf = xxgb.XGBClassifier()
    clf.fit(X, y_classifier)
    pred = clf.predict(X)
    print(pred)
    print(y_classifier)
    assert len(pred) == len(y_classifier)
    assert set(pred.to_numpy()).issubset({0, 1})

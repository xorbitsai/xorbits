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

try:
    import xgboost
except ImportError:  # pragma: no cover
    xgboost = None

import pytest

from ... import xgboost as xgb


@pytest.mark.skipif(xgboost is None, reason="XGBoost not installed")
def test_xgbclassifier_docstring():
    docstring = xgb.XGBClassifier.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from xgboost."
    )

    docstring = xgb.XGBClassifier.fit.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from xgboost.sklearn.XGBClassifier."
    )


@pytest.mark.skipif(xgboost is None, reason="XGBoost not installed")
def test_xgbregressor_docstring():
    docstring = xgb.XGBRegressor.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from xgboost."
    )

    docstring = xgb.XGBRegressor.fit.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from xgboost.sklearn.XGBRegressor."
    )

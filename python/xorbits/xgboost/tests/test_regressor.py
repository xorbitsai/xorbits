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

import numpy as np
import pytest

from ... import xgboost as xxgb


def test_XGBRegressor_array(setup, dummy_xgb_cls_array):
    X, y = dummy_xgb_cls_array

    regressor = xxgb.XGBRegressor(verbosity=1, n_estimators=2)
    regressor.fit(X, y, eval_set=[(X, y)])
    pred = regressor.predict(X)

    assert pred.ndim == 1
    assert pred.shape[0] == len(X)

    history = regressor.evals_result()

    assert isinstance(history, dict)
    assert list(history)[0] == "validation_0"

    assert len(pred) == len(y)


def test_XGBRegressor_df(setup, dummy_xgb_cls_df):
    X, y = dummy_xgb_cls_df

    regressor = xxgb.XGBRegressor(verbosity=1, n_estimators=2)
    regressor.fit(X, y, eval_set=[(X, y)])
    pred = regressor.predict(X)

    assert pred.ndim == 1
    assert pred.shape[0] == len(X)

    history = regressor.evals_result()

    assert isinstance(history, dict)

    assert list(history)[0] == "validation_0"

    assert len(pred) == len(y)

    # test weight
    weight = np.random.rand(X.shape[0])
    regressor.set_params(tree_method="hist")
    regressor.fit(X, y, sample_weight=weight)
    prediction = regressor.predict(X)

    assert prediction.ndim == 1
    assert prediction.shape[0] == len(X)

    # test wrong argument
    with pytest.raises(TypeError):
        regressor.fit(X, y, wrong_param=1)

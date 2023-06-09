# Copyright 2022-2023 XProbe Inc.
# derived from copyright 1999-2021 Alibaba Group Holding Ltd.
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
except ImportError:
    xgboost = None

import numpy as np
import pytest

from ... import xgboost as xxgb

X = np.random.rand(100, 10)
y = np.random.randint(0, 2, 100)


@pytest.mark.skipif(xgboost is None, reason="XGBoost not installed")
def test_local_predict_tensor(setup):
    from xgboost import Booster

    DMatrix = xxgb.DMatrix()
    dtrain = DMatrix(X, label=y)
    booster = xxgb.train({}, dtrain, num_boost_round=2)
    assert isinstance(booster, Booster)

    prediction = xxgb.predict(booster, X)
    assert isinstance(prediction.to_numpy(), np.ndarray)

    prediction = xxgb.predict(booster, dtrain)
    assert isinstance(prediction.fetch(), np.ndarray)

    with pytest.raises(TypeError):
        xxgb.predict(None, X)

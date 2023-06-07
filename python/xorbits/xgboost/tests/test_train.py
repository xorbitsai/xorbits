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


@pytest.mark.skipif(xgboost is None, reason="XGBoost not installed")
def test_train_evals(setup, dummy_xgb_cls_array):
    from xgboost import Booster

    X, y = dummy_xgb_cls_array
    base_margin = np.random.rand(X.shape[0])
    DMatrix = xxgb.DMatrix()
    dtrain = DMatrix(X, label=y, base_margin=base_margin)
    eval_x = DMatrix(X, label=y)
    evals = [(eval_x, "eval_x")]
    evals_result = dict()

    booster = xxgb.train(
        {}, dtrain, num_boost_round=2, evals=evals, evals_result=evals_result
    )
    assert isinstance(booster, Booster)
    assert len(evals_result) > 0

    prediction = xxgb.predict(booster, X)
    assert isinstance(prediction.to_numpy(), np.ndarray)

    with pytest.raises(TypeError):
        xxgb.train(
            {},
            dtrain,
            num_boost_round=2,
            evals=[("eval_x", eval_x)],
            evals_result=evals_result,
        )

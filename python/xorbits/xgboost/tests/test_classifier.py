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
except ImportError:
    xgboost = None

import numpy as np
import pandas as pd
import pytest

from ... import xgboost as xxgb

X = np.random.rand(100, 10)
X_df = pd.DataFrame(X)
y = np.random.randint(0, 2, 100)


@pytest.mark.skipif(xgboost is None, reason="XGBoost not installed")
def test_XGBClassifier_array(setup):
    classifier = xxgb.XGBClassifier(verbosity=1, n_estimators=2)

    classifier.fit(X, y, eval_set=[(X, y)])
    pred = classifier.predict(X)

    assert pred.ndim == 1
    assert pred.shape[0] == len(X)

    history = classifier.evals_result()

    assert isinstance(history, dict)

    assert list(history)[0] == "validation_0"

    prob = classifier.predict_proba(X)

    assert prob.shape[0] == X.shape[0]

    assert len(pred) == len(y)
    assert set(pred.to_numpy()).issubset({0, 1})


@pytest.mark.skipif(xgboost is None, reason="XGBoost not installed")
def test_XGBClassifier_df(setup):
    classifier = xxgb.XGBClassifier(verbosity=1, n_estimators=2)

    classifier.fit(X_df, y, eval_set=[(X_df, y)])
    pred = classifier.predict(X_df)

    assert pred.ndim == 1
    assert pred.shape[0] == len(X_df)

    history = classifier.evals_result()

    assert isinstance(history, dict)

    assert list(history)[0] == "validation_0"

    prob = classifier.predict_proba(X_df)

    assert prob.shape[0] == X_df.shape[0]

    assert len(pred) == len(y)
    assert set(pred.to_numpy().to_numpy()).issubset({0, 1})

    # test weight
    weights = [
        np.random.rand(X_df.shape[0]),
        pd.Series(np.random.rand(X_df.shape[0])),
        pd.DataFrame(np.random.rand(X_df.shape[0])),
    ]
    y_df = pd.DataFrame(y)
    for weight in weights:
        classifier.fit(X_df, y_df, sample_weight=weight)
        prediction = classifier.predict(X_df)

        assert prediction.ndim == 1
        assert prediction.shape[0] == len(X_df)

    # should raise error if weight.ndim > 1
    with pytest.raises(ValueError):
        classifier.fit(X_df, y_df, sample_weight=np.random.rand(1, 1))

    # test wrong argument
    with pytest.raises(TypeError):
        classifier.fit(X_df, y, wrong_param=1)

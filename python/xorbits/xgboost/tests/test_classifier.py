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
import pandas as pd
import pytest
import xgboost as xgb

from ... import xgboost as xxgb


def test_XGBClassifier_array(setup, dummy_xgb_cls_array):
    X, y = dummy_xgb_cls_array

    classifier = xxgb.XGBClassifier(verbosity=1, n_estimators=2)
    classifier.fit(X, y, eval_set=[(X, y)])
    pred = classifier.predict(X)

    assert pred.ndim == 1
    assert pred.shape[0] == len(X)

    history = classifier.evals_result()

    assert isinstance(history, dict)

    assert list(history)[0] == "validation_0"

    prob = classifier.predict_proba(X)
    # import pdb; pdb.set_trace()
    assert prob.shape[0] == X.shape[0]

    assert len(pred) == len(y)
    assert set(pred.to_numpy()).issubset({0, 1})


def test_XGBClassifier_df(setup, dummy_xgb_cls_df):
    X, y = dummy_xgb_cls_df

    classifier = xxgb.XGBClassifier(verbosity=1, n_estimators=2)
    classifier.fit(X, y, eval_set=[(X, y)])
    pred = classifier.predict(X)

    assert pred.ndim == 1
    assert pred.shape[0] == len(X)

    history = classifier.evals_result()

    assert isinstance(history, dict)

    assert list(history)[0] == "validation_0"

    prob = classifier.predict_proba(X)
    # import pdb; pdb.set_trace()
    assert prob.shape[0] == X.shape[0]

    assert len(pred) == len(y)
    assert set(pred.to_numpy().to_numpy()).issubset({0, 1})

    # test weight
    weights = [
        np.random.rand(X.shape[0]),
        pd.Series(np.random.rand(X.shape[0])),
        pd.DataFrame(np.random.rand(X.shape[0])),
    ]
    y_df = pd.DataFrame(y)
    for weight in weights:
        classifier.fit(X, y_df, sample_weight=weight)
        prediction = classifier.predict(X)

        assert prediction.ndim == 1
        assert prediction.shape[0] == len(X)

    # should raise error if weight.ndim > 1
    with pytest.raises(ValueError):
        classifier.fit(X, y_df, sample_weight=np.random.rand(1, 1))

    # test wrong argument
    with pytest.raises(TypeError):
        classifier.fit(X, y, wrong_param=1)


def test_xorbits_xgb_consistency(setup, dummy_xgb_cls_array):
    X, y = dummy_xgb_cls_array

    classifier = xgb.XGBClassifier(verbosity=1, n_estimators=2)
    classifier.fit(X, y, eval_set=[(X, y)])
    pred = classifier.predict(X)

    xclassifier = xxgb.XGBClassifier(verbosity=1, n_estimators=2)
    xclassifier.fit(X, y, eval_set=[(X, y)])
    xpred = xclassifier.predict(X)

    assert (xpred.to_numpy() == pred).all()

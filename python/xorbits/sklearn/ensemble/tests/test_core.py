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
    import sklearn
except ImportError:  # pragma: no cover
    sklearn = None

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC

from ...datasets import make_classification, make_regression
from ...ensemble import BaggingClassifier, BaggingRegressor, IsolationForest


@pytest.mark.skipif(sklearn is None, reason="scikit-learn not installed")
def test_doc():
    docstring = BaggingClassifier.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from sklearn.ensemble."
    )

    docstring = BaggingRegressor.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from sklearn.ensemble."
    )

    docstring = IsolationForest.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from sklearn.ensemble."
    )

    docstring = BaggingClassifier.fit.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from sklearn.ensemble._bagging.BaggingClassifier."
    )

    docstring = BaggingRegressor.fit.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from sklearn.ensemble._bagging.BaggingRegressor."
    )

    docstring = IsolationForest.fit.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from sklearn.ensemble._iforest.IsolationForest."
    )


@pytest.mark.skipif(sklearn is None, reason="scikit-learn not installed")
def test_baggingclassifier():
    rs = np.random.RandomState(0)

    raw_x, raw_y = make_classification(
        n_samples=100,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        random_state=rs,
        shuffle=False,
    )

    clf = BaggingClassifier(
        base_estimator=SVC(),
        n_estimators=10,
        max_samples=10,
        max_features=1,
        random_state=rs,
        warm_start=True,
    )

    clf.fit(raw_x, raw_y)
    log_proba = clf.predict_log_proba(raw_x)
    log_proba = log_proba.fetch()
    exp_log_proba_array = np.exp(log_proba)
    assert clf.n_estimators == 10
    assert np.all((exp_log_proba_array >= 0) & (exp_log_proba_array <= 1))
    assert np.allclose(np.sum(exp_log_proba_array, axis=1), 1.0)


def test_bagging_regression():
    rs = np.random.RandomState(0)

    raw_x, raw_y = make_regression(
        n_samples=100, n_features=4, n_informative=2, random_state=rs, shuffle=False
    )
    clf = BaggingRegressor(
        base_estimator=LinearRegression(),
        n_estimators=10,
        max_samples=10,
        max_features=0.5,
        random_state=rs,
        warm_start=True,
    )
    clf.fit(raw_x, raw_y)

    predict_y = clf.predict(raw_x)
    predict_y_array = predict_y.fetch()
    assert predict_y_array.shape == raw_y.shape


def test_iforest():
    rs = np.random.RandomState(0)
    raw_train = rs.poisson(size=(100, 10))
    raw_test = rs.poisson(size=(200, 10))

    clf = IsolationForest(random_state=rs, n_estimators=10, max_samples=1)
    pred = clf.fit(raw_train).predict(raw_test).fetch()
    score = clf.score_samples(raw_test).fetch()

    assert clf.n_estimators == 10
    assert pred.shape == (200,)
    assert score.shape == (200,)

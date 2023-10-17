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

from .. import LinearRegression, LogisticRegression

n_rows = 100
n_columns = 5
X = np.random.rand(n_rows, n_columns)
y = np.random.rand(n_rows)
y_cat = np.random.randint(0, 2, n_rows)
X_new = np.random.rand(n_rows, n_columns)


@pytest.mark.skipif(sklearn is None, reason="scikit-learn not installed")
def test_doc():
    docstring = LogisticRegression.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from sklearn.linear_model."
    )

    docstring = LogisticRegression.fit.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from sklearn.linear_model._logistic.LogisticRegression."
    )

    docstring = LinearRegression.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from sklearn.linear_model."
    )

    docstring = LinearRegression.fit.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from sklearn.linear_model._base.LinearRegression."
    )


@pytest.mark.skipif(sklearn is None, reason="scikit-learn not installed")
def test_linear_regression():
    lr = LinearRegression()
    lr.fit(X, y)
    predict = lr.predict(X_new)

    assert np.shape(lr.coef_.fetch()) == (n_columns,)
    assert np.shape(lr.intercept_.fetch()) == ()
    assert np.shape(predict) == (n_rows,)


@pytest.mark.skipif(sklearn is None, reason="scikit-learn not installed")
def test_logistic_regression():
    lr = LogisticRegression(max_iter=1)
    lr.fit(X, y_cat)
    predict = lr.predict(X_new).fetch()

    assert np.shape(predict) == (n_rows,)

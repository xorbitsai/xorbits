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
import sys

try:
    import sklearn
except ImportError:  # pragma: no cover
    sklearn = None

import inspect

import numpy as np
import pytest

from ... import metrics


@pytest.mark.skipif(sklearn is None, reason="scikit-learn not installed")
def test_doc():
    for name, f in inspect.getmembers(metrics, inspect.isfunction):
        if name.startswith("_"):
            continue
        docstring = f.__doc__
        assert docstring is not None


@pytest.mark.skipif(
    sklearn is None or sys.maxsize <= 2**32, reason="scikit-learn not installed"
)
def test_classification():
    from sklearn.metrics import f1_score as sklearn_f1_score
    from sklearn.metrics import fbeta_score as sklearn_fbeta_score
    from sklearn.metrics import (
        multilabel_confusion_matrix as sklearn_multilabel_confusion_matrix,
    )
    from sklearn.metrics import (
        precision_recall_fscore_support as sklearn_precision_recall_fscore_support,
    )
    from sklearn.metrics import precision_score as sklearn_precision_score
    from sklearn.metrics import recall_score as sklearn_recall_score

    from ...metrics import (
        f1_score,
        fbeta_score,
        multilabel_confusion_matrix,
        precision_recall_fscore_support,
        precision_score,
        recall_score,
    )

    y_true = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    y_pred = np.array([0, 2, 1, 0, 0, 1], dtype=np.int64)

    np.testing.assert_array_almost_equal(
        f1_score(y_true, y_pred, average="macro").execute().fetch(),
        sklearn_f1_score(y_true, y_pred, average="macro"),
    )
    np.testing.assert_array_almost_equal(
        fbeta_score(y_true, y_pred, beta=0.5, average="macro").execute().fetch(),
        sklearn_fbeta_score(y_true, y_pred, beta=0.5, average="macro"),
    )

    np.testing.assert_array_almost_equal(
        precision_score(y_true, y_pred, average="macro").execute().fetch(),
        sklearn_precision_score(y_true, y_pred, average="macro"),
    )

    np.testing.assert_array_almost_equal(
        recall_score(y_true, y_pred, average="macro").execute().fetch(),
        sklearn_recall_score(y_true, y_pred, average="macro"),
    )

    np.testing.assert_array_almost_equal(
        multilabel_confusion_matrix(y_true, y_pred).execute().fetch(),
        sklearn_multilabel_confusion_matrix(y_true, y_pred),
    )

    np.testing.assert_array_almost_equal(
        precision_recall_fscore_support(y_true, y_pred)[0].execute().fetch(),
        sklearn_precision_recall_fscore_support(y_true, y_pred)[0],
    )


@pytest.mark.skipif(sklearn is None, reason="scikit-learn not installed")
def test_scorer():
    from sklearn.metrics import r2_score

    from ...metrics import get_scorer

    assert get_scorer("r2") is not None
    assert get_scorer(r2_score) is not None


@pytest.mark.skipif(sklearn is None, reason="scikit-learn not installed")
def test_r2_score():
    from ...metrics import r2_score

    y_true = np.array([[1, 0, 0, 1], [0, 1, 1, 1], [1, 1, 0, 1]])
    y_pred = np.array([[0, 0, 0, 1], [1, 0, 1, 1], [0, 0, 0, 1]])

    error = r2_score(y_true, y_pred, multioutput="variance_weighted")
    np.testing.assert_almost_equal(error.fetch(), 1.0 - 5.0 / 2)


@pytest.mark.skipif(sklearn is None, reason="scikit-learn not installed")
def test_ranking():
    from sklearn.metrics import accuracy_score as sklearn_accuracy_score
    from sklearn.metrics import auc as sklearn_auc
    from sklearn.metrics import roc_curve as sklearn_roc_curve
    from sklearn.metrics.tests.test_ranking import make_prediction

    from ...metrics import accuracy_score, auc, roc_auc_score, roc_curve

    y_true, y_score, _ = make_prediction(binary=True)

    np.testing.assert_almost_equal(
        accuracy_score(y_true, y_score).fetch(),
        sklearn_accuracy_score(y_true, y_score),
    )
    rs = np.random.RandomState(0)
    y = rs.randint(0, 10, (10,))
    pred = rs.rand(10)
    fpr, tpr, thresholds = roc_curve(y, pred, pos_label=2)
    m = auc(fpr, tpr)

    sk_fpr, sk_tpr, sk_threshod = sklearn_roc_curve(
        y,
        pred,
        pos_label=2,
    )
    expect_m = sklearn_auc(sk_fpr, sk_tpr)
    assert pytest.approx(m.fetch()) == expect_m
    y_true = np.array([0, 0, 1, 1], dtype=np.int64)
    assert roc_auc_score(y_true, y_true, max_fpr=1) == 1

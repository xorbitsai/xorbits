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

import pytest

import xorbits.numpy as np

from ... import datasets
from ...datasets import (
    make_blobs,
    make_classification,
    make_low_rank_matrix,
    make_regression,
)


@pytest.mark.skipif(sklearn is None, reason="scikit-learn not installed")
def test_doc():
    docstring = datasets.make_blobs.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from sklearn.datasets."
    )

    docstring = datasets.make_classification.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from sklearn.datasets."
    )

    docstring = datasets.make_low_rank_matrix.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from sklearn.datasets."
    )

    docstring = datasets.make_regression.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from sklearn.datasets."
    )


def test_make_classification():
    weights = [0.1, 0.25]
    X, y = make_classification(
        n_samples=100,
        n_features=20,
        n_informative=5,
        n_redundant=1,
        n_repeated=1,
        n_classes=3,
        n_clusters_per_class=1,
        hypercube=False,
        shift=None,
        scale=None,
        weights=weights,
        random_state=0,
        flip_y=-1,
    )
    X, y = X.execute().fetch(), y.execute().fetch()
    assert X.shape == (100, 20)
    assert y.shape == (100,)
    assert np.unique(y).shape == (3,)
    assert (y == 0).sum() == 10
    assert (y == 1).sum() == 25
    assert (y == 2).sum() == 65


@pytest.mark.skipif(sklearn is None, reason="scikit-learn not installed")
def test_make_regression():
    X, y, c = make_regression(
        n_samples=100,
        n_features=10,
        n_informative=3,
        effective_rank=5,
        coef=True,
        bias=0.0,
        noise=1.0,
        random_state=0,
    )
    X, y, c = X.execute().fetch(), y.execute().fetch(), c.execute().fetch()
    assert X.shape == (100, 10), "X shape mismatch"
    assert y.shape == (100,), "y shape mismatch"
    assert c.shape == (10,), "coef shape mismatch"
    assert sum(c != 0.0) == 3, "Unexpected number of informative features"


@pytest.mark.skipif(sklearn is None, reason="scikit-learn not installed")
def test_make_blobs():
    cluster_stds = np.array([0.05, 0.2, 0.4])
    cluster_centers = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    X, y = make_blobs(
        random_state=0,
        n_samples=50,
        n_features=2,
        centers=cluster_centers,
        cluster_std=cluster_stds,
    )
    X, y = X.execute().fetch(), y.execute().fetch()
    assert X.shape == (50, 2)
    assert y.shape == (50,)
    assert np.unique(y).shape == (3,)


@pytest.mark.skipif(sklearn is None, reason="scikit-learn not installed")
def test_make_low_rank_matrix():
    X = make_low_rank_matrix(
        n_samples=50,
        n_features=25,
        effective_rank=5,
        tail_strength=0.01,
        random_state=0,
    )
    X = X.execute().fetch()
    assert X.shape == (50, 25)
    _, s, _ = np.linalg.svd(X)
    s = s.execute().fetch()
    assert (s.sum() - 5) < 0.1

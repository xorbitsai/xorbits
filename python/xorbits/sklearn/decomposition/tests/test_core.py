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
import scipy.sparse as sp
from numpy.testing import assert_array_almost_equal, assert_equal
from sklearn import datasets
from sklearn.utils import check_random_state

from .. import PCA, TruncatedSVD

iris = np.asarray(datasets.load_iris().data)


@pytest.mark.skipif(sklearn is None, reason="scikit-learn not installed")
def test_doc():
    docstring = PCA.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from sklearn.decomposition."
    )

    docstring = PCA.fit.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from sklearn.decomposition._pca.PCA."
    )

    docstring = TruncatedSVD.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from sklearn.decomposition."
    )

    docstring = TruncatedSVD.fit.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from sklearn.decomposition._truncated_svd.TruncatedSVD."
    )


@pytest.mark.skipif(sklearn is None, reason="scikit-learn not installed")
def test_pca():
    X = iris

    for n_comp in np.arange(X.shape[1]):
        pca = PCA(n_components=n_comp, svd_solver="full")
        pca.fit(X)
        X_r = pca.transform(X).fetch()
        assert_equal(X_r.shape[1], n_comp)

        X_r2 = pca.fit_transform(X).fetch()
        assert_array_almost_equal(X_r, X_r2)

        X_r = pca.transform(X).fetch()
        X_r2 = pca.fit_transform(X).fetch()
        assert_array_almost_equal(X_r, X_r2)

        # Test get_covariance and get_precision
        cov = pca.get_covariance()
        precision = pca.get_precision()
        assert_array_almost_equal(np.dot(cov, precision), np.eye(X.shape[1]), 12)


@pytest.mark.skipif(sklearn is None, reason="scikit-learn not installed")
def test_truncated_svd():
    shape = 60, 55
    n_samples, n_features = shape
    rng = check_random_state(42)
    X = rng.randint(-100, 20, np.product(shape)).reshape(shape)
    X = sp.csr_matrix(np.maximum(X, 0), dtype=np.float64)
    for n_components in (10, 25, 41):
        tsvd = TruncatedSVD(n_components).fit(X)
        assert tsvd.n_components == n_components
        assert tsvd.components_.shape == (n_components, n_features)

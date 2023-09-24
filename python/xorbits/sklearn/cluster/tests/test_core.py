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

from .... import numpy as xnp
from .. import KMeans

n_rows = 1000
n_clusters = 8
n_columns = 10
chunk_size = 200
rs = xnp.random.RandomState(0)
X = rs.rand(n_rows, n_columns, chunk_size=chunk_size)
X_new = rs.rand(n_rows, n_columns, chunk_size=chunk_size)


@pytest.mark.skipif(sklearn is None, reason="scikit-learn not installed")
def test_doc():
    docstring = KMeans.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from sklearn.cluster."
    )

    docstring = KMeans.fit.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from sklearn.cluster._kmeans.KMeans."
    )


@pytest.mark.skipif(sklearn is None, reason="sci-kit-learn not installed")
def test_kmeans_cluster():
    kms = KMeans(n_clusters=n_clusters, random_state=0)
    kms.fit(X)
    predict = kms.predict(X_new).fetch()

    assert kms.n_clusters == n_clusters
    assert np.shape(kms.labels_.fetch()) == (n_rows,)
    assert np.shape(kms.cluster_centers_.fetch()) == (n_clusters, n_columns)
    assert np.shape(predict) == (n_rows,)

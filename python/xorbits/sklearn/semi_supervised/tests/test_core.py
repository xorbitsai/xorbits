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

from ...semi_supervised import LabelPropagation


@pytest.mark.skipif(sklearn is None, reason="scikit-learn not installed")
def test_doc():
    docstring = LabelPropagation.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from sklearn.semi_supervised."
    )

    docstring = LabelPropagation.fit.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from sklearn.semi_supervised._label_propagation.LabelPropagation."
    )


@pytest.mark.skipif(sklearn is None, reason="scikit-learn not installed")
def test_label_propagation():
    rng = np.random.RandomState(0)
    X = rng.rand(10, 5)
    y = np.array([0, 0, 0, 1, 1, -1, -1, -1, -1, -1])
    lp = LabelPropagation()
    lp.fit(X, y)
    assert lp.classes_.tolist() == [0, 1]
    assert lp.transduction_.tolist() == [0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
    assert lp.predict(X).tolist() == [0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
    assert lp.score(X, y) == 0.5

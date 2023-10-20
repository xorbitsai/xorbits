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

from ...preprocessing import LabelBinarizer, LabelEncoder, MinMaxScaler


@pytest.mark.skipif(sklearn is None, reason="scikit-learn not installed")
def test_doc():
    docstring = MinMaxScaler.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from sklearn.preprocessing."
    )

    docstring = LabelBinarizer.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from sklearn.preprocessing."
    )

    docstring = LabelEncoder.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from sklearn.preprocessing."
    )

    docstring = MinMaxScaler.fit.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from sklearn.preprocessing._data.MinMaxScaler."
    )

    docstring = LabelBinarizer.fit.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from sklearn.preprocessing._label.LabelBinarizer."
    )

    docstring = LabelEncoder.fit.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from sklearn.preprocessing._label.LabelEncoder."
    )


@pytest.mark.skipif(sklearn is None, reason="scikit-learn not installed")
def test_min_max_scaler():
    X = np.array([[1, 2], [2, 4], [4, 8], [8, 16]], dtype=np.float64)
    scaler = MinMaxScaler()
    scaler.fit(X)
    np.testing.assert_array_equal(scaler.data_min_, [1.0, 2.0])
    np.testing.assert_array_equal(scaler.data_max_, [8.0, 16.0])
    np.testing.assert_array_equal(scaler.data_range_, [7.0, 14.0])

    X_transformed = scaler.transform(X).fetch()
    assert X_transformed.shape == (4, 2)


@pytest.mark.skipif(sklearn is None, reason="scikit-learn not installed")
def test_label_binarizer():
    lb = LabelBinarizer()
    lb.fit([1, 2, 6, 4, 2])
    assert lb.classes_.tolist() == [1, 2, 4, 6]


@pytest.mark.skipif(sklearn is None, reason="scikit-learn not installed")
def test_label_encoder():
    le = LabelEncoder()
    le.fit([1, 2, 2, 6])
    assert le.classes_.tolist() == [1, 2, 6]

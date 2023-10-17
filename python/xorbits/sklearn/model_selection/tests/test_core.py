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

from typing import Iterable, Sized

import numpy as np
import pytest

from ...model_selection import KFold, ParameterGrid, train_test_split


@pytest.mark.skipif(sklearn is None, reason="scikit-learn not installed")
def test_doc():
    docstring = KFold.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from sklearn.model_selection."
    )

    docstring = ParameterGrid.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from sklearn.model_selection."
    )


@pytest.mark.skipif(sklearn is None, reason="scikit-learn not installed")
def test_parameter_grid():
    arr1 = [1, 2, 3]
    params1 = {"foo": arr1}
    grid1 = ParameterGrid(params1)
    assert isinstance(grid1, Iterable)
    assert isinstance(grid1, Sized)
    assert len(grid1) == 3


@pytest.mark.skipif(sklearn is None, reason="scikit-learn not installed")
def test_kfold():
    X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    kf = KFold(n_splits=2)
    splits = kf.get_n_splits(X)
    assert splits == 2


@pytest.mark.skipif(sklearn is None, reason="scikit-learn not installed")
def test_train_test_split():
    X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    y = np.array([1, 2, 3, 4])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    assert X_train.shape == (2, 2)
    assert X_test.shape == (2, 2)
    assert y_train.shape == (2,)
    assert y_test.shape == (2,)

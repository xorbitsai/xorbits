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

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from ... import numpy as xnp
from ... import pandas as xpd

try:
    import lightgbm
except ImportError:
    lightgbm = None
from ... import lightgbm as lgb

n_rows = 1000
n_columns = 10
chunk_size = 200
rs = xnp.random.RandomState(0)
X_raw = rs.rand(n_rows, n_columns, chunk_size=chunk_size)
y_raw = rs.rand(n_rows, chunk_size=chunk_size)
filter = rs.rand(n_rows, chunk_size=chunk_size) < 0.8
X = X_raw[filter]
y = y_raw[filter]

X_df = xpd.DataFrame(X)
x_sparse = np.random.rand(n_rows, n_columns)
x_sparse[np.arange(n_rows), np.random.randint(n_columns, size=n_rows)] = np.nan
X_sparse = xnp.array(x_sparse, chunk_size=chunk_size).tosparse(missing=np.nan)[filter]


@pytest.mark.skipif(lightgbm is None, reason="LightGBM not installed")
def test_local_classifier(setup):
    y_data = (y * 10).astype(xnp.int64)
    classifier = lgb.LGBMClassifier(n_estimators=2)
    classifier.fit(X, y_data, eval_set=[(X, y_data)])
    # import pdb;pdb.set_trace()
    prediction = classifier.predict(X)

    assert prediction.ndim == 1
    assert prediction.shape[0] == len(X)

    assert isinstance(prediction, xnp.ndarray)

    # test sparse tensor
    X_sparse_data = X_sparse
    classifier = lgb.LGBMClassifier(n_estimators=2)
    classifier.fit(X_sparse_data, y_data, eval_set=[(X_sparse_data, y_data)])
    prediction = classifier.predict(X_sparse_data)

    assert prediction.ndim == 1
    assert prediction.shape[0] == len(X)

    assert isinstance(prediction, xnp.ndarray)

    prob = classifier.predict_proba(X)
    assert prob.shape == X.shape

    prediction_empty = classifier.predict(xnp.array([]).reshape((0, X.shape[1])))
    assert prediction_empty.shape == (0,)

    # test dataframe
    X_df_data = X_df
    classifier = lgb.LGBMClassifier(n_estimators=2)
    classifier.fit(X_df_data, y_data)
    prediction = classifier.predict(X_df_data)

    assert prediction.ndim == 1
    assert prediction.shape[0] == len(X)

    prob = classifier.predict_proba(X_df)

    assert prob.ndim == 2
    assert prob.shape == (len(X), 10)

    # test weight
    weights = [xnp.random.rand(X.shape[0]), xpd.Series(xnp.random.rand(X.shape[0]))]
    y_df = xpd.DataFrame(y_data)
    for weight in weights:
        classifier = lgb.LGBMClassifier(n_estimators=2)
        classifier.fit(X, y_df, sample_weight=weight)
        prediction = classifier.predict(X)

        assert prediction.ndim == 1
        assert prediction.shape[0] == len(X)

    # should raise error if weight.ndim > 1
    with pytest.raises(ValueError):
        lgb.LGBMClassifier(n_estimators=2).fit(
            X, y_df, sample_weight=xnp.random.rand(1, 1)
        )

    # test binary classifier
    new_y = (y_data > 0.5).astype(xnp.int64)
    classifier = lgb.LGBMClassifier(n_estimators=2)
    classifier.fit(X, new_y)

    prediction = classifier.predict(X)
    assert prediction.ndim == 1
    assert prediction.shape[0] == len(X)

    prediction = classifier.predict_proba(X)
    assert prediction.ndim == 2
    assert prediction.shape[0] == len(X)

    # test with existing model
    X_np = X.execute().fetch()
    new_y_np = new_y.execute().fetch()
    raw_classifier = lightgbm.LGBMClassifier(n_estimators=2)
    raw_classifier.fit(X_np, new_y_np)

    classifier = lgb.LGBMClassifier(raw_classifier)
    label_result = classifier.predict(X_df)
    assert label_result.ndim == 1
    assert label_result.shape[0] == len(X)

    proba_result = classifier.predict_proba(X_df)
    assert proba_result.ndim == 2
    assert proba_result.shape[0] == len(X)


@pytest.mark.skipif(lightgbm is None, reason="LightGBM not installed")
def test_local_classifier_from_to_parquet(setup):
    n_rows = 1000
    n_columns = 10
    rs = np.random.RandomState(0)
    X = rs.rand(n_rows, n_columns)
    y = (rs.rand(n_rows) > 0.5).astype(np.int32)
    df = pd.DataFrame(X, columns=[f"c{i}" for i in range(n_columns)])

    # test with existing model
    classifier = lightgbm.LGBMClassifier(n_estimators=2)
    classifier.fit(X, y)

    with tempfile.TemporaryDirectory() as d:
        result_dir = os.path.join(d, "result")
        os.mkdir(result_dir)
        data_dir = os.path.join(d, "data")
        os.mkdir(data_dir)

        df.iloc[:500].to_parquet(os.path.join(d, "data", "data1.parquet"))
        df.iloc[500:].to_parquet(os.path.join(d, "data", "data2.parquet"))

        df = xpd.read_parquet(data_dir, dtype_backend="numpy_nullable")
        model = lgb.LGBMClassifier()
        model.load_model(classifier)
        result = model.predict(df, run=False)
        r = xpd.DataFrame(result).to_parquet(result_dir)

        r.execute()

        ret = (
            xpd.read_parquet(result_dir, dtype_backend="numpy_nullable")
            .to_pandas()
            .iloc[:, 0]
            .to_numpy()
        )
        expected = classifier.predict(X)
        expected = np.stack([1 - expected, expected]).argmax(axis=0)
        np.testing.assert_array_equal(ret, expected)

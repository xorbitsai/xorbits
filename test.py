import pytest
from python.xorbits import numpy as np
from python.xorbits import pandas as pd
from python.xorbits import xgboost as xxgb
from python.xorbits._mars.core.entity.objects import ObjectData


X = np.random.rand(100, 10)
X_df = pd.DataFrame(X)
y = np.random.randint(0, 2, 100)

classifier = xxgb.XGBClassifier(verbosity=1, n_estimators=2)

classifier.fit(X_df, y, eval_set=[(X_df, y)])
pred = classifier.predict(X_df)

assert pred.ndim == 1
assert pred.shape[0] == len(X_df)

history = classifier.evals_result()

assert isinstance(history, dict)

assert list(history)[0] == "validation_0"

prob = classifier.predict_proba(X_df)

assert prob.shape[0] == X_df.shape[0]

assert len(pred) == len(y)
assert set(pred.to_numpy().to_numpy()).issubset({0, 1})

# test weight
weights = [
    np.random.rand(X_df.shape[0]),
    pd.Series(np.random.rand(X_df.shape[0])),
    pd.DataFrame(np.random.rand(X_df.shape[0])),
]
y_df = pd.DataFrame(y)
for weight in weights:
    classifier.fit(X_df, y_df, sample_weight=weight)
    prediction = classifier.predict(X_df)

    assert prediction.ndim == 1
    assert prediction.shape[0] == len(X_df)

# should raise error if weight.ndim > 1
with pytest.raises(ValueError):
    classifier.fit(X_df, y_df, sample_weight=np.random.rand(1, 1))

# test wrong argument
with pytest.raises(TypeError):
    classifier.fit(X_df, y, wrong_param=1)

# test wrong attribute
with pytest.raises(AttributeError):
    classifier.wrong_attribute()

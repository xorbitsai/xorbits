import time
from sklearn.datasets import load_iris
from xorbits._mars.learn.glm import LogisticRegression

X, y = load_iris(return_X_y=True)

start = time.time()
clf = LogisticRegression(random_state=0, max_iter=3).fit(X, y)
print(clf.predict(X[:2, :]))
print(time.time() - start)

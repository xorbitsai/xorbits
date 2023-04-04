# import time
# from sklearn.datasets import load_iris
# from xorbits._mars.learn.glm import LogisticRegression

# X, y = load_iris(return_X_y=True)

# start = time.time()
# clf = LogisticRegression(random_state=0, max_iter=3).fit(X, y)
# print(clf.predict(X[:2, :]))
# print(time.time() - start)

import xorbits._mars.tensor as mt
import xorbits._mars.remote as mr

def test(a):
    a = a.reshape((1, 5))
    return a

l = mr.ExecutableTuple([mr.spawn(test, args=(mt.ones((5, 1)).to_numpy())) for _ in range(10)]).execute().fetch()

print(l)
print(mt.sum(l, axis=0).execute().fetch())


# Copyright 2022-2023 XProbe Inc.
# derived from copyright 1999-2021 Alibaba Group Holding Ltd.
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

import math

import numpy as np

from ... import remote as mr
from ... import tensor as mt
from ...tensor.datasource import tensor as astensor


def batch_generator(idx, X, y):
    for i in idx:
        yield X[i], y[i]


def single_softmax_grad(W, X, y, reg):
    X = X.reshape((1, X.shape[0]))
    K = W.shape[1]
    y_obs = np.eye(K)[y]

    dW = np.zeros(shape=(X.shape[1], K))

    # Matrix approach
    dW = (
        -1
        * X.T
        @ (y_obs - (np.exp(X @ W) / np.sum(np.exp(X @ W), axis=1).reshape(-1, 1)))
        + reg * W
    )

    return dW


def gradient_descent(
    X,
    y,
    learning_rate=1e-3,
    reg=1e-5,
    max_epochs=100,
    batch_size=20,
    fit_intercept=True,
    verbose=0,
):
    # assume y takes values 0...K-1 where K is number of classes
    num_classes = (mt.max(y) + 1).to_numpy()

    num_train, dim = X.shape
    num_iters_per_epoch = int(math.floor(1.0 * num_train / batch_size))

    # need extra entries if fit_intercept
    if fit_intercept:
        X = mt.hstack((X, mt.ones((num_train, 1))))
        W = 0.001 * mt.random.randn(dim + 1, num_classes).execute()
    else:
        X = astensor(X)
        W = 0.001 * mt.random.randn(dim, num_classes).execute()

    X = X.to_numpy()
    W = W.to_numpy()
    y = y.to_numpy()

    for _ in range(max_epochs):
        # perform mini-batch SGD update
        perm_idx = np.random.permutation(num_train)
        for it in range(num_iters_per_epoch):
            idx = perm_idx[it * batch_size : (it + 1) * batch_size]
            funcs = mr.ExecutableTuple(
                [
                    mr.spawn(
                        single_softmax_grad,
                        args=(W, X[i], y[i], reg),
                    )
                    for i in idx
                ]
            )
            grad_tensors = funcs.execute().fetch()
            grad_sum = np.sum(grad_tensors, axis=0) / len(idx)

            # update parameters
            W = W - learning_rate * grad_sum

    return W

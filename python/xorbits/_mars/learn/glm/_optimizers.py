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


def instance_softmax_loss_and_sgd(W, X, y, loss_reg_W, i):
    X = X[i].reshape((1, X[i].shape[0]))
    y = y[i]

    N, D = X.shape
    K = W.shape[1]

    y_obs = mt.eye(K)[y]

    exp_X_W_over_sum = mt.exp(X @ W) / mt.sum(mt.exp(X @ W), axis=1).reshape(-1, 1)

    loss = -1 * mt.sum(y_obs * mt.log(exp_X_W_over_sum)) + loss_reg_W

    dW = mt.zeros(shape=(D, K))

    # Matrix approach
    dW = -1 * X.T @ (y_obs - (exp_X_W_over_sum))

    return loss, dW


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

    X = X.execute()
    y = y.execute()

    for _ in range(max_epochs):
        # perform mini-batch SGD update
        perm_idx = np.random.permutation(num_train)
        for it in range(num_iters_per_epoch):
            idx = perm_idx[it * batch_size : (it + 1) * batch_size]
            cur_W = W.execute()
            loss_reg_W = (0.5 * reg * mt.sum(mt.square(W))).execute()
            dW_reg_W = (reg * cur_W).execute()

            funcs = [
                mr.spawn(
                    instance_softmax_loss_and_sgd,
                    args=(
                        cur_W,
                        X,
                        y,
                        loss_reg_W,
                        i,
                    ),
                )
                for i in idx
            ]

            outs = mr.ExecutableTuple(funcs).execute().fetch()
            grad_tensors = [out[1] for out in outs]

            # update parameters
            W = (
                cur_W
                - learning_rate * (mt.sum(grad_tensors, axis=0) / len(idx) + dW_reg_W)
            ).execute()

    return W

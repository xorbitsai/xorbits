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

import numpy as np
import pytest

from .... import numpy as xnp


@pytest.mark.parametrize(
    "mod_name, func, params",
    [
        ("", "isneginf", [np.NINF]),
        ("", "isposinf", [np.PINF]),
        (
            "",
            "einsum_path",
            [
                "ij,jk,kl->il",
                np.random.rand(2, 2),
                np.random.rand(2, 5),
                np.random.rand(5, 2),
            ],
        ),
        ("", "outer", [np.ones((5,)), np.linspace(-2, 2, 5)]),
        ("", "kron", [[1, 10, 100], [5, 6, 7]]),
        ("", "trace", [np.arange(8).reshape((2, 2, 2))]),
        ("linalg", "cond", [np.array([[1, 0, -1], [0, 1, 0], [1, 0, 1]])]),
        ("linalg", "det", [np.array([[1, 2], [3, 4]])]),
        ("linalg", "eig", [np.diag((1, 2, 3))]),
        ("linalg", "eigh", [np.array([[1, -2j], [2j, 5]])]),
        ("linalg", "eigvals", [np.diag((-1, 1))]),
        ("linalg", "eigvalsh", [np.array([[5 + 2j, 9 - 2j], [0 + 2j, 2 - 1j]])]),
        (
            "linalg",
            "multi_dot",
            [
                [
                    np.random.random((10000, 100)),
                    np.random.random((100, 1000)),
                    np.random.random((1000, 5)),
                    np.random.random((5, 333)),
                ]
            ],
        ),
        ("linalg", "matrix_power", [np.array([[0, 1], [-1, 0]]), 3]),
        ("linalg", "matrix_rank", [np.eye(4)]),
        (
            "linalg",
            "lstsq",
            [
                np.vstack(
                    [np.array([0, 1, 2, 3]), np.ones(len(np.array([0, 1, 2, 3])))]
                ).T,
                np.array([-1, 0.2, 0.9, 2.1]),
            ],
        ),
        (
            "linalg",
            "slogdet",
            [np.array([[[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]]])],
        ),
        ("linalg", "pinv", [np.random.randn(9, 6)]),
        ("random", "default_rng", []),
        ("random", "PCG64", []),
        ("random", "MT19937", []),
        ("random", "Generator", [np.random.PCG64()]),
    ],
)
def test_numpy_fallback(mod_name, func, params):
    with pytest.warns(Warning) as w:
        xnp_func = (
            getattr(getattr(xnp, mod_name), func)
            if mod_name != ""
            else getattr(xnp, func)
        )
        np_func = (
            getattr(getattr(np, mod_name), func)
            if mod_name != ""
            else getattr(np, func)
        )

        xnp_output = xnp_func(*params).execute().fetch()
        np_output = np_func(*params)

        assert (
            f"xorbits.numpy{'.' + mod_name if mod_name != '' else ''}.{func} will fallback to NumPy"
            == str(w[0].message)
        )

        if isinstance(xnp_output, int):
            xnp_output = np.int64(xnp_output)
        if isinstance(xnp_output, float):
            xnp_output = np.float64(xnp_output)
        assert type(xnp_output) == type(np_output)

        if isinstance(np_output, np.ndarray):
            assert isinstance(xnp_output, np.ndarray)
            assert np.equal(xnp_output.all(), np_output.all())
        if isinstance(np_output, object):
            assert isinstance(xnp_output, object)
            assert dir(xnp_output) == dir(np_output)


def test_tensorsolve_fallback(setup):
    with pytest.warns(Warning) as w:
        a = np.eye(2 * 3 * 4)
        a.shape = (2 * 3, 4, 2, 3, 4)
        b = np.random.randn(2 * 3, 4)

        xnp_output = xnp.linalg.tensorsolve(a, b).execute().fetch()
        np_output = np.linalg.tensorsolve(a, b)

        assert f"xorbits.numpy.linalg.tensorsolve will fallback to NumPy" == str(
            w[0].message
        )

        assert np.equal(xnp_output.all(), np_output.all())


def test_tensorinv_fallback(setup):
    with pytest.warns(Warning) as w:
        a = np.eye(4 * 6)
        a.shape = (4, 6, 8, 3)

        xnp_output = xnp.linalg.tensorinv(a, ind=2).execute().fetch()
        np_output = np.linalg.tensorinv(a, ind=2)

        assert f"xorbits.numpy.linalg.tensorinv will fallback to NumPy" == str(
            w[0].message
        )

        assert np.equal(xnp_output.all(), np_output.all())

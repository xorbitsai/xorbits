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
import pandas as pd
import pytest

from .... import numpy as xnp
from .... import pandas as xpd


def valid_func(xnp_output, np_output):
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


def test_numpy_fallback(setup):
    # TODO: Expected dtype cannot be None
    # with pytest.warns(Warning) as w:
    #     xnp_output = xnp.busday_count("2011-01", "2011-02").execute().fetch()
    #     np_output = np.busday_count("2011-01", "2011-02")
    #     assert f"xorbits.numpy.busday_count will fallback to NumPy" == str(
    #         w[0].message
    #     )
    #     valid_func(xnp_output, np_output)

    with pytest.warns(Warning) as w:
        xnp_output = xnp.isneginf(-np.inf).execute().fetch()
        np_output = np.isneginf(-np.inf)
        assert f"xorbits.numpy.isneginf will fallback to NumPy" == str(w[0].message)
        valid_func(xnp_output, np_output)

    with pytest.warns(Warning) as w:
        xnp_output = xnp.isposinf(np.inf).execute().fetch()
        np_output = np.isposinf(np.inf)
        assert f"xorbits.numpy.isposinf will fallback to NumPy" == str(w[0].message)
        valid_func(xnp_output, np_output)

    with pytest.warns(Warning) as w:
        xnp_output = (
            xnp.einsum_path(
                "ij,jk,kl->il",
                np.random.rand(2, 2),
                np.random.rand(2, 5),
                np.random.rand(5, 2),
            )
            .execute()
            .fetch()
        )
        np_output = np.einsum_path(
            "ij,jk,kl->il",
            np.random.rand(2, 2),
            np.random.rand(2, 5),
            np.random.rand(5, 2),
        )
        assert f"xorbits.numpy.einsum_path will fallback to NumPy" == str(w[0].message)
        valid_func(xnp_output, np_output)

    with pytest.warns(Warning) as w:
        xnp_output = xnp.outer(np.ones((5,)), np.linspace(-2, 2, 5)).execute().fetch()
        np_output = np.outer(np.ones((5,)), np.linspace(-2, 2, 5))
        assert f"xorbits.numpy.outer will fallback to NumPy" == str(w[0].message)
        valid_func(xnp_output, np_output)

    # TODO AssertionError: Expected dtype cannot be None
    # with pytest.warns(Warning) as w:
    #     xnp_output = xnp.kron([1, 10, 100], [5, 6, 7]).execute().fetch()
    #     np_output = np.kron([1, 10, 100], [5, 6, 7])
    #     assert f"xorbits.numpy.kron will fallback to NumPy" == str(
    #         w[0].message
    #     )
    #     valid_func(xnp_output, np_output)

    # TODO: AssertionError: Expected dtype cannot be None
    # with pytest.warns(Warning) as w:
    #     xnp_output = xnp.trace(np.arange(8).reshape((2, 2, 2))).execute().fetch()
    #     np_output = np.trace(np.arange(8).reshape((2, 2, 2)))
    #     assert f"xorbits.numpy.trace will fallback to NumPy" == str(
    #         w[0].message
    #     )
    #     valid_func(xnp_output, np_output)


def test_linalg_fallback(setup):
    with pytest.warns(Warning) as w:
        xnp_output = (
            xnp.linalg.cond(np.array([[1, 0, -1], [0, 1, 0], [1, 0, 1]]))
            .execute()
            .fetch()
        )
        np_output = np.linalg.cond(np.array([[1, 0, -1], [0, 1, 0], [1, 0, 1]]))
        assert f"xorbits.numpy.linalg.cond will fallback to NumPy" == str(w[0].message)
        valid_func(xnp_output, np_output)

    with pytest.warns(Warning) as w:
        xnp_output = xnp.linalg.det(np.array([[1, 2], [3, 4]])).execute().fetch()
        np_output = np.linalg.det(np.array([[1, 2], [3, 4]]))
        assert f"xorbits.numpy.linalg.det will fallback to NumPy" == str(w[0].message)
        valid_func(xnp_output, np_output)

    with pytest.warns(Warning) as w:
        xnp_output = xnp.linalg.eig(np.diag((1, 2, 3))).execute().fetch()
        np_output = np.linalg.eig(np.diag((1, 2, 3)))
        assert f"xorbits.numpy.linalg.eig will fallback to NumPy" == str(w[0].message)
        valid_func(xnp_output, np_output)

    with pytest.warns(Warning) as w:
        xnp_output = xnp.linalg.eigh(np.array([[1, -2j], [2j, 5]])).execute().fetch()
        np_output = np.linalg.eigh(np.array([[1, -2j], [2j, 5]]))
        assert f"xorbits.numpy.linalg.eigh will fallback to NumPy" == str(w[0].message)
        valid_func(xnp_output, np_output)

    with pytest.warns(Warning) as w:
        xnp_output = xnp.linalg.eigvals(np.diag((-1, 1))).execute().fetch()
        np_output = np.linalg.eigvals(np.diag((-1, 1)))
        assert f"xorbits.numpy.linalg.eigvals will fallback to NumPy" == str(
            w[0].message
        )
        valid_func(xnp_output, np_output)

    with pytest.warns(Warning) as w:
        xnp_output = (
            xnp.linalg.eigvalsh(np.array([[5 + 2j, 9 - 2j], [0 + 2j, 2 - 1j]]))
            .execute()
            .fetch()
        )
        np_output = np.linalg.eigvalsh(np.array([[5 + 2j, 9 - 2j], [0 + 2j, 2 - 1j]]))
        assert f"xorbits.numpy.linalg.eigvalsh will fallback to NumPy" == str(
            w[0].message
        )
        valid_func(xnp_output, np_output)

    with pytest.warns(Warning) as w:
        xnp_output = (
            xnp.linalg.multi_dot(
                [
                    np.random.random((10000, 100)),
                    np.random.random((100, 1000)),
                    np.random.random((1000, 5)),
                    np.random.random((5, 333)),
                ]
            )
            .execute()
            .fetch()
        )
        np_output = np.linalg.multi_dot(
            [
                np.random.random((10000, 100)),
                np.random.random((100, 1000)),
                np.random.random((1000, 5)),
                np.random.random((5, 333)),
            ]
        )
        assert f"xorbits.numpy.linalg.multi_dot will fallback to NumPy" == str(
            w[0].message
        )
        valid_func(xnp_output, np_output)

    # TODO: Actor caller has created too many clients
    # with pytest.warns(Warning) as w:
    #     xnp_output = xnp.linalg.matrix_power(np.array([[0, 1], [-1, 0]]), 3).execute().fetch()
    #     np_output = np.linalg.matrix_power(np.array([[0, 1], [-1, 0]]), 3)
    #     assert f"xorbits.numpy.linalg.matrix_power will fallback to NumPy" == str(
    #         w[0].message
    #     )
    #     valid_func(xnp_output, np_output)

    # TODO: Actor caller has created too many clients
    #  with pytest.warns(Warning) as w:
    #     xnp_output = xnp.linalg.matrix_rank(np.eye(4)).execute().fetch()
    #     np_output = np.linalg.matrix_rank(np.eye(4))
    #     assert f"xorbits.numpy.linalg.matrix_rank will fallback to NumPy" == str(
    #         w[0].message
    #     )
    #     valid_func(xnp_output, np_output)

    with pytest.warns(Warning) as w:
        xnp_output = (
            xnp.linalg.lstsq(
                np.vstack(
                    [np.array([0, 1, 2, 3]), np.ones(len(np.array([0, 1, 2, 3])))]
                ).T,
                np.array([-1, 0.2, 0.9, 2.1]),
            )
            .execute()
            .fetch()
        )
        np_output = np.linalg.lstsq(
            np.vstack([np.array([0, 1, 2, 3]), np.ones(len(np.array([0, 1, 2, 3])))]).T,
            np.array([-1, 0.2, 0.9, 2.1]),
        )
        assert f"xorbits.numpy.linalg.lstsq will fallback to NumPy" == str(w[0].message)
        valid_func(xnp_output, np_output)

    with pytest.warns(Warning) as w:
        xnp_output = (
            xnp.linalg.slogdet(
                np.array([[[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]]])
            )
            .execute()
            .fetch()
        )
        np_output = np.linalg.slogdet(
            np.array([[[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]]])
        )
        assert f"xorbits.numpy.linalg.slogdet will fallback to NumPy" == str(
            w[0].message
        )
        valid_func(xnp_output, np_output)

    with pytest.warns(Warning) as w:
        xnp_output = xnp.linalg.pinv(np.random.randn(9, 6)).execute().fetch()
        np_output = np.linalg.pinv(np.random.randn(9, 6))
        assert f"xorbits.numpy.linalg.pinv will fallback to NumPy" == str(w[0].message)
        valid_func(xnp_output, np_output)


def test_random_fallback(setup):
    with pytest.warns(Warning) as w:
        xnp_output = xnp.random.default_rng().execute().fetch()
        np_output = np.random.default_rng()
        assert f"xorbits.numpy.random.default_rng will fallback to NumPy" == str(
            w[0].message
        )
        valid_func(xnp_output, np_output)

    with pytest.warns(Warning) as w:
        xnp_output = xnp.random.PCG64().execute().fetch()
        np_output = np.random.PCG64()
        assert f"xorbits.numpy.random.PCG64 will fallback to NumPy" == str(w[0].message)
        valid_func(xnp_output, np_output)

    with pytest.warns(Warning) as w:
        xnp_output = xnp.random.MT19937().execute().fetch()
        np_output = np.random.MT19937()
        assert f"xorbits.numpy.random.MT19937 will fallback to NumPy" == str(
            w[0].message
        )
        valid_func(xnp_output, np_output)

    with pytest.warns(Warning) as w:
        xnp_output = xnp.random.Generator(np.random.PCG64()).execute().fetch()
        np_output = np.random.Generator(np.random.PCG64())
        assert f"xorbits.numpy.random.Generator will fallback to NumPy" == str(
            w[0].message
        )
        valid_func(xnp_output, np_output)


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


def test_busday_offset(setup):
    with pytest.warns(Warning) as w:
        xnp_output = xnp.busday_offset("2011-10", 0, roll="forward").execute().fetch()
        np_output = np.busday_offset("2011-10", 0, roll="forward")

        assert f"xorbits.numpy.busday_offset will fallback to NumPy" == str(
            w[0].message
        )

        assert np.equal(xnp_output.all(), np_output.all())


def test_is_busday(setup):
    with pytest.warns(Warning) as w:
        xnp_output = (
            xnp.is_busday(
                ["2011-07-01", "2011-07-02", "2011-07-18"],
                holidays=["2011-07-01", "2011-07-04", "2011-07-17"],
            )
            .execute()
            .fetch()
        )
        np_output = np.is_busday(
            ["2011-07-01", "2011-07-02", "2011-07-18"],
            holidays=["2011-07-01", "2011-07-04", "2011-07-17"],
        )

        assert f"xorbits.numpy.is_busday will fallback to NumPy" == str(w[0].message)

        assert np.equal(xnp_output.all(), np_output.all())


def test_docstring():
    docstring = xnp.trace.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from numpy."
    )

    docstring = xnp.linalg.det.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from numpy.linalg."
    )

    docstring = xnp.random.default_rng.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from numpy.random."
    )

    docstring = xnp.ndarray.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from numpy."
    )

    docstring = xnp.ndarray.tolist.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from numpy.ndarray."
    )


def test_tensor_tolist(setup):
    data = np.random.rand(15, 25)
    tensor = xnp.array(data)
    assert data.tolist() == tensor.tolist()

    expected = pd.unique(pd.Series([i for i in range(100)])).tolist()
    result = xpd.unique(xpd.Series([i for i in range(100)])).tolist()
    assert expected == result

    data = np.array([1, 2, 3, 4])
    tensor = xnp.array([1, 2, 3, 4])
    assert data.tolist() == tensor.tolist()

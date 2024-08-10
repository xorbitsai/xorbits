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

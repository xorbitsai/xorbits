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

import numpy

from .... import numpy as xnp
from ..index_tricks import CClass, MGridClass, OGridClass, RClass


def test_ogrid(setup):
    numpy.testing.assert_array_equal(numpy.ogrid[-1:1:5j], xnp.ogrid[-1:1:5j].to_numpy())


def test_ogrid_docstring():
    assert isinstance(xnp.ogrid, OGridClass)

    # class docstring.
    docstring = OGridClass.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from numpy."
    )

    # object docstring.
    docstring = xnp.ogrid.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from numpy."
    )


def test_mgrid(setup):
    numpy.testing.assert_array_equal(numpy.mgrid[0:5, 0:5], xnp.mgrid[0:5, 0:5].to_numpy())


def test_mgrid_docstring():
    assert isinstance(xnp.mgrid, MGridClass)

    # class docstring.
    docstring = MGridClass.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from numpy."
    )

    # object docstring.
    docstring = xnp.mgrid.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from numpy."
    )


def test_c_(setup):
    numpy.testing.assert_array_equal(
        numpy.c_[numpy.array([1, 2, 3]), numpy.array([4, 5, 6])],
        xnp.c_[xnp.array([1, 2, 3]), xnp.array([4, 5, 6])].to_numpy(),
    )


def test_c__docstring():
    assert isinstance(xnp.c_, CClass)

    # class docstring.
    docstring = CClass.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from numpy."
    )

    # object docstring.
    docstring = xnp.c_.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from numpy."
    )


def test_r_(setup):
    numpy.testing.assert_array_equal(
        numpy.r_["0,2,0", [1, 2, 3], [4, 5, 6]],
        xnp.r_["0,2,0", [1, 2, 3], [4, 5, 6]].to_numpy(),
    )


def test_r__docstring():
    assert isinstance(xnp.r_, RClass)

    # class docstring.
    docstring = RClass.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from numpy."
    )

    # object docstring.
    docstring = xnp.r_.__doc__
    assert docstring is not None and docstring.endswith(
        "This docstring was copied from numpy."
    )

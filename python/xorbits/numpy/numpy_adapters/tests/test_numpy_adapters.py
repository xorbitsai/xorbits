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
from ..core import wrap_numpy_module_method


@pytest.mark.parametrize(
    "xnp_cls,np_cls,func",
    [
        (xnp.random, np.random, "default_rng"),
        (xnp.random, np.random, "PCG64"),
        (xnp.random, np.random, "MT19937"),
    ],
)
def test_zero_param_funcs(xnp_cls, np_cls, func):
    with pytest.warns(Warning) as w:
        xnp_func = getattr(xnp_cls, func)
        np_func = getattr(np_cls, func)

        res = xnp_func()
        assert f"xorbits.numpy.{func} will fallback to NumPy" == str(w[0].message)

        xnp_output = res.execute().fetch()
        np_output = np_func()
        wrapped_func = wrap_numpy_module_method(np_cls, func)
        wrapped_func_output = wrapped_func().fetch()

        assert type(wrapped_func_output) == type(np_output)
        assert type(xnp_output) == type(np_output)

        if isinstance(np_output, np.ndarray):
            assert isinstance(xnp_output, np.ndarray)
            assert np.equal(xnp_output, np_output)
            assert isinstance(wrapped_func_output, np.ndarray)
            assert np.equal(wrapped_func_output, np_output)
        if isinstance(np_output, object):
            assert isinstance(xnp_output, object)
            assert dir(xnp_output) == dir(np_output)
            assert isinstance(wrapped_func_output, object)
            assert dir(wrapped_func_output) == dir(np_output)


@pytest.mark.parametrize(
    "xnp_cls,np_cls,func",
    [
        (xnp.random, np.random, "Generator"),
    ],
)
def test_one_param_funcs(xnp_cls, np_cls, func):
    with pytest.warns(Warning) as w:
        xnp_func = getattr(xnp_cls, func)
        np_func = getattr(np_cls, func)

        res = xnp_func(np.random.PCG64())
        assert f"xorbits.numpy.{func} will fallback to NumPy" == str(w[0].message)

        xnp_output = res.execute().fetch()
        np_output = np_func(np.random.PCG64())
        wrapped_func = wrap_numpy_module_method(np_cls, func)
        wrapped_func_output = wrapped_func(np.random.PCG64()).fetch()

        assert type(wrapped_func_output) == type(np_output)
        assert type(xnp_output) == type(np_output)

        if isinstance(np_output, np.ndarray):
            assert isinstance(xnp_output, np.ndarray)
            assert np.equal(xnp_output, np_output)
            assert isinstance(wrapped_func_output, np.ndarray)
            assert np.equal(wrapped_func_output, np_output)
        if isinstance(np_output, object):
            assert isinstance(xnp_output, object)
            assert dir(xnp_output) == dir(np_output)
            assert isinstance(wrapped_func_output, object)
            assert dir(wrapped_func_output) == dir(np_output)

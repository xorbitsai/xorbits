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

import functools
from typing import Callable

from ...core.adapter import from_mars, mars_remote, to_mars


def _wrap_remote_func(c: Callable):
    """
    Making the inputs and outputs of the wrapped callable be xorbits types.
    """

    @functools.wraps(c)
    def wrapped(*args, **kwargs):
        return to_mars(c(*from_mars(args), **from_mars(kwargs)))

    return wrapped


def spawn(
    func,
    args=(),
    kwargs=None,
    retry_when_fail=False,
    n_output=None,
    output_types=None,
    **kw,
):
    """
    Spawn a function and return a Mars Object which can be executed later.

    Parameters
    ----------
    func : function
        Function to spawn.
    args: tuple
       Args to pass to function
    kwargs: dict
       Kwargs to pass to function
    retry_when_fail: bool, default False
       If True, retry when function failed.
    n_output: int
       Count of outputs for the function
    output_types: str or list, default "object"
        Specify type of returned objects.

    Returns
    -------
    Object
        Mars Object.

    Examples
    --------
    >>> import xorbits.remote as xr
    >>> def inc(x):
    ...     return x + 1
    >>>
    >>> result = xr.spawn(inc, args=(0,))
    >>> result.to_object()
    1


    >>> results = [xr.spawn(inc, args=(i,)) for i in range(10)]
    >>> [r.to_object() for r in results]
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    Xorbits objects returned by :meth:`xorbits.remote.spawn` can be used as arguments when spawning
     another function.

    >>> def sum_all(xs):
    ...     return sum(xs)
    >>> xr.spawn(sum_all, args=(results,)).to_object()
    55

    functions can be spawned recursively.

    >>> def driver():
    ...     results = [xr.spawn(inc, args=(i,)) for i in range(10)]
    ...     return [r.to_object() for r in results]
    >>> xr.spawn(driver).to_object()
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    Xorbits objects can also be used in spawned functions.

    >>> import xorbits.numpy as np
    >>> def driver2():
    ...     t = np.ones((10, 10))
    ...     return t.sum().to_numpy()
    >>> xr.spawn(driver2).to_object()
    100.0

    The argument `n_output` indicates that the function to spawn has multiple outputs.

    >>> def triage(alist):
    ...     ret = [], []
    ...     for i in alist:
    ...         if i < 0.5:
    ...             ret[0].append(i)
    ...         else:
    ...             ret[1].append(i)
    ...     return ret
    >>>
    >>> def sum_all(xs):
    ...     return sum(xs)
    >>>
    >>> l = [0.4, 0.7, 0.2, 0.8]
    >>> la, lb = xr.spawn(triage, args=(l,), n_output=2)
    >>>
    >>> sa = xr.spawn(sum_all, args=(la,))
    >>> sb = xr.spawn(sum_all, args=(lb,))
    >>> [sa.to_object(), sb.to_object()]
    [0.6000000000000001, 1.5]
    """

    func = _wrap_remote_func(func)
    args = to_mars(args)
    kwargs = to_mars(kwargs)

    ret = mars_remote.spawn(
        func,
        args,
        kwargs,
        retry_when_fail=retry_when_fail,
        n_output=n_output,
        output_types=output_types,
        **kw,
    )

    return from_mars(ret)

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

from .flip import flip


def fliplr(m):
    """
    Flip tensor in the left/right direction.

    Flip the entries in each row in the left/right direction.
    Columns are preserved, but appear in a different order than before.

    Parameters
    ----------
    m : array_like
        Input tensor, must be at least 2-D.

    Returns
    -------
    f : Tensor
        A view of `m` with the columns reversed.  Since a view
        is returned, this operation is :math:`\\mathcal O(1)`.

    See Also
    --------
    flipud : Flip array in the up/down direction.
    rot90 : Rotate array counterclockwise.

    Notes
    -----
    Equivalent to m[:,::-1]. Requires the tensor to be at least 2-D.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> A = mt.diag([1.,2.,3.])
    >>> A.execute()
    array([[ 1.,  0.,  0.],
           [ 0.,  2.,  0.],
           [ 0.,  0.,  3.]])
    >>> mt.fliplr(A).execute()
    array([[ 0.,  0.,  1.],
           [ 0.,  2.,  0.],
           [ 3.,  0.,  0.]])

    >>> A = mt.random.randn(2,3,5)
    >>> mt.all(mt.fliplr(A) == A[:,::-1,...]).execute()
    True

    """
    return flip(m, 1)

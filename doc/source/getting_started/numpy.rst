.. _10min_numpy:

===================================
10 minutes to :code:`xorbits.numpy`
===================================

.. currentmodule:: xorbits.numpy

This is a short introduction to :code:`xorbits.numpy` which is originated from NumPy's quickstart.

Array Creation
--------------

Popular mechanisms for creating arrays in NumPy are supported in :code:`xorbits.numpy`.

For example, you can create an array from a regular Python list or tuple using the ``array``
function. The type of the resulting array is deduced from the type of the elements in the sequences.

::

    >>> import xorbits.numpy as np
    >>> a = np.array([2, 3, 4])
    >>> a
    array([2, 3, 4])
    >>> a.dtype
    dtype('int64')
    >>> b = np.array([1.2, 3.5, 5.1])
    >>> b.dtype
    dtype('float64')

In addition, creating an array from existing NumPy ndarray is supported.

::

    >>> import numpy as np
    >>> import xorbits.numpy as xnp
    >>> a = np.arange(6)
    >>> b = xnp.array(a)
    >>> print(b)
    array([0 1 2 3 4 5])

A frequent error consists in calling ``array`` with multiple arguments,
rather than providing a single sequence as an argument.

::

    >>> a = np.array(1, 2, 3, 4)    # WRONG
    Traceback (most recent call last):
      ...
    TypeError: array() takes from 1 to 2 positional arguments but 4 were given
    >>> a = np.array([1, 2, 3, 4])  # RIGHT

``array`` transforms sequences of sequences into two-dimensional arrays, sequences of sequences
of sequences into three-dimensional arrays, and so on.

::

    >>> b = np.array([(1.5, 2, 3), (4, 5, 6)])
    >>> b
    array([[1.5, 2. , 3. ],
           [4. , 5. , 6. ]])

The type of the array can also be explicitly specified at creation time:

::

    >>> c = np.array([[1, 2], [3, 4]], dtype=complex)
    >>> c
    array([[1.+0.j, 2.+0.j],
           [3.+0.j, 4.+0.j]])

Often, the elements of an array are originally unknown, but its size is known. Hence, several
functions are offered to create arrays with initial placeholder content. These minimize the
necessity of growing arrays, an expensive operation.

The function ``zeros`` creates an array full of zeros, the function ``ones`` creates an array full
of ones, and the function ``empty`` creates an array whose initial content is random and depends on the
state of the memory. By default, the dtype of the created array is ``float64``, but it can be
specified via the key word argument ``dtype``.

::

    >>> np.zeros((3, 4))
    array([[0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.]])
    >>> np.ones((2, 3, 4), dtype=np.int16)
    array([[[1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1]],
    <BLANKLINE>
           [[1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1]]], dtype=int16)
    >>> np.empty((2, 3)) #doctest: +SKIP
    array([[3.73603959e-262, 6.02658058e-154, 6.55490914e-260],  # may vary
           [5.30498948e-313, 3.14673309e-307, 1.00000000e+000]])

To create sequences of numbers, use ``arange`` function which is analogous to the Python built-in
``range``, but returns an array.

::

    >>> np.arange(10, 30, 5)
    array([10, 15, 20, 25])
    >>> np.arange(0, 2, 0.3)  # it accepts float arguments
    array([0. , 0.3, 0.6, 0.9, 1.2, 1.5, 1.8])

When ``arange`` is used with floating point arguments, it is generally not possible to predict the
number of elements obtained, due to the finite floating point precision. For this reason, it is
usually better to use the function ``linspace`` that receives as an argument the number of
elements that we want, instead of the step::

    >>> from xorbits.numpy import pi
    >>> np.linspace(0, 2, 9)                   # 9 numbers from 0 to 2
    array([0.  , 0.25, 0.5 , 0.75, 1.  , 1.25, 1.5 , 1.75, 2.  ])
    >>> x = np.linspace(0, 2 * pi, 100)        # useful to evaluate function at lots of points
    >>> f = np.sin(x)


However, the way of loading and saving arrays is quite different. Please see :ref:`io <routines.io>` for
detailed info. Here's an example of creating and loading an HDF5 file::

    >>> import h5py                            # if you don't have h5py installed, run "pip install h5py" first
    >>> arr = np.random.randn(1000)
    >>> with h5py.File('t.hdf5', 'w') as f:
    >>>     dset = f.create_dataset("t", data=arr)
    >>>
    >>> a = np.from_hdf5("t.hdf5", dataset="t")

Once an ndarray is created, use ``to_numpy`` to convert it to a local NumPy ndarray::

    >>> a = np.array([1, 2, 3, 4])
    >>> a.to_numpy()
    array([1, 2, 3, 4])


Printing Arrays
---------------

Xorbits displays an array in a similar way to NumPy:

-  the last axis is printed from left to right,
-  the second-to-last is printed from top to bottom,
-  the rest are also printed from top to bottom, with each slice
   separated from the next by an empty line.

One-dimensional arrays are then printed as rows, bidimensionals as matrices and tridimensionals
as lists of matrices.

::

    >>> a = np.arange(6)                    # 1d array
    >>> print(a)
    [0 1 2 3 4 5]
    >>>
    >>> b = np.arange(12).reshape(4, 3)     # 2d array
    >>> print(b)
    [[ 0  1  2]
     [ 3  4  5]
     [ 6  7  8]
     [ 9 10 11]]
    >>>
    >>> c = np.arange(24).reshape(2, 3, 4)  # 3d array
    >>> print(c)
    [[[ 0  1  2  3]
      [ 4  5  6  7]
      [ 8  9 10 11]]
    <BLANKLINE>
     [[12 13 14 15]
      [16 17 18 19]
      [20 21 22 23]]]

If an array is too large to be printed, the central part of the array will be automatically skipped
and only prints the corners::

    >>> print(np.arange(10000))
    [   0    1    2 ... 9997 9998 9999]
    >>>
    >>> print(np.arange(10000).reshape(100, 100))
    [[   0    1    2 ...   97   98   99]
     [ 100  101  102 ...  197  198  199]
     [ 200  201  202 ...  297  298  299]
     ...
     [9700 9701 9702 ... 9797 9798 9799]
     [9800 9801 9802 ... 9897 9898 9899]
     [9900 9901 9902 ... 9997 9998 9999]]


.. _quickstart.basic-operations:

Basic Operations
----------------

Arithmetic operators on arrays apply *elementwise*. A new array is created and filled with the
result.

::

    >>> a = np.array([20, 30, 40, 50])
    >>> b = np.arange(4)
    >>> b
    array([0, 1, 2, 3])
    >>> c = a - b
    >>> c
    array([20, 29, 38, 47])
    >>> b**2
    array([0, 1, 4, 9])
    >>> 10 * np.sin(a)
    array([ 9.12945251, -9.88031624,  7.4511316 , -2.62374854])
    >>> a < 35
    array([ True,  True, False, False])

Unlike in many matrix languages, the product operator ``*`` operates elementwise in
:code:`xorbits.numpy` arrays. The matrix product can be performed using the ``@`` operator (in
python >=3.5) or the ``dot`` function or method::

    >>> A = np.array([[1, 1],
    ...               [0, 1]])
    >>> B = np.array([[2, 0],
    ...               [3, 4]])
    >>> A * B     # elementwise product
    array([[2, 0],
           [0, 4]])
    >>> A @ B     # matrix product
    array([[5, 4],
           [3, 4]])
    >>> A.dot(B)  # another matrix product
    array([[5, 4],
           [3, 4]])

Some operations, such as ``+=`` and ``*=``, act in place to modify an existing array rather than
create a new one.

::

    >>> a = np.ones((2, 3), dtype=int)
    >>> b = np.random.random((2, 3))
    >>> a *= 3
    >>> a
    array([[3, 3, 3],
           [3, 3, 3]])
    >>> b += a
    >>> b
    array([[3.51182162, 3.9504637 , 3.14415961],
           [3.94864945, 3.31183145, 3.42332645]])

When operating with arrays of different types, the type of the resulting array corresponds to the
more general or precise one (a behavior known as upcasting).

::

    >>> a = np.ones(3, dtype=np.int32)
    >>> b = np.linspace(0, np.pi, 3)
    >>> b.dtype.name
    'float64'
    >>> c = a + b
    >>> c
    array([1.        , 2.57079633, 4.14159265])
    >>> c.dtype.name
    'float64'
    >>> d = np.exp(c * 1j)
    >>> d
    array([ 0.54030231+0.84147098j, -0.84147098+0.54030231j,
           -0.54030231-0.84147098j])
    >>> d.dtype.name
    'complex128'

Many unary operations, such as computing the sum of all the elements in the array, are implemented
as methods of the ``ndarray`` class.

::

    >>> a = np.random.random((2, 3))
    >>> a
    array([[0.82770259, 0.40919914, 0.54959369],
           [0.02755911, 0.75351311, 0.53814331]])
    >>> a.sum()
    3.1057109529998157
    >>> a.min()
    0.027559113243068367
    >>> a.max()
    0.8277025938204418

By default, these operations apply to the array as though it were a list of numbers, regardless of
its shape. However, by specifying the ``axis`` parameter you can apply an operation along the
specified axis of an array::

    >>> b = np.arange(12).reshape(3, 4)
    >>> b
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11]])
    >>>
    >>> b.sum(axis=0)     # sum of each column
    array([12, 15, 18, 21])
    >>>
    >>> b.min(axis=1)     # min of each row
    array([0, 4, 8])
    >>>
    >>> b.cumsum(axis=1)  # cumulative sum along each row
    array([[ 0,  1,  3,  6],
           [ 4,  9, 15, 22],
           [ 8, 17, 27, 38]])


Universal Functions
-------------------

Mathematical functions such as sin, cos, and exp are provided. These are called "universal
functions" (\ ``ufunc``). These functions operate elementwise on an array, producing an array as
output.

::

    >>> B = np.arange(3)
    >>> B
    array([0, 1, 2])
    >>> np.exp(B)
    array([1.        , 2.71828183, 7.3890561 ])
    >>> np.sqrt(B)
    array([0.        , 1.        , 1.41421356])
    >>> C = np.array([2., -1., 4.])
    >>> np.add(B, C)
    array([2., 0., 6.])

.. _quickstart.indexing-slicing-and-iterating:

Indexing, Slicing and Iterating
-------------------------------

**One-dimensional** arrays can be indexed, sliced and iterated over,
much like
`lists <https://docs.python.org/tutorial/introduction.html#lists>`__
and other Python sequences.

::

    >>> a = np.arange(10)**3
    >>> a
    array([  0,   1,   8,  27,  64, 125, 216, 343, 512, 729])
    >>> a[2]
    8
    >>> a[2:5]
    array([ 8, 27, 64])
    >>> # equivalent to a[0:6:2] = 1000;
    >>> # from start to position 6, exclusive, set every 2nd element to 1000
    >>> a[:6:2] = 1000
    >>> a
    array([1000,    1, 1000,   27, 1000,  125,  216,  343,  512,  729])
    >>> a[::-1]  # reversed a
    array([ 729,  512,  343,  216,  125, 1000,   27, 1000,    1, 1000])
    >>> for i in a:
    ...     print(i**(1 / 3.))
    ...
    9.999999999999998
    1.0
    9.999999999999998
    3.0
    9.999999999999998
    4.999999999999999
    5.999999999999999
    6.999999999999999
    7.999999999999999
    8.999999999999998


**Multidimensional** arrays can have one index per axis. These indices
are given in a tuple separated by commas::

    >>> def f(x, y):
    ...     return 10 * x + y
    ...
    >>> b = np.arange(12).reshape(3, 4)
    >>> b
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11]])
    >>> b[2, 3]
    11
    >>> b[0:3, 1]  # each row in the second column of b
    array([1, 5, 9])
    >>> b[:, 1]    # equivalent to the previous example
    array([1, 5, 9])
    >>> b[1:3, :]  # each column in the second and third row of b
    array([[ 4,  5,  6,  7],
           [ 8,  9, 10, 11]])

When fewer indices are provided than the number of axes, the missing
indices are considered complete slices\ ``:``

::

    >>> b[-1]   # the last row. Equivalent to b[-1, :]
    array([ 8,  9, 10, 11])

The expression within brackets in ``b[i]`` is treated as an ``i``
followed by as many instances of ``:`` as needed to represent the
remaining axes. You can also write this using dots as
``b[i, ...]``.

The **dots** (``...``) represent as many colons as needed to produce a
complete indexing tuple. For example, if ``x`` is an array with 5
axes, then

-  ``x[1, 2, ...]`` is equivalent to ``x[1, 2, :, :, :]``,
-  ``x[..., 3]`` to ``x[:, :, :, :, 3]`` and
-  ``x[4, ..., 5, :]`` to ``x[4, :, :, 5, :]``.

::

    >>> c = np.array([[[  0,  1,  2],  # a 3D array (two stacked 2D arrays)
    ...                [ 10, 12, 13]],
    ...               [[100, 101, 102],
    ...                [110, 112, 113]]])
    >>> c.shape
    (2, 2, 3)
    >>> c[1, ...]  # same as c[1, :, :] or c[1]
    array([[100, 101, 102],
           [110, 112, 113]])
    >>> c[..., 2]  # same as c[:, :, 2]
    array([[  2,  13],
           [102, 113]])

**Iterating** over multidimensional arrays is done with respect to the
first axis::

    >>> for row in b:
    ...     print(row)
    ...
    [0 1 2 3]
    [4 5 6 7]
    [ 8  9 10 11]

However, if one wants to perform an operation on each element in the
array, one can use the ``flat`` attribute which is an
`iterator <https://docs.python.org/tutorial/classes.html#iterators>`__
over all the elements of the array::

    >>> for element in b.flat:
    ...     print(element)
    ...
    0
    1
    2
    3
    4
    5
    6
    7
    8
    9
    10
    11

.. _quickstart.shape-manipulation:

Changing the shape of an array
------------------------------

An array has a shape given by the number of elements along each axis::

    >>> a = np.floor(10 * np.random.random((3, 4)))
    >>> a
    array([[6., 9., 5., 2.],
           [4., 9., 8., 9.],
           [4., 7., 2., 8.]])
    >>> a.shape
    (3, 4)

The shape of an array can be changed with various commands. Note that the following three commands
all return a modified array, but do not change the original array::

    >>> a.ravel()  # returns the array, flattened
    array([6., 9., 5., 2., 4., 9., 8., 9., 4., 7., 2., 8.])
    >>> a.reshape(6, 2)  # returns the array with a modified shape
    array([[6., 9.],
           [5., 2.],
           [4., 9.],
           [8., 9.],
           [4., 7.],
           [2., 8.]])
    >>> a.T  # returns the array, transposed
    array([[6., 4., 4.],
           [9., 9., 7.],
           [5., 8., 2.],
           [2., 9., 8.]])
    >>> a.T.shape
    (4, 3)
    >>> a.shape
    (3, 4)

The order of the elements in the array resulting from ``ravel`` is normally "C-style", that is, the
rightmost index "changes the fastest", so the element after ``a[0, 0]`` is ``a[0, 1]``. If the
array is reshaped to some other shape, again the array is treated as "C-style". Normally arrays
are created stored in this order, so ``ravel`` will usually not need to copy its argument, but if
the array was made by taking slices of another array or created with unusual options, it may need
to be copied. The functions ``ravel`` and ``reshape`` can also be instructed, using an optional
argument, to use FORTRAN-style arrays, in which the leftmost index changes the fastest.

If a dimension is given as ``-1`` in a reshaping operation, the other dimensions are automatically
calculated::

    >>> a.reshape(3, -1)
    array([[6., 9., 5., 2.],
           [4., 9., 8., 9.],
           [4., 7., 2., 8.]])

.. _quickstart.stacking-arrays:

Stacking together different arrays
----------------------------------

Several arrays can be stacked together along different axes::

    >>> a = np.floor(10 * np.random.random((2, 2)))
    >>> a
    array([[9., 7.],
           [5., 2.]])
    >>> b = np.floor(10 * np.random.random((2, 2)))
    >>> b
    array([[1., 9.],
           [5., 1.]])
    >>> np.vstack((a, b))
    array([[9., 7.],
           [5., 2.],
           [1., 9.],
           [5., 1.]])
    >>> np.hstack((a, b))
    array([[9., 7., 1., 9.],
           [5., 2., 5., 1.]])

The function ``column_stack`` stacks 1D arrays as columns into a 2D array. It is equivalent to
``hstack`` only for 2D arrays::

    >>> from xorbits.numpy import newaxis
    >>> np.column_stack((a, b))  # with 2D arrays
    array([[9., 7., 1., 9.],
           [5., 2., 5., 1.]])
    >>> a = np.array([4., 2.])
    >>> b = np.array([3., 8.])
    >>> np.column_stack((a, b))  # returns a 2D array
    array([[4., 3.],
           [2., 8.]])
    >>> np.hstack((a, b))        # the result is different
    array([4., 2., 3., 8.])
    >>> a[:, newaxis]  # view `a` as a 2D column vector
    array([[4.],
           [2.]])
    >>> np.column_stack((a[:, newaxis], b[:, newaxis]))
    array([[4., 3.],
           [2., 8.]])
    >>> np.hstack((a[:, newaxis], b[:, newaxis]))  # the result is the same
    array([[4., 3.],
           [2., 8.]])

On the other hand, the function ``row_stack`` is equivalent to ``vstack`` for any input arrays. In
fact, ``row_stack`` is an alias for ``vstack``::

    >>> np.column_stack is np.hstack
    False
    >>> np.row_stack is np.vstack
    True

In general, for arrays with more than two dimensions, ``hstack`` stacks along their second axes,
``vstack`` stacks along their first axes, and ``concatenate`` allows for an optional arguments giving
the number of the axis along which the concatenation should happen.

.. note::

    In complex cases, ``r_`` and ``c_`` are useful for creating arrays by stacking numbers along one axis.
    They allow the use of range literals ``:``. ::

           >>> np.r_[1:4, 0, 4]
           array([1, 2, 3, 0, 4])

    When used with arrays as arguments, ``r_`` and ``c_`` are similar to ``vstack`` and ``hstack`` in their
    default behavior, but allow for an optional argument giving the number of the axis along which to
    concatenate.

Splitting one array into several smaller ones
---------------------------------------------

Using ``hsplit``, you can split an
array along its horizontal axis, either by specifying the number of
equally shaped arrays to return, or by specifying the columns after
which the division should occur::

    >>> a = np.floor(10 * np.random.random((2, 12)))
    >>> a
    array([[6., 7., 6., 9., 0., 5., 4., 0., 6., 8., 5., 2.],
           [8., 5., 5., 7., 1., 8., 6., 7., 1., 8., 1., 0.]])
    >>> # Split `a` into 3
    >>> np.hsplit(a, 3)
    [array([[6., 7., 6., 9.],
           [8., 5., 5., 7.]]), array([[0., 5., 4., 0.],
           [1., 8., 6., 7.]]), array([[6., 8., 5., 2.],
           [1., 8., 1., 0.]])]
    >>> # Split `a` after the third and the fourth column
    >>> np.hsplit(a, (3, 4))
    [array([[6., 7., 6.],
           [8., 5., 5.]]), array([[9.],
           [7.]]), array([[0., 5., 4., 0., 6., 8., 5., 2.],
           [1., 8., 6., 7., 1., 8., 1., 0.]])]

``vsplit`` splits along the vertical
axis, and ``array_split`` allows
one to specify along which axis to split.

.. _routines.linalg:

Linear algebra (:mod:`xorbits.numpy.linalg`)
============================================

The following table lists both implemented and not implemented methods. If you have need
of an operation that is listed as not implemented, feel free to open an issue on the
`GitHub repository`_, or give a thumbs up to already created issues. Contributions are
also welcome!

The following table is structured as follows: The first column contains the method name.
The second column contains link to a description of corresponding numpy method.
The third column is a flag for whether or not there is an implementation in Xorbits
for the method in the left column. ``Y`` stands for yes, ``N`` stands for no, ``P`` standsfor partial 
(meaning some parameters may not be supported yet), and ``D`` stands for default to numpy.

The NumPy linear algebra functions rely on BLAS and LAPACK to provide efficient
low level implementations of standard linear algebra algorithms. Those
libraries may be provided by NumPy itself using C versions of a subset of their
reference implementations but, when possible, highly optimized libraries that
take advantage of specialized processor functionality are preferred. Examples
of such libraries are OpenBLAS_, MKL (TM), and ATLAS. Because those libraries
are multithreaded and processor dependent, environmental variables and external
packages such as threadpoolctl_ may be needed to control the number of threads
or specify the processor architecture.

.. _OpenBLAS: https://www.openblas.net/
.. _threadpoolctl: https://github.com/joblib/threadpoolctl

The SciPy library also contains a `~scipy.linalg` submodule, and there is
overlap in the functionality provided by the SciPy and NumPy submodules.  SciPy
contains functions not found in `xorbits.numpy.linalg`, such as functions related to
LU decomposition and the Schur decomposition, multiple ways of calculating the
pseudoinverse, and matrix transcendentals such as the matrix logarithm.  Some
functions that exist in both have augmented functionality in `scipy.linalg`.
For example, `scipy.linalg.eig` can take a second matrix argument for solving
generalized eigenvalue problems.  Some functions in NumPy, however, have more
flexible broadcasting options.  For example, `xorbits.numpy.linalg.solve` can handle
"stacked" arrays, while `scipy.linalg.solve` accepts only a single square
array as its first argument.

.. note::

   The term *matrix* as it is used on this page indicates a 2d `xorbits.numpy.array`
   object, and *not* a `xorbits.numpy.matrix` object. The latter is no longer
   recommended, even for linear algebra. See
   :ref:`the matrix object documentation<matrix-objects>` for
   more information.

The ``@`` operator
------------------

Introduced in NumPy 1.10.0, the ``@`` operator is preferable to
other methods when computing the matrix product between 2d arrays. The
:func:`xorbits.numpy.matmul` function implements the ``@`` operator.

Matrix and vector products
--------------------------

+-------------------------+------------------------+------------------------+----------------------------------+
| ``xorbits.numpy``       | ``numpy``              | Implemented? (Y/N/P/D) | Notes for Current implementation |
+-------------------------+------------------------+------------------------+----------------------------------+
| ``dot``                 | `dot`_                 | Y                      |                                  |
+-------------------------+------------------------+------------------------+----------------------------------+
| ``linalg.multi_dot``    | `linalg.multi_dot`_    | Y                      |                                  |
+-------------------------+------------------------+------------------------+----------------------------------+
| ``vdot``                | `vdot`_                | Y                      |                                  |
+-------------------------+------------------------+------------------------+----------------------------------+
| ``inner``               | `inner`_               | Y                      |                                  |
+-------------------------+------------------------+------------------------+----------------------------------+
| ``outer``               | `outer`_               | Y                      |                                  |
+-------------------------+------------------------+------------------------+----------------------------------+
| ``matmul``              | `matmul`_              | Y                      |                                  |
+-------------------------+------------------------+------------------------+----------------------------------+
| ``tensordot``           | `tensordot`_           | Y                      |                                  |
+-------------------------+------------------------+------------------------+----------------------------------+
| ``einsum``              | `einsum`_              | Y                      |                                  |
+-------------------------+------------------------+------------------------+----------------------------------+
| ``einsum_path``         | `einsum_path`_         | Y                      |                                  |
+-------------------------+------------------------+------------------------+----------------------------------+
| ``linalg.matrix_power`` | `linalg.matrix_power`_ | Y                      |                                  |
+-------------------------+------------------------+------------------------+----------------------------------+
| ``kron``                | `kron`_                | Y                      |                                  |
+-------------------------+------------------------+------------------------+----------------------------------+

Decompositions
--------------

+---------------------+--------------------+------------------------+----------------------------------+
| ``xorbits.numpy``   | ``numpy``          | Implemented? (Y/N/P/D) | Notes for Current implementation |
+---------------------+--------------------+------------------------+----------------------------------+
| ``linalg.cholesky`` | `linalg.cholesky`_ | Y                      |                                  |
+---------------------+--------------------+------------------------+----------------------------------+
| ``linalg.qr``       | `linalg.qr`_       | Y                      |                                  |
+---------------------+--------------------+------------------------+----------------------------------+
| ``linalg.svd``      | `linalg.svd`_      | Y                      |                                  |
+---------------------+--------------------+------------------------+----------------------------------+

Matrix eigenvalues
------------------

+---------------------+--------------------+------------------------+----------------------------------+
| ``xorbits.numpy``   | ``numpy``          | Implemented? (Y/N/P/D) | Notes for Current implementation |
+---------------------+--------------------+------------------------+----------------------------------+
| ``linalg.eig``      | `linalg.eig`_      | Y                      |                                  |
+---------------------+--------------------+------------------------+----------------------------------+
| ``linalg.eigh``     | `linalg.eigh`_     | Y                      |                                  |
+---------------------+--------------------+------------------------+----------------------------------+
| ``linalg.eigvals``  | `linalg.eigvals`_  | Y                      |                                  |
+---------------------+--------------------+------------------------+----------------------------------+
| ``linalg.eigvalsh`` | `linalg.eigvalsh`_ | Y                      |                                  |
+---------------------+--------------------+------------------------+----------------------------------+

Norms and other numbers
-----------------------

+------------------------+-----------------------+------------------------+----------------------------------+
| ``xorbits.numpy``      | ``numpy``             | Implemented? (Y/N/P/D) | Notes for Current implementation |
+------------------------+-----------------------+------------------------+----------------------------------+
| ``linalg.norm``        | `linalg.norm`_        | Y                      |                                  |
+------------------------+-----------------------+------------------------+----------------------------------+
| ``linalg.cond``        | `linalg.cond`_        | Y                      |                                  |
+------------------------+-----------------------+------------------------+----------------------------------+
| ``linalg.det``         | `linalg.det`_         | Y                      |                                  |
+------------------------+-----------------------+------------------------+----------------------------------+
| ``linalg.matrix_rank`` | `linalg.matrix_rank`_ | Y                      |                                  |
+------------------------+-----------------------+------------------------+----------------------------------+
| ``linalg.slogdet``     | `linalg.slogdet`_     | Y                      |                                  |
+------------------------+-----------------------+------------------------+----------------------------------+
| ``trace``              | `trace`_              | Y                      |                                  |
+------------------------+-----------------------+------------------------+----------------------------------+

Solving equations and inverting matrices
----------------------------------------

+------------------------+-----------------------+------------------------+----------------------------------+
| ``xorbits.numpy``      | ``numpy``             | Implemented? (Y/N/P/D) | Notes for Current implementation |
+------------------------+-----------------------+------------------------+----------------------------------+
| ``linalg.solve``       | `linalg.solve`_       | Y                      |                                  |
+------------------------+-----------------------+------------------------+----------------------------------+
| ``linalg.tensorsolve`` | `linalg.tensorsolve`_ | Y                      |                                  |
+------------------------+-----------------------+------------------------+----------------------------------+
| ``linalg.lstsq``       | `linalg.lstsq`_       | Y                      |                                  |
+------------------------+-----------------------+------------------------+----------------------------------+
| ``linalg.inv``         | `linalg.inv`_         | Y                      |                                  |
+------------------------+-----------------------+------------------------+----------------------------------+
| ``linalg.pinv``        | `linalg.pinv`_        | Y                      |                                  |
+------------------------+-----------------------+------------------------+----------------------------------+
| ``linalg.tensorinv``   | `linalg.tensorinv`_   | Y                      |                                  |
+------------------------+-----------------------+------------------------+----------------------------------+

Exceptions
----------

+------------------------+-----------------------+------------------------+----------------------------------+
| ``xorbits.numpy``      | ``numpy``             | Implemented? (Y/N/P/D) | Notes for Current implementation |
+------------------------+-----------------------+------------------------+----------------------------------+
| ``linalg.LinAlgError`` | `linalg.LinAlgError`_ | Y                      |                                  |
+------------------------+-----------------------+------------------------+----------------------------------+

.. _routines.linalg-broadcasting:

Linear algebra on several matrices at once
------------------------------------------

.. versionadded:: 1.8.0

Several of the linear algebra routines listed above are able to
compute results for several matrices at once, if they are stacked into
the same array.

This is indicated in the documentation via input parameter
specifications such as ``a : (..., M, M) array_like``. This means that
if for instance given an input array ``a.shape == (N, M, M)``, it is
interpreted as a "stack" of N matrices, each of size M-by-M. Similar
specification applies to return values, for instance the determinant
has ``det : (...)`` and will in this case return an array of shape
``det(a).shape == (N,)``. This generalizes to linear algebra
operations on higher-dimensional arrays: the last 1 or 2 dimensions of
a multidimensional array are interpreted as vectors or matrices, as
appropriate for each operation.

.. _`GitHub repository`: https://github.com/xorbitsai/xorbits/issues
.. _`dot`: https://numpy.org/doc/stable/reference/generated/numpy.dot.html
.. _`linalg.multi_dot`: https://numpy.org/doc/stable/reference/generated/numpy.linalg.multi_dot.html
.. _`vdot`: https://numpy.org/doc/stable/reference/generated/numpy.vdot.html
.. _`inner`: https://numpy.org/doc/stable/reference/generated/numpy.inner.html
.. _`outer`: https://numpy.org/doc/stable/reference/generated/numpy.outer.html
.. _`matmul`: https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
.. _`tensordot`: https://numpy.org/doc/stable/reference/generated/numpy.tensordot.html
.. _`einsum`: https://numpy.org/doc/stable/reference/generated/numpy.einsum.html
.. _`einsum_path`: https://numpy.org/doc/stable/reference/generated/numpy.einsum_path.html
.. _`linalg.matrix_power`: https://numpy.org/doc/stable/reference/generated/numpy.linalg.matrix_power.html
.. _`kron`: https://numpy.org/doc/stable/reference/generated/numpy.kron.html
.. _`linalg.cholesky`: https://numpy.org/doc/stable/reference/generated/numpy.linalg.cholesky.html
.. _`linalg.qr`: https://numpy.org/doc/stable/reference/generated/numpy.linalg.qr.html
.. _`linalg.svd`: https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html
.. _`linalg.eig`: https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html
.. _`linalg.eigh`: https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html
.. _`linalg.eigvals`: https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigvals.html
.. _`linalg.eigvalsh`: https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigvalsh.html
.. _`linalg.norm`: https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html
.. _`linalg.cond`: https://numpy.org/doc/stable/reference/generated/numpy.linalg.cond.html
.. _`linalg.det`: https://numpy.org/doc/stable/reference/generated/numpy.linalg.det.html
.. _`linalg.matrix_rank`: https://numpy.org/doc/stable/reference/generated/numpy.linalg.matrix_rank.html
.. _`linalg.slogdet`: https://numpy.org/doc/stable/reference/generated/numpy.linalg.slogdet.html
.. _`trace`: https://numpy.org/doc/stable/reference/generated/numpy.trace.html
.. _`linalg.solve`: https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html
.. _`linalg.tensorsolve`: https://numpy.org/doc/stable/reference/generated/numpy.linalg.tensorsolve.html
.. _`linalg.lstsq`: https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html
.. _`linalg.inv`: https://numpy.org/doc/stable/reference/generated/numpy.linalg.inv.html
.. _`linalg.pinv`: https://numpy.org/doc/stable/reference/generated/numpy.linalg.pinv.html
.. _`linalg.tensorinv`: https://numpy.org/doc/stable/reference/generated/numpy.linalg.tensorinv.html
.. _`linalg.LinAlgError`: https://numpy.org/doc/stable/reference/generated/numpy.linalg.LinAlgError.html

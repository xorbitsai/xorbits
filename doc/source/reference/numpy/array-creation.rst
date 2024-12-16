.. _routines.array-creation:

Array creation routines
=======================

The following table lists both implemented and not implemented methods. If you have need
of an operation that is listed as not implemented, feel free to open an issue on the
`GitHub repository`_, or give a thumbs up to already created issues. Contributions are
also welcome!

The following table is structured as follows: The first column contains the method name.
The second column contains link to a description of corresponding numpy method.
The third column is a flag for whether or not there is an implementation in Xorbits
for the method in the left column. ``Y`` stands for yes, ``N`` stands for no, ``P`` standsfor partial 
(meaning some parameters may not be supported yet), and ``D`` stands for default to numpy.

From shape or value
-------------------

+-------------------+---------------+------------------------+----------------------------------+
| ``xorbits.numpy`` | ``numpy``     | Implemented? (Y/N/P/D) | Notes for Current implementation |
+-------------------+---------------+------------------------+----------------------------------+
| ``empty``         | `empty`_      | Y                      |                                  |
+-------------------+---------------+------------------------+----------------------------------+
| ``empty_like``    | `empty_like`_ | Y                      |                                  |
+-------------------+---------------+------------------------+----------------------------------+
| ``eye``           | `eye`_        | Y                      |                                  |
+-------------------+---------------+------------------------+----------------------------------+
| ``identity``      | `identity`_   | Y                      |                                  |
+-------------------+---------------+------------------------+----------------------------------+
| ``ones``          | `ones`_       | Y                      |                                  |
+-------------------+---------------+------------------------+----------------------------------+
| ``ones_like``     | `ones_like`_  | Y                      |                                  |
+-------------------+---------------+------------------------+----------------------------------+
| ``zeros``         | `zeros`_      | Y                      |                                  |
+-------------------+---------------+------------------------+----------------------------------+
| ``zeros_like``    | `zeros_like`_ | Y                      |                                  |
+-------------------+---------------+------------------------+----------------------------------+
| ``full``          | `full`_       | Y                      |                                  |
+-------------------+---------------+------------------------+----------------------------------+
| ``full_like``     | `full_like`_  | Y                      |                                  |
+-------------------+---------------+------------------------+----------------------------------+

From existing data
------------------

+-----------------------+----------------------+------------------------+----------------------------------+
| ``xorbits.numpy``     | ``numpy``            | Implemented? (Y/N/P/D) | Notes for Current implementation |
+-----------------------+----------------------+------------------------+----------------------------------+
| ``array``             | `array`_             | Y                      |                                  |
+-----------------------+----------------------+------------------------+----------------------------------+
| ``asarray``           | `asarray`_           | Y                      |                                  |
+-----------------------+----------------------+------------------------+----------------------------------+
| ``asanyarray``        | `asanyarray`_        | Y                      |                                  |
+-----------------------+----------------------+------------------------+----------------------------------+
| ``ascontiguousarray`` | `ascontiguousarray`_ | Y                      |                                  |
+-----------------------+----------------------+------------------------+----------------------------------+
| ``asmatrix``          | `asmatrix`_          | Y                      |                                  |
+-----------------------+----------------------+------------------------+----------------------------------+
| ``copy``              | `copy`_              | Y                      |                                  |
+-----------------------+----------------------+------------------------+----------------------------------+
| ``frombuffer``        | `frombuffer`_        | Y                      |                                  |
+-----------------------+----------------------+------------------------+----------------------------------+
| ``from_dlpack``       | `from_dlpack`_       | Y                      |                                  |
+-----------------------+----------------------+------------------------+----------------------------------+
| ``fromfile``          | `fromfile`_          | Y                      |                                  |
+-----------------------+----------------------+------------------------+----------------------------------+
| ``fromfunction``      | `fromfunction`_      | Y                      |                                  |
+-----------------------+----------------------+------------------------+----------------------------------+
| ``fromiter``          | `fromiter`_          | Y                      |                                  |
+-----------------------+----------------------+------------------------+----------------------------------+
| ``fromstring``        | `fromstring`_        | Y                      |                                  |
+-----------------------+----------------------+------------------------+----------------------------------+
| ``loadtxt``           | `loadtxt`_           | Y                      |                                  |
+-----------------------+----------------------+------------------------+----------------------------------+

Numerical ranges
----------------

+-------------------+--------------+------------------------+----------------------------------+
| ``xorbits.numpy`` | ``numpy``    | Implemented? (Y/N/P/D) | Notes for Current implementation |
+-------------------+--------------+------------------------+----------------------------------+
| ``arange``        | `arange`_    | Y                      |                                  |
+-------------------+--------------+------------------------+----------------------------------+
| ``linspace``      | `linspace`_  | Y                      |                                  |
+-------------------+--------------+------------------------+----------------------------------+
| ``logspace``      | `logspace`_  | Y                      |                                  |
+-------------------+--------------+------------------------+----------------------------------+
| ``geomspace``     | `geomspace`_ | Y                      |                                  |
+-------------------+--------------+------------------------+----------------------------------+
| ``meshgrid``      | `meshgrid`_  | Y                      |                                  |
+-------------------+--------------+------------------------+----------------------------------+
| ``mgrid``         | `mgrid`_     | Y                      |                                  |
+-------------------+--------------+------------------------+----------------------------------+
| ``ogrid``         | `ogrid`_     | Y                      |                                  |
+-------------------+--------------+------------------------+----------------------------------+

Building matrices
-----------------

+-------------------+-------------+------------------------+----------------------------------+
| ``xorbits.numpy`` | ``numpy``   | Implemented? (Y/N/P/D) | Notes for Current implementation |
+-------------------+-------------+------------------------+----------------------------------+
| ``diag``          | `diag`_     | Y                      |                                  |
+-------------------+-------------+------------------------+----------------------------------+
| ``diagflat``      | `diagflat`_ | Y                      |                                  |
+-------------------+-------------+------------------------+----------------------------------+
| ``tri``           | `tri`_      | Y                      |                                  |
+-------------------+-------------+------------------------+----------------------------------+
| ``tril``          | `tril`_     | Y                      |                                  |
+-------------------+-------------+------------------------+----------------------------------+
| ``triu``          | `triu`_     | Y                      |                                  |
+-------------------+-------------+------------------------+----------------------------------+
| ``vander``        | `vander`_   | Y                      |                                  |
+-------------------+-------------+------------------------+----------------------------------+

The Matrix class
----------------

+-------------------+-----------+------------------------+----------------------------------+
| ``xorbits.numpy`` | ``numpy`` | Implemented? (Y/N/P/D) | Notes for Current implementation |
+-------------------+-----------+------------------------+----------------------------------+
| ``matrix``        | `matrix`_ | Y                      |                                  |
+-------------------+-----------+------------------------+----------------------------------+
| ``bmat``          | `bmat`_   | Y                      |                                  |
+-------------------+-----------+------------------------+----------------------------------+

.. _`GitHub repository`: https://github.com/xorbitsai/xorbits/issues
.. _`empty`: https://numpy.org/doc/stable/reference/generated/numpy.empty.html
.. _`empty_like`: https://numpy.org/doc/stable/reference/generated/numpy.empty_like.html
.. _`eye`: https://numpy.org/doc/stable/reference/generated/numpy.eye.html
.. _`identity`: https://numpy.org/doc/stable/reference/generated/numpy.identity.html
.. _`ones`: https://numpy.org/doc/stable/reference/generated/numpy.ones.html
.. _`ones_like`: https://numpy.org/doc/stable/reference/generated/numpy.ones_like.html
.. _`zeros`: https://numpy.org/doc/stable/reference/generated/numpy.zeros.html
.. _`zeros_like`: https://numpy.org/doc/stable/reference/generated/numpy.zeros_like.html
.. _`full`: https://numpy.org/doc/stable/reference/generated/numpy.full.html
.. _`full_like`: https://numpy.org/doc/stable/reference/generated/numpy.full_like.html
.. _`array`: https://numpy.org/doc/stable/reference/generated/numpy.array.html
.. _`asarray`: https://numpy.org/doc/stable/reference/generated/numpy.asarray.html
.. _`asanyarray`: https://numpy.org/doc/stable/reference/generated/numpy.asanyarray.html
.. _`ascontiguousarray`: https://numpy.org/doc/stable/reference/generated/numpy.ascontiguousarray.html
.. _`asmatrix`: https://numpy.org/doc/stable/reference/generated/numpy.asmatrix.html
.. _`copy`: https://numpy.org/doc/stable/reference/generated/numpy.copy.html
.. _`frombuffer`: https://numpy.org/doc/stable/reference/generated/numpy.frombuffer.html
.. _`from_dlpack`: https://numpy.org/doc/stable/reference/generated/numpy.from_dlpack.html
.. _`fromfile`: https://numpy.org/doc/stable/reference/generated/numpy.fromfile.html
.. _`fromfunction`: https://numpy.org/doc/stable/reference/generated/numpy.fromfunction.html
.. _`fromiter`: https://numpy.org/doc/stable/reference/generated/numpy.fromiter.html
.. _`fromstring`: https://numpy.org/doc/stable/reference/generated/numpy.fromstring.html
.. _`loadtxt`: https://numpy.org/doc/stable/reference/generated/numpy.loadtxt.html
.. _`arange`: https://numpy.org/doc/stable/reference/generated/numpy.arange.html
.. _`linspace`: https://numpy.org/doc/stable/reference/generated/numpy.linspace.html
.. _`logspace`: https://numpy.org/doc/stable/reference/generated/numpy.logspace.html
.. _`geomspace`: https://numpy.org/doc/stable/reference/generated/numpy.geomspace.html
.. _`meshgrid`: https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html
.. _`mgrid`: https://numpy.org/doc/stable/reference/generated/numpy.mgrid.html
.. _`ogrid`: https://numpy.org/doc/stable/reference/generated/numpy.ogrid.html
.. _`diag`: https://numpy.org/doc/stable/reference/generated/numpy.diag.html
.. _`diagflat`: https://numpy.org/doc/stable/reference/generated/numpy.diagflat.html
.. _`tri`: https://numpy.org/doc/stable/reference/generated/numpy.tri.html
.. _`tril`: https://numpy.org/doc/stable/reference/generated/numpy.tril.html
.. _`triu`: https://numpy.org/doc/stable/reference/generated/numpy.triu.html
.. _`vander`: https://numpy.org/doc/stable/reference/generated/numpy.vander.html
.. _`matrix`: https://numpy.org/doc/stable/reference/generated/numpy.matrix.html
.. _`bmat`: https://numpy.org/doc/stable/reference/generated/numpy.bmat.html

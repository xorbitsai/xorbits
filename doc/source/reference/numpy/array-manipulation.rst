Array manipulation routines
***************************

The following table lists both implemented and not implemented methods. If you have need
of an operation that is listed as not implemented, feel free to open an issue on the
`GitHub repository`_, or give a thumbs up to already created issues. Contributions are
also welcome!

The following table is structured as follows: The first column contains the method name.
The second column contains link to a description of corresponding numpy method.
The third column is a flag for whether or not there is an implementation in Xorbits
for the method in the left column. ``Y`` stands for yes, ``N`` stands for no, ``P`` standsfor partial 
(meaning some parameters may not be supported yet), and ``D`` stands for default to numpy.

Basic operations
================

+-------------------+-----------+------------------------+----------------------------------+
| ``xorbits.numpy`` | ``numpy`` | Implemented? (Y/N/P/D) | Notes for Current implementation |
+-------------------+-----------+------------------------+----------------------------------+
| ``copyto``        | `copyto`_ | Y                      |                                  |
+-------------------+-----------+------------------------+----------------------------------+
| ``shape``         | `shape`_  | Y                      |                                  |
+-------------------+-----------+------------------------+----------------------------------+

Changing array shape
====================

+---------------------+--------------------+------------------------+----------------------------------+
| ``xorbits.numpy``   | ``numpy``          | Implemented? (Y/N/P/D) | Notes for Current implementation |
+---------------------+--------------------+------------------------+----------------------------------+
| ``reshape``         | `reshape`_         | Y                      |                                  |
+---------------------+--------------------+------------------------+----------------------------------+
| ``ravel``           | `ravel`_           | Y                      |                                  |
+---------------------+--------------------+------------------------+----------------------------------+
| ``ndarray.flat``    | `ndarray.flat`_    | Y                      |                                  |
+---------------------+--------------------+------------------------+----------------------------------+
| ``ndarray.flatten`` | `ndarray.flatten`_ | Y                      |                                  |
+---------------------+--------------------+------------------------+----------------------------------+

Transpose-like operations
=========================

+-------------------+--------------+------------------------+----------------------------------+
| ``xorbits.numpy`` | ``numpy``    | Implemented? (Y/N/P/D) | Notes for Current implementation |
+-------------------+--------------+------------------------+----------------------------------+
| ``moveaxis``      | `moveaxis`_  | Y                      |                                  |
+-------------------+--------------+------------------------+----------------------------------+
| ``rollaxis``      | `rollaxis`_  | Y                      |                                  |
+-------------------+--------------+------------------------+----------------------------------+
| ``swapaxes``      | `swapaxes`_  | Y                      |                                  |
+-------------------+--------------+------------------------+----------------------------------+
| ``ndarray.T``     | `ndarray.T`_ | Y                      |                                  |
+-------------------+--------------+------------------------+----------------------------------+
| ``transpose``     | `transpose`_ | Y                      |                                  |
+-------------------+--------------+------------------------+----------------------------------+

Changing number of dimensions
=============================

+----------------------+---------------------+------------------------+----------------------------------+
| ``xorbits.numpy``    | ``numpy``           | Implemented? (Y/N/P/D) | Notes for Current implementation |
+----------------------+---------------------+------------------------+----------------------------------+
| ``atleast_1d``       | `atleast_1d`_       | Y                      |                                  |
+----------------------+---------------------+------------------------+----------------------------------+
| ``atleast_2d``       | `atleast_2d`_       | Y                      |                                  |
+----------------------+---------------------+------------------------+----------------------------------+
| ``atleast_3d``       | `atleast_3d`_       | Y                      |                                  |
+----------------------+---------------------+------------------------+----------------------------------+
| ``broadcast``        | `broadcast`_        | Y                      |                                  |
+----------------------+---------------------+------------------------+----------------------------------+
| ``broadcast_to``     | `broadcast_to`_     | Y                      |                                  |
+----------------------+---------------------+------------------------+----------------------------------+
| ``broadcast_arrays`` | `broadcast_arrays`_ | Y                      |                                  |
+----------------------+---------------------+------------------------+----------------------------------+
| ``expand_dims``      | `expand_dims`_      | Y                      |                                  |
+----------------------+---------------------+------------------------+----------------------------------+
| ``squeeze``          | `squeeze`_          | Y                      |                                  |
+----------------------+---------------------+------------------------+----------------------------------+

Changing kind of array
======================

+-----------------------+----------------------+------------------------+----------------------------------+
| ``xorbits.numpy``     | ``numpy``            | Implemented? (Y/N/P/D) | Notes for Current implementation |
+-----------------------+----------------------+------------------------+----------------------------------+
| ``asarray``           | `asarray`_           | Y                      |                                  |
+-----------------------+----------------------+------------------------+----------------------------------+
| ``asanyarray``        | `asanyarray`_        | Y                      |                                  |
+-----------------------+----------------------+------------------------+----------------------------------+
| ``asmatrix``          | `asmatrix`_          | Y                      |                                  |
+-----------------------+----------------------+------------------------+----------------------------------+
| ``asfarray``          | `asfarray`_          | Y                      |                                  |
+-----------------------+----------------------+------------------------+----------------------------------+
| ``asfortranarray``    | `asfortranarray`_    | Y                      |                                  |
+-----------------------+----------------------+------------------------+----------------------------------+
| ``ascontiguousarray`` | `ascontiguousarray`_ | Y                      |                                  |
+-----------------------+----------------------+------------------------+----------------------------------+
| ``asarray_chkfinite`` | `asarray_chkfinite`_ | Y                      |                                  |
+-----------------------+----------------------+------------------------+----------------------------------+
| ``require``           | `require`_           | Y                      |                                  |
+-----------------------+----------------------+------------------------+----------------------------------+

Joining arrays
==============

+-------------------+-----------------+------------------------+----------------------------------+
| ``xorbits.numpy`` | ``numpy``       | Implemented? (Y/N/P/D) | Notes for Current implementation |
+-------------------+-----------------+------------------------+----------------------------------+
| ``concatenate``   | `concatenate`_  | Y                      |                                  |
+-------------------+-----------------+------------------------+----------------------------------+
| ``stack``         | `stack`_        | Y                      |                                  |
+-------------------+-----------------+------------------------+----------------------------------+
| ``block``         | `block`_        | Y                      |                                  |
+-------------------+-----------------+------------------------+----------------------------------+
| ``vstack``        | `vstack`_       | Y                      |                                  |
+-------------------+-----------------+------------------------+----------------------------------+
| ``hstack``        | `hstack`_       | Y                      |                                  |
+-------------------+-----------------+------------------------+----------------------------------+
| ``dstack``        | `dstack`_       | Y                      |                                  |
+-------------------+-----------------+------------------------+----------------------------------+
| ``column_stack``  | `column_stack`_ | Y                      |                                  |
+-------------------+-----------------+------------------------+----------------------------------+
| ``row_stack``     | `row_stack`_    | Y                      |                                  |
+-------------------+-----------------+------------------------+----------------------------------+

Splitting arrays
================

+-------------------+----------------+------------------------+----------------------------------+
| ``xorbits.numpy`` | ``numpy``      | Implemented? (Y/N/P/D) | Notes for Current implementation |
+-------------------+----------------+------------------------+----------------------------------+
| ``split``         | `split`_       | Y                      |                                  |
+-------------------+----------------+------------------------+----------------------------------+
| ``array_split``   | `array_split`_ | Y                      |                                  |
+-------------------+----------------+------------------------+----------------------------------+
| ``dsplit``        | `dsplit`_      | Y                      |                                  |
+-------------------+----------------+------------------------+----------------------------------+
| ``hsplit``        | `hsplit`_      | Y                      |                                  |
+-------------------+----------------+------------------------+----------------------------------+
| ``vsplit``        | `vsplit`_      | Y                      |                                  |
+-------------------+----------------+------------------------+----------------------------------+

Tiling arrays
=============

+-------------------+-----------+------------------------+----------------------------------+
| ``xorbits.numpy`` | ``numpy`` | Implemented? (Y/N/P/D) | Notes for Current implementation |
+-------------------+-----------+------------------------+----------------------------------+
| ``tile``          | `tile`_   | Y                      |                                  |
+-------------------+-----------+------------------------+----------------------------------+
| ``repeat``        | `repeat`_ | Y                      |                                  |
+-------------------+-----------+------------------------+----------------------------------+

Adding and removing elements
============================

+-------------------+---------------+------------------------+----------------------------------+
| ``xorbits.numpy`` | ``numpy``     | Implemented? (Y/N/P/D) | Notes for Current implementation |
+-------------------+---------------+------------------------+----------------------------------+
| ``delete``        | `delete`_     | Y                      |                                  |
+-------------------+---------------+------------------------+----------------------------------+
| ``insert``        | `insert`_     | Y                      |                                  |
+-------------------+---------------+------------------------+----------------------------------+
| ``append``        | `append`_     | Y                      |                                  |
+-------------------+---------------+------------------------+----------------------------------+
| ``resize``        | `resize`_     | Y                      |                                  |
+-------------------+---------------+------------------------+----------------------------------+
| ``trim_zeros``    | `trim_zeros`_ | Y                      |                                  |
+-------------------+---------------+------------------------+----------------------------------+
| ``unique``        | `unique`_     | Y                      |                                  |
+-------------------+---------------+------------------------+----------------------------------+

Rearranging elements
====================

+-------------------+------------+------------------------+----------------------------------+
| ``xorbits.numpy`` | ``numpy``  | Implemented? (Y/N/P/D) | Notes for Current implementation |
+-------------------+------------+------------------------+----------------------------------+
| ``flip``          | `flip`_    | Y                      |                                  |
+-------------------+------------+------------------------+----------------------------------+
| ``fliplr``        | `fliplr`_  | Y                      |                                  |
+-------------------+------------+------------------------+----------------------------------+
| ``flipud``        | `flipud`_  | Y                      |                                  |
+-------------------+------------+------------------------+----------------------------------+
| ``reshape``       | `reshape`_ | Y                      |                                  |
+-------------------+------------+------------------------+----------------------------------+
| ``roll``          | `roll`_    | Y                      |                                  |
+-------------------+------------+------------------------+----------------------------------+
| ``rot90``         | `rot90`_   | Y                      |                                  |
+-------------------+------------+------------------------+----------------------------------+

.. _`GitHub repository`: https://github.com/xorbitsai/xorbits/issues
.. _`copyto`: https://numpy.org/doc/stable/reference/generated/numpy.copyto.html
.. _`shape`: https://numpy.org/doc/stable/reference/generated/numpy.shape.html
.. _`reshape`: https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
.. _`ravel`: https://numpy.org/doc/stable/reference/generated/numpy.ravel.html
.. _`ndarray.flat`: https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flat.html
.. _`ndarray.flatten`: https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flatten.html
.. _`moveaxis`: https://numpy.org/doc/stable/reference/generated/numpy.moveaxis.html
.. _`rollaxis`: https://numpy.org/doc/stable/reference/generated/numpy.rollaxis.html
.. _`swapaxes`: https://numpy.org/doc/stable/reference/generated/numpy.swapaxes.html
.. _`ndarray.T`: https://numpy.org/doc/stable/reference/generated/numpy.ndarray.T.html
.. _`transpose`: https://numpy.org/doc/stable/reference/generated/numpy.transpose.html
.. _`atleast_1d`: https://numpy.org/doc/stable/reference/generated/numpy.atleast_1d.html
.. _`atleast_2d`: https://numpy.org/doc/stable/reference/generated/numpy.atleast_2d.html
.. _`atleast_3d`: https://numpy.org/doc/stable/reference/generated/numpy.atleast_3d.html
.. _`broadcast`: https://numpy.org/doc/stable/reference/generated/numpy.broadcast.html
.. _`broadcast_to`: https://numpy.org/doc/stable/reference/generated/numpy.broadcast_to.html
.. _`broadcast_arrays`: https://numpy.org/doc/stable/reference/generated/numpy.broadcast_arrays.html
.. _`expand_dims`: https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html
.. _`squeeze`: https://numpy.org/doc/stable/reference/generated/numpy.squeeze.html
.. _`asarray`: https://numpy.org/doc/stable/reference/generated/numpy.asarray.html
.. _`asanyarray`: https://numpy.org/doc/stable/reference/generated/numpy.asanyarray.html
.. _`asmatrix`: https://numpy.org/doc/stable/reference/generated/numpy.asmatrix.html
.. _`asfarray`: https://numpy.org/doc/stable/reference/generated/numpy.asfarray.html
.. _`asfortranarray`: https://numpy.org/doc/stable/reference/generated/numpy.asfortranarray.html
.. _`ascontiguousarray`: https://numpy.org/doc/stable/reference/generated/numpy.ascontiguousarray.html
.. _`asarray_chkfinite`: https://numpy.org/doc/stable/reference/generated/numpy.asarray_chkfinite.html
.. _`require`: https://numpy.org/doc/stable/reference/generated/numpy.require.html
.. _`concatenate`: https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html
.. _`stack`: https://numpy.org/doc/stable/reference/generated/numpy.stack.html
.. _`block`: https://numpy.org/doc/stable/reference/generated/numpy.block.html
.. _`vstack`: https://numpy.org/doc/stable/reference/generated/numpy.vstack.html
.. _`hstack`: https://numpy.org/doc/stable/reference/generated/numpy.hstack.html
.. _`dstack`: https://numpy.org/doc/stable/reference/generated/numpy.dstack.html
.. _`column_stack`: https://numpy.org/doc/stable/reference/generated/numpy.column_stack.html
.. _`row_stack`: https://numpy.org/doc/stable/reference/generated/numpy.row_stack.html
.. _`split`: https://numpy.org/doc/stable/reference/generated/numpy.split.html
.. _`array_split`: https://numpy.org/doc/stable/reference/generated/numpy.array_split.html
.. _`dsplit`: https://numpy.org/doc/stable/reference/generated/numpy.dsplit.html
.. _`hsplit`: https://numpy.org/doc/stable/reference/generated/numpy.hsplit.html
.. _`vsplit`: https://numpy.org/doc/stable/reference/generated/numpy.vsplit.html
.. _`tile`: https://numpy.org/doc/stable/reference/generated/numpy.tile.html
.. _`repeat`: https://numpy.org/doc/stable/reference/generated/numpy.repeat.html
.. _`delete`: https://numpy.org/doc/stable/reference/generated/numpy.delete.html
.. _`insert`: https://numpy.org/doc/stable/reference/generated/numpy.insert.html
.. _`append`: https://numpy.org/doc/stable/reference/generated/numpy.append.html
.. _`resize`: https://numpy.org/doc/stable/reference/generated/numpy.resize.html
.. _`trim_zeros`: https://numpy.org/doc/stable/reference/generated/numpy.trim_zeros.html
.. _`unique`: https://numpy.org/doc/stable/reference/generated/numpy.unique.html
.. _`flip`: https://numpy.org/doc/stable/reference/generated/numpy.flip.html
.. _`fliplr`: https://numpy.org/doc/stable/reference/generated/numpy.fliplr.html
.. _`flipud`: https://numpy.org/doc/stable/reference/generated/numpy.flipud.html
.. _`reshape`: https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
.. _`roll`: https://numpy.org/doc/stable/reference/generated/numpy.roll.html
.. _`rot90`: https://numpy.org/doc/stable/reference/generated/numpy.rot90.html

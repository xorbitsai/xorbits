.. _routines.io:

Input and output
================

The following table lists both implemented and not implemented methods. If you have need
of an operation that is listed as not implemented, feel free to open an issue on the
`GitHub repository`_, or give a thumbs up to already created issues. Contributions are
also welcome!

The following table is structured as follows: The first column contains the method name.
The second column contains link to a description of corresponding numpy method.
The third column is a flag for whether or not there is an implementation in Xorbits
for the method in the left column. ``Y`` stands for yes, ``N`` stands for no, ``P`` standsfor partial 
(meaning some parameters may not be supported yet), and ``D`` stands for default to numpy.

NumPy binary files (NPY, NPZ)
-----------------------------

+----------------------+---------------------+------------------------+----------------------------------+
| ``xorbits.numpy``    | ``numpy``           | Implemented? (Y/N/P/D) | Notes for Current implementation |
+----------------------+---------------------+------------------------+----------------------------------+
| ``load``             | `load`_             | Y                      |                                  |
+----------------------+---------------------+------------------------+----------------------------------+
| ``save``             | `save`_             | Y                      |                                  |
+----------------------+---------------------+------------------------+----------------------------------+
| ``savez``            | `savez`_            | Y                      |                                  |
+----------------------+---------------------+------------------------+----------------------------------+
| ``savez_compressed`` | `savez_compressed`_ | Y                      |                                  |
+----------------------+---------------------+------------------------+----------------------------------+

The format of these binary file types is documented in
:py:mod:`xorbits.numpy.lib.format`

Text files
----------

+--------------------+-------------------+------------------------+----------------------------------+
| ``xorbits.numpy``  | ``numpy``         | Implemented? (Y/N/P/D) | Notes for Current implementation |
+--------------------+-------------------+------------------------+----------------------------------+
| ``loadtxt``        | `loadtxt`_        | Y                      |                                  |
+--------------------+-------------------+------------------------+----------------------------------+
| ``savetxt``        | `savetxt`_        | Y                      |                                  |
+--------------------+-------------------+------------------------+----------------------------------+
| ``genfromtxt``     | `genfromtxt`_     | Y                      |                                  |
+--------------------+-------------------+------------------------+----------------------------------+
| ``fromregex``      | `fromregex`_      | Y                      |                                  |
+--------------------+-------------------+------------------------+----------------------------------+
| ``fromstring``     | `fromstring`_     | Y                      |                                  |
+--------------------+-------------------+------------------------+----------------------------------+
| ``ndarray.tofile`` | `ndarray.tofile`_ | Y                      |                                  |
+--------------------+-------------------+------------------------+----------------------------------+
| ``ndarray.tolist`` | `ndarray.tolist`_ | Y                      |                                  |
+--------------------+-------------------+------------------------+----------------------------------+

Raw binary files
----------------

+--------------------+-------------------+------------------------+----------------------------------+
| ``xorbits.numpy``  | ``numpy``         | Implemented? (Y/N/P/D) | Notes for Current implementation |
+--------------------+-------------------+------------------------+----------------------------------+
| ``fromfile``       | `fromfile`_       | Y                      |                                  |
+--------------------+-------------------+------------------------+----------------------------------+
| ``ndarray.tofile`` | `ndarray.tofile`_ | Y                      |                                  |
+--------------------+-------------------+------------------------+----------------------------------+
| ``from_hdf5``      |                   | Y                      |                                  |
+--------------------+-------------------+------------------------+----------------------------------+
| ``to_hdf5``        |                   | Y                      |                                  |
+--------------------+-------------------+------------------------+----------------------------------+
| ``from_zarr``      |                   | Y                      |                                  |
+--------------------+-------------------+------------------------+----------------------------------+
| ``to_zarr``        |                   | Y                      |                                  |
+--------------------+-------------------+------------------------+----------------------------------+
| ``from_tiledb``    |                   | Y                      |                                  |
+--------------------+-------------------+------------------------+----------------------------------+
| ``to_tiledb``      |                   | Y                      |                                  |
+--------------------+-------------------+------------------------+----------------------------------+

String formatting
-----------------

+-----------------------------+----------------------------+------------------------+----------------------------------+
| ``xorbits.numpy``           | ``numpy``                  | Implemented? (Y/N/P/D) | Notes for Current implementation |
+-----------------------------+----------------------------+------------------------+----------------------------------+
| ``array2string``            | `array2string`_            | Y                      |                                  |
+-----------------------------+----------------------------+------------------------+----------------------------------+
| ``array_repr``              | `array_repr`_              | Y                      |                                  |
+-----------------------------+----------------------------+------------------------+----------------------------------+
| ``array_str``               | `array_str`_               | Y                      |                                  |
+-----------------------------+----------------------------+------------------------+----------------------------------+
| ``format_float_positional`` | `format_float_positional`_ | Y                      |                                  |
+-----------------------------+----------------------------+------------------------+----------------------------------+
| ``format_float_scientific`` | `format_float_scientific`_ | Y                      |                                  |
+-----------------------------+----------------------------+------------------------+----------------------------------+

Memory mapping files
--------------------

+----------------------------+---------------------------+------------------------+----------------------------------+
| ``xorbits.numpy``          | ``numpy``                 | Implemented? (Y/N/P/D) | Notes for Current implementation |
+----------------------------+---------------------------+------------------------+----------------------------------+
| ``memmap``                 | `memmap`_                 | Y                      |                                  |
+----------------------------+---------------------------+------------------------+----------------------------------+
| ``lib.format.open_memmap`` | `lib.format.open_memmap`_ | Y                      |                                  |
+----------------------------+---------------------------+------------------------+----------------------------------+

Text formatting options
-----------------------

+-------------------------+------------------------+------------------------+----------------------------------+
| ``xorbits.numpy``       | ``numpy``              | Implemented? (Y/N/P/D) | Notes for Current implementation |
+-------------------------+------------------------+------------------------+----------------------------------+
| ``set_printoptions``    | `set_printoptions`_    | Y                      |                                  |
+-------------------------+------------------------+------------------------+----------------------------------+
| ``get_printoptions``    | `get_printoptions`_    | Y                      |                                  |
+-------------------------+------------------------+------------------------+----------------------------------+
| ``set_string_function`` | `set_string_function`_ | Y                      |                                  |
+-------------------------+------------------------+------------------------+----------------------------------+
| ``printoptions``        | `printoptions`_        | Y                      |                                  |
+-------------------------+------------------------+------------------------+----------------------------------+

Base-n representations
----------------------

+-------------------+----------------+------------------------+----------------------------------+
| ``xorbits.numpy`` | ``numpy``      | Implemented? (Y/N/P/D) | Notes for Current implementation |
+-------------------+----------------+------------------------+----------------------------------+
| ``binary_repr``   | `binary_repr`_ | Y                      |                                  |
+-------------------+----------------+------------------------+----------------------------------+
| ``base_repr``     | `base_repr`_   | Y                      |                                  |
+-------------------+----------------+------------------------+----------------------------------+

Data sources
------------

+-------------------+---------------------+------------------------+------------------------------------------------------------------------------------------------------------+
| ``xorbits.numpy`` | ``numpy.lib.npyio`` | Implemented? (Y/N/P/D) | Notes for Current implementation                                                                           |
+-------------------+---------------------+------------------------+------------------------------------------------------------------------------------------------------------+
| ``DataSource``    | `DataSource`_       | Y                      | In NumPy 2.0, ``numpy.DataSource`` was moved to ``numpy.lib.npyio.DataSource`` for backward compatibility. |
+-------------------+---------------------+------------------------+------------------------------------------------------------------------------------------------------------+

Binary Format Description
-------------------------

+-------------------+---------------+------------------------+----------------------------------+
| ``xorbits.numpy`` | ``numpy``     | Implemented? (Y/N/P/D) | Notes for Current implementation |
+-------------------+---------------+------------------------+----------------------------------+
| ``lib.format``    | `lib.format`_ | Y                      |                                  |
+-------------------+---------------+------------------------+----------------------------------+

.. _`GitHub repository`: https://github.com/xorbitsai/xorbits/issues
.. _`load`: https://numpy.org/doc/stable/reference/generated/numpy.load.html
.. _`save`: https://numpy.org/doc/stable/reference/generated/numpy.save.html
.. _`savez`: https://numpy.org/doc/stable/reference/generated/numpy.savez.html
.. _`savez_compressed`: https://numpy.org/doc/stable/reference/generated/numpy.savez_compressed.html
.. _`loadtxt`: https://numpy.org/doc/stable/reference/generated/numpy.loadtxt.html
.. _`savetxt`: https://numpy.org/doc/stable/reference/generated/numpy.savetxt.html
.. _`genfromtxt`: https://numpy.org/doc/stable/reference/generated/numpy.genfromtxt.html
.. _`fromregex`: https://numpy.org/doc/stable/reference/generated/numpy.fromregex.html
.. _`fromstring`: https://numpy.org/doc/stable/reference/generated/numpy.fromstring.html
.. _`ndarray.tofile`: https://numpy.org/doc/stable/reference/generated/numpy.ndarray.tofile.html
.. _`ndarray.tolist`: https://numpy.org/doc/stable/reference/generated/numpy.ndarray.tolist.html
.. _`fromfile`: https://numpy.org/doc/stable/reference/generated/numpy.fromfile.html
.. _`ndarray.tofile`: https://numpy.org/doc/stable/reference/generated/numpy.ndarray.tofile.html
.. _`array2string`: https://numpy.org/doc/stable/reference/generated/numpy.array2string.html
.. _`array_repr`: https://numpy.org/doc/stable/reference/generated/numpy.array_repr.html
.. _`array_str`: https://numpy.org/doc/stable/reference/generated/numpy.array_str.html
.. _`format_float_positional`: https://numpy.org/doc/stable/reference/generated/numpy.format_float_positional.html
.. _`format_float_scientific`: https://numpy.org/doc/stable/reference/generated/numpy.format_float_scientific.html
.. _`memmap`: https://numpy.org/doc/stable/reference/generated/numpy.memmap.html
.. _`lib.format.open_memmap`: https://numpy.org/doc/stable/reference/generated/numpy.lib.format.open_memmap.html
.. _`set_printoptions`: https://numpy.org/doc/stable/reference/generated/numpy.set_printoptions.html
.. _`get_printoptions`: https://numpy.org/doc/stable/reference/generated/numpy.get_printoptions.html
.. _`set_string_function`: https://numpy.org/doc/stable/reference/generated/numpy.set_string_function.html
.. _`printoptions`: https://numpy.org/doc/stable/reference/generated/numpy.printoptions.html
.. _`binary_repr`: https://numpy.org/doc/stable/reference/generated/numpy.binary_repr.html
.. _`base_repr`: https://numpy.org/doc/stable/reference/generated/numpy.base_repr.html
.. _`DataSource`: https://numpy.org/doc/stable/reference/generated/numpy.lib.npyio.DataSource.html
.. _`lib.format`: https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html

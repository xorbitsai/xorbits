Binary operations
=================

The following table lists both implemented and not implemented methods. If you have need
of an operation that is listed as not implemented, feel free to open an issue on the
`GitHub repository`_, or give a thumbs up to already created issues. Contributions are
also welcome!

The following table is structured as follows: The first column contains the method name.
The second column contains link to a description of corresponding numpy method.
The third column is a flag for whether or not there is an implementation in Xorbits
for the method in the left column. ``Y`` stands for yes, ``N`` stands for no, ``P`` standsfor partial 
(meaning some parameters may not be supported yet), and ``D`` stands for default to numpy.

Elementwise bit operations
--------------------------

+-------------------+----------------+------------------------+----------------------------------+
| ``xorbits.numpy`` | ``numpy``      | Implemented? (Y/N/P/D) | Notes for Current implementation |
+-------------------+----------------+------------------------+----------------------------------+
| ``bitwise_and``   | `bitwise_and`_ | Y                      |                                  |
+-------------------+----------------+------------------------+----------------------------------+
| ``bitwise_or``    | `bitwise_or`_  | Y                      |                                  |
+-------------------+----------------+------------------------+----------------------------------+
| ``bitwise_xor``   | `bitwise_xor`_ | Y                      |                                  |
+-------------------+----------------+------------------------+----------------------------------+
| ``invert``        | `invert`_      | Y                      |                                  |
+-------------------+----------------+------------------------+----------------------------------+
| ``left_shift``    | `left_shift`_  | Y                      |                                  |
+-------------------+----------------+------------------------+----------------------------------+
| ``right_shift``   | `right_shift`_ | Y                      |                                  |
+-------------------+----------------+------------------------+----------------------------------+

Bit packing
-----------

+-------------------+---------------+------------------------+----------------------------------+
| ``xorbits.numpy`` | ``numpy``     | Implemented? (Y/N/P/D) | Notes for Current implementation |
+-------------------+---------------+------------------------+----------------------------------+
| ``packbits``      | `packbits`_   | Y                      |                                  |
+-------------------+---------------+------------------------+----------------------------------+
| ``unpackbits``    | `unpackbits`_ | Y                      |                                  |
+-------------------+---------------+------------------------+----------------------------------+

Output formatting
-----------------

+-------------------+----------------+------------------------+----------------------------------+
| ``xorbits.numpy`` | ``numpy``      | Implemented? (Y/N/P/D) | Notes for Current implementation |
+-------------------+----------------+------------------------+----------------------------------+
| ``binary_repr``   | `binary_repr`_ | Y                      |                                  |
+-------------------+----------------+------------------------+----------------------------------+

.. _`GitHub repository`: https://github.com/xorbitsai/xorbits/issues
.. _`bitwise_and`: https://numpy.org/doc/stable/reference/generated/numpy.bitwise_and.html
.. _`bitwise_or`: https://numpy.org/doc/stable/reference/generated/numpy.bitwise_or.html
.. _`bitwise_xor`: https://numpy.org/doc/stable/reference/generated/numpy.bitwise_xor.html
.. _`invert`: https://numpy.org/doc/stable/reference/generated/numpy.invert.html
.. _`left_shift`: https://numpy.org/doc/stable/reference/generated/numpy.left_shift.html
.. _`right_shift`: https://numpy.org/doc/stable/reference/generated/numpy.right_shift.html
.. _`packbits`: https://numpy.org/doc/stable/reference/generated/numpy.packbits.html
.. _`unpackbits`: https://numpy.org/doc/stable/reference/generated/numpy.unpackbits.html
.. _`binary_repr`: https://numpy.org/doc/stable/reference/generated/numpy.binary_repr.html

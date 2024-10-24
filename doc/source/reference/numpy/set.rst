Set routines
============

The following table lists both implemented and not implemented methods. If you have need
of an operation that is listed as not implemented, feel free to open an issue on the
`GitHub repository`_, or give a thumbs up to already created issues. Contributions are
also welcome!

The following table is structured as follows: The first column contains the method name.
The second column contains link to a description of corresponding numpy method.
The third column is a flag for whether or not there is an implementation in Xorbits
for the method in the left column. ``Y`` stands for yes, ``N`` stands for no, ``P`` standsfor partial 
(meaning some parameters may not be supported yet), and ``D`` stands for default to numpy.

+---------------------+-----------+------------------------+----------------------------------+
| ``xorbits.numpy``   | ``numpy`` | Implemented? (Y/N/P/D) | Notes for Current implementation |
+---------------------+-----------+------------------------+----------------------------------+
| ``lib.arraysetops`` |           | Y                      |                                  |
+---------------------+-----------+------------------------+----------------------------------+

Making proper sets
------------------
+-------------------+-----------+------------------------+----------------------------------+
| ``xorbits.numpy`` | ``numpy`` | Implemented? (Y/N/P/D) | Notes for Current implementation |
+-------------------+-----------+------------------------+----------------------------------+
| ``unique``        | `unique`_ | Y                      |                                  |
+-------------------+-----------+------------------------+----------------------------------+

Boolean operations
------------------

+-------------------+----------------+------------------------+----------------------------------+
| ``xorbits.numpy`` | ``numpy``      | Implemented? (Y/N/P/D) | Notes for Current implementation |
+-------------------+----------------+------------------------+----------------------------------+
| ``in1d``          | `in1d`_        | Y                      |                                  |
+-------------------+----------------+------------------------+----------------------------------+
| ``intersect1d``   | `intersect1d`_ | Y                      |                                  |
+-------------------+----------------+------------------------+----------------------------------+
| ``isin``          | `isin`_        | Y                      |                                  |
+-------------------+----------------+------------------------+----------------------------------+
| ``setdiff1d``     | `setdiff1d`_   | Y                      |                                  |
+-------------------+----------------+------------------------+----------------------------------+
| ``setxor1d``      | `setxor1d`_    | Y                      |                                  |
+-------------------+----------------+------------------------+----------------------------------+
| ``union1d``       | `union1d`_     | Y                      |                                  |
+-------------------+----------------+------------------------+----------------------------------+

.. _`GitHub repository`: https://github.com/xorbitsai/xorbits/issues
.. _`unique`: https://numpy.org/doc/stable/reference/generated/numpy.unique.html
.. _`in1d`: https://numpy.org/doc/stable/reference/generated/numpy.in1d.html
.. _`intersect1d`: https://numpy.org/doc/stable/reference/generated/numpy.intersect1d.html
.. _`isin`: https://numpy.org/doc/stable/reference/generated/numpy.isin.html
.. _`setdiff1d`: https://numpy.org/doc/stable/reference/generated/numpy.setdiff1d.html
.. _`setxor1d`: https://numpy.org/doc/stable/reference/generated/numpy.setxor1d.html
.. _`union1d`: https://numpy.org/doc/stable/reference/generated/numpy.union1d.html

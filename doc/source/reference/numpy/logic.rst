Logic functions
===============

The following table lists both implemented and not implemented methods. If you have need
of an operation that is listed as not implemented, feel free to open an issue on the
`GitHub repository`_, or give a thumbs up to already created issues. Contributions are
also welcome!

The following table is structured as follows: The first column contains the method name.
The second column contains link to a description of corresponding numpy method.
The third column is a flag for whether or not there is an implementation in Xorbits
for the method in the left column. ``Y`` stands for yes, ``N`` stands for no, ``P`` standsfor partial 
(meaning some parameters may not be supported yet), and ``D`` stands for default to numpy.

Truth value testing
-------------------

+-------------------+-----------+------------------------+----------------------------------+
| ``xorbits.numpy`` | ``numpy`` | Implemented? (Y/N/P/D) | Notes for Current implementation |
+-------------------+-----------+------------------------+----------------------------------+
| ``all``           | `all`_    | Y                      |                                  |
+-------------------+-----------+------------------------+----------------------------------+
| ``any``           | `any`_    | Y                      |                                  |
+-------------------+-----------+------------------------+----------------------------------+

Array contents
--------------

+-------------------+-------------+------------------------+----------------------------------+
| ``xorbits.numpy`` | ``numpy``   | Implemented? (Y/N/P/D) | Notes for Current implementation |
+-------------------+-------------+------------------------+----------------------------------+
| ``isfinite``      | `isfinite`_ | Y                      |                                  |
+-------------------+-------------+------------------------+----------------------------------+
| ``isinf``         | `isinf`_    | Y                      |                                  |
+-------------------+-------------+------------------------+----------------------------------+
| ``isnan``         | `isnan`_    | Y                      |                                  |
+-------------------+-------------+------------------------+----------------------------------+
| ``isnat``         | `isnat`_    | Y                      |                                  |
+-------------------+-------------+------------------------+----------------------------------+
| ``isneginf``      | `isneginf`_ | Y                      |                                  |
+-------------------+-------------+------------------------+----------------------------------+
| ``isposinf``      | `isposinf`_ | Y                      |                                  |
+-------------------+-------------+------------------------+----------------------------------+

Array type testing
------------------

+-------------------+-----------------+------------------------+----------------------------------+
| ``xorbits.numpy`` | ``numpy``       | Implemented? (Y/N/P/D) | Notes for Current implementation |
+-------------------+-----------------+------------------------+----------------------------------+
| ``iscomplex``     | `iscomplex`_    | Y                      |                                  |
+-------------------+-----------------+------------------------+----------------------------------+
| ``iscomplexobj``  | `iscomplexobj`_ | Y                      |                                  |
+-------------------+-----------------+------------------------+----------------------------------+
| ``isfortran``     | `isfortran`_    | Y                      |                                  |
+-------------------+-----------------+------------------------+----------------------------------+
| ``isreal``        | `isreal`_       | Y                      |                                  |
+-------------------+-----------------+------------------------+----------------------------------+
| ``isrealobj``     | `isrealobj`_    | Y                      |                                  |
+-------------------+-----------------+------------------------+----------------------------------+
| ``isscalar``      | `isscalar`_     | Y                      |                                  |
+-------------------+-----------------+------------------------+----------------------------------+

Logical operations
------------------

+-------------------+----------------+------------------------+----------------------------------+
| ``xorbits.numpy`` | ``numpy``      | Implemented? (Y/N/P/D) | Notes for Current implementation |
+-------------------+----------------+------------------------+----------------------------------+
| ``logical_and``   | `logical_and`_ | Y                      |                                  |
+-------------------+----------------+------------------------+----------------------------------+
| ``logical_or``    | `logical_or`_  | Y                      |                                  |
+-------------------+----------------+------------------------+----------------------------------+
| ``logical_not``   | `logical_not`_ | Y                      |                                  |
+-------------------+----------------+------------------------+----------------------------------+
| ``logical_xor``   | `logical_xor`_ | Y                      |                                  |
+-------------------+----------------+------------------------+----------------------------------+

Comparison
----------

+-------------------+------------------+------------------------+----------------------------------+
| ``xorbits.numpy`` | ``numpy``        | Implemented? (Y/N/P/D) | Notes for Current implementation |
+-------------------+------------------+------------------------+----------------------------------+
| ``allclose``      | `allclose`_      | Y                      |                                  |
+-------------------+------------------+------------------------+----------------------------------+
| ``isclose``       | `isclose`_       | Y                      |                                  |
+-------------------+------------------+------------------------+----------------------------------+
| ``array_equal``   | `array_equal`_   | Y                      |                                  |
+-------------------+------------------+------------------------+----------------------------------+
| ``array_equiv``   | `array_equiv`_   | Y                      |                                  |
+-------------------+------------------+------------------------+----------------------------------+
| ``greater``       | `greater`_       | Y                      |                                  |
+-------------------+------------------+------------------------+----------------------------------+
| ``greater_equal`` | `greater_equal`_ | Y                      |                                  |
+-------------------+------------------+------------------------+----------------------------------+
| ``less``          | `less`_          | Y                      |                                  |
+-------------------+------------------+------------------------+----------------------------------+
| ``less_equal``    | `less_equal`_    | Y                      |                                  |
+-------------------+------------------+------------------------+----------------------------------+
| ``equal``         | `equal`_         | Y                      |                                  |
+-------------------+------------------+------------------------+----------------------------------+
| ``not_equal``     | `not_equal`_     | Y                      |                                  |
+-------------------+------------------+------------------------+----------------------------------+

.. _`GitHub repository`: https://github.com/xorbitsai/xorbits/issues
.. _`all`: https://numpy.org/doc/stable/reference/generated/numpy.all.html
.. _`any`: https://numpy.org/doc/stable/reference/generated/numpy.any.html
.. _`isfinite`: https://numpy.org/doc/stable/reference/generated/numpy.isfinite.html
.. _`isinf`: https://numpy.org/doc/stable/reference/generated/numpy.isinf.html
.. _`isnan`: https://numpy.org/doc/stable/reference/generated/numpy.isnan.html
.. _`isnat`: https://numpy.org/doc/stable/reference/generated/numpy.isnat.html
.. _`isneginf`: https://numpy.org/doc/stable/reference/generated/numpy.isneginf.html
.. _`isposinf`: https://numpy.org/doc/stable/reference/generated/numpy.isposinf.html
.. _`iscomplex`: https://numpy.org/doc/stable/reference/generated/numpy.iscomplex.html
.. _`iscomplexobj`: https://numpy.org/doc/stable/reference/generated/numpy.iscomplexobj.html
.. _`isfortran`: https://numpy.org/doc/stable/reference/generated/numpy.isfortran.html
.. _`isreal`: https://numpy.org/doc/stable/reference/generated/numpy.isreal.html
.. _`isrealobj`: https://numpy.org/doc/stable/reference/generated/numpy.isrealobj.html
.. _`isscalar`: https://numpy.org/doc/stable/reference/generated/numpy.isscalar.html
.. _`logical_and`: https://numpy.org/doc/stable/reference/generated/numpy.logical_and.html
.. _`logical_or`: https://numpy.org/doc/stable/reference/generated/numpy.logical_or.html
.. _`logical_not`: https://numpy.org/doc/stable/reference/generated/numpy.logical_not.html
.. _`logical_xor`: https://numpy.org/doc/stable/reference/generated/numpy.logical_xor.html
.. _`allclose`: https://numpy.org/doc/stable/reference/generated/numpy.allclose.html
.. _`isclose`: https://numpy.org/doc/stable/reference/generated/numpy.isclose.html
.. _`array_equal`: https://numpy.org/doc/stable/reference/generated/numpy.array_equal.html
.. _`array_equiv`: https://numpy.org/doc/stable/reference/generated/numpy.array_equiv.html
.. _`greater`: https://numpy.org/doc/stable/reference/generated/numpy.greater.html
.. _`greater_equal`: https://numpy.org/doc/stable/reference/generated/numpy.greater_equal.html
.. _`less`: https://numpy.org/doc/stable/reference/generated/numpy.less.html
.. _`less_equal`: https://numpy.org/doc/stable/reference/generated/numpy.less_equal.html
.. _`equal`: https://numpy.org/doc/stable/reference/generated/numpy.equal.html
.. _`not_equal`: https://numpy.org/doc/stable/reference/generated/numpy.not_equal.html

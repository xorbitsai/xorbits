============
Installation
============

Xorbits can be installed via pip from `PyPI <https://pypi.org/project/xorbits>`__.

::

    pip install xorbits


Python version support
----------------------

Officially Python 3,7, 3.8, 3.9 and 3.10.

Dependencies
------------

================================================================ ==========================
Package                                                          Minimum supported version
================================================================ ==========================
`NumPy <https://numpy.org>`__                                    1.20.3
`pandas <https://pandas.pydata.org>`__                           1.0.0
`scipy <https://scipy.org>`__                                    1.0.0
`scikit-learn <https://scikit-learn.org/stable>`__               0.20
cloudpickle                                                      1.5.0
pyyaml                                                           5.1
psutil                                                           5.9.0
pickle5 (for python version < 3.8)                               0.0.1
shared-memory38 (for python version < 3.8)                       0.1.0
tornado                                                          6.0
sqlalchemy                                                       1.2.0
defusedxml                                                       0.5.0
tqdm                                                             4.1.0
uvloop (for systems other than win32)                            0.14.0
================================================================ ==========================

Recommended dependencies
~~~~~~~~~~~~~~~~~~~~~~~~

* `numexpr <https://github.com/pydata/numexpr>`__: for accelerating certain numerical operations.
  ``numexpr`` uses multiple cores as well as smart chunking and caching to achieve large speedups.
  If installed, must be Version 2.6.4 or higher.

.. note::

   You are highly encouraged to install these libraries, as they provide speed improvements,
   especially when working with large data sets.
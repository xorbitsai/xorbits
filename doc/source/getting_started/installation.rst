.. _installation:

============
Installation
============

Xorbits can be installed via pip from `PyPI <https://pypi.org/project/xorbits>`__.

::

    pip install xorbits


.. _install.version:

Python version support
----------------------

Officially Python 3.9, 3.10, 3.11, and 3.12.

Packages support
----------------

Xorbits partitions large datasets into chunks and processes each individual 
chunk using single-node packages (such as pandas). Currently, our latest version strives 
to be compatible with the latest single-node packages. The table below lists the highest 
versions of the single-node packages that Xorbits are compatible with. If you are using 
an older version of pandas, you should either upgrade your pandas or downgrade Xorbits.

======= =================== ======== ========= ========== =========== ===========
Xorbits Python              `NumPy`_ `pandas`_ `xgboost`_ `lightgbm`_ `datasets`_
======= =================== ======== ========= ========== =========== ===========
0.7.4   3.9,3.10,3.11       1.26.4   2.2.3     2.1.1      4.5.0       3.0.1
0.8.0   3.9,3.10,3.11,3.12  2.1.3    2.2.3     2.1.2      4.5.0       3.1.0
======= =================== ======== ========= ========== =========== ===========

.. _`NumPy`: https://numpy.org
.. _`pandas`: https://pandas.pydata.org
.. _`xgboost`: https://xgboost.readthedocs.io
.. _`lightgbm`: https://lightgbm.readthedocs.io
.. _`datasets`: https://huggingface.co/docs/datasets/index


Dependencies
------------

Required packages
~~~~~~~~~~~~~~~~~

Xorbits depends on the following libraries, which are mandatory. When you run 
``pip install xorbits``, ``pip`` will download the latest versions of these packages from the PyPI.

- cloudpickle                                                      
- pyyaml                                                          
- psutil                                                          
- tornado                                                         
- sqlalchemy                                                      
- defusedxml                                                      
- tqdm                                                            
- uvloop (for systems other than win32)                           

Recommended dependencies
~~~~~~~~~~~~~~~~~~~~~~~~

.. note::

   You are highly encouraged to install these libraries, as they provide speed improvements,
   especially when working with large datasets.

Recommended dependencies can be installed using ``pip``.

::

    pip install 'xorbits[extra]'


The following extra dependencies will be installed.

.. _install.optional_dependencies:

* `numexpr <https://github.com/pydata/numexpr>`__: for accelerating certain numerical operations.
  ``numexpr`` uses multiple cores as well as smart chunking and caching to achieve large speedups.

* `pillow <https://python-pillow.org/>`__: the Python Imaging Library.

* `pyarrow <https://pypi.org/project/pyarrow/>`__: python API for Arrow C++ libraries.

* `lz4 <https://github.com/python-lz4/python-lz4>`__: python bindings for the LZ4 compression
  library.

* `fsspec <https://github.com/fsspec/filesystem_spec>`__: for cloud data accessing.

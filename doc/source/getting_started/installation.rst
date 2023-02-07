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

Recommended dependencies can be installed conveniently using pip.

::

    pip install 'xorbits[extra]'


The following extra dependencies will be installed.

.. _install.optional_dependencies:

* `numexpr <https://github.com/pydata/numexpr>`__: for accelerating certain numerical operations.
  ``numexpr`` uses multiple cores as well as smart chunking and caching to achieve large speedups.
  If installed, must be Version 2.6.4 or higher.

* `pillow <https://python-pillow.org/>`__: the Python Imaging Library. If installed, must be
  Version 7.0.0 or higher.

* `pyarrow <https://pypi.org/project/pyarrow/>`__: python API for Arrow C++ libraries. If
  installed, must be Version 5.0.0 or higher.

* `lz4 <https://github.com/python-lz4/python-lz4>`__: python bindings for the LZ4 compression
  library. If installed, must be Version 1.0.0 or higher.

* `fsspec <https://github.com/fsspec/filesystem_spec>`__: for cloud data accessing. If installed,
  must be Version 2022.7.1 or higher and cannot be Version 2022.8.0.

.. note::

   You are highly encouraged to install these libraries, as they provide speed improvements,
   especially when working with large data sets.

Deployment
~~~~~~~~~~

Xorbits can be deployed on many platforms, see below for what you need.

========================================= ============================================================
Deployment                                Description
========================================= ============================================================
:ref:`Local <deployment_local>`           Run Xorbits on a local machine, e.g. your laptop
:ref:`Cluster <deployment_cluster>`       Deploy Xorbits to existing cluster via command lines
:ref:`Kubernetes <deployment_kubernetes>` Deploy Xorbits to existing k8s cluster via python code
:ref:`Cloud <deployment_cloud>`           Deploy Xorbits to various cloud platforms via python code
========================================= ============================================================
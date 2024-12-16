.. _installation:

============
Installation
============

Xorbits can be installed via pip from `PyPI <https://pypi.org/project/xorbits>`__.

::

    pip install xorbits

It will install the latest version of Xorbits and dependencies like ``pandas``, ``numpy``, etc.
We recommend you to use environment management tools like ``conda`` or ``venv`` to create 
a new environment. ``conda`` will install the pre-compiled packages, while ``pip`` will
install the wheel (which is pre-compiled) or compile the packages from source code if no wheel
is available.

Python version support
----------------------

Officially support Python 3.9, 3.10, 3.11, and 3.12.

Packages support
----------------

Xorbits partitions large datasets into chunks and processes each individual 
chunk using single-node packages (such as pandas). 
Currently, our latest version strives 
to be compatible with the latest single-node packages. The table below lists the highest 
versions of the single-node packages that Xorbits are compatible with. If you are using 
an older version of pandas, you should either upgrade your pandas or downgrade Xorbits.

======= =================== ======== ========= ========== =========== ===========
Xorbits Python              `NumPy`_ `pandas`_ `xgboost`_ `lightgbm`_ `datasets`_
======= =================== ======== ========= ========== =========== ===========
0.8.1   3.9,3.10,3.11,3.12  2.1.3    2.2.3     2.1.3      4.5.0       3.1.0
0.8.0   3.9,3.10,3.11,3.12  2.1.3    2.2.3     2.1.2      4.5.0       3.1.0
0.7.4   3.9,3.10,3.11       1.26.4   2.2.3     2.1.1      4.5.0       3.0.1
======= =================== ======== ========= ========== =========== ===========

.. _`NumPy`: https://numpy.org
.. _`pandas`: https://pandas.pydata.org
.. _`xgboost`: https://xgboost.readthedocs.io
.. _`lightgbm`: https://lightgbm.readthedocs.io
.. _`datasets`: https://huggingface.co/docs/datasets/index

GPU support
-----------

Xorbits can also scale GPU-accelerated data science tools like `CuPy`_ and `cuDF`_. To enable GPU support, you need to install
GPU-accelerated packages. As GPU software stacks (i.e.,GPU driver, CUDA, etc.)
are complicated from CPU, you need to make sure NVIDIA driver and CUDA toolkit are properly installed.
We recommend you to use ``conda`` to install ``cuDF`` first, it will install both ``cudf`` and ``cupy``,
and then install ``xorbits`` with ``pip``. 
``conda`` will help resolve the dependencies of ``cuDF`` and provides supporting software like CUDA.
Refer to `RAPIDS_INSTALL_DOCS`_ for more details about how to install ``cuDF``.

When using Xorbits with GPU, you need to add the :code:`gpu=True` parameter to the data loading method.
For example:

.. code-block:: python

    import xorbits.pandas as pd
    df = pd.read_parquet(path, gpu=True)

======= =================== ======== =========
Xorbits Python              `CuPy`_  `cuDF`_  
======= =================== ======== =========
0.8.1   3.10,3.11,3.12      13.3.0    24.10   
======= =================== ======== =========

If you find installing GPU-accelerated packages too complicated, you can use our docker images
with pre-installed GPU drivers and CUDA toolkit. Please refer to :ref:`docker` for more details.

.. _`Cupy`: https://cupy.dev
.. _`cuDF`: https://docs.rapids.ai/api/cudf/stable/
.. _`RAPIDS_INSTALL_DOCS`: https://docs.rapids.ai/install/

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

.. _docker:
Docker image
------------

To simplify the installation of Xorbits, we provide docker images with pre-installed
Xorbits and its dependencies.

* CPU image: ``xprobe/xorbits:v{version}-py{python_version}``, e.g., ``xprobe/xorbits:v0.8.0-py3.12``
* GPU image: ``xprobe/xorbits:v{version}-cuda{cuda_version}-py{python_version}``, e.g., ``xprobe/xorbits:v0.8.0-cuda12.0-py3.12``

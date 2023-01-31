.. _index:

.. raw:: html

    <img class="align-center" alt="Xorbits Logo" src="_static/xorbits.svg" style="background-color: transparent", width="77%">

====


Xorbits: scalable Python data science, familiar & fast.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""

Xorbits is a scalable Python data science framework that aims to **scale the whole Python data science world,
including numpy, pandas, scikit-learn and many other libraries**. It can leverage multi cores or GPUs to
accelerate computation on a single machine, or scale out up to thousands of machines to support processing
terabytes of data. In our benchmark test, **Xorbits is the fastest framework among
the most popular distributed data science frameworks**.

As for the name of ``xorbits``, it has many meanings, you can treat it as ``X-or-bits`` or ``X-orbits`` or ``xor-bits``,
just have fun to comprehend it in your own way.

At a glance
-----------

.. image:: _static/pandas_vs_xorbits.gif

Codes are almost identical except for the import,
replace ``import pandas`` with ``import xorbits.pandas`` will just work,
so does numpy and so forth.

Where to get it?
----------------

The source code is currently hosted on GitHub at: https://github.com/xprobe-inc/xorbits

Binary installers for the latest released version are available at the
`Python Package Index (PyPI) <https://pypi.org/project/xorbits>`_

.. code-block:: shell

   # PyPI
   pip install xorbits

API compatibility
-----------------

As long as you know how to use numpy, pandas and so forth, you would probably know how to use Xorbits.

All Xorbits APIs implemented or planned include:

======================================= =========================================================
API                                     Implemented version or plan
======================================= =========================================================
:ref:`xorbits.pandas <pandas_api>`      v0.1.0
:ref:`xorbits.numpy <numpy_api>`        v0.1.0
``xorbits.sklearn``                     Planned in the near future
``xorbits.xgboost``                     Planned in the near future
``xorbits.lightgbm``                    Planned in the near future
``xorbits.xarray``                      Planned in the future
======================================= =========================================================

Lightning fast speed
--------------------

Xorbits is the fastest compared to other popular frameworks according to our benchmark tests.

We did benchmarks for TPC-H at scale factor 100 (~100 GB datasets) and 1000 (~1 TB datasets).
The performances are shown as below.

TPC-H SF100: Xorbits vs Dask
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: https://xorbits.io/res/benchmark_dask.png

Q21 was excluded since Dask ran out of memory. Across all queries, Xorbits was found to be 7.3x faster than Dask.

TPC-H SF100: Xorbits vs Pandas API on Spark
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: https://xorbits.io/res/benchmark_spark.png

Across all queries, the two systems have roughly similar performance, but Xorbits provided much better API compatibility.
Pandas API on Spark failed on Q1, Q4, Q7, Q21, and ran out of memory on Q20.

TPC-H SF100: Xorbits vs Modin
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: https://xorbits.io/res/benchmark_modin.png

Although Modin ran out of memory for most of the queries that involve heavy data shuffles,
making the performance difference less obvious, Xorbits was still found to be 3.2x faster than Modin.

TPC-H SF1000: Xorbits
~~~~~~~~~~~~~~~~~~~~~

.. image:: https://xorbits.io/res/xorbits_1t.png

Although Xorbits is able to pass all the queries in a row,
Dask, Pandas API on Spark and Modin failed on most of the queries.
Thus, we are not able to compare the performance difference now, and we plan to try again later.

For more information, see `performance benchmarks <https://xorbits.io/benchmark>`_.

Deployment
----------

Xorbits can be deployed on your local machine, largely deployed to a cluster via command lines,
or deploy via Kubernetes and clouds.

========================================= ============================================================
Deployment                                Description
========================================= ============================================================
:ref:`Local <deployment_local>`           Running Xorbits on a local machine, e.g. laptop
:ref:`Cluster <deployment_cluster>`       Deploy Xorbits to existing cluster via command lines
:ref:`Kubernetes <deployment_kubernetes>` Deploy Xorbits to existing k8s cluster via python codes
:ref:`Cloud <deployment_cloud>`           Deploy Xorbits to various cloud platforms via python codes
========================================= ============================================================

Getting involved
----------------

.. only:: html and not zh_cn

    +--------------------------------------------------------------------------------------------------+----------------------------------------------------+
    | **Platform**                                                                                     | **Purpose**                                        |
    +--------------------------------------------------------------------------------------------------+----------------------------------------------------+
    | `Discourse Forum <https://discuss.xorbits.io/>`_                                                 | Asking usage questions and discussing development. |
    +--------------------------------------------------------------------------------------------------+----------------------------------------------------+
    | `Github Issues <https://github.com/xprobe-inc/xorbits/issues>`_                                  | Reporting bugs and filing feature requests.        |
    +--------------------------------------------------------------------------------------------------+----------------------------------------------------+
    | `Slack <https://join.slack.com/t/xorbitsio/shared_invite/zt-1o3z9ucdh-RbfhbPVpx7prOVdM1CAuxg>`_  | Collaborating with other Xorbits users.            |
    +--------------------------------------------------------------------------------------------------+----------------------------------------------------+
    | `StackOverflow <https://stackoverflow.com/questions/tagged/xorbits>`_                            | Asking questions about how to use Xorbits.         |
    +--------------------------------------------------------------------------------------------------+----------------------------------------------------+
    | `Twitter <https://twitter.com/xorbitsio>`_                                                       | Staying up-to-date on new features.                |
    +--------------------------------------------------------------------------------------------------+----------------------------------------------------+

.. only:: html and zh_cn

    +--------------------------------------------------------------------------------------------------+----------------------------------------------------+
    | **Platform**                                                                                     | **Purpose**                                        |
    +--------------------------------------------------------------------------------------------------+----------------------------------------------------+
    | `Github Issues <https://github.com/xprobe-inc/xorbits/issues>`_                                  | Reporting bugs and filing feature requests.        |
    +--------------------------------------------------------------------------------------------------+----------------------------------------------------+


.. toctree::
   :maxdepth: 2
   :hidden:

   getting_started/index
   user_guide/index
   deployment/index
   reference/index

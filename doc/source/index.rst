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

As long as you know how to use numpy, pandas and so forth, you would probably know how to use xorbits.

Here is an example.

+--------------------------------------------------+--------------------------------------------------+
| **pandas**                                       | **Xorbits**                                      |
+--------------------------------------------------+--------------------------------------------------+
|.. code-block:: python                            |.. code-block:: python                            |
|                                                  |                                                  |
|    import pandas as pd                           |    import xorbits.pandas as pd                   |
|                                                  |                                                  |
|    ratings = pd.read_csv('ratings.csv')          |    ratings = pd.read_csv('ratings.csv')          |
|    movies = pd.read_csv('movies.csv')            |    movies = pd.read_csv('movies.csv')            |
|                                                  |                                                  |
|    m = ratings.groupby(                          |    m = ratings.groupby(                          |
|        'MOVIE_ID', as_index=False).agg(          |        'MOVIE_ID', as_index=False).agg(          |
|        {'RATING': ['mean', 'count']})            |        {'RATING': ['mean', 'count']})            |
|    m.columns = ['MOVIE_ID', 'RATING', 'COUNT']   |    m.columns = ['MOVIE_ID', 'RATING', 'COUNT']   |
|    m = m[m['COUNT'] > 100]                       |    m = m[m['COUNT'] > 100]                       |
|    top_100 = m.sort_values(                      |    top_100 = m.sort_values(                      |
|        'RATING', ascending=False)[:100]          |        'RATING', ascending=False)[:100]          |
|    top_100 = top_100.merge(                      |    top_100 = top_100.merge(                      |
|        movies[['MOVIE_ID', 'NAME']])             |        movies[['MOVIE_ID', 'NAME']])             |
|    print(top_100)                                |    print(top_100)                                |
|                                                  |                                                  |
+--------------------------------------------------+--------------------------------------------------+

Codes are almost identical except for the import,
replace ``import pandas`` with ``import xorbits.pandas`` will just work,
so does numpy and so forth.

Lightning fast speed
--------------------

Xorbits is the fastest compared to other popular frameworks according to our benchmark tests.

We did a benchmark for TPC-H at scale factor 100 and 1000. The performances are shown as below.

Deployment
----------

Xorbits can be deployed on your local machine, or largely deployed to a cluster via command lines.

======================================= =========================================================
Deployment                              Description
======================================= =========================================================
:ref:`Local <deployment_local>`         Running Xorbits on a local machine, e.g. laptop
:ref:`Cluster <deployment_cluster>`     Deploy Xorbits to existing cluster via command lines
======================================= =========================================================

Getting involved
----------------

+-----------------------------------------------------------------------+----------------------------------------------------+
| **Platform**                                                          | **Purpose**                                        |
+-----------------------------------------------------------------------+----------------------------------------------------+
| `Discourse Forum <https://discuss.xorbits.io/>`_                      | Asking usage questions and discussing development. |
+-----------------------------------------------------------------------+----------------------------------------------------+
| `Github Issues <https://github.com/xprobe-inc/xorbits/issues>`_       | Reporting bugs and filing feature requests.        |
+-----------------------------------------------------------------------+----------------------------------------------------+
| `Slack <https://slack.xorbits.io/>`_                                  | Collaborating with other Xorbits users.            |
+-----------------------------------------------------------------------+----------------------------------------------------+
| `StackOverflow <https://stackoverflow.com/questions/tagged/xorbits>`_ | Asking questions about how to use Xorbits.         |
+-----------------------------------------------------------------------+----------------------------------------------------+
| `Twitter <https://twitter.com/xorbitsio>`_                            | Staying up-to-date on new features.                |
+-----------------------------------------------------------------------+----------------------------------------------------+


.. toctree::
   :maxdepth: 2
   :hidden:

   getting_started/index
   user_guide/index
   deployment/index
   reference/index

.. _index:

.. figure:: _static/xorbits.svg
   :align: center
   :width: 77%

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

Local
~~~~~

On a single machine e.g. your laptop, optionally, you can initialize Xorbits on your own:

.. code-block:: python

    import xorbits
    xorbits.init()

Or Xorbits will try to init for you when the first time some computation is triggered.


Bare metal
~~~~~~~~~~

Deploy to bare metal machines is dead easy as well. Ensure xorbits is installed on each machine,
then:

For supervisor which schedules tasks to workers.

.. code-block:: shell

   # for supervisor
   python -m xorbits.supervisor -H <host_name> -p <supervisor_port> -w <web_port>

For workers which run actual computations.

.. code-block:: shell

   # for worker
   python -m xorbits.worker -H <host_name> -p <worker_port> -s <supervisor_ip>:<supervisor_port>


Then connect to the supervisor anywhere that can run Python code.

.. code-block:: python

    import xorbits
    xorbits.init("http://<supervisor_ip>:<supervisor_web_port>")

Replace the ``<supervisor_ip>`` with the supervisor host name that you just specified and
``<supervisor_web_port>`` with the supervisor web port.


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

   reference/index
   getting_started/index

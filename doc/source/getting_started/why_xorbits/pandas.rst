.. _xorbits_vs_pandas:

Why choose Xorbits Pandas over pandas?
======================================

Boosting performance and scalability with Xorbits 
-------------------------------------------------

In brief, Xorbits Pandas offers the pandas API and makes it easy to scale. The original
implementation of pandas is inherently single-threaded, which means at any given time, 
only one of your CPU cores can be employed. In a laptop setting, using pandas would look
something like this:

.. image:: /_static/pandas_multicore.png
   :alt: pandas is single threaded
   :align: center
   :scale: 25%

However, Xorbits changes the game by enabling the use of all the cores on your machine, 
or even in an entire cluster, if available. This maximized use of resources results in 
better performance. Here's how it would look like on a laptop when using Xorbits:

.. image:: /_static/xorbits_multicore.png
   :alt: Xorbits uses 100% resource
   :align: center
   :scale: 25%

When it comes to scaling across an entire cluster, Xorbits stands out by efficiently 
utilizing all the available hardware resources (even GPUs):

.. image:: /_static/xorbits_cluster.png
   :alt: Xorbits works on a cluster
   :align: center
   :scale: 30%


Overcoming memory limitations in large datasets with Xorbits
------------------------------------------------------------

Pandas uses in-memory data structures to store and manipulate data. This means that
a dataset too large for memory will trigger an error in pandas.

This problem is effectively addressed by Xorbits, which utilizes disk space as an 
extension for memory, enabling you to handle datasets that are too large to be accommodated
in the memory. By default, Xorbits employs out-of-core methods to manage datasets that don't
fit in memory. More specifically, Xorbits adopts a streaming approach to load dataset, 
which means it doesn't need to load the entire dataset into memory. Instead, only the data 
required for computations is loaded into memory, while the remainder can be stored on the 
disk/cloud storage. This method is efficient and saves memory resources.

Xorbits not only enables you to work with datasets too large for memory, but it also lets you 
perform memory-intensive operations (e.g. joins) on them without being constrained by memory 
limits.
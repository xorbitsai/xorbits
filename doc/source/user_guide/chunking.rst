.. _chunking:

========
Chunking
========

Xorbits divides large datasets into multiple chunks, with each chunk executed independently using 
single-node libraries such as pandas and numpy. Chunking significantly impacts performance. Too 
many chunks can lead to a large computation graph, causing the supervisor to spend excessive time 
on scheduling. Conversely, too few chunks may result in OOM (Out-Of-Memory) issues for some chunks 
that exceed memory capacity. Therefore, a single chunk should not be too large or too small, and 
chunking needs to align with both the computation and the available storage. Users familiar with 
Dask know that Dask requires manual setting of chunk shape or sizes using certain operations, such as :code:`repartition()`.

Automatically
-------------

Unlike Dask, Xorbits does not require users to manually set chunk sizes or perform :code:`repartition()` 
operations, as our chunking process occurs automatically in the background, transparent to the user. 
This automatic chunking mechanism simplifies user interfaces (no more extra :code:`repartition` code) and 
optimizes performance (no more OOM issues). We call this process **Dynamic Tiling**. Interested 
readers can refer to our `research paper <https://arxiv.org/abs/2401.00865>`_ for more detailed 
information.

Xorbits' operator partitioning is referred to as tiling. We have a predefined option called 
:code:`chunk_store_limit`. This option controls the upper limit of each chunk. During the tiling 
process, Xorbits calculates the data size incoming from upstream operators. Each chunk's data size 
is \<= the :code:`chunk_store_limit`. Any data exceeding the :code:`chunk_store_limit` is 
partitioned into a new chunk.

We have set this :code:`chunk_store_limit` option to :code:`512 * 1024 ** 2`, which is equivalent to 
512 M. It's important to note that this value may not be optimal for all scenarios and workloads. 
In CPU environments, setting this value higher may not yield substantial benefits even if you 
have a large amount of RAM available. However, in GPU scenarios, it's advisable to set this value 
higher to maximize the data size within each chunk, thereby minimizing data transfer between GPUs.

You can set this value inside a context:

.. code-block:: python

    with xorbits.option_context({"chunk_store_limit": 1024 ** 3}):
        # your xorbits code

Or you can set this value at the begining of your Python script:

.. code-block:: python

    xorbits.options.chunk_store_limit = 1024 ** 3
    # your xorbits code

Manually
--------

We recommend using either the :code:`xorbits.option_context()` method or the :code:`xorbits.options` 
attribute mentioned above to configure the setting. If you wish to specify the number of chunks 
(typically for debugging purposes), you can do so as follows by specifying the :code:`chunk_size` 
when creating a Xorbits DataFrame or Array.

.. code-block:: python

    import numpy as np
    import pandas as pd
    import xorbits.pandas as xpd
    
    data = pd.DataFrame(
        np.random.rand(10, 10), index=np.arange(10), columns=np.arange(3, 13)
    )
    xdf = xpd.DataFrame(data, chunk_size=5)



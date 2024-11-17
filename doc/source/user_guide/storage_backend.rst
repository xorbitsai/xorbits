.. _storage_backend:

===============
Storage Backend
===============

Xorbits is a distributed computing system that utilize computation graph to decompose big data. 
The smallest unit in a computation graph is a subtask graph. Executing a subtask graph is calling 
single-machine numpy or pandas, and the generated intermediate data is stored in memory by default. 
The backend used for storing this intermediate data is called the **Storage Backend**.

Specifically, the default memory storage backend is called shared memory. Besides shared memory, 
there are several other types of storage backends: GPU, filesystem, mmap, etc.

Shared Memory
-------------

Shared memory is the default storage backend in Xorbits. It is fast and efficient when the data size 
can fit into the memory of a computing node. Share memory means that the data can be accessed by one 
or more processes on a multicore or symmetric multiprocessor (SMP) machine. We implemented this backend 
using Python 3's `multiprocessing.shared_memory <https://docs.python.org/3/library/multiprocessing.shared_memory.html>`_.
In most scenarios, shared memory performs well. We don't need to switch to other storage backends.

Filesystem
----------

Usually, the available capacity of the filesystem is much larger than that of RAM.
When your RAM is not large enough, you can use the filesystem to store these intermediate results. 
The filesystem can be further divided into two categories: local and distributed.
Examples of local filesystem are local disk or mmap; and distributed filesystems are systems like 
JuiceFS and Alluxio. 

Local
^^^^^

Suppose you want to use the `/tmp` directory on your local disk as the storage backend. You should 
create a YAML configuration file named `file.yml` which specify `backends` and `root_dirs`.

.. code-block:: yaml
    
    "@inherits": "@default"
    storage:
        backends: [disk]
        disk:
            root_dirs: "/tmp"

Start the worker using the :code:`-f file.yml` option:

.. code-block:: bash

    xorbits-worker -H <worker_ip> -p <worker_port> -s <supervisor_ip>:<supervisor_port> -f file.yml


mmap
^^^^

mmap (memory-mapped file) is a technique that maps a filesystem file or a device into RAM. Pandas may 
encounter OOM(Out-of-Memory) issues when processing large datasets. Xorbits' mmap storage backend enables 
users to handle datasets much larger than the available RAM capacity. Basically, Xorbits's mmap controls 
the amount of data in runtime memory at a stable level, loading data from disk when necessary. Therefore, 
mmap can handle datasets that are sizes of the available disk space.

Xorbits's mmap now can run on a single-node setup. Just initialize in the :code:`xorbits.init()` method like
and specify :code:`root_dirs` to a disk file path:

.. code-block:: python
    
    import xorbits
    xorbits.init(storage_config={"mmap": {"root_dirs": "<your_dir>"}})



Distributed Filesystems
^^^^^^^^^^^^^^^^^^^^^^^

Both Alluxio and JuiceFS provide a FUSE (Filesystem in Userspace) interface, enabling users to access 
large-scale data using standard POSIX interfaces. They allow users to interact with distributed data 
as if it is on a local filesystem path like `/path/to/data`. When a user reads from or writes to a local 
path, the data is first cached in memory, and then persisted in a remote underlying big data storage engine, 
such as HDFS or S3.

Suppose you mount Alluxio or JuiceFS on `/mnt/xorbits`. You can write a YAML file just like the local filesystem
and start the worker by adding :code:`-f file.yml` option.

.. code-block:: yaml
    
    "@inherits": "@default"
    storage:
        backends: [disk]
        disk:
            root_dirs: "/mnt/xorbits"


GPU
---

If you want to run tasks on GPUs, add the :code:`gpu=True` parameter to the data loading method. For example:

.. code-block:: python
    
    import xorbits.pandas as pd
    import xorbits.numpy as np
    
    df = pd.read_parquet(path, gpu=True)
    ...

    a = np.ones((1000, 1000), gpu=True)
    b = np.ones((1000, 1000), gpu=True)
    c = np.matmul(a, b)
    ...


All subsequent operations will run on GPUs.
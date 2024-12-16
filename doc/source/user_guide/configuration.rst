.. _configuration:

=============
Configuration
=============

In Xorbits, there are two types of configuration and option setting approaches: 

- cluster-level: applied to the whole cluster when starting the supervisor and the workers.
- job-level: applied to a specific Xorbits job or Python script.

Cluster-Level Configuration
---------------------------

Cluster-level configurations are applied to the entire Xorbits cluster and affect all jobs 
running on it. These settings are typically defined when starting the Xorbits cluster 
(i.e., the supervisor or the workers) and remain constant throughout the cluster's lifetime.

Examples of cluster-level configurations include:

- Network: use TCP Socket or UCX.
- Storage: use Shared Memory or Filesystem.

These configurations are usually set through command-line arguments and configuration files 
when launching the Xorbits cluster. Specifically, users should create a YAML configuration 
file (e.g., `config.yml`) and starting the when starting the 
supervisor and workers using the ``-f config.yml`` option. Find more details on how to use ``-f`` in :ref:`custom configuration 
in cluster deployment <cluster_custom_configuration>`. The default YAML file is 
`base_config.yml <https://github.com/xorbitsai/xorbits/blob/main/python/xorbits/_mars/deploy/oscar/base_config.yml>`_.
Write your own one like this:

.. code-block:: yaml
    :caption: config.yml

    "@inherits": "@default"
    storage:
        default_config: 
            transfer_block_size: 10 * 1024 ** 2
    cluster:
        node_timeout: 1200

Job-Level Configuration
-----------------------

Job-level configurations are specific to individual Xorbits jobs or sessions. These settings 
allow users to fine-tune the behavior of their specific workloads without affecting other 
jobs running on the same cluster.

Job-level configurations can be set using the following methods:

1. Using ``xorbits.options.set_option()`` or ``xorbits.pandas.set_option()``.

``xorbits.options.set_option()`` and ``xorbits.pandas.set_option()`` are effective for all packages within Xorbits. 

.. code-block:: python
   
   from xorbits import options

   options.set_option("chunk_store_limit", 1024 ** 3)


Using ``xorbits.pandas.set_option()`` to configure both pandas and Xoribts, 
as ``xorbits.pandas.set_option()`` can also be used to configure not only 
Xorbits but also pandas-native settings.

.. code-block:: python

   import xorbits.pandas as xpd
   
   xpd.set_option("chunk_store_limit", 1024 ** 3)
   xpd.set_option("display.max_rows", 100)


2. Using ``xorbits.option_context()`` or ``xorbits.pandas.option_context()``.

Note that the argument of ``option_context()`` is a ``dict``. These two ``option_context()`` configuration methods are only effective within a specific 
context. Similar to ``xorbits.pandas.set_option()``, ``xorbits.pandas.option_context()`` can also be used to configure pandas-native settings.

.. code-block:: python

   import xorbits.pandas as xpd

   with xpd.option_context({"chunk_store_limit": 1024 ** 3}):
      # Your Xorbits code here
      # The chunk_store_limit will be set to 1 GB 
      # only within this context
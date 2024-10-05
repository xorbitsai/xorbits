.. _config:

=============
Configuration
=============

In Xorbits, there are two types of configuration and option setting approaches: 

- cluster-level: applied to the whole cluster when starting the supervisor or the workers.
- job-level: applied to a specific Xorbits job or Python script.

Cluster-Level Configuration
---------------------------

Cluster-level configurations are applied to the entire Xorbits cluster and affect all jobs 
running on it. These settings are typically defined when starting the Xorbits cluster 
(i.e., the supervisor or the workers) and remain constant throughout the cluster's lifetime.

Examples of cluster-level configurations include:

- Network: use TCP Socket or UCX.
- Storage: use Shared Memory or Filesystem.

These configurations are usually set through command-line arguments or configuration files 
when launching the Xorbits cluster.

Job-Level Configuration
-----------------------

Job-level configurations are specific to individual Xorbits jobs or sessions. These settings allow users to fine-tune the behavior of their specific workloads without affecting other jobs running on the same cluster.

Job-level configurations can be set using the following methods:

1. Using `xorbits.set_option()`:

   .. code-block:: python

      import xorbits.pandas as xpd
      
      xpd.set_option("chunk_store_limit", 1024 ** 3)  # Set chunk store limit to 1 GB

2. Using `xorbits.option_context()`:

   .. code-block:: python

      import xorbits.pandas as xpd

      with xpd.option_context({"chunk_store_limit": 1024 ** 3}):
          # Your Xorbits code here
          # The chunk_store_limit will be set to 1 GB only within this context

These job-level configurations allow users to optimize their workloads for specific requirements without impacting other jobs or the overall cluster configuration.

By providing both cluster-level and job-level configuration options, Xorbits offers flexibility in managing resources and optimizing performance for various use cases and workloads.

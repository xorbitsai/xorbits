.. _deployment_slurm:


SLURM deployment
=============


If you have access to a SLURM cluster, you can refer to the following guide to run an Xorbits job. Other HPC job schedulers like Torque or LSF are similar.
You are recommended to read the :ref:`cluster deployment <deployment_cluster>` first to know some basic knowledge of a Xorbits cluster.

Installation
------------

HPC clusters have shared storage among all compute nodes. You can use ``module``, ``conda`` or ``mamba`` to create a Python environment and use ``pip`` to install Xorbits on the shared storage, see :ref:`installation document <installation>` for reference. 

Walkthrough using Xorbits with SLURM
------------------------------------

On a SLURM cluster, you are required to interact with the compute resources via ``sbatch`` command and a SLURM script file declaring specific compute resources. You can get a list of hostnames after the compute resources are allocated.

In the SLURM script file, you will want to start a Xorbits cluster with multiple ``srun`` commands (tasks), and then execute your python script that connects to the Xorbits cluster. You need to first start a supervisor and then start the workers.

The below walkthrough will do the following:

1. Set the proper headers where you ask for resources from the SLURM cluster.

2. Load the proper environment/modules.

3. Fetch a list of available compute nodes and their hostnames.

4. Launch a supervisor process on one of the nodes (called the head node).

5. Launch worker processes on other worker nodes with the head node's address.

6. After the underlying Xorbits cluster is ready, submit the user-specified task.


Script Method
--------------

SLURM script file
~~~~~~~~~~~~~~~~~

In the SLURM script, you'll need to tell SLURM to allocate nodes for your Xorbits job. 
In this example, we ask for 4 nodes, and on each node, we've set ``--cpus-per-task=24`` and ``--ntasks-per-node=1`` which means we need 24 CPUs per node. 
Modify this setting according to your workload. Similarly, you can also specify the number of GPUs per node via ``--gpus=1``.
You need to change ``--partition`` to select the partition in your site. You can also add other optional flags to your ``sbatch`` directives.

.. code-block:: bash

    #!/bin/bash
    #SBATCH --job-name=xorbits
    #SBATCH --nodes=4
    #SBATCH --cpus-per-task=24
    #SBATCH --ntasks-per-node=1
    #SBATCH --partition=cpu24c
    #SBATCH --time=00:30:00


Load your environment
~~~~~~~~~~~~~~~~~~~~~~~~

You'll need to install Xorbits into a specific environment using ``conda`` or ``module``. 
In the SLURM script, you should load modules or your own conda environment. 
And on the compute nodes allocated, the environment will be switched to the one where Xorbits is installed.
In this case, we install Xorbits in a conda environment called ``df``.

.. code-block:: bash

    # Example: module load xorbits
    # Example: source activate my-env

    source activate df


Obtain the nodes
~~~~~~~~~~~~~~~~~~~~~~~~~~

Next, we'll want to obtain all the allocated compute nodes.

.. code-block:: bash

    # Getting the node names
    nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
    nodes_array=($nodes)

Now the ``nodes_array`` is the hostname list of all the nodes allocated for this job.

Start the supervisor
~~~~~~~~~~~~~~~~~~~~~~~

Choose the first node of the ``nodes_array`` as the head node. The head node is for supervisor and other nodes are for workers. 

.. code-block:: bash

    head_node=${nodes_array[0]}

After getting the head node hostname, we'll want to run the supervisor on the head node. 
We'll do this by using ``srun`` to start the supervisor on the head node. 
``xorbits-supervisor`` is the command line tool to start the supervisor.
You should specify the hostname, port, and the web port.
Note that you should sleep a few seconds as the supervisor need some time to start. Otherwise, worker nodes may not be able to connect to the supervisor.

.. code-block:: bash

    port=16380
    web_port=16379

    echo "Starting SUPERVISOR at ${head_node}"
    srun --nodes=1 --ntasks=1 -w "${head_node}" \
        xorbits-supervisor -H "${head_node}" -p "${port}" -w "${web_port}" &
    sleep 10

Start Workers
~~~~~~~~~~~~~~~~
The rest of the machines can be started as workers via command:

.. code-block:: bash

    # number of nodes other than the head node
    worker_num=$((SLURM_JOB_NUM_NODES - 1))

    for ((i = 1; i <= worker_num; i++)); do
        node_i=${nodes_array[$i]}
        port_i=$((port + i))
        
        echo "Starting WORKER $i at ${node_i}"
        srun --nodes=1 --ntasks=1 -w "${node_i}" \
            xorbits-worker -H "${node_i}"  -p "${port_i}" -s "${head_node}":"${port}" &
    done
    sleep 5

Connect to The Cluster
~~~~~~~~~~~~~~~~~~~~~~

Now, the Xorbits cluster is created, and ``address`` is the endpoint to connect.
You can connect to the supervisor and submit your Xorbits job.

.. code-block:: bash

    address=http://"${head_node}":"${web_port}"

    python -u test.py --endpoint "${address}"

The ``test.py`` is like the following: 

.. code-block:: python

    import argparse

    import xorbits
    import xorbits.numpy as np

    parser = argparse.ArgumentParser(description="test")
    parser.add_argument(
        "--endpoint",
        type=str,
        default="0.0.0.0",
        required=True,
    )

    args = parser.parse_args()

    xorbits.init(args.endpoint)
    print(np.random.rand(100, 100).mean())


Name this SLURM script file as ``xorbits_slurm.sh``. Submit the job via:

.. code-block:: bash

    sbatch xorbits_slurm.sh


Put all together
~~~~~~~~~~~~~~~~~~~~~~

The SLURM script looks like this:

.. code-block:: bash

    #!/bin/bash

    #SBATCH --job-name=xorbits
    #SBATCH --nodes=4
    #SBATCH --cpus-per-task=24
    #SBATCH --ntasks-per-node=1
    #SBATCH --partition=cpu24c
    #SBATCH --time=00:30:00

    source activate df

    ### Use the debug mode to see if the shell commands are correct.
    ### If you do not want the shell command logs, delete the following line.
    set -x

    # Getting the node names
    nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
    nodes_array=($nodes)

    head_node=${nodes_array[0]}
    port=16380
    web_port=16379

    echo "Starting SUPERVISOR at ${head_node}"
    srun --nodes=1 --ntasks=1 -w "${head_node}" \
        xorbits-supervisor -H "${head_node}" -p "${port}" -w "${web_port}" &
    sleep 10

    # number of nodes other than the head node
    worker_num=$((SLURM_JOB_NUM_NODES - 1))

    for ((i = 1; i <= worker_num; i++)); do
        node_i=${nodes_array[$i]}
        port_i=$((port + i))
        
        echo "Starting WORKER $i at ${node_i}"
        srun --nodes=1 --ntasks=1 -w "${node_i}" \
            xorbits-worker -H "${node_i}"  -p "${port_i}" -s "${head_node}":"${port}" &
    done
    sleep 5

    address=http://"${head_node}":"${web_port}"

    python -u test.py --endpoint "${address}"


Code Method
-----------
   

Initialization
~~~~~~~~~~~~~~

To create an instance of the `SLURMCluster` class, you can use the following parameters:

   - `job_name` (str, optional): Name of the Slurm job.
   - `num_nodes` (int, optional): Number of nodes in the Slurm cluster.
   - `partition_option` (str, optional): Request a specific partition for resource allocation.
   - `load_env` (str, optional): Conda Environment to load.
   - `output_path` (str, optional): Path for log output.
   - `error_path` (str, optional): Path for log errors.
   - `work_dir` (str, optional): Slurm's working directory, the default location for logs and results.
   - `time` (str, optional): Minimum time limit for job allocation.
   - `processes` (int, optional): Number of processes.
   - `cores` (int, optional): Number of cores.
   - `memory` (str, optional): Specify the real memory required per node. Default units are megabytes.
   - `account` (str, optional): Charge resources used by this job to the specified account.
   - `webport` (int, optional): Xorbits' web port.
   - `**kwargs`: Additional parameters that can be added using the Slurm interface.


.. code-block:: python

    from xorbits.deploy.slurm import SLURMCluster
    cluster = SLURMCluster(
          job_name="my_job",
          num_nodes=4,
          partition_option="compute",
          load_env="my_env",
          output_path="logs/output.log",
          error_path="logs/error.log",
          work_dir="/path/to/work_dir",
          time="1:00:00",
          processes=8,
          cores=2,
          memory="8G",
          account="my_account",
          webport=16379,
          custom_param1="value1",
          custom_param2="value2"
    )


.. note::
    Modify the parameters as needed for your specific use case.

Running the Job
~~~~~~~~~~~~~~~

To submit the job to SLURM, use the `run()` method. It will return the job's address.

.. code-block:: python

    address = cluster.run()

Getting Job Information
~~~~~~~~~~~~~~~~~~~~~~~~


- `get_job_id()`: This method extracts the job ID from the output of the `sbatch` command.

.. code-block:: python

    job_id = cluster.get_job_id()

- `cancel_job()`: This method cancels the job using the `scancel` command. A hook is designed so that while canceling the program, the Slurm task will also be canceled.

.. code-block:: python

    cluster.cancel_job(job_id)

- `update_head_node()`: This method retrieves the head node information from the SLURM job.

.. code-block:: python

    cluster.update_head_node()

- `get_job_address(retry_attempts=10, sleep_interval=30)`: This method retrieves the job address after deployment. It retries several times to get the job data.

.. code-block:: python

    job_address = cluster.get_job_address()


Example
~~~~~~~

Here's an example of how to use the `SLURMCluster` class::

.. code-block:: python

    import pandas as pd
    from xorbits.deploy.slurm import SLURMCluster

    test_cluster = SLURMCluster(
          job_name="xorbits",
          num_nodes=2,
          output_path="/shared_space/output.out",
          time="00:30:00",
      )
    address = test_cluster.run()
    xorbits.init(address)
    assert pd.Series([1, 2, 3]).sum() == 6


.. _deployment_slurm:

==================
SLURM deployment
==================

If you have access to a SLURM cluster, you can refer to the following guide to run an Xorbits job. Other HPC job schedulers like Torque or LSF are similar.
You are recommended to read the :ref:`cluster deployment <deployment_cluster>` first to know some basic knowledge of a Xorbits cluster.

Installation
------------

HPC clusters have shared storage among all compute nodes. You can use ``module``, ``conda`` or ``mamba`` to create a Python environment and use ``pip`` to install Xorbits on the shared storage, see :ref:`installation document <installation>` for reference. 

Walkthrough using Xorbits with SLURM
------------------------------------

On a SLURM cluster, you are required to interact with the compute resources via ``sbatch`` command and a SLURM script file declaring specific compute resources.

In the SLURM script file, you will want to start a Xorbits cluster with multiple ``srun`` commands (tasks), and then execute your python script that connects to the Xorbits cluster. You need to first start a supervisor and then start the workers.

The below walkthrough will do the following:

1. Set the proper headers where you ask for resources from the SLURM cluster.

2. Load the proper environment/modules.

3. Fetch a list of available compute nodes and their IP addresses.

4. Launch a supervisor process on one of the nodes (called the head node).

5. Launch worker processes on other worker nodes with the head node's address.

6. After the underlying Xorbits cluster is ready, submit the user-specified task.

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

Next, we'll want to obtain the head node and the IP address.

.. code-block:: bash

    # Getting the node names
    nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
    nodes_array=($nodes)

Now the ``nodes_array`` is the list of all the nodes allocated for this job.

Choose the first node of the ``nodes_array`` as the head node. 
The ``hostname --ip-address`` command will return the IP address of the specific node. 
Get the IP address of the head node:

.. code-block:: bash

    head_node=${nodes_array[0]}
    head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

    # if we detect a space character in the head node IP, we'll
    # convert it to an ipv4 address. This step is optional.
    if [[ "$head_node_ip" == *" "* ]]; then
        IFS=' ' read -ra ADDR <<<"$head_node_ip"
        if [[ ${#ADDR[0]} -gt 16 ]]; then
            head_node_ip=${ADDR[1]}
        else
            head_node_ip=${ADDR[0]}
        fi
        echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
    fi

Start the supervisor
~~~~~~~~~~~~~~~~~~~~~~~

After detecting the head node hostname and IP address, we'll want to run the supervisor on the head node. 
We'll do this by using ``srun`` to start the supervisor on the head node. 
``xorbits-supervisor`` is the command line tool to start the supervisor.
You should specify the IP, port, the web port.
Note that you should sleep a few seconds as the supervisor need some time to start. Otherwise, other worker nodes may not able to connect to the supervisor.

.. code-block:: bash

    port=16380
    web_port=16379

    echo "Starting HEAD at $head_node"
    srun --nodes=1 --ntasks=1 -w "$head_node" \
        xorbits-supervisor -H "$head_node_ip" -p "$port" -w "$web_port" &
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
        
        echo "Starting WORKER $i at $node_i"
        srun --nodes=1 --ntasks=1 -w "$node_i" \
            xorbits-worker -H "$node_i"  -p "$port_i" -s "$head_node_ip":"$port" &
    done
    sleep 5

Connect to The Cluster
~~~~~~~~~~~~~~~~~~~~~~

Now, the Xorbits cluster is created, and ``address`` is the endpoint to connect.
You can connect to the supervisor and submit your Xorbits job.

.. code-block:: bash

    address=http://"$head_node_ip":"$web_port"

    python -u test.py --endpoint "$address"

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
----------------

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
    head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

    # if we detect a space character in the head node IP, we'll
    # convert it to an ipv4 address. This step is optional.
    if [[ "$head_node_ip" == *" "* ]]; then
        IFS=' ' read -ra ADDR <<<"$head_node_ip"
        if [[ ${#ADDR[0]} -gt 16 ]]; then
            head_node_ip=${ADDR[1]}
        else
            head_node_ip=${ADDR[0]}
        fi
        echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
    fi

    port=16380
    web_port=16379

    echo "Starting HEAD at $head_node"
    srun --nodes=1 --ntasks=1 -w "$head_node" \
        xorbits-supervisor -H "$head_node_ip" -p "$port" -w "$web_port" &
    sleep 10

    # number of nodes other than the head node
    worker_num=$((SLURM_JOB_NUM_NODES - 1))

    for ((i = 1; i <= worker_num; i++)); do
        node_i=${nodes_array[$i]}
        port_i=$((port + i))
        
        echo "Starting WORKER $i at $node_i"
        srun --nodes=1 --ntasks=1 -w "$node_i" \
            xorbits-worker -H "$node_i"  -p "$port_i" -s "$head_node_ip":"$port" &
    done
    sleep 5

    address=http://"$head_node_ip":"$web_port"

    python -u test.py --endpoint "$address"


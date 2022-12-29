.. _deployment_cluster:

==================
Cluster deployment
==================

If you have some machines either on the cloud or not, and you want to deploy Xorbits via command lines,
you can refer to this documentation.

Installation
------------

First, ensure Xorbits is correctly installed on each machine, if not, see :ref:`installation document <installation>`.

Starting Supervisor
-------------------

Among the machines, pick at least one as the supervisor which ships with a web service as well,
starting supervisor via command:

.. code-block:: bash

    xorbits-supervisor -H <host_name> -p <supervisor_port> -w <web_port>

Or using ``python -m``:

.. code-block:: bash

    python -m xorbits.supervisor -H <host_name> -p <supervisor_port> -w <web_port>

Starting Workers
----------------

The rest of the machines can be started as workers via command:

.. code-block:: bash

    xorbits-worker -H <host_name> -p <worker_port> -s <supervisor_ip>:<supervisor_port>

Or using ``python -m``:

.. code-block:: bash

    python -m xorbits.worker -H <host_name> -p <worker_port> -s <supervisor_ip>:<supervisor_port>

Connecting to Created Cluster
-----------------------------

Now, you can connect to the supervisor from anywhere that can run Python code.

.. code-block:: python

    import xorbits
    xorbits.init("http://<supervisor_ip>:<supervisor_web_port>")


Replace the ``<supervisor_ip>`` with the supervisor host name that you just specified and
``<supervisor_web_port>`` with the supervisor web port.

Xorbits Web UI
--------------

You can open a web browser and type ``http://<supervisor_ip>:<supervisor_web_port>`` to open Xorbits Web UI to
look up resource usage of workers and execution progress of submitted tasks.

Command Line Options
-------------------

Common Options
~~~~~~~~~~~~~~

Common Command line options are listed below.

+------------------+----------------------------------------------------------------+
| Argument         | Description                                                    |
+==================+================================================================+
| ``-H``           | Service IP binding, ``0.0.0.0`` by default                     |
+------------------+----------------------------------------------------------------+
| ``-p``           | Port of the service. If absent, a randomized port will be used |
+------------------+----------------------------------------------------------------+
| ``-f``           | Path to service configuration file. Absent when use default    |
|                  | configuration.                                                 |
+------------------+----------------------------------------------------------------+
| ``-s``           | List of supervisor endpoints, separated by commas. Useful for  |
|                  | workers and webs to spot supervisors, or when you want to run  |
|                  | more than one supervisor                                       |
+------------------+----------------------------------------------------------------+
| ``--log-level``  | Log level, can be ``debug``, ``info``, ``warning``, ``error``  |
+------------------+----------------------------------------------------------------+
| ``--log-format`` | Log format, can be Python logging format                       |
+------------------+----------------------------------------------------------------+
| ``--log-conf``   | Python logging configuration file, ``logging.conf`` by default |
+------------------+----------------------------------------------------------------+
| ``--use-uvloop`` | Whether to use ``uvloop`` to accelerate, ``auto`` by default   |
+------------------+----------------------------------------------------------------+

Extra Options for Supervisors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+------------------+----------------------------------------------------------------+
| Argument         | Description                                                    |
+==================+================================================================+
| ``-w``           | Port of web service in supervisor                              |
+------------------+----------------------------------------------------------------+

Extra Options for Workers
~~~~~~~~~~~~~~~~~~~~~~~~~

+--------------------+----------------------------------------------------------------+
| Argument           | Description                                                    |
+====================+================================================================+
| ``--n-cpu``        | Number of CPU cores to use. If absent, the value will be       |
|                    | the available number of cores                                  |
+--------------------+----------------------------------------------------------------+
| ``--n-io-process`` | Number of IO processes for network operations. 1 by default    |
+--------------------+----------------------------------------------------------------+
| ``--cuda-devices`` | Index of CUDA devices to use. If not specified, all devices    |
|                    | will be used. Specifying an empty string will ignore all       |
|                    | devices                                                        |
+--------------------+----------------------------------------------------------------+

Example
-------

For instance, if you want to start a Xorbits cluster with two supervisors and two
workers, you can run commands below (memory and CPU tunings are omitted):

On Supervisor 1 (192.168.1.10):

.. code-block:: bash

    xorbits-supervisor -H 192.168.1.10 -p 7001 -w 7005 -s 192.168.1.10:7001,192.168.1.11:7002

On Supervisor 2 (192.168.1.11):

.. code-block:: bash

    xorbits-supervisor -H 192.168.1.11 -p 7002 -s 192.168.1.10:7001,192.168.1.11:7002

On Worker 1 (192.168.1.20):

.. code-block:: bash

    xorbits-worker -H 192.168.1.20 -p 7003 -s 192.168.1.10:7001,192.168.1.11:7002

On Worker 2 (192.168.1.21):

.. code-block:: bash

    xorbits-worker -H 192.168.1.21 -p 7004 -s 192.168.1.10:7001,192.168.1.11:7002

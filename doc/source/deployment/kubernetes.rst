.. _deployment_kubernetes:

=====================
Kubernetes deployment
=====================

Prerequisites
-------------
Install Xorbits on the machine where you plan to run the kubernetes deploy code.
Refer to :ref:`installation document <installation>`.

Currently we support Xorbits kubernetes cluster deployment based on Minikube.

Minikube
--------
Make sure Minikube is properly installed on your machine, create a cluster and enable the NGINX Ingress controller:

.. code-block:: bash

    $ minikube start
    $ minikube addons enable ingress

Follow `minikube documentation <https://kubernetes.io/docs/tasks/access-application-cluster/ingress-minikube/>`_ to verify whether ingress is enabled correctly.

For MacOS with docker driver, `docker-mac-net-connect <https://github.com/chipmk/docker-mac-net-connect>`_ is needed due to its `limitation <https://github.com/kubernetes/minikube/issues/7332>`_:

.. code-block:: bash

    # Install via Homebrew
    $ brew install chipmk/tap/docker-mac-net-connect

    # Run the service and register it to launch at boot
    # Notice that this command must be executed with sudo
    $ sudo brew services start chipmk/tap/docker-mac-net-connect

Then deploy Xorbits cluster, for example:

.. code-block:: python

    from kubernetes import config
    from xorbits.deploy.kubernetes import new_cluster
    cluster = new_cluster(config.new_client_from_config(), worker_cpu=1, worker_mem='4g')


When a log of the form ``Xorbits endpoint http://<ingress_service_ip>:80 is ready!`` appears,
you can access the web page of your xorbits cluster through the endpoint in the log.

To verify the cluster:

.. code-block:: python

    import xorbits.pandas as pd
    print(pd.DataFrame({'a': [1,2,3,4]}).sum())


API Parameters
-----------

``new_cluster`` Parameters
~~~~~~~~~~~~~~

+---------------------+------------------------------+------------------------------------------------------+-----------------------------------+
| Parameter           | Type                         | Description                                          | Required / Default value          |
+=====================+==============================+======================================================+===================================+
| kube_api_client     | kubernetes.client.ApiClient  | Kubernetes API client                                | required                          |
+---------------------+------------------------------+------------------------------------------------------+-----------------------------------+
| image               | str                          | Docker image to use                                  | xprobe/xorbits:<xorbits version>  |
+---------------------+------------------------------+------------------------------------------------------+-----------------------------------+
| supervisor_num      | int                          | Number of supervisors in the cluster                 | 1                                 |
+---------------------+------------------------------+------------------------------------------------------+-----------------------------------+
| supervisor_cpu      | int                          | Number of CPUs for every supervisor                  | 1                                 |
+---------------------+------------------------------+------------------------------------------------------+-----------------------------------+
| supervisor_mem      | str                          | Memory size for every supervisor                     | 4G                                |
+---------------------+------------------------------+------------------------------------------------------+-----------------------------------+
| worker_num          | int                          | Number of workers in the cluster                     | 1                                 |
+---------------------+------------------------------+------------------------------------------------------+-----------------------------------+
| worker_cpu          | int                          | Number of CPUs for every worker                      | required                          |
+---------------------+------------------------------+------------------------------------------------------+-----------------------------------+
| worker_mem          | str                          | Memory size for every worker                         | required                          |
+---------------------+------------------------------+------------------------------------------------------+-----------------------------------+
| worker_spill_paths  | List[str]                    | Spill paths for worker pods on host                  | None                              |
+---------------------+------------------------------+------------------------------------------------------+-----------------------------------+
| worker_cache_mem    | str                          | Size or ratio of cache memory for every worker       | None                              |
+---------------------+------------------------------+------------------------------------------------------+-----------------------------------+
| min_worker_num      | int                          | Minimal ready workers                                | None (equal to worker_num)        |
+---------------------+------------------------------+------------------------------------------------------+-----------------------------------+
| timeout             | int                          | Timeout seconds when creating clusters               | None (never timeout)              |
+---------------------+------------------------------+------------------------------------------------------+-----------------------------------+
| cluster_type        | str                          | K8s Cluster type, ``minikube`` or ``eks`` supported  | minikube                          |
+---------------------+------------------------------+------------------------------------------------------+-----------------------------------+

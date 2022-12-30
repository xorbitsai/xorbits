.. _deployment_kubernetes:

=====================
Kubernetes deployment
=====================

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
you can access the web page of your Xorbits cluster through the endpoint in the log.

To verify the cluster:

.. code-block:: python

    import xorbits.pandas as pd
    print(pd.DataFrame({'a': [1,2,3,4]}).sum())


Amazon Elastic Kubernetes Service (Amazon EKS)
----------------------------------------------
Firstly, make sure you have an EKS cluster and it can access `our Dockerhub <https://hub.docker.com/repository/docker/xprobe/xorbits>`_ to pull the Xorbits image.

Secondly, install the `AWS Load Balancer Controller <https://docs.aws.amazon.com/eks/latest/userguide/aws-load-balancer-controller.html>`_.

Then, deploy Xorbits cluster, for example:

.. code-block:: python

    from kubernetes import config
    from xorbits.deploy.kubernetes import new_cluster
    cluster = new_cluster(config.new_client_from_config(), worker_cpu=1, worker_mem='4g', cluster_type='eks')


Note that the option *cluster_type=\'eks\'* cannot be ignored.
When a log of the form ``Xorbits endpoint http://<ingress_service_ip>:80 is ready!`` appears,
you can access the web page of your Xorbits cluster through the endpoint in the log.

To verify the cluster:

.. code-block:: python

    import xorbits.numpy as np
    a = np.ones((100, 100), chunk_size=30) * 2 * 1 + 1
    b = np.ones((100, 100), chunk_size=20) * 2 * 1 + 1
    c = (a * b * 2 + 1).sum()
    print(c)

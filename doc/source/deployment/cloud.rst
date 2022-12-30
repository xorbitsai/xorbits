.. _deployment_cloud:

================
Cloud deployment
================

Prerequisites
-------------
Currently we support deploying Xorbits to Amazon EKS cluster.

Install Xorbits on the machine where you can control your Amazon EKS cluster.
Refer to :ref:`installation document <installation>`.

Amazon EKS
----------
Firstly, make sure you have an EKS cluster and it can access `our Dockerhub <https://hub.docker.com/repository/docker/xprobe/xorbits>`_ to pull the Xorbits image.

Secondly, install the `AWS Load Balancer Controller <https://docs.aws.amazon.com/eks/latest/userguide/aws-load-balancer-controller.html>`_.

Then, deploy Xorbits cluster, for example:

.. code-block:: python

    from kubernetes import config
    from xorbits.deploy.kubernetes import new_cluster
    cluster = new_cluster(config.new_client_from_config(), worker_cpu=1, worker_mem='4g', cluster_type='eks')


Note that the option ``cluster_type='eks'`` cannot be ignored.
When a log of the form ``Xorbits endpoint http://<ingress_service_ip>:80 is ready!`` appears,
you can access the web page of your xorbits cluster through the endpoint in the log.

To verify the cluster:

.. code-block:: python

    import xorbits.numpy as np
    a = np.ones((100, 100), chunk_size=30) * 2 * 1 + 1
    b = np.ones((100, 100), chunk_size=20) * 2 * 1 + 1
    c = (a * b * 2 + 1).sum()
    print(c)

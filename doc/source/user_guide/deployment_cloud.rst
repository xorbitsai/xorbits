.. _deployment_cloud:

================
Cloud deployment
================

Prerequisites
-------------
Currently, we support deploying Xorbits on an existing Amazon EKS cluster.

Install Xorbits on the machine for Amazon EKS cluster management.
Refer to :ref:`installation document <installation>`.

Amazon EKS
----------
Firstly, make sure your EKS cluster can access `our Dockerhub <https://hub.docker.com/repository/docker/xprobe/xorbits>`_ to pull the Xorbits image.

Secondly, install the `AWS Load Balancer Controller <https://docs.aws.amazon.com/eks/latest/userguide/aws-load-balancer-controller.html>`_.

Then, deploy Xorbits cluster, for example:

.. code-block:: python

    from kubernetes import config
    from xorbits.deploy.kubernetes import new_cluster
    cluster = new_cluster(config.new_client_from_config(), worker_cpu=1, worker_mem='4g')


Note that the option ``cluster_type`` of the function ``new_cluster`` has default value ``auto``, which means that
Xorbits will detect the ``kubectl`` context automatically. Please make sure ``kubectl`` is using the correct EKS context.

You should be able to see ``Xorbits endpoint http://<ingress_service_ip>:80 is ready!`` soon, and
you can access the web UI of your Xorbits cluster using the endpoint.

Refer :ref:`Kubernetes deployment <deployment_image>` to deploy Xorbits with your own image.

Refer :ref:`Install Python Packages <deployment_install>` to install additional python packages for the Xorbits supervisors and workers.

``new_cluster`` api refers to :meth:`xorbits.deploy.kubernetes.client.new_cluster`.

To verify the cluster:

.. code-block:: python

    import xorbits.numpy as np
    a = np.ones((100, 100), chunk_size=30) * 2 * 1 + 1
    b = np.ones((100, 100), chunk_size=20) * 2 * 1 + 1
    c = (a * b * 2 + 1).sum()
    print(c)


External Storage
----------

Great news! Xorbits starts to support separation of storage and compute. Start to enjoy better scalability, flexibility, and performance it brings.
Check out the last section ``JuiceFS on Kubernetes`` on the page :ref:`Kubernetes Deployment <deployment_kubernetes>`.
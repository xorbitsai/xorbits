.. _deployment_kubernetes:

=====================
Kubernetes deployment
=====================

Prerequisites
-------------
Install Xorbits on the machine where you plan to run the kubernetes deploy code.
Refer to :ref:`installation document <installation>`.

Kubernetes
----------
Make sure a K8s cluster is properly installed on your machine(s), and enable the `ingress service <https://kubernetes.io/docs/concepts/services-networking/ingress/>`_.

For example, if you use Minikube, start a cluster and enable ingress like this:

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


Please make sure ``kubectl`` uses your kubernetes cluster context.

You should be able to see ``Xorbits endpoint http://<ingress_service_ip>:80 is ready!`` soon, and
you can access the web UI of your Xorbits cluster using the endpoint.

``new_cluster`` api refers to :meth:`xorbits.deploy.kubernetes.client.new_cluster`.

To verify the cluster:

.. code-block:: python

    import xorbits.pandas as pd
    print(pd.DataFrame({'a': [1,2,3,4]}).sum())


.. _deployment_image:

Docker Image
------------
By default, the image tagged by ``xprobe/xorbits:<xorbits version>`` on `our Dockerhub <https://hub.docker.com/repository/docker/xprobe/xorbits>`_
is used in the kubernetes deployment. Each released version of Xorbits has its image, distinguished by the ``<xorbits version>``.

.. note::
   Since v0.1.2, each release image of xorbits supports python ``3.7``, ``3.8``, ``3.9`` and ``3.10``,
   with ``-py<python_version>`` as the suffix of the image tag.

   For example, ``xprobe/xorbits:v0.1.2-py3.10`` means the image is built on python ``3.10``.

   By default, the image tagged by ``xprobe/xorbits:<xorbits version>`` still exists, and it is built on python ``3.9``.


If you need to build an image from source, the related Dockerfiles exists at `this position <https://github.com/xprobe-inc/xorbits/tree/main/python/xorbits/deploy/docker>`_ for reference.
You can follow the `Docker document <https://docs.docker.com/engine/reference/commandline/build/>`_ to build your own Xorbits image.

After you build your own image, push it to a image repository accessible by your K8s cluster, e.g. your own DockerHub namespace.

Finally, specify your own image during the deployment process through the ``image`` option of the :meth:`xorbits.deploy.kubernetes.client.new_cluster` api.


.. _deployment_install:

Install Python Packages
-----------------------
Refer `DockerFile <https://github.com/xprobe-inc/xorbits/blob/main/python/xorbits/deploy/docker/Dockerfile.base>`_ for the python packages included in the Xorbits image.
If you want to install additional python packages in your Xorbits K8s cluster, use ``pip`` and ``conda`` options of the :meth:`xorbits.deploy.kubernetes.client.new_cluster` api.

Please make sure your K8s cluster can access the corresponding `channel of conda <https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/channels.html>`_ or `PyPi <https://pypi.org/>`_, when using ``pip`` and ``conda`` options.

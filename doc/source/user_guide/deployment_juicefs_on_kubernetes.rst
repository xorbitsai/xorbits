.. _deployment_juicefs_on_kubernetes:

=====================
JuiceFS on Kubernetes
=====================

Xorbits is able to utilize `JuiceFS <https://juicefs.com/en/>`_ as one of the storage backend.

Prerequisites
-------------
Xorbits
~~~~~~~~~~~~~~
Install Xorbits on the machine where you plan to deploy Kubernetes with JuiceFS.
Refer to :ref:`installation document <installation>`.

Metadata Storage
~~~~~~~~~~~~~~
JuiceFS decouples data and metadata. Many databases are supported. See `How to Set Up Metadata Engine <https://juicefs.com/docs/community/databases_for_metadata>`_ and choose an appropriate metadata storage.
In our example here, we select ``Redis`` as our metadata storage. Follow `Configuring Redis using a ConfigMap <https://kubernetes.io/docs/tutorials/configuration/configure-redis-using-configmap/>`_ and create a pod inside default namespace.
You should set its maxmemory as 50mb since 2mb in the example is too small.

.. code-block:: bash

    $ kubectl get po redis
    NAME    READY   STATUS    RESTARTS    AGE
    redis   1/1     Running   0           6d6h


Kubernetes
----------
Follow :ref:`kubernetes deployment document <deployment_kubernetes>` to initialize a K8s cluster on your machine.

Then, deploy Xorbits cluster, for example:

.. code-block:: python

    from kubernetes import config
    from xorbits.deploy.kubernetes
    import new_cluster

    cluster = new_cluster(config.new_client_from_config(), worker_num=1, worker_cpu=1, worker_mem='1g', supervisor_cpu=1, supervisor_mem='1g',external_storage='juicefs', metadata_url='redis://10.244.0.45:6379/1', bucket='/var')

Currently, only juicefs is supported as one of our storage backend. When you want to switch from shared memory to JuiceFS, You must specify ``external_storage='juicefs'`` explicitly when you initialize a new cluster.

You must explicitly specify connection URL ``metadata_url``, in our case ``redis://10.244.0.45:6379/1``.

Specify bucket URL with ``bucket`` or use its default value ``/var`` if you do not want to change the directory for bucket. See `Set Up Object Storage <https://juicefs.com/docs/community/how_to_setup_object_storage/>`_ to set up different object storage.

After several minutes, you would see ``Xorbits endpoint http://<ingress_service_ip>:80`` is ready!

Verify the cluster by running a simple task.

.. code-block:: python

    import xorbits
    xorbits.init('http://<ingress_service_ip>:80')
    import xorbits.pandas as pd
    pd.DataFrame({'a': [1,2,3,4]}).sum()

If the cluster is working, the output should be 10.

Verify the storage
----------
Currently, we mount JuiceFS storage data in ``/juicefs-data``.

Execute an interactive shell (bash) inside a pod which belongs to the Xorbits namespace to check if data is stored in ``/juicefs-data``.

You should see a similar hex string like 9c3e069a-70d9-4874-bad6-d608979746a0, meaning that data inside JuiceFS is successfully mounted!

.. code-block:: bash

    $ kubectl get namespaces
    NAME                                          STATUS   AGE
    default                                       Active   38d
    kube-node-lease                               Active   38d
    kube-public                                   Active   38d
    kube-system                                   Active   38d
    xorbits-ns-cc53e351744f4394b20180a0dafd8b91   Active   4m5s

    $ kubectl get po -n xorbits-ns-cc53e351744f4394b20180a0dafd8b91
    NAME                                 READY   STATUS             RESTARTS   AGE
    xorbitssupervisor-84754bf5f4-dcstd   0/1     Running            0          80s
    xorbitsworker-5b9b976767-sfpkk       0/1     Running            0          80s

    $ kubectl exec -it xorbitssupervisor-84754bf5f4-dcstd -n xorbits-ns-cc53e351744f4394b20180a0dafd8b91 -- /bin/bash
    $ cd ..
    $ cd data
    $ ls
    9c3e069a-70d9-4874-bad6-d608979746a0
    $ cat 9c3e069a-70d9-4874-bad6-d608979746a0
..
You should see the serialized output of the simple task which may not be human-readable. It should contain ``pandas``, meaning that it matches our simple task!

Manage the Xorbits cluster & Debug
----------

You can get Xorbits namespace, check the status of Xorbits pods, and check Xorbits UI by following `Detailed tutorial: Deploying and Running Xorbits on Amazon EKS. <https://zhuanlan.zhihu.com/p/610955102>`_.
If everything works fine, you can easily scale up and down the storage resources by adding or deleting pods inside the namespace.
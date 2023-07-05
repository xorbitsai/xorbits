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
   Since ``v0.1.2``, each release image of xorbits supports python ``3.7``, ``3.8``, ``3.9`` and ``3.10``,
   with ``-py<python_version>`` as the suffix of the image tag.

   For example, ``xprobe/xorbits:v0.1.2-py3.10`` means the image is built on python ``3.10``.

   By default, the image tagged by ``xprobe/xorbits:<xorbits version>`` still exists, and it is built on python ``3.9``.

   Since ``v0.2.0``, Xorbits automatically selects the deployment image according to your local python version by default.
   For example, if your local python version is ``3.9``, Xorbits uses the image tagged by ``xprobe/xorbits:<xorbits version>-py3.9`` during deployment.

   Since ``v0.2.1``, Xorbits image has removed support for python ``3.7`` and introduced support for python ``3.11``.
   The image tagged by ``xprobe/xorbits:<xorbits version>`` is built on python ``3.10``.


If you need to build an image from source, the related Dockerfiles exists at `this position <https://github.com/xorbitsai/xorbits/tree/main/python/xorbits/deploy/docker>`_ for reference.
You can follow the `Docker document <https://docs.docker.com/engine/reference/commandline/build/>`_ to build your own Xorbits image.

After you build your own image, push it to a image repository accessible by your K8s cluster, e.g. your own DockerHub namespace.

Finally, specify your own image during the deployment process through the ``image`` option of the :meth:`xorbits.deploy.kubernetes.client.new_cluster` api.


.. _deployment_install:

Install Python Packages
-----------------------
Refer `DockerFile <https://github.com/xorbitsai/xorbits/blob/main/python/xorbits/deploy/docker/Dockerfile.base>`_ for the python packages included in the Xorbits image.
If you want to install additional python packages in your Xorbits K8s cluster, use ``pip`` and ``conda`` options of the :meth:`xorbits.deploy.kubernetes.client.new_cluster` api.

Please make sure your K8s cluster can access the corresponding `channel of conda <https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/channels.html>`_ or `PyPi <https://pypi.org/>`_, when using ``pip`` and ``conda`` options.


.. _deployment_juicefs_on_k8s:

JuiceFS on Kubernetes
---------------------
Xorbits is now able to integrate `JuiceFS <https://juicefs.com/en/>`_, a distributed POSIX file system that can be easily integrated with Kubernetes to provide persistent storage, as one of the storage backend.

Prerequisites
~~~~~~~~~~~~~

Xorbits
+++++++
Install Xorbits on the machine where you plan to deploy Kubernetes with JuiceFS. Refer to :ref:`installation document <installation>`.

Metadata Storage
++++++++++++++++
JuiceFS decouples data and metadata. Many databases are supported. See `How to Set Up Metadata Engine <https://juicefs.com/docs/community/databases_for_metadata>`_ and choose an appropriate metadata storage.

In our example here, we select ``Redis`` as our metadata storage.

Follow `Configuring Redis using a ConfigMap <https://kubernetes.io/docs/tutorials/configuration/configure-redis-using-configmap/>`_ and create a pod inside default namespace.

You should set its maxmemory as 50mb since 2mb in the example is too small.

Make sure redis pod is running:

.. code-block:: bash

    $ kubectl get po redis
    NAME    READY   STATUS    RESTARTS    AGE
    redis   1/1     Running   0           6d6h

..


Check redis pod's IP address. In this example, IP for redis is 172.17.0.8.

.. code-block:: bash

    $ kubectl get po redis -o custom-columns=IP:.status.podIP --no-headers
    172.17.0.8

..

You can also check how much CPU and memory resources redis pod gets by running

.. code-block:: bash

    $ kubectl get po redis

..

and check the corresponding fields.

Kubernetes
~~~~~~~~~~

.. code-block:: bash

    $ pip install kubernetes

..

Follow previous Kubernetes section to initialize a K8s cluster on your machine.

Install ``kubectl``, a command-line tool for interacting with Kubernetes clusters and verify its installation.

.. code-block:: bash

    $ kubectl version --client
    WARNING: This version information is deprecated and will be replaced with the output from kubectl version --short.  Use --output=yaml|json to get the full version.
    Client Version: version.Info{Major:"1", Minor:"25", GitVersion:"v1.25.4", GitCommit:"872a965c6c6526caa949f0c6ac028ef7aff3fb78", GitTreeState:"clean", BuildDate:"2022-11-09T13:36:36Z", GoVersion:"go1.19.3", Compiler:"gc", Platform:"linux/amd64"}
    Kustomize Version: v4.5.7

..

JuiceFS Installation
~~~~~~~~~~~~~~~~~~~~


We will walk you through the process of installing JuiceFS on a Kubernetes cluster, enabling you to leverage its features and benefits. There are three ways to use JuiceFS on K8S  `Use JuiceFS on Kubernetes <https://juicefs.com/docs/zh/community/how_to_use_on_kubernetes>`_.


But our implementation in k8s must rely on CSI since CSI provides better portability, enhanced isolation, and more advanced features.

Reference Page: `JuiceFS CSI Driver <https://juicefs.com/docs/csi/getting_started/>`_


JuiceFS CSI Driver
++++++++++++++++++

Installation with Helm
^^^^^^^^^^^^^^^^^^^^^^
Reference Page: `JuiceFS Installation with Helm <https://juicefs.com/docs/csi/getting_started#helm-1>`_

Firstly, `Install Helm <https://helm.sh/docs/intro/install/>`_

Secondly, download the Helm chart for JuiceFS CSI Driver

.. code-block:: bash

    $ helm repo add juicefs https://juicedata.github.io/charts/
    $ helm repo update
    $ helm fetch --untar juicefs/juicefs-csi-driver

..

.. code-block:: bash

    $ cd juicefs-csi-driver
    # Installation configurations is included in values.yaml, review this file and modify to your needs
    $ cat values.yaml

..

You should be careful with limits and requests of cpu and memory. Change according to your system settings. Here we give you the minimal configuration.


.. code-block:: bash

  resources:
    limits:
      cpu: 100m
      memory: 50Mi
    requests:
      cpu: 100m
      memory: 50Mi

..


Thirdly, execute below commands to deploy JuiceFS CSI Driver:

.. code-block:: bash

    $ helm install juicefs-csi-driver juicefs/juicefs-csi-driver -n kube-system -f ./values.yaml`

..

Fourthly, verify installation

.. code-block:: bash

    $ kubectl -n kube-system get pods -l app.kubernetes.io/name=juicefs-csi-driver
    NAME                       READY   STATUS    RESTARTS   AGE
    juicefs-csi-controller-0   3/3     Running   0          22m
    juicefs-csi-node-v9tzb     3/3     Running   0          14m

..

If you want to delete JuiceFS CSI Driver in the future, you can run:

.. code-block:: bash

    $ helm list -n kube-system
    NAME              	NAMESPACE  	REVISION	UPDATED                                	STATUS  	CHART                    	APP VERSION
    juicefs-csi-driver	kube-system	1       	2023-05-31 03:12:16.717087425 +0000 UTC	deployed	juicefs-csi-driver-0.15.1	0.19.0

    $ helm uninstall juicefs-csi-driver -n kube-system

..


Create and use PV
^^^^^^^^^^^^^^^^^

**If you want to directly use JuiceFS on K8S, you can skip this Create and use PV section because in Xorbits, new_cluster function would create secret, pv, and pvc for you.**

If you want to understand how the mounting works and the meaning of each parameter in the configurations, you can walk through this section.

JuiceFS leverages persistent volumes to store data.

Reference Page: `Create and use PV <https://juicefs.com/docs/csi/guide/pv>`_

We would create several YAML files. Validate their formats on `YAML validator <https://www.yamllint.com/>`_ before usage.

First, create Kubernetes Secret:

.. code-block:: bash

    $ vim secret.yaml

..

Write the following into the yaml file:

.. code-block:: bash

    apiVersion: v1
    kind: Secret
    metadata:
      name: juicefs-secret
    type: Opaque
    stringData:
      name: jfs
      metaurl: redis://172.17.0.8:6379/1 # Replace with your own metadata storage URL
      storage: file # Check out full supported list on `Set Up Object Storage <https://juicefs.com/docs/community/how_to_setup_object_storage/>`_.
      bucket: /var # Bucket URL. Read `Set Up Object Storage <https://juicefs.com/docs/community/how_to_setup_object_storage/>`_ to learn how to setup different object storage.

..

In our case, we do not need access-key and secret-key. Add if you need object storage credentials.

Secondly, create Persistent Volume and Persistent Volume Claim with static provisioning

Read `Usage <https://juicefs.com/docs/csi/introduction#usage>`_ to learn the difference between static and dynamic provisioning.

.. code-block:: bash

    $ vim static_provisioning.yaml

..

Write the following into the yaml file:

.. code-block:: bash

    apiVersion: v1
    kind: PersistentVolume
    metadata:
      name: juicefs-pv
      labels:
        juicefs-name: juicefs-fs # Works as a match-label for selector
    spec:
      # For now, JuiceFS CSI Driver doesn't support setting storage capacity for static PV. Fill in any valid string is fine.
      capacity:
        storage: 10Pi
      volumeMode: Filesystem
      mountOptions: ["subdir=/data/subdir"]  # Mount in sub directory to achieve data isolation. See https://juicefs.com/docs/csi/guide/pv/#create-storage-class for more references.
      accessModes:
        - ReadWriteMany # accessModes is restricted to ReadWriteMany because it's the most suitable mode for our system. See https://kubernetes.io/docs/concepts/storage/persistent-volumes/#access-modes for more reference.
      persistentVolumeReclaimPolicy: Retain # persistentVolumeReclaimPolicy is restricted to Retain for Static provisioning. See https://juicefs.com/docs/csi/guide/resource-optimization/#reclaim-policy for more references.
      csi:
        # A CSIDriver named csi.juicefs.com is created during installation
        driver: csi.juicefs.com
        # volumeHandle needs to be unique within the cluster, simply using the PV name is recommended
        volumeHandle: juicefs-pv
        fsType: juicefs
        # Reference the volume credentials (Secret) created in previous step
        # If you need to use different credentials, or even use different JuiceFS volumes, you'll need to create different volume credentials
        nodePublishSecretRef:
          name: juicefs-secret
          namespace: xorbits-ns-cc53e351744f4394b20180a0dafd8b91 # change the namespace to your Xorbits namespace
    ---
    apiVersion: v1
    kind: PersistentVolumeClaim
    metadata:
      name: juicefs-pvc
      namespace: xorbits-ns-cc53e351744f4394b20180a0dafd8b91 # change the namespace to your Xorbits namespace
    spec:
      accessModes:
        - ReadWriteMany
      volumeMode: Filesystem
      # Must use an empty string as storageClassName
      # Meaning that this PV will not use any StorageClass, instead will use the PV specified by selector
      storageClassName: ""
      # For now, JuiceFS CSI Driver doesn't support setting storage capacity for static PV. Fill in any valid string that's lower than the PV capacity.
      resources:
        requests:
          storage: 10Pi
      selector:
        matchLabels:
          juicefs-name: juicefs-fs

..

Thirdly, apply Secret, PV, and PVC to your namespace and verify:

Create your namespace (or Xorbits namespace) and run the following:

.. code-block:: bash

    $ kubectl apply -f secret.yaml -n {your_namespace}
    $ kubectl apply -f static_provisioning -n {your_namespace}

..

.. code-block:: bash

    $ kubectl get pv
    NAME          CAPACITY   ACCESS MODES   RECLAIM POLICY   STATUS   CLAIM                                                       STORAGECLASS   REASON   AGE
    juicefs-pv    10Pi       RWX            Retain           Bound    xorbits-ns-cc53e351744f4394b20180a0dafd8b91/juicefs-pvc                             17h

    $ kubectl get pvc --all-namespaces
    NAMESPACE                                     NAME           STATUS   VOLUME        CAPACITY   ACCESS MODES   STORAGECLASS   AGE
    xorbits-ns-cc53e351744f4394b20180a0dafd8b91   juicefs-pvc    Bound    juicefs-pv    10Pi       RWX                           17h

..

Fourthly, create a pod

.. code-block:: bash

    $ vim pod.yaml

..

Write the following into the yaml file:

.. code-block:: bash

    apiVersion: v1
    kind: Pod
    metadata:
      name: juicefs-app
      namespace: xorbits-ns-cc53e351744f4394b20180a0dafd8b91 # Replace with your namespace
    spec:
      containers:
      - args:
        - -c
        - while true; do echo $(date -u) >> /juicefs-data/out.txt; sleep 5; done
        command:
        - /bin/sh
        image: centos
        name: app
        volumeMounts:
        - mountPath: /juicefs-data
          name: data
        resources:
          requests:
            cpu: 10m
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: juicefs-pvc

..

After pod is up and running, you'll see out.txt being created by the container inside the JuiceFS mount point ``/juicefs-data``.

Congratulations! You have successfully set up JuiceFS on Kubernetes by yourself.

Deploy Cluster
~~~~~~~~~~~~~~

Before you deploy, go to our `DockerHub <https://hub.docker.com/r/xprobe/xorbits>`_ and get the latest docker image.

Check the image by running:

.. code-block:: bash

    $ docker image ls
    REPOSITORY                              TAG                                      IMAGE ID       CREATED         SIZE
    xprobe/xorbits                          v0.4.0-py3.10                            4166e1b0676d   5 days ago      3.38GB

..

Deploy Xorbits cluster, for example:

.. code-block:: python

    from kubernetes import config
    from xorbits.deploy.kubernetes import new_cluster


    cluster = new_cluster(config.new_client_from_config(), worker_num=1, worker_cpu=1, worker_mem='1g', supervisor_cpu=1, supervisor_mem='1g', image='xprobe/xorbits:v0.4.0-py3.10', external_storage='juicefs',external_storage_config={"metadata_url": "redis://172.17.0.8:6379/1","bucket": "/var", "mountPath": "/juicefs-data"},)

..


Currently, only juicefs is supported as one of our storage backend. When you want to switch from shared memory to JuiceFS, You must specify ``external_storage='juicefs'`` explicitly when you initialize a new cluster.

JuiceFS has corresponding parameters which you should specify in a dictionary named ``external_storage_config``.

You must explicitly specify connection URL ``metadata_url``, in our case ``redis://172.17.0.8:6379/1``. 172.17.0.8 is the IP address of the Redis server, and 6379 is the default port number on which the Redis server is listening. 1 represents the Redis database number.

Specify bucket URL with ``bucket`` or use its default value ``/var`` if you do not want to change the directory for bucket. See `Set Up Object Storage <https://juicefs.com/docs/community/how_to_setup_object_storage/>`_ to set up different object storage.

Specify mount path with ``mountPath`` or use its default value ``/juicefs-data``.

After several minutes, you would see ``Xorbits endpoint http://<ingress_service_ip>:80`` is ready!

Verify the cluster by running a simple task.

.. code-block:: python

    import xorbits
    xorbits.init('http://<ingress_service_ip>:80')
    import xorbits.pandas as pd
    pd.DataFrame({'a': [1,2,3,4]}).sum()

..


If the cluster is working, the output should be 10.

Verify the storage
~~~~~~~~~~~~~~~~~~

In our example, we mount JuiceFS storage data in ``/juicefs-data``, which is also the default path.

Firstly, get the namespace that starts with ``xorbits`` and get its pods.

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
    xorbitssupervisor-84754bf5f4-dcstd   1/1     Running            0          80s
    xorbitsworker-5b9b976767-sfpkk       1/1     Running            0          80s

..

Then, execute an interactive shell (bash) inside a pod which belongs to the Xorbits namespace. You can verify either supervisor pod or worker pod, or both.

.. code-block:: bash

    $ kubectl exec -it xorbitssupervisor-84754bf5f4-dcstd -n xorbits-ns-cc53e351744f4394b20180a0dafd8b91 -- /bin/bash

..

Check if data is stored in ``/juicefs-data``.

You should see a similar hex string like 9c3e069a-70d9-4874-bad6-d608979746a0, meaning that data inside JuiceFS is successfully mounted!

.. code-block:: bash

    $ cd ..
    $ cd juicefs-data
    $ ls
    9c3e069a-70d9-4874-bad6-d608979746a0
    $ cat 9c3e069a-70d9-4874-bad6-d608979746a0

..

You should see the serialized output of the simple task which may not be human-readable. It should contain ``pandas``, meaning that it matches our simple task!

Manage the Xorbits cluster & Debug
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can get Xorbits namespace, check the status of Xorbits pods, and check Xorbits UI by following `Detailed tutorial: Deploying and Running Xorbits on Amazon EKS. <https://zhuanlan.zhihu.com/p/610955102>`_.

If everything works fine, now you can easily scale up and down the storage resources by adding or deleting pods inside the namespace.

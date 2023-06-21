.. _deployment_juicefs_on_kubernetes:

=====================
JuiceFS on Kubernetes
=====================

Xorbits is able to utilize `JuiceFS <https://juicefs.com/en/>`_, distributed POSIX file system that can be easily integrated with Kubernetes to provide persistent storage, as one of the storage backend.

Prerequisites
-------------
Xorbits
~~~~~~~~~~~~~~
Install Xorbits on the machine where you plan to deploy Kubernetes with JuiceFS.
Refer to :ref:`installation document <installation>`.

Metadata Storage
~~~~~~~~~~~~~~
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
----------
Follow :ref:`kubernetes deployment document <deployment_kubernetes>` to initialize a K8s cluster on your machine.

Install ``kubectl``, a command-line tool for interacting with Kubernetes clusters and verify its installation.

.. code-block:: bash

    $ kubectl version --client
    WARNING: This version information is deprecated and will be replaced with the output from kubectl version --short.  Use --output=yaml|json to get the full version.
    Client Version: version.Info{Major:"1", Minor:"25", GitVersion:"v1.25.4", GitCommit:"872a965c6c6526caa949f0c6ac028ef7aff3fb78", GitTreeState:"clean", BuildDate:"2022-11-09T13:36:36Z", GoVersion:"go1.19.3", Compiler:"gc", Platform:"linux/amd64"}
    Kustomize Version: v4.5.7

..


JuiceFS Installation
----------

We will still walk you through the process of installing JuiceFS on a Kubernetes cluster, enabling you to leverage its features and benefits.

There are three ways to use JuiceFS on K8S  `Use JuiceFS on Kubernetes <https://juicefs.com/docs/zh/community/how_to_use_on_kubernetes>`_.

But our implementation in k8s must rely on CSI since CSI provides better portability, enhanced isolation, and more advanced features.

Reference Page: `JuiceFS CSI Driver <https://juicefs.com/docs/csi/getting_started/>`_

JuiceFS CSI Driver
~~~~~~~~~~~~~~~~~~~~~~~

Installation with Helm
++++++++++++++++++++++++++

Reference Page: `JuiceFS Installation with Helm <https://juicefs.com/docs/csi/getting_started#helm-1>`_

1. `Install Helm <https://helm.sh/docs/intro/install/>`_

2. Download the Helm chart for JuiceFS CSI Driver

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

You should be careful with limits and requests of cpu and memory. Change according to your system settings.

.. code-block:: bash

  resources:
    limits:
      cpu: 100m
      memory: 50Mi
    requests:
      cpu: 100m
      memory: 50Mi

..


3. Execute below commands to deploy JuiceFS CSI Driver:

.. code-block:: bash

    $ helm repo add juicefs https://juicedata.github.io/charts/
    $ helm repo update
    $ helm install juicefs-csi-driver juicefs/juicefs-csi-driver -n kube-system -f ./values.yaml`

..

4. Verify installation

.. code-block:: bash

    $ kubectl -n kube-system get pods -l app.kubernetes.io/name=juicefs-csi-driver
    NAME                       READY   STATUS    RESTARTS   AGE
    juicefs-csi-controller-0   3/3     Running   0          22m
    juicefs-csi-node-v9tzb     3/3     Running   0          14m

..

Create and use PV
++++++++++++++++++++++++++

You can skip this ``Create and use PV`` section because in Xorbits, the ``new_cluster`` function would create secret, pv, and pvc for you.

You can still walk through this section as it would give you a better understanding of each parameter in the configurations.

JuiceFS leverages persistent volumes to store data.

Reference Page: `Create and use pv <https://juicefs.com/docs/csi/guide/pv>`_

We would create several YAML files. Validate their formats on `YAML validator <https://www.yamllint.com/>` before usage.

1. Create Kubernetes Secret:

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

2. Create Persistent Volume and Persistent Volume Claim with static provisioning

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
        juicefs-name: ten-pb-fs # Works as a match-label for selector
    spec:
      # For now, JuiceFS CSI Driver doesn't support setting storage capacity for static PV. Fill in any valid string is fine.
      capacity:
        storage: 10Pi
      volumeMode: Filesystem
      mountOptions: ["subdir=/data/subdir"],  # Mount in sub directory to achieve data isolation. See https://juicefs.com/docs/csi/guide/pv/#create-storage-class for more references.
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
          namespace: default # change the namespace to our Xorbits or your own namespace
    ---
    apiVersion: v1
    kind: PersistentVolumeClaim
    metadata:
      name: juicefs-pvc
      namespace: default # change the namespace to our Xorbits or your own namespace
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
          juicefs-name: ten-pb-fs

..

3. Apply Secret, PV, and PVC to your namespace and verify:

Create your namespace (or Xorbits namespace) and run the following:

.. code-block:: bash

    $ kubectl apply -f secret.yaml -n {your_namespace}
    $ kubectl apply -f static_provisioning -n {your_namespace}

..

.. code-block:: bash

    $ kubectl get pv
    NAME          CAPACITY   ACCESS MODES   RECLAIM POLICY   STATUS   CLAIM                  STORAGECLASS   REASON   AGE
    juicefs-pv    10Pi       RWX            Retain           Bound    testns/juicefs-pvc                             17h
    juicefs-pv1   10Pi       RWX            Retain           Bound    testns1/juicefs-pvc1                           17h

    $ kubectl get pvc --all-namespaces
    NAMESPACE                                     NAME           STATUS   VOLUME        CAPACITY   ACCESS MODES   STORAGECLASS   AGE
    testns                                        juicefs-pvc    Bound    juicefs-pv    10Pi       RWX                           17h
    testns1                                       juicefs-pvc1   Bound    juicefs-pv1   10Pi       RWX                           17h

..

4. Create a pod

.. code-block:: bash

    $ vim pod.yaml

..

Write the following into the yaml file:

.. code-block:: bash

    apiVersion: v1
    kind: Pod
    metadata:
      name: juicefs-app
      namespace: default # Replace with your namespace
    spec:
      containers:
      - args:
        - -c
        - while true; do echo $(date -u) >> /data/out.txt; sleep 5; done
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

After pod is up and running, you'll see out.txt being created by the container inside the JuiceFS mount point.

Congratulations! You have successfully set up JuiceFS on Kubernetes by yourself.


Deploy Cluster
----------

Deploy Xorbits cluster, for example:

.. code-block:: python

    from kubernetes import config
    from xorbits.deploy.kubernetes
    import new_cluster

    cluster = new_cluster(config.new_client_from_config(), worker_num=1, worker_cpu=1, worker_mem='1g', supervisor_cpu=1, supervisor_mem='1g',external_storage='juicefs', metadata_url='redis://10.244.0.45:6379/1', bucket='/var')

..


Currently, only juicefs is supported as one of our storage backend. When you want to switch from shared memory to JuiceFS, You must specify ``external_storage='juicefs'`` explicitly when you initialize a new cluster.

You must explicitly specify connection URL ``metadata_url``, in our case ``redis://172.17.0.8:6379/1``. 172.17.0.8 is the IP address of the Redis server, and 6379 is the default port number on which the Redis server is listening. 1 represents the Redis database number.

Specify bucket URL with ``bucket`` or use its default value ``/var`` if you do not want to change the directory for bucket. See `Set Up Object Storage <https://juicefs.com/docs/community/how_to_setup_object_storage/>`_ to set up different object storage.

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
----------
Currently, we mount JuiceFS storage data in ``/juicefs-data``.

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
    xorbitssupervisor-84754bf5f4-dcstd   0/1     Running            0          80s
    xorbitsworker-5b9b976767-sfpkk       0/1     Running            0          80s

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
----------

You can get Xorbits namespace, check the status of Xorbits pods, and check Xorbits UI by following `Detailed tutorial: Deploying and Running Xorbits on Amazon EKS. <https://zhuanlan.zhihu.com/p/610955102>`_.
If everything works fine, now you can easily scale up and down the storage resources by adding or deleting pods inside the namespace.
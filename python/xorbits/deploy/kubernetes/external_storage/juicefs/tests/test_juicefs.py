# Copyright 2022-2023 XProbe Inc.
# derived from copyright 1999-2021 Alibaba Group Holding Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import subprocess as sp

import pytest

from ....tests.test_kubernetes import (
    _start_kube_cluster,
    k8s,
    kube_available,
    simple_job,
)
from ..config import PersistentVolumeClaimConfig, PersistentVolumeConfig, SecretConfig

juicefs_available = "juicefs" in sp.getoutput(
    "kubectl get pods --all-namespaces -o wide | grep 'juicefs-csi'"
)


@pytest.mark.skipif(not kube_available, reason="Cannot run without kubernetes")
@pytest.mark.skipif(not juicefs_available, reason="Cannot run without juicefs")
def test_juicefs_secret_object():
    secret_config = SecretConfig(
        metadata_url="redis://172.17.0.5:6379/1", bucket="/var"
    )
    secret_config_dict = secret_config.build()
    assert secret_config_dict["metadata"]["name"] == "juicefs-secret"
    assert secret_config_dict["stringData"]["metaurl"] == "redis://172.17.0.5:6379/1"


@pytest.mark.skipif(not kube_available, reason="Cannot run without kubernetes")
@pytest.mark.skipif(not juicefs_available, reason="Cannot run without juicefs")
def test_juicefs_persistent_volume_object():
    persistent_volume_config = PersistentVolumeConfig(
        namespace="xorbits-ns-07ba172af1e94b9e84352e1aa790fefe"
    )
    persistent_volume_config_dict = persistent_volume_config.build()
    assert (
        persistent_volume_config_dict["spec"]["csi"]["nodePublishSecretRef"][
            "namespace"
        ]
        == "xorbits-ns-07ba172af1e94b9e84352e1aa790fefe"
    )


@pytest.mark.skipif(not kube_available, reason="Cannot run without kubernetes")
@pytest.mark.skipif(not juicefs_available, reason="Cannot run without juicefs")
def test_juicefs_persistent_volume_claim_object():
    persistent_volume_claim_config = PersistentVolumeClaimConfig(
        namespace="xorbits-ns-07ba172af1e94b9e84352e1aa790fefe"
    )
    persistent_volume_claim_config_dict = persistent_volume_claim_config.build()
    assert (
        persistent_volume_claim_config_dict["metadata"]["namespace"]
        == "xorbits-ns-07ba172af1e94b9e84352e1aa790fefe"
    )


@pytest.mark.skipif(not kube_available, reason="Cannot run without kubernetes")
@pytest.mark.skipif(not juicefs_available, reason="Cannot run without juicefs")
@pytest.mark.asyncio
async def test_external_storage_juicefs():
    redis_ip = sp.getoutput(
        "echo $(kubectl get po redis -o wide) | grep -o '[0-9]*\\.[0-9]*\\.[0-9]*\\.[0-9]*'"
    )
    with _start_kube_cluster(
        supervisor_cpu=0.1,
        supervisor_mem="1G",
        worker_num=1,
        worker_cpu=0.1,
        worker_mem="1G",
        external_storage="juicefs",
        external_storage_config={
            "metadata_url": "redis://" + redis_ip + ":6379/1",
            "bucket": "/var",
            "mountPath": "/juicefs-data",
        },
        use_local_image=True,
    ) as cluster_client:
        import xorbits.pandas as pd

        a = pd.DataFrame({"col": [1, 2, 3]}).sum()
        print(a)

        from kubernetes.stream import stream

        api_client = k8s.config.new_client_from_config()
        kube_api = k8s.client.CoreV1Api(api_client)
        pods_name_list = sp.getoutput(
            "kubectl get pods -o name --no-headers=true -n {ns}".format(
                ns=cluster_client.namespace
            )
        ).split("\n")
        pods_name_list = list(map(lambda x: x[x.index("/") + 1 :], pods_name_list))
        for pod in pods_name_list:
            exec_cmd = ["/bin/sh", "-c", "ls /juicefs-data"]
            resp = stream(
                kube_api.connect_get_namespaced_pod_exec,
                name=pod,
                namespace=cluster_client.namespace,
                command=exec_cmd,
                stderr=True,
                stdin=False,
                stdout=True,
                tty=False,
            )
            assert len(resp.split(" ")) == 1


@pytest.mark.skipif(not kube_available, reason="Cannot run without kubernetes")
@pytest.mark.skipif(not juicefs_available, reason="Cannot run without juicefs")
@pytest.mark.asyncio
async def test_external_storage_juicefs_missing_metadata_url():
    with pytest.raises(
        ValueError,
        match="For external storage JuiceFS, you must specify the metadata url for its metadata storage in external storage config, for example external_storage_config={'metadata_url': 'redis://172.17.0.5:6379/1',}.",
    ):
        with _start_kube_cluster(
            supervisor_cpu=0.1,
            supervisor_mem="1G",
            worker_num=1,
            worker_cpu=0.1,
            worker_mem="1G",
            external_storage="juicefs",
            use_local_image=True,
        ):
            simple_job()


@pytest.mark.skipif(not kube_available, reason="Cannot run without kubernetes")
@pytest.mark.skipif(not juicefs_available, reason="Cannot run without juicefs")
@pytest.mark.asyncio
async def test_external_storage_invalid_option():
    with pytest.raises(
        ValueError,
        match="Currently, only juicefs is supported as one of our storage backend.",
    ):
        with _start_kube_cluster(
            supervisor_cpu=0.1,
            supervisor_mem="1G",
            worker_num=1,
            worker_cpu=0.1,
            worker_mem="1G",
            external_storage="Ceph",
            use_local_image=True,
            external_storage_config={
                "metadata_url": "fakedb://fakeip",
                "bucket": "/var",
                "mountPath": "/juicefs-data",
            },
        ):
            simple_job()

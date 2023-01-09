# Copyright 2022-2023 XProbe Inc.
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
import os
import tempfile

import pytest

from ...._mars.utils import lazy_import
from ..client import KubernetesCluster
from ..config import (
    EmptyDirVolumeConfig,
    HostPathVolumeConfig,
    IngressConfig,
    NamespaceConfig,
    RoleBindingConfig,
    RoleConfig,
    ServiceConfig,
    XorbitsSupervisorsConfig,
    XorbitsWorkersConfig,
)

kubernetes = lazy_import("kubernetes")


@pytest.mark.skipif(kubernetes is None, reason="Cannot run without kubernetes")
def test_simple_objects():
    ns_config_dict = NamespaceConfig("ns_name").build()
    assert ns_config_dict["metadata"]["name"] == "ns_name"

    role_config_dict = RoleConfig(
        "xorbits-pod-reader", "ns_name", "", "pods", "get,watch,list"
    ).build()
    assert role_config_dict["metadata"]["name"] == "xorbits-pod-reader"
    assert "get" in role_config_dict["rules"][0]["verbs"]

    role_binding_config_dict = RoleBindingConfig(
        "xorbits-pod-reader-binding", "ns_name", "xorbits-pod-reader", "default"
    ).build()
    assert role_binding_config_dict["metadata"]["name"] == "xorbits-pod-reader-binding"

    service_config_dict = ServiceConfig(
        "xorbits-test-service",
        "NodePort",
        "xorbits/service-type=xorbits-supervisor",
        7103,
        7103,
    ).build()
    assert service_config_dict["metadata"]["name"] == "xorbits-test-service"


@pytest.mark.skipif(kubernetes is None, reason="Cannot run without kubernetes")
def test_supervisor_object():
    supervisor_config = XorbitsSupervisorsConfig(
        1, cpu=2, memory="10g", limit_resources=False, modules=["xorbits.test_mod"]
    )
    supervisor_config.add_simple_envs(dict(TEST_ENV="test_val"))

    supervisor_config_dict = supervisor_config.build()
    assert supervisor_config_dict["metadata"]["name"] == "xorbitssupervisor"
    assert supervisor_config_dict["spec"]["replicas"] == 1

    container_dict = supervisor_config_dict["spec"]["template"]["spec"]["containers"][0]
    assert int(container_dict["resources"]["requests"]["memory"]) == 10 * 1024**3

    container_envs = dict((p["name"], p) for p in container_dict["env"])
    assert container_envs["TEST_ENV"]["value"] == "test_val"
    assert container_envs["MKL_NUM_THREADS"]["value"] == "2"
    assert container_envs["MARS_CPU_TOTAL"]["value"] == "2"
    assert int(container_envs["MARS_MEMORY_TOTAL"]["value"]) == 10 * 1024**3
    assert container_envs["MARS_LOAD_MODULES"]["value"] == "xorbits.test_mod"

    supervisor_config = XorbitsSupervisorsConfig(
        1,
        cpu=2,
        memory="10g",
        limit_resources=False,
        modules=["xorbits.test_mod"],
        service_port=11111,
        web_port=11112,
        kind="Pod",
    )
    supervisor = supervisor_config.build()
    assert (
        supervisor["spec"]["selector"]["xorbits/service-type"]
        == XorbitsSupervisorsConfig.rc_name
    )
    assert supervisor_config.api_version == "v1"


@pytest.mark.skipif(kubernetes is None, reason="Cannot run without kubernetes")
def test_worker_object():
    worker_config_dict = XorbitsWorkersConfig(
        4,
        cpu=2,
        memory=10 * 1024**3,
        limit_resources=True,
        memory_limit_ratio=2,
        spill_volumes=[
            "/tmp/spill_vol",
            EmptyDirVolumeConfig("empty-dir", "/tmp/empty"),
        ],
        worker_cache_mem="20%",
        min_cache_mem="10%",
        modules="xorbits.test_mod",
        mount_shm=True,
    ).build()
    assert worker_config_dict["metadata"]["name"] == "xorbitsworker"
    assert worker_config_dict["spec"]["replicas"] == 4

    container_dict = worker_config_dict["spec"]["template"]["spec"]["containers"][0]
    assert int(container_dict["resources"]["requests"]["memory"]) == 10 * 1024**3
    assert int(container_dict["resources"]["limits"]["memory"]) == 20 * 1024**3

    container_envs = dict((p["name"], p) for p in container_dict["env"])
    assert container_envs["MKL_NUM_THREADS"]["value"] == "2"
    assert container_envs["MARS_CPU_TOTAL"]["value"] == "2"
    assert int(container_envs["MARS_MEMORY_TOTAL"]["value"]) == 10 * 1024**3
    assert container_envs["MARS_LOAD_MODULES"]["value"] == "xorbits.test_mod"
    assert set(container_envs["MARS_SPILL_DIRS"]["value"].split(":")) == {
        "/tmp/empty",
        "/mnt/hostpath0",
    }
    assert container_envs["MARS_CACHE_MEM_SIZE"]["value"] == "20%"

    volume_list = worker_config_dict["spec"]["template"]["spec"]["volumes"]
    volume_envs = dict((v["name"], v) for v in volume_list)
    assert "empty-dir" in volume_envs
    assert volume_envs["host-path-vol-0"]["hostPath"]["path"] == "/tmp/spill_vol"

    volume_mounts = dict((v["name"], v) for v in container_dict["volumeMounts"])
    assert volume_mounts["empty-dir"]["mountPath"] == "/tmp/empty"
    assert volume_mounts["host-path-vol-0"]["mountPath"] == "/mnt/hostpath0"

    worker_config_dict = XorbitsWorkersConfig(
        4,
        cpu=2,
        memory=10 * 1024**3,
        limit_resources=False,
        spill_volumes=[
            "/tmp/spill_vol",
            EmptyDirVolumeConfig("empty-dir", "/tmp/empty"),
        ],
        modules="xorbits.test_mod",
        mount_shm=False,
    ).build()

    volume_list = worker_config_dict["spec"]["template"]["spec"]["volumes"]
    assert "shm-volume" not in volume_list

    container_dict = worker_config_dict["spec"]["template"]["spec"]["containers"][0]
    volume_mounts = dict((v["name"], v) for v in container_dict["volumeMounts"])
    assert "shm-volume" not in volume_mounts

    worker_config_dict = XorbitsWorkersConfig(
        4,
        cpu=2,
        memory=None,
        limit_resources=True,
        supervisor_web_port=11111,
        service_port=11112,
        service_name="worker",
        volumes=[HostPathVolumeConfig("vol", "/tmp/test1", "/tmp/test2")],
    ).build()
    container_dict = worker_config_dict["spec"]["template"]["spec"]["containers"][0]
    container_envs = dict((p["name"], p) for p in container_dict["env"])
    assert "MARS_CACHE_MEM_SIZE" not in container_envs
    assert container_envs["MARS_K8S_SERVICE_NAME"]["value"] == "worker"
    assert container_envs["MARS_K8S_SERVICE_PORT"]["value"] == str(11112)


@pytest.mark.skipif(kubernetes is None, reason="Cannot run without kubernetes")
def test_ingress_object():
    from kubernetes.client import V1Ingress

    ingress_config = IngressConfig(
        namespace="ns",
        name="ingress",
        service_name="ing-service",
        service_port=7777,
        cluster_type="eks",
    )
    ingress = ingress_config.build()
    assert type(ingress) is V1Ingress
    assert ingress.spec.ingress_class_name == "alb"


@pytest.mark.skipif(kubernetes is None, reason="Cannot run without kubernetes")
def test_cluster_type():
    from kubernetes import config

    context = {"cluster": "minikube"}
    res = KubernetesCluster._get_k8s_context(context)
    assert res == "kubernetes"

    context = {"context": {"user": "yyyawsxxx"}}
    res = KubernetesCluster._get_k8s_context(context)
    assert res == "eks"

    context = {"name": "arn:eks:abcd"}
    res = KubernetesCluster._get_k8s_context(context)
    assert res == "eks"

    res = KubernetesCluster._get_cluster_type("kubernetes")
    assert res == "kubernetes"

    res = KubernetesCluster._get_cluster_type("auto")
    expected = KubernetesCluster._get_k8s_context(config.list_kube_config_contexts()[1])
    assert res == expected


@pytest.mark.skipif(kubernetes is None, reason="Cannot run without kubernetes")
def test_install_command():
    _, file_path = tempfile.mkstemp()

    with open(file_path, "w") as f:
        f.write("package1\n")
        f.write("package2")

    rc = XorbitsSupervisorsConfig(replicas=1, pip=file_path)
    content = rc.get_install_content(rc._pip)
    assert "package1\npackage2" in content
    assert not ("conda" in content)

    rc = XorbitsWorkersConfig(replicas=1, conda=file_path)
    content = rc.get_install_content(rc._conda)
    assert "package1\npackage2" in content
    assert not ("pip" in content)
    commands = rc.build_container_command()
    assert "conda_yaml" in commands[0]

    with pytest.raises(ValueError):
        rc = XorbitsWorkersConfig(replicas=1, pip="/aa/bb.txt")
        _ = rc.get_install_content(rc._pip)

    with pytest.raises(ValueError):
        rc = XorbitsWorkersConfig(replicas=1, conda="/aa/bb.yaml")
        _ = rc.get_install_content(rc._conda)

    rc = XorbitsWorkersConfig(replicas=1, pip=["p1", "p2"])
    content = rc.get_install_content(rc._pip)
    assert "p1\np2" in content

    rc = XorbitsWorkersConfig(replicas=1, conda=["p1==1.0.0", "p2"])
    content = rc.get_install_content(rc._conda)
    assert "p1==1.0.0\np2" in content

    rc = XorbitsWorkersConfig(replicas=1, conda=["p1", "p2"], pip=file_path)
    content1 = rc.get_install_content(rc._pip)
    content2 = rc.get_install_content(rc._conda)
    assert "package1\npackage2" in content1
    assert "p1\np2" in content2

    commands = rc.build_container_command()
    assert len(commands) == 1
    assert "install.sh" in commands[0]
    assert "entrypoint.sh" in commands[0]
    assert "conda_default" in commands[0]

    try:
        os.remove(file_path)
    except:
        pass

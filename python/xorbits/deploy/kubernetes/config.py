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

import abc
import functools
import math
import os
import re
from typing import Any, Dict, List, Optional, Union

from ... import __version__
from ..._mars.utils import calc_size_by_str, parse_readable_size
from ...compat._constants import COMPATIBLE_DEPS
from ...utils import get_local_package_version, get_local_py_version
from ._constants import SERVICE_PID_FILE

try:
    from kubernetes.client import ApiClient
except ImportError:  # pragma: no cover
    ApiClient = None

DEFAULT_WORKER_CACHE_MEM = "40%"


def _remove_nones(cfg) -> Dict:
    return dict((k, v) for k, v in cfg.items() if v is not None)


_kube_api_mapping = {
    "v1": "CoreV1Api",
    "apps/v1": "AppsV1Api",
    "rbac.authorization.k8s.io/v1": "RbacAuthorizationV1Api",
}


@functools.lru_cache(10)
def _get_k8s_api(api_version: str, k8s_api_client: ApiClient):
    from kubernetes import client as kube_client

    return getattr(kube_client, _kube_api_mapping[api_version])(k8s_api_client)


@functools.lru_cache(10)
def _camel_to_underline(name: str) -> str:
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


class KubeConfig(abc.ABC):
    api_version = "v1"

    def create_namespaced(self, k8s_api_client: ApiClient, namespace: str):
        api = _get_k8s_api(self.api_version, k8s_api_client)
        config = self.build()
        method_name = f'create_namespaced_{_camel_to_underline(config["kind"])}'
        return getattr(api, method_name)(namespace, config)

    @abc.abstractmethod
    def build(self):
        """Build config dict of the object"""


class RoleConfig(KubeConfig):
    """
    Configuration builder for Kubernetes RBAC roles
    """

    api_version = "rbac.authorization.k8s.io/v1"

    def __init__(
        self, name: str, namespace: str, api_groups: str, resources: str, verbs: str
    ):
        self._name = name
        self._namespace = namespace
        self._api_groups = api_groups.split(",")
        self._resources = resources.split(",")
        self._verbs = verbs.split(",")

    def build(self):
        return {
            "kind": "Role",
            "metadata": {"name": self._name, "namespace": self._namespace},
            "rules": [
                {
                    "apiGroups": self._api_groups,
                    "resources": self._resources,
                    "verbs": self._verbs,
                },
                {
                    "apiGroups": ["apps"],
                    "resources": ["deployments", "deployments/scale"],
                    "verbs": ["get", "watch", "list", "patch"],
                },
            ],
        }


class RoleBindingConfig(KubeConfig):
    """
    Configuration builder for Kubernetes RBAC role bindings
    """

    api_version = "rbac.authorization.k8s.io/v1"

    def __init__(
        self, name: str, namespace: str, role_name: str, service_account_name: str
    ):
        self._name = name
        self._namespace = namespace
        self._role_name = role_name
        self._service_account_name = service_account_name

    def build(self):
        return {
            "kind": "RoleBinding",
            "metadata": {"name": self._name, "namespace": self._namespace},
            "roleRef": {
                "apiGroup": "rbac.authorization.k8s.io",
                "kind": "Role",
                "name": self._role_name,
            },
            "subjects": [
                {
                    "kind": "ServiceAccount",
                    "name": self._service_account_name,
                    "namespace": self._namespace,
                }
            ],
        }


class NamespaceConfig(KubeConfig):
    """
    Configuration builder for Kubernetes namespaces
    """

    def __init__(self, name: str):
        self._name = name

    def build(self):
        return {
            "kind": "Namespace",
            "metadata": {
                "name": self._name,
                "labels": {
                    "name": self._name,
                },
            },
        }


class ServiceConfig(KubeConfig):
    """
    Configuration builder for Kubernetes services
    """

    def __init__(
        self,
        name: str,
        service_type: Optional[str],
        selector: Union[str, Dict],
        port: int,
        target_port: Optional[int] = None,
        protocol: Optional[str] = None,
    ):
        self._name = name
        self._type = service_type
        self._protocol = protocol or "TCP"
        self._selector = selector
        self._port = port
        self._target_port = target_port

    def build(self):
        return {
            "kind": "Service",
            "metadata": {
                "name": self._name,
                "labels": {
                    "xorbits/service-name": self._name,
                },
            },
            "spec": _remove_nones(
                {
                    "type": self._type,
                    "selector": self._selector,
                    "ports": [
                        _remove_nones(
                            {
                                "protocol": self._protocol,
                                "port": self._port,
                                "targetPort": self._target_port,
                            }
                        ),
                    ],
                }
            ),
        }


class IngressConfig(KubeConfig):
    api_version = "networking.k8s.io/v1"

    def __init__(
        self,
        namespace: str,
        name: str,
        service_name: str,
        service_port: int,
        cluster_type: str,
    ):
        self._namespace = namespace
        self._name = name
        self._service_name = service_name
        self._service_port = service_port
        self._cluster_type = cluster_type

    def build(self):
        from kubernetes import client

        annotations = None
        ingress_cls_name = None
        if self._cluster_type == "eks":
            annotations = {
                "alb.ingress.kubernetes.io/scheme": "internet-facing",
                "alb.ingress.kubernetes.io/target-type": "ip",
            }
            ingress_cls_name = "alb"

        body = client.V1Ingress(
            api_version="networking.k8s.io/v1",
            kind="Ingress",
            metadata=client.V1ObjectMeta(
                name=self._name, namespace=self._namespace, annotations=annotations
            ),
            spec=client.V1IngressSpec(
                rules=[
                    client.V1IngressRule(
                        http=client.V1HTTPIngressRuleValue(
                            paths=[
                                client.V1HTTPIngressPath(
                                    path="/",
                                    path_type="Prefix",
                                    backend=client.V1IngressBackend(
                                        service=client.V1IngressServiceBackend(
                                            name=self._service_name,
                                            port=client.V1ServiceBackendPort(
                                                number=self._service_port
                                            ),
                                        )
                                    ),
                                )
                            ]
                        )
                    )
                ],
                ingress_class_name=ingress_cls_name,
            ),
        )

        return body


class ResourceConfig:
    """
    Configuration builder for Kubernetes computation resources
    """

    def __init__(self, cpu, memory):
        self._cpu = cpu
        self._memory, ratio = (
            parse_readable_size(memory) if memory is not None else (None, False)
        )
        assert not ratio

    @property
    def cpu(self) -> int:
        return self._cpu

    @property
    def memory(self) -> float:
        return self._memory

    def build(self):
        return _remove_nones(
            {
                "cpu": f"{int(self._cpu * 1000)}m" if self._cpu else None,
                "memory": str(int(self._memory)) if self._memory else None,
            }
        )


class PortConfig:
    """
    Configuration builder for Kubernetes ports definition for containers
    """

    def __init__(self, container_port: Union[int, str]):
        self._container_port = int(container_port)

    def build(self):
        return {
            "containerPort": self._container_port,
        }


class VolumeConfig(abc.ABC):
    """
    Base configuration builder for Kubernetes volumes
    """

    def __init__(self, name: str, mount_path: str):
        self.name = name
        self.mount_path = mount_path

    @abc.abstractmethod
    def build(self):
        """Build volume config"""

    def build_mount(self):
        return {
            "name": self.name,
            "mountPath": self.mount_path,
        }


class HostPathVolumeConfig(VolumeConfig):
    """
    Configuration builder for Kubernetes host volumes
    """

    def __init__(
        self,
        name: str,
        mount_path: str,
        host_path: str,
        volume_type: Optional[str] = None,
    ):
        super().__init__(name, mount_path)
        self._host_path = host_path
        self._volume_type = volume_type or "DirectoryOrCreate"

    def build(self):
        return {
            "name": self.name,
            "hostPath": {"path": self._host_path, "type": self._volume_type},
        }


class EmptyDirVolumeConfig(VolumeConfig):
    """
    Configuration builder for Kubernetes empty-dir volumes
    """

    def __init__(
        self,
        name: str,
        mount_path: str,
        use_memory: bool = True,
        size_limit: Optional[int] = None,
    ):
        super().__init__(name, mount_path)
        self._medium = "Memory" if use_memory else None
        self._size_limit = size_limit

    def build(self):
        result = {"name": self.name, "emptyDir": {}}
        if self._medium:
            result["emptyDir"]["medium"] = self._medium
        if self._size_limit:
            result["emptyDir"]["sizeLimit"] = str(int(self._size_limit))
        return result


class ContainerEnvConfig:
    """
    Configuration builder for Kubernetes container environments
    """

    def __init__(
        self, name: str, value: Optional[Any] = None, field_path: Optional[str] = None
    ):
        self._name = name
        self._value = value
        self._field_path = field_path

    def build(self):
        result = dict(name=self._name)
        if self._value is not None:
            result["value"] = str(self._value)
        elif self._field_path is not None:  # pragma: no branch
            result["valueFrom"] = {"fieldRef": {"fieldPath": self._field_path}}
        return result


class ProbeConfig:
    """
    Base configuration builder for Kubernetes liveness and readiness probes
    """

    def __init__(
        self,
        initial_delay: int = 5,
        period: int = 5,
        timeout: Optional[int] = None,
        success_thresh: Optional[int] = None,
        failure_thresh: Optional[int] = None,
    ):
        self._initial_delay = initial_delay
        self._period = period
        self._timeout = timeout
        self._success_thresh = success_thresh
        self._failure_thresh = failure_thresh

    def build(self):
        return _remove_nones(
            {
                "initialDelaySeconds": self._initial_delay,
                "periodSeconds": self._period,
                "timeoutSeconds": self._timeout,
                "successThreshold": self._success_thresh,
                "failureThreshold": self._failure_thresh,
            }
        )


class TcpSocketProbeConfig(ProbeConfig):
    """
    Configuration builder for TCP liveness and readiness probes
    """

    def __init__(self, port: int, **kwargs):
        super().__init__(**kwargs)
        self._port = port

    def build(self):
        ret = super().build()
        ret["tcpSocket"] = {"port": self._port}
        return ret


class CommandExecProbeConfig(ProbeConfig):
    """
    Configuration builder for command probe
    """

    def __init__(self, commands: List[str], **kwargs):
        super().__init__(**kwargs)
        self._commands = commands

    def build(self):
        ret = super().build()
        ret["exec"] = {"command": self._commands}
        return ret


class ReplicationConfig(KubeConfig):
    """
    Base configuration builder for Kubernetes replication controllers
    """

    _default_kind = "Deployment"

    def __init__(
        self,
        name: Optional[str],
        image: str,
        replicas: int,
        resource_request: Optional["ResourceConfig"] = None,
        resource_limit: Optional["ResourceConfig"] = None,
        liveness_probe: Optional["ProbeConfig"] = None,
        startup_probe: Optional["ProbeConfig"] = None,
        pre_stop_command: Optional[List[str]] = None,
        kind: Optional[str] = None,
        external_storage: Optional[str] = None,
        external_storage_config: Optional[dict] = None,
        **kwargs,
    ):
        self._name = name
        self._kind = kind or self._default_kind
        self._image = image
        self._replicas = replicas
        self._ports: List[PortConfig] = []
        self._volumes: List[VolumeConfig] = []
        self._envs: Dict[str, ContainerEnvConfig] = dict()
        self._labels: Dict[str, Any] = dict()

        self.add_default_envs()

        self._resource_request = resource_request
        self._resource_limit = resource_limit

        self._liveness_probe = liveness_probe
        self._startup_probe = startup_probe

        self._pre_stop_command = pre_stop_command
        self._external_storage = external_storage
        self._external_storage_config = external_storage_config
        self._use_local_image = kwargs.pop("use_local_image", False)
        self._pip: Optional[Union[str, List[str]]] = kwargs.pop("pip", None)
        self._conda: Optional[Union[str, List[str]]] = kwargs.pop("conda", None)

    @property
    def api_version(self):
        return "apps/v1" if self._kind in ("Deployment", "ReplicaSet") else "v1"

    def add_env(self, name, value=None, field_path=None):
        self._envs[name] = ContainerEnvConfig(name, value=value, field_path=field_path)

    def remove_env(self, name):  # pragma: no cover
        self._envs.pop(name, None)

    def add_simple_envs(self, envs):
        for k, v in envs.items() or ():
            self.add_env(k, v)

    def add_labels(self, labels):
        self._labels.update(labels)

    def add_port(self, container_port):
        self._ports.append(PortConfig(container_port))

    def add_default_envs(self):
        pass  # pragma: no cover

    def add_volume(self, vol):
        self._volumes.append(vol)

    def config_init_containers(self) -> List[Dict]:
        return []

    @staticmethod
    def get_install_content(source: Union[str, List[str]]) -> str:
        if isinstance(source, str):
            if not os.path.exists(source):
                raise ValueError(f"Cannot find the file {source} for installation.")
            with open(source, "r") as f:
                content = f.read()
        else:
            content = "\n".join(source)
        return content

    @staticmethod
    def get_compatible_packages() -> List[str]:
        deps = []
        for dep in COMPATIBLE_DEPS:
            version = get_local_package_version(dep)
            if version is not None:
                deps.append(f"{dep}=={version}")
        return deps

    @abc.abstractmethod
    def build_container_command(self) -> List:
        """Output container command"""
        cmd = ""
        # All install contents must be wrapped in double quotes to ensure the correctness
        # when passing to the shell script.
        # At the same time, each command is followed by a semicolon to separate the individual commands.
        deps = self.get_compatible_packages()
        if deps:
            # Install for consistent packages must be at the top of all commands.
            # This does not affect user behavior.
            cmd += (
                f'/srv/install.sh pip_compatible "{self.get_install_content(deps)}" ; '
            )

        if self._pip is not None:
            cmd += f'/srv/install.sh pip "{self.get_install_content(self._pip)}" ; '
        if self._conda is not None:
            if isinstance(self._conda, str):
                cmd += f'/srv/install.sh conda_yaml "{self.get_install_content(self._conda)}" ; '

            else:
                cmd += f'/srv/install.sh conda_default "{self.get_install_content(self._conda)}" ; '

        return [cmd]

    def build_container(self):
        resources_dict = {
            "requests": self._resource_request.build()
            if self._resource_request
            else None,
            "limits": self._resource_limit.build() if self._resource_limit else None,
        }
        lifecycle_dict = _remove_nones(
            {
                "preStop": {
                    "exec": {"command": self._pre_stop_command},
                }
                if self._pre_stop_command
                else None,
            }
        )
        volume_juicefs = []
        if self._external_storage == "juicefs":
            volume_juicefs.append(
                {
                    "mountPath": self._external_storage_config["mountPath"],
                    "name": "data",
                }
            )
        return _remove_nones(
            {
                "command": ["/bin/sh", "-c"],
                "args": self.build_container_command(),
                "env": [env.build() for env in self._envs.values()] or None,
                "image": self._image,
                "name": self._name,
                "resources": dict((k, v) for k, v in resources_dict.items() if v)
                or None,
                "ports": [p.build() for p in self._ports] or None,
                "volumeMounts": [vol.build_mount() for vol in self._volumes]
                + volume_juicefs,
                "livenessProbe": self._liveness_probe.build()
                if self._liveness_probe
                else None,
                "startupProbe": self._startup_probe.build()
                if self._startup_probe
                else None,
                "lifecycle": lifecycle_dict or None,
                "imagePullPolicy": "Never" if self._use_local_image else None,
            }
        )

    def build_template_spec(self) -> Dict:
        result = {
            "containers": [self.build_container()],
            "initContainers": self.config_init_containers(),
            "volumes": [vol.build() for vol in self._volumes],
        }
        if self._external_storage == "juicefs":
            external_volumes = [
                {"name": "data", "persistentVolumeClaim": {"claimName": "juicefs-pvc"}}
            ]
            result["volumes"].extend(external_volumes)

        return dict((k, v) for k, v in result.items() if v)

    def build(self):
        return {
            "kind": self._kind,
            "metadata": {
                "name": self._name,
            },
            "spec": {
                "replicas": int(self._replicas),
                "template": {
                    "metadata": {
                        "labels": _remove_nones(self._labels) or None,
                    },
                    "spec": self.build_template_spec(),
                },
            },
        }


class XorbitsReplicationConfig(ReplicationConfig, abc.ABC):
    """
    Base configuration builder for replication controllers for Xorbits
    """

    rc_name: Optional[str] = None
    default_readiness_port = 15031

    def __init__(
        self,
        replicas: int,
        cpu: Optional[int] = None,
        memory: Optional[str] = None,
        limit_resources: bool = False,
        memory_limit_ratio: Optional[float] = None,
        image: Optional[str] = None,
        modules: Optional[Union[str, List[str]]] = None,
        volumes: Optional[List[VolumeConfig]] = None,
        service_name: Optional[str] = None,
        service_port: Optional[int] = None,
        external_storage: Optional[str] = None,
        external_storage_config: Optional[dict] = None,
        **kwargs,
    ):
        self._cpu = cpu
        self._memory, ratio = (
            parse_readable_size(memory) if memory is not None else (None, False)
        )
        assert not ratio

        if isinstance(modules, str):
            self._modules: Optional[List[str]] = modules.split(",")
        else:
            self._modules = modules

        req_res = ResourceConfig(cpu, memory) if cpu or memory else None
        limit_res = (
            ResourceConfig(req_res.cpu, req_res.memory * (memory_limit_ratio or 1))
            if req_res and memory
            else None
        )

        self._service_name = service_name
        self._service_port = service_port
        self._external_storage = external_storage
        self._external_storage_config = external_storage_config

        super().__init__(
            self.rc_name,
            image or self.get_default_image(),
            replicas,
            resource_request=req_res,
            resource_limit=limit_res if limit_resources else None,
            liveness_probe=self.config_liveness_probe(),
            startup_probe=self.config_startup_probe(),
            external_storage=self._external_storage,
            external_storage_config=self._external_storage_config,
            **kwargs,
        )
        if service_port:
            self.add_port(service_port)

        for vol in volumes or ():
            self.add_volume(vol)

        self.add_labels({"xorbits/service-type": self.rc_name})

    @staticmethod
    def get_default_image():
        image = "xprobe/xorbits:v" + __version__
        image += f"-py{get_local_py_version()}"
        return image

    def add_default_envs(self):
        self.add_env("MARS_K8S_POD_NAME", field_path="metadata.name")
        self.add_env("MARS_K8S_POD_NAMESPACE", field_path="metadata.namespace")
        self.add_env("MARS_K8S_POD_IP", field_path="status.podIP")

        if self._service_name:
            self.add_env("MARS_K8S_SERVICE_NAME", str(self._service_name))
        if self._service_port:
            self.add_env("MARS_K8S_SERVICE_PORT", str(self._service_port))

        self.add_env("MARS_CONTAINER_IP", field_path="status.podIP")

        if self._cpu:
            self.add_env("MKL_NUM_THREADS", str(self._cpu))
            self.add_env("MARS_CPU_TOTAL", str(self._cpu))
            if getattr(self, "stat_type", "cgroup") == "cgroup":
                self.add_env("MARS_USE_CGROUP_STAT", "1")

        if self._memory:
            self.add_env("MARS_MEMORY_TOTAL", str(int(self._memory)))

        if self._modules:
            self.add_env("MARS_LOAD_MODULES", ",".join(self._modules))

        if self._external_storage == "juicefs":
            self.add_env("XORBITS_EXTERNAL_STORAGE", "juicefs")
            self.add_env(
                "JUICEFS_MOUNT_PATH", self._external_storage_config["mountPath"]
            )

    def config_liveness_probe(self):
        """
        Liveness probe works after the startup probe.
        If the startup probe exists, the initial_delay of the liveness probe can be smaller.
        """
        raise NotImplementedError  # pragma: no cover

    @staticmethod
    def config_startup_probe() -> "ProbeConfig":
        """
        The startup probe is used to check whether the startup is smooth.
        The initial_delay of the startup probe can be sensitive to the system.
        """
        return CommandExecProbeConfig(
            ["sh", "-c", f'until [ -f "{SERVICE_PID_FILE}" ]; do sleep 3s; done'],
            failure_thresh=5,
            initial_delay=10,
            timeout=300,
        )

    @staticmethod
    def get_local_app_module(mod_name) -> str:
        return __name__.rsplit(".", 1)[0] + "." + mod_name

    def build(self):
        result = super().build()
        if self._kind in ("Deployment", "ReplicaSet"):
            result["spec"]["selector"] = {
                "matchLabels": {"xorbits/service-type": self.rc_name}
            }
        else:
            result["spec"]["selector"] = {"xorbits/service-type": self.rc_name}
        return result


class XorbitsSupervisorsConfig(XorbitsReplicationConfig):
    """
    Configuration builder for Xorbits supervisor service
    """

    rc_name = "xorbitssupervisor"

    def __init__(self, *args, **kwargs):
        self._web_port = kwargs.pop("web_port", None)
        self._readiness_port = kwargs.pop("readiness_port", self.default_readiness_port)
        super().__init__(*args, **kwargs)
        if self._web_port:
            self.add_port(self._web_port)

    def config_liveness_probe(self) -> "TcpSocketProbeConfig":
        return TcpSocketProbeConfig(
            self._readiness_port, timeout=60, failure_thresh=10, initial_delay=0
        )

    def build_container_command(self):
        cmd = super().build_container_command()
        start_command = f"/srv/entrypoint.sh {self.get_local_app_module('supervisor')}"
        if self._service_port:
            start_command += f" -p {str(self._service_port)}"
        if self._web_port:
            start_command += f" -w {str(self._web_port)}"
        if self._cpu:
            start_command += f" --n-process {str(int(math.ceil(self._cpu)))}"
        start_command += ";"

        cmd[0] += start_command
        return cmd


class XorbitsWorkersConfig(XorbitsReplicationConfig):
    """
    Configuration builder for Xorbits worker service
    """

    rc_name = "xorbitsworker"

    def __init__(self, *args, **kwargs):
        spill_volumes = kwargs.pop("spill_volumes", None) or ()
        mount_shm = kwargs.pop("mount_shm", True)
        self._limit_resources = kwargs["limit_resources"] = kwargs.get(
            "limit_resources", True
        )
        worker_cache_mem = (
            kwargs.pop("worker_cache_mem", None) or DEFAULT_WORKER_CACHE_MEM
        )
        min_cache_mem = kwargs.pop("min_cache_mem", None)
        self._readiness_port = kwargs.pop("readiness_port", self.default_readiness_port)
        self._readiness_service_name = kwargs.pop("readiness_service_name", None)
        supervisor_web_port = kwargs.pop("supervisor_web_port", None)

        super().__init__(*args, **kwargs)

        self._spill_volumes = []
        for idx, vol in enumerate(spill_volumes):
            if isinstance(vol, str):
                path = f"/mnt/hostpath{idx}"
                self.add_volume(HostPathVolumeConfig(f"host-path-vol-{idx}", path, vol))
                self._spill_volumes.append(path)
            else:
                self.add_volume(vol)
                self._spill_volumes.append(vol.mount_path)
        if self._spill_volumes:
            self.add_env("MARS_SPILL_DIRS", ":".join(self._spill_volumes))

        if self._memory:
            size_limit = calc_size_by_str(worker_cache_mem, self._memory)
            self.add_env("MARS_CACHE_MEM_SIZE", worker_cache_mem)
        else:
            size_limit = None

        if mount_shm:
            self.add_volume(
                EmptyDirVolumeConfig(
                    "xorbits-shared", "/dev/shm", size_limit=size_limit
                )
            )

        if min_cache_mem:
            self.add_env("MARS_MIN_CACHE_MEM_SIZE", min_cache_mem)
        if supervisor_web_port:
            self.add_env("MARS_K8S_SUPERVISOR_WEB_PORT", supervisor_web_port)

    def config_liveness_probe(self) -> "TcpSocketProbeConfig":
        return TcpSocketProbeConfig(
            self._readiness_port, timeout=60, failure_thresh=10, initial_delay=0
        )

    def config_init_containers(self) -> List[Dict]:
        """
        The worker pod checks whether the readiness port of the supervisor is successfully opened in the init container.
        The worker pod will not start until the init container is executed successfully.
        This ensures that the worker must start after the supervisor.
        The image and command used by the init container refer to the Kubernetes official documentation (https://kubernetes.io/docs/concepts/workloads/pods/init-containers/).
        """
        return [
            {
                "name": "waiting-for-supervisors",
                "image": "busybox:1.28",
                "command": [
                    "sh",
                    "-c",
                    f"until nslookup {self._readiness_service_name}.$(cat /var/run/secrets/kubernetes.io/serviceaccount/namespace).svc.cluster.local; do echo waiting for supervisors; sleep 5; done",
                ],
            }
        ]

    def build_container_command(self):
        cmd = super().build_container_command()
        start_command = f"/srv/entrypoint.sh {self.get_local_app_module('worker')}"
        if self._service_port:
            start_command += f" -p {str(self._service_port)}"
        start_command += ";"

        cmd[0] += start_command
        return cmd

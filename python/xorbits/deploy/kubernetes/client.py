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

import asyncio
import functools
import logging
import time
import uuid
from typing import Collection, Dict, List, Optional, Union

from ..._mars.deploy.utils import wait_services_ready
from ..._mars.lib.aio import new_isolation, stop_isolation
from ..._mars.services.cluster.api import WebClusterAPI
from ..._mars.session import new_session
from ..._mars.utils import calc_size_by_str
from ...compat._constants import SUPPORTED_PY_VERSIONS
from ...utils import get_local_py_version
from .config import (
    IngressConfig,
    NamespaceConfig,
    RoleBindingConfig,
    RoleConfig,
    ServiceConfig,
    XorbitsReplicationConfig,
    XorbitsSupervisorsConfig,
    XorbitsWorkersConfig,
)
from .external_storage.external_storage import JuicefsK8SStorage

try:
    from kubernetes.client import ApiClient
    from kubernetes.client.rest import ApiException as K8SApiException
except ImportError:  # pragma: no cover
    K8SApiException = None
    ApiClient = None

logger = logging.getLogger(__name__)


class KubernetesClusterClient:
    def __init__(self, cluster: "KubernetesCluster"):
        self._cluster = cluster
        self._endpoint = None
        self._session = None

    @property
    def endpoint(self) -> Optional[str]:
        return self._endpoint

    @property
    def namespace(self) -> Optional[str]:
        return self._cluster.namespace

    @property
    def session(self) -> Optional[str]:
        return self._session

    def start(self):
        try:
            self._endpoint = self._cluster.start()
            self._session = new_session(self._endpoint)
        except:  # noqa: E722  # nosec  # pylint: disable=bare-except   # pragma: no cover
            self.stop()
            raise

    def stop(self, wait=False, timeout=0):
        self._cluster.stop(wait=wait, timeout=timeout)


class KubernetesCluster:
    _supervisor_config_cls = XorbitsSupervisorsConfig
    _worker_config_cls = XorbitsWorkersConfig
    _default_service_port: int = 7103
    _default_web_port: int = 7104

    def __init__(
        self,
        kube_api_client: ApiClient,
        worker_cpu: int,
        worker_mem: str,
        image: Optional[str] = None,
        namespace: Optional[str] = None,
        supervisor_num: int = 1,
        supervisor_cpu: int = 1,
        supervisor_mem: str = "4G",
        supervisor_mem_limit_ratio: Optional[float] = None,
        worker_num: int = 1,
        worker_spill_paths: Optional[str] = None,
        worker_cache_mem: Optional[str] = None,
        min_worker_num: Optional[int] = None,
        worker_min_cache_mem: Optional[str] = None,
        worker_mem_limit_ratio: Optional[float] = None,
        web_port: Optional[int] = None,
        service_name: Optional[str] = None,
        service_type: Optional[str] = None,
        pip: Optional[Union[str, List[str]]] = None,
        conda: Optional[Union[str, List[str]]] = None,
        timeout: Optional[int] = None,
        cluster_type: str = "auto",
        external_storage: Optional[str] = None,
        external_storage_config: Optional[dict] = None,
        **kwargs,
    ):
        from kubernetes import client as kube_client

        if worker_cpu is None or worker_mem is None:  # pragma: no cover
            raise TypeError("`worker_cpu` and `worker_mem` must be specified")
        if cluster_type not in ["auto", "eks", "kubernetes"]:  # pragma: no cover
            raise ValueError("`cluster_type` must be `auto`, `kubernetes` or `eks`")
        if pip is not None and not isinstance(pip, (str, list)):  # pragma: no cover
            raise TypeError("`pip` must be str or List[str] type.")
        if conda is not None and not isinstance(conda, (str, list)):  # pragma: no cover
            raise TypeError("`conda` must be str or List[str] type.")
        local_py_version = get_local_py_version()
        if (
            image is None and local_py_version not in SUPPORTED_PY_VERSIONS
        ):  # pragma: no cover
            raise RuntimeError(
                f"Xorbits does not support Kubernetes deployment under your python version {local_py_version}. "
                f"Please change the python version to one of the following versions: {SUPPORTED_PY_VERSIONS}."
            )

        self._api_client = kube_api_client
        self._core_api = kube_client.CoreV1Api(kube_api_client)
        self._networking_api = kube_client.NetworkingV1Api(self._api_client)

        self._namespace = namespace
        self._image = image
        self._timeout = timeout
        self._service_name = service_name or "xorbits-service"
        self._service_type = service_type or "NodePort"
        self._extra_volumes = kwargs.pop("extra_volumes", ())
        self._pre_stop_command = kwargs.pop("pre_stop_command", None)
        self._log_when_fail = kwargs.pop("log_when_fail", False)
        self._cluster_type = self._get_cluster_type(cluster_type)
        self._ingress_name = "xorbits-ingress"
        self._use_local_image = kwargs.pop("use_local_image", False)
        self._external_storage = external_storage
        self._external_storage_config = external_storage_config

        if self._external_storage and self._external_storage not in ["juicefs"]:
            raise ValueError(
                "Currently, only juicefs is supported as one of our storage backend."
            )
        if self._external_storage == "juicefs":
            if (
                not self._external_storage_config
                or "metadata_url" not in self._external_storage_config
            ):
                raise ValueError(
                    "For external storage JuiceFS, you must specify the metadata url for its metadata storage in external storage config, for example external_storage_config={'metadata_url': 'redis://172.17.0.5:6379/1',}."
                )

        extra_modules = kwargs.pop("extra_modules", None) or []
        extra_modules = (
            extra_modules.split(",")
            if isinstance(extra_modules, str)
            else extra_modules
        )
        extra_envs = kwargs.pop("extra_env", None) or dict()
        extra_labels = kwargs.pop("extra_labels", None) or dict()
        service_port = kwargs.pop("service_port", None) or self._default_service_port

        def _override_modules(updates: Union[str, Collection]):
            modules = set(extra_modules)
            updates = updates.split(",") if isinstance(updates, str) else updates
            modules.update(updates)
            return sorted(modules)

        def _override_dict(d, updates):
            updates = updates or dict()
            ret = d.copy()
            ret.update(updates)
            return ret

        _override_envs = functools.partial(_override_dict, extra_envs)
        _override_labels = functools.partial(_override_dict, extra_labels)

        self._supervisor_num = supervisor_num
        self._supervisor_cpu = supervisor_cpu
        self._supervisor_mem = calc_size_by_str(supervisor_mem, None)
        self._supervisor_mem_limit_ratio = supervisor_mem_limit_ratio
        self._supervisor_extra_modules = _override_modules(
            kwargs.pop("supervisor_extra_modules", [])
        )
        self._supervisor_extra_env = _override_envs(
            kwargs.pop("supervisor_extra_env", None)
        )
        self._supervisor_extra_labels = _override_labels(
            kwargs.pop("supervisor_extra_labels", None)
        )
        self._supervisor_service_port = (
            kwargs.pop("supervisor_service_port", None) or service_port
        )
        self._web_port = web_port or self._default_web_port
        self._external_web_endpoint = None

        self._worker_num = worker_num
        self._worker_cpu = worker_cpu
        self._worker_mem = calc_size_by_str(worker_mem, None)
        self._worker_mem_limit_ratio = worker_mem_limit_ratio
        self._worker_spill_paths = worker_spill_paths
        self._worker_cache_mem = worker_cache_mem
        self._worker_min_cache_men = worker_min_cache_mem
        self._min_worker_num = min_worker_num
        self._worker_extra_modules = _override_modules(
            kwargs.pop("worker_extra_modules", [])
        )
        self._worker_extra_env = _override_envs(kwargs.pop("worker_extra_env", None))
        self._worker_extra_labels = _override_labels(
            kwargs.pop("worker_extra_labels", None)
        )
        self._worker_service_port = (
            kwargs.pop("worker_service_port", None) or service_port
        )
        self._pip = pip
        self._conda = conda
        self._readiness_service_name = kwargs.pop(
            "readiness_service_name", "xorbits-readiness-service"
        )

    @property
    def namespace(self) -> Optional[str]:
        return self._namespace

    @staticmethod
    def _get_k8s_context(context: Dict) -> str:
        cluster = context.get("context", {}).get("cluster", "")
        user = context.get("context", {}).get("user", "")
        name = context.get("name", "")

        for s in [cluster, user, name]:
            if ("eks" in s) or ("aws" in s):
                return "eks"
        return "kubernetes"

    @staticmethod
    def _get_cluster_type(cluster_type: str) -> str:
        from kubernetes import config

        if cluster_type == "auto":
            try:
                curr_context = config.list_kube_config_contexts()[1]
                return KubernetesCluster._get_k8s_context(curr_context)
            except:  # pragma: no cover
                raise ValueError(
                    "`cluster_type` cannot be `auto`, please check your kubectl configuration or specify it as `eks` or `kubernetes`"
                )
        else:
            return cluster_type

    def _get_free_namespace(self) -> str:
        while True:
            namespace = "xorbits-ns-" + str(uuid.uuid4().hex)
            try:
                self._core_api.read_namespace(namespace)
            except K8SApiException as ex:
                if ex.status != 404:  # pragma: no cover
                    raise
                return namespace

    def _create_kube_service(self):
        if self._service_type != "NodePort":  # pragma: no cover
            raise NotImplementedError(
                f"Service type {self._service_type} not supported"
            )

        service_config = ServiceConfig(
            self._service_name,
            service_type="NodePort",
            port=self._web_port,
            selector={"xorbits/service-type": XorbitsSupervisorsConfig.rc_name},
        )
        self._core_api.create_namespaced_service(
            self._namespace, service_config.build()
        )

    def _create_readiness_service(self):
        """
        Start a simple ClusterIP service for the workers to detect whether the supervisor is successfully started.
        """
        readiness_service_config = ServiceConfig(
            name=self._readiness_service_name,
            service_type=None,
            selector={"xorbits/service-type": XorbitsSupervisorsConfig.rc_name},
            port=XorbitsReplicationConfig.default_readiness_port,
            target_port=XorbitsReplicationConfig.default_readiness_port,
            protocol="TCP",
        )
        self._core_api.create_namespaced_service(
            self._namespace, readiness_service_config.build()
        )

    def _get_ready_pod_count(self, label_selector: str) -> int:  # pragma: no cover
        query = self._core_api.list_namespaced_pod(
            namespace=self._namespace, label_selector=label_selector
        ).to_dict()
        cnt = 0
        for el in query["items"]:
            if el["status"]["phase"] in ("Error", "Failed"):
                logger.warning(
                    "Error in starting pod, message: %s", el["status"]["message"]
                )
                continue
            if "status" not in el or "conditions" not in el["status"]:
                cnt += 1
            elif any(
                cond["type"] == "Ready" and cond["status"] == "True"
                for cond in el["status"].get("conditions") or ()
            ):
                cnt += 1
        return cnt

    def _create_namespace(self):
        namespace = self._namespace = self._get_free_namespace()
        self._core_api.create_namespace(NamespaceConfig(namespace).build())

    def _create_roles_and_bindings(self):
        # create role and binding
        role_config = RoleConfig(
            "xorbits-pod-operator",
            self._namespace,
            api_groups="",
            resources="pods,endpoints,services",
            verbs="get,watch,list,patch",
        )
        role_config.create_namespaced(self._api_client, self._namespace)
        role_binding_config = RoleBindingConfig(
            "xorbits-pod-operator-binding",
            self._namespace,
            "xorbits-pod-operator",
            "default",
        )
        role_binding_config.create_namespaced(self._api_client, self._namespace)

    def _create_supervisors(self):
        supervisors_config = self._supervisor_config_cls(
            self._supervisor_num,
            image=self._image,
            cpu=self._supervisor_cpu,
            memory=self._supervisor_mem,
            memory_limit_ratio=self._supervisor_mem_limit_ratio,
            modules=self._supervisor_extra_modules,
            volumes=self._extra_volumes,
            service_name=self._service_name,
            service_port=self._supervisor_service_port,
            web_port=self._web_port,
            pre_stop_command=self._pre_stop_command,
            use_local_image=self._use_local_image,
            pip=self._pip,
            conda=self._conda,
            external_storage=self._external_storage,
            external_storage_config=self._external_storage_config,
        )
        supervisors_config.add_simple_envs(self._supervisor_extra_env)
        supervisors_config.add_labels(self._supervisor_extra_labels)
        supervisors_config.create_namespaced(self._api_client, self._namespace)

    def _create_workers(self):
        workers_config = self._worker_config_cls(
            self._worker_num,
            image=self._image,
            cpu=self._worker_cpu,
            memory=self._worker_mem,
            memory_limit_ratio=self._worker_mem_limit_ratio,
            spill_volumes=self._worker_spill_paths,
            modules=self._worker_extra_modules,
            volumes=self._extra_volumes,
            worker_cache_mem=self._worker_cache_mem,
            min_cache_mem=self._worker_min_cache_men,
            service_name=self._service_name,
            service_port=self._worker_service_port,
            pre_stop_command=self._pre_stop_command,
            supervisor_web_port=self._web_port,
            use_local_image=self._use_local_image,
            pip=self._pip,
            conda=self._conda,
            readiness_service_name=self._readiness_service_name,
            external_storage=self._external_storage,
            external_storage_config=self._external_storage_config,
        )
        workers_config.add_simple_envs(self._worker_extra_env)
        workers_config.add_labels(self._worker_extra_labels)
        workers_config.create_namespaced(self._api_client, self._namespace)

    def _create_services(self):
        self._create_supervisors()
        self._create_workers()

    def _wait_services_ready(self):
        min_worker_num = int(self._min_worker_num or self._worker_num)
        limits = [self._supervisor_num, min_worker_num]
        selectors = [
            "xorbits/service-type=" + XorbitsSupervisorsConfig.rc_name,
            "xorbits/service-type=" + XorbitsWorkersConfig.rc_name,
        ]
        start_time = time.time()
        logger.debug("Start waiting pods to be ready")
        wait_services_ready(
            selectors,
            limits,
            lambda sel: self._get_ready_pod_count(sel),
            timeout=self._timeout,
        )
        logger.info("All service pods ready.")
        if self._timeout is not None:  # pragma: no branch
            self._timeout -= time.time() - start_time

    def _create_ingress(self):
        self._networking_api.create_namespaced_ingress(
            self._namespace,
            IngressConfig(
                self._namespace,
                self._ingress_name,
                self._service_name,
                self._web_port,
                self._cluster_type,
            ).build(),
        )

    def _get_ingress_address(self):
        from kubernetes import watch

        w = watch.Watch()
        for event in w.stream(
            func=self._networking_api.list_namespaced_ingress, namespace=self._namespace
        ):
            if (
                event["object"].kind == "Ingress"
                and event["object"].status.load_balancer.ingress is not None
                and len(event["object"].status.load_balancer.ingress) > 0
            ):
                ingress = event["object"].status.load_balancer.ingress[0]
                ip = ingress.ip
                host = ingress.hostname if hasattr(ingress, "hostname") else None
                address = ip or host
                logger.debug(f"Ingress ip: {ip}, host: {host}, address: {address}")
                if address is not None:
                    return f"http://{address}:80"

    def _wait_web_ready(self):
        loop = new_isolation().loop

        async def get_supervisors():
            start_time = time.time()
            while True:
                try:
                    time.sleep(1)
                    # TODO use Xorbits API instead of mars.
                    cluster_api = WebClusterAPI(self._external_web_endpoint)
                    supervisors = await cluster_api.get_supervisors()

                    if len(supervisors) == self._supervisor_num:
                        break
                except:  # noqa: E722  # nosec  # pylint: disable=bare-except  # pragma: no cover
                    if (
                        self._timeout is not None
                        and time.time() - start_time > self._timeout
                    ):
                        logger.exception("Error when fetching supervisors")
                        raise TimeoutError(
                            "Wait for kubernetes cluster timed out"
                        ) from None

        asyncio.run_coroutine_threadsafe(get_supervisors(), loop).result()

    def _load_cluster_logs(self) -> Dict:  # pragma: no cover
        log_dict = dict()
        pod_items = self._core_api.list_namespaced_pod(self._namespace).to_dict()
        for item in pod_items["items"]:
            log_dict[item["metadata"]["name"]] = self._core_api.read_namespaced_pod_log(
                name=item["metadata"]["name"], namespace=self._namespace
            )
        return log_dict

    def _create_external_storage(self, storage_name: str):
        if storage_name == "juicefs":
            juicefs_k8s_storage = JuicefsK8SStorage(
                namespace=self.namespace,
                api_client=self._api_client,
                external_storage_config=self._external_storage_config,
            )
            juicefs_k8s_storage.build()

    def start(self):
        try:
            self._create_namespace()
            self._create_roles_and_bindings()

            if self._external_storage:
                self._create_external_storage(storage_name=self._external_storage)

            self._create_services()
            self._create_kube_service()
            self._create_readiness_service()

            self._wait_services_ready()

            self._create_ingress()

            self._external_web_endpoint = self._get_ingress_address()
            self._wait_web_ready()
            logger.warning(f"Xorbits endpoint {self._external_web_endpoint} is ready!")
            return self._external_web_endpoint
        except:  # noqa: E722   # pragma: no cover
            if self._log_when_fail:
                logger.error("Error when creating cluster")
                for name, log in self._load_cluster_logs().items():
                    logger.error("Error logs for %s:\n%s", name, log)
            self.stop()
            raise

    def _delete_pv(self):
        from kubernetes.client import CoreV1Api

        api = CoreV1Api(self._api_client)
        pv_name = f"juicefs-pv-{self._namespace}"
        try:
            # K8s issue: https://github.com/kubernetes/kubernetes/issues/69697
            api.patch_persistent_volume(
                name=pv_name, body={"metadata": {"finalizers": None}}
            )
            api.delete_persistent_volume(pv_name)
        except Exception as e:  # pragma: no cover
            logger.warning(f"Error happens during deleting pv: {e}")

    def stop(self, wait: bool = False, timeout: int = 0):
        # stop isolation
        stop_isolation()

        from kubernetes.client import CoreV1Api

        api = CoreV1Api(self._api_client)
        api.delete_namespace(self._namespace)
        if self._external_storage == "juicefs":
            self._delete_pv()
        if wait:
            start_time = time.time()
            while True:
                try:
                    api.read_namespace(self._namespace)
                except K8SApiException as ex:  # pragma: no cover
                    if ex.status != 404:
                        raise
                    break
                else:
                    time.sleep(1)
                    if (
                        timeout and time.time() - start_time > timeout
                    ):  # pragma: no cover
                        raise TimeoutError


def new_cluster(
    kube_api_client: ApiClient,
    worker_cpu: int,
    worker_mem: str,
    image: Optional[str] = None,
    supervisor_num: int = 1,
    supervisor_cpu: int = 1,
    supervisor_mem: str = "4G",
    worker_num: int = 1,
    worker_spill_paths: Optional[List[str]] = None,
    worker_cache_mem: Optional[str] = None,
    min_worker_num: Optional[int] = None,
    pip: Optional[Union[str, List[str]]] = None,
    conda: Optional[Union[str, List[str]]] = None,
    timeout: Optional[int] = None,
    cluster_type: str = "auto",
    external_storage: Optional[str] = None,
    **kwargs,
) -> "KubernetesClusterClient":
    """
    The entrance of deploying xorbits cluster.

    Parameters
    ----------
    kube_api_client :
        Kubernetes API client, required, can be created with ``new_client_from_config``
    worker_cpu :
        Number of CPUs for every worker, required
    worker_mem :
        Memory size for every worker, required
    image :
        Docker image to use, ``xprobe/xorbits:<xorbits version>`` by default
    supervisor_num :
        Number of supervisors in the cluster, 1 by default
    supervisor_cpu :
        Number of CPUs for every supervisor, 1 by default
    supervisor_mem :
        Memory size for every supervisor, 4G by default
    worker_num :
        Number of workers in the cluster, 1 by default
    worker_spill_paths :
        Spill paths for worker pods on host
    worker_cache_mem :
        Size or ratio of cache memory for every worker
    min_worker_num :
        Minimal ready workers, equal to ``worker_num`` by default
    pip :
        Either a list of pip requirements specifiers,
        or a string containing the path to a pip `requirements.txt <https://pip.pypa.io/en/stable/user_guide/#requirements-files>`_ file.
        None by default.
        Both supervisor and worker will install the specified pip packages.
        Examples:

            * ``pip=["requests==1.0.0", "aiohttp"]``
            * ``pip="/path/to/requirements.txt"``
    conda :
        Either a string containing the path to a `conda environment.yml <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#create-env-file-manually>`_ file,
        or a list of conda packages used for `conda install <https://docs.conda.io/projects/conda/en/latest/commands/install.html>`_ command.
        None by default.
        Both supervisor and worker will install the specified conda packages.
        When this parameter is list type, install the conda packages from the `default channel <https://repo.anaconda.com/pkgs/>`_.
        When this parameter is string type, the ``name`` attribute in the environment.yml file will not take effect,
        since all package changes will occur in the ``base`` conda environment where Xorbits exists.
        Examples:

            * ``conda=["tensorflow", "tensorboard"]``
            * ``conda="/path/to/environment.yml"``
    timeout :
        Timeout in seconds when creating clusters, never timeout by default
    cluster_type :
        K8s cluster type, ``auto``, ``kubernetes`` or ``eks`` supported, ``auto`` by default.
        ``auto`` means that it will automatically detect whether the kubectl context is ``eks``.
        You can also manually specify ``kubernetes`` or ``eks`` in some special cases.
    external_storage:
        You can specify an external storage. Currently, only juicefs is supported as one of our storage backend.
    kwargs :
        Extra kwargs

    Returns
    -------
    KubernetesClusterClient
        a KubernetesClusterClient instance
    """
    cluster_cls = kwargs.pop("cluster_cls", KubernetesCluster)

    cluster = cluster_cls(
        kube_api_client,
        worker_cpu,
        worker_mem,
        image=image,
        supervisor_num=supervisor_num,
        supervisor_cpu=supervisor_cpu,
        supervisor_mem=supervisor_mem,
        worker_num=worker_num,
        worker_spill_paths=worker_spill_paths,
        worker_cache_mem=worker_cache_mem,
        min_worker_num=min_worker_num,
        pip=pip,
        conda=conda,
        timeout=timeout,
        cluster_type=cluster_type,
        external_storage=external_storage,
        **kwargs,
    )
    client = KubernetesClusterClient(cluster)
    client.start()
    return client

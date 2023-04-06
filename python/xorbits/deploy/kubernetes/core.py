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
import json
import logging
import os
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, TypeVar

from ..._mars.deploy.utils import next_in_thread, wait_all_supervisors_ready
from ..._mars.services.cluster import WebClusterAPI
from ..._mars.services.cluster.backends import (
    AbstractClusterBackend,
    register_cluster_backend,
)
from ..._mars.services.cluster.core import NodeRole
from ._constants import SERVICE_PID_FILE
from .config import XorbitsReplicationConfig, XorbitsWorkersConfig

logger = logging.getLogger(__name__)
RetType = TypeVar("RetType")


@register_cluster_backend
class K8SClusterBackend(AbstractClusterBackend):
    name = "k8s"

    def __init__(
        self,
        node_role: Optional[NodeRole] = None,
        pool_address: Optional[str] = None,
        k8s_config: Optional[Any] = None,
        k8s_namespace: Optional[str] = None,
    ):
        from kubernetes import client

        self._node_role = node_role
        self._pool_address = pool_address
        self._k8s_config = k8s_config

        verify_ssl = bool(int(os.environ.get("KUBE_VERIFY_SSL", "1")))
        if not verify_ssl:
            c = client.Configuration()
            c.verify_ssl = False
            client.Configuration.set_default(c)

        self._k8s_namespace = (
            k8s_namespace or os.environ.get("MARS_K8S_POD_NAMESPACE") or "default"
        )
        self._service_name = os.environ.get("MARS_K8S_SERVICE_NAME")
        self._full_label_selector = None
        self._client = client.CoreV1Api(client.ApiClient(self._k8s_config))
        self._apps_client = client.AppsV1Api(client.ApiClient(self._k8s_config))

    @classmethod
    async def create(
        cls, node_role: NodeRole, lookup_address: Optional[str], pool_address: str
    ) -> "AbstractClusterBackend":
        from kubernetes import client, config

        if lookup_address is None:
            k8s_namespace = None
            k8s_config = config.load_incluster_config()
        else:
            address_parts = lookup_address.rsplit("?", 1)
            k8s_namespace = None if len(address_parts) == 1 else address_parts[1]

            k8s_config = client.Configuration()
            if "://" in address_parts[0]:
                k8s_config.host = address_parts[0]
            else:
                config.load_kube_config(
                    address_parts[0], client_configuration=k8s_config
                )
        return cls(node_role, pool_address, k8s_config, k8s_namespace)

    def __reduce__(self):
        return (
            type(self),
            (
                self._node_role,
                self._pool_address,
                self._k8s_config,
                self._k8s_namespace,
            ),
        )

    @staticmethod
    def _format_endpoint_query_result(result: Dict, filter_ready: bool = True):
        port = os.environ["MARS_K8S_SERVICE_PORT"]
        endpoints = [
            f"{addr['ip']}:{port}" for addr in result["subsets"][0]["addresses"] or []
        ]
        if not filter_ready:
            endpoints = [
                f"{addr['ip']}:{port}"
                for addr in result["subsets"][0]["not_ready_addresses"] or []
            ]
        return endpoints

    def _get_web_cluster_api(self):
        supervisor_web_port = os.environ["MARS_K8S_SUPERVISOR_WEB_PORT"]
        web_url = (
            f"http://{self._service_name}.{self._k8s_namespace}:{supervisor_web_port}"
        )
        api = WebClusterAPI(web_url)
        return api

    async def _watch_supervisors_by_service_api(
        self,
    ) -> AsyncGenerator[List[str], None]:
        from kubernetes.watch import Watch as K8SWatch
        from urllib3.exceptions import ReadTimeoutError

        w = K8SWatch()

        while True:
            streamer = w.stream(
                self._client.list_namespaced_endpoints,
                namespace=self._k8s_namespace,
                label_selector=f"xorbits/service-name={self._service_name}",
                timeout_seconds=60,
            )
            while True:
                try:
                    event = await next_in_thread(streamer)
                    obj_dict = event["object"].to_dict()
                    yield self._format_endpoint_query_result(obj_dict)
                except (ReadTimeoutError, StopAsyncIteration):
                    break
                except:  # noqa: E722  # pragma: no cover  # pylint: disable=bare-except
                    logger.exception("Unexpected error when watching on kubernetes")
                    break

    async def _watch_supervisors_by_cluster_web_api(self):
        while True:
            try:
                api = self._get_web_cluster_api()
                async for supervisors in api.watch_supervisors():
                    yield supervisors
            except (OSError, asyncio.TimeoutError):
                pass

    async def _get_supervisors_by_service_api(
        self, filter_ready: bool = True
    ) -> List[str]:
        result = (
            await asyncio.to_thread(
                self._client.read_namespaced_endpoints,
                name=self._service_name,
                namespace=self._k8s_namespace,
            )
        ).to_dict()
        return self._format_endpoint_query_result(result, filter_ready=filter_ready)

    async def _get_supervisors_by_cluster_web_api(self, filter_ready: bool = True):
        api = self._get_web_cluster_api()
        try:
            supervisors = await api.get_supervisors(filter_ready=filter_ready)
            return supervisors
        except (OSError, asyncio.TimeoutError):  # pragma: no cover
            return []

    async def get_supervisors(self, filter_ready: bool = True) -> List[str]:
        if self._node_role == NodeRole.SUPERVISOR:
            return await self._get_supervisors_by_service_api(filter_ready)
        else:
            return await self._get_supervisors_by_cluster_web_api(filter_ready)

    async def watch_supervisors(self) -> AsyncGenerator[List[str], None]:
        if self._node_role == NodeRole.SUPERVISOR:
            watch_fun = self._watch_supervisors_by_service_api
        else:
            watch_fun = self._watch_supervisors_by_cluster_web_api

        try:
            async for supervisors in watch_fun():
                yield supervisors
        except asyncio.CancelledError:
            pass

    async def request_worker(
        self,
        worker_cpu: Optional[int] = None,
        worker_mem: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> str:
        raise NotImplementedError

    def _list_workers(self):
        workers = []
        pods = self._client.list_namespaced_pod(
            namespace=self._k8s_namespace,
            _preload_content=False,
            label_selector="xorbits/service-type=" + XorbitsWorkersConfig.rc_name,
        )
        pods_list = json.loads(pods.data)["items"]
        port = os.environ["MARS_K8S_SERVICE_PORT"]
        for p in pods_list:
            if "podIP" in p["status"]:
                workers.append(p["status"]["podIP"] + ":" + port)
            elif (
                "conditions" in p["status"]
                and "reason" in p["status"]["conditions"][0]
                and p["status"]["conditions"][0]["reason"] == "Unschedulable"
            ):
                raise SystemError(p["status"]["conditions"][0]["message"])
        return workers

    async def request_workers(
        self, worker_num: int, timeout: Optional[int] = None
    ) -> List[str]:
        if worker_num <= 0:
            raise ValueError("Please specify a `worker_num` that is greater than zero")
        if timeout and timeout < 0:
            raise ValueError("Please specify a `timeout` that is greater than zero")
        start_time = time.time()
        deployment = self._apps_client.read_namespaced_deployment(
            name=XorbitsWorkersConfig.rc_name,
            namespace=self._k8s_namespace,
            _preload_content=False,
        )
        deployment_data = json.loads(deployment.data)
        old_replica = deployment_data["status"]["replicas"]
        old_workers = self._list_workers()
        new_replica = old_replica + worker_num
        body = {"spec": {"replicas": new_replica}}
        self._apps_client.patch_namespaced_deployment_scale(
            name=XorbitsWorkersConfig.rc_name, namespace=self._k8s_namespace, body=body
        )
        while True:
            if timeout is not None and (timeout + start_time) < time.time():
                raise TimeoutError("Request worker timeout")
            new_workers = self._list_workers()
            if len(new_workers) == new_replica:
                return list(set(new_workers) - set(old_workers))
            await asyncio.sleep(1)

    async def release_worker(self, address: str):
        raise NotImplementedError

    async def reconstruct_worker(self, address: str):
        raise NotImplementedError


class K8SServiceMixin:
    @staticmethod
    def write_pid_file():
        with open(SERVICE_PID_FILE, "w") as pid_file:
            pid_file.write(str(os.getpid()))

    async def wait_all_supervisors_ready(self):
        """
        Wait till all containers are ready
        """
        await wait_all_supervisors_ready(self.args.endpoint)

    async def start_readiness_server(self):
        readiness_port = os.environ.get(
            "MARS_K8S_READINESS_PORT", XorbitsReplicationConfig.default_readiness_port
        )
        self._readiness_server = await asyncio.start_server(
            lambda r, w: None, port=readiness_port
        )

    async def stop_readiness_server(self):
        self._readiness_server.close()
        await self._readiness_server.wait_closed()

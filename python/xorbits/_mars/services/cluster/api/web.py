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

import json
import time
from typing import Callable, Dict, List, Optional, Set

from ....lib.aio import alru_cache
from ....resource import Resource
from ....typing import BandType
from ....utils import deserialize_serializable, serialize_serializable
from ...core import NodeRole
from ...web import MarsServiceWebAPIHandler, MarsWebAPIClientMixin, web_api
from ..core import NodeStatus, watch_method
from .core import AbstractClusterAPI


class ClusterWebAPIHandler(MarsServiceWebAPIHandler):
    _root_pattern = "/api/cluster"

    @alru_cache(cache_exceptions=False)
    async def _get_cluster_api(self):
        from ...cluster import ClusterAPI

        return await ClusterAPI.create(self._supervisor_addr)

    @staticmethod
    def _convert_node_dict(node_info_list: Dict[str, Dict]):
        res = {}
        for node_addr, node in node_info_list.items():
            res_dict = node.copy()
            res_dict["status"] = res_dict["status"].value
            res[node_addr] = res_dict
        return res

    @web_api("nodes", method=["get", "post"], cache_blocking=True)
    async def get_nodes_info(self):
        watch = bool(int(self.get_argument("watch", "0")))
        env = bool(int(self.get_argument("env", "0")))
        resource = bool(int(self.get_argument("resource", "0")))
        detail = bool(int(self.get_argument("detail", "0")))

        nodes_arg = self.get_argument("nodes", None)
        nodes = nodes_arg.split(",") if nodes_arg is not None else None

        role_arg = self.get_argument("role", None)
        role = NodeRole(int(role_arg)) if role_arg is not None else None

        statuses_arg = self.get_argument("statuses", None)
        statuses = (
            set(NodeStatus(int(v)) for v in statuses_arg.split(","))
            if statuses_arg
            else None
        )

        exclude_statuses_arg = self.get_argument("exclude_statuses", None)
        exclude_statuses = (
            set(NodeStatus(int(v)) for v in exclude_statuses_arg.split(","))
            if exclude_statuses_arg
            else None
        )

        statuses = WebClusterAPI._calc_statuses(statuses, exclude_statuses)

        cluster_api = await self._get_cluster_api()
        result = {}
        if watch:
            assert nodes is None
            version = self.get_argument("version", "") or None
            if version:
                version = int(version)

            async for version, node_infos in cluster_api.watch_nodes(
                role,
                env=env,
                resource=resource,
                detail=detail,
                statuses=statuses,
                version=version,
            ):
                result["version"] = version
                result["nodes"] = self._convert_node_dict(node_infos)
                break
        else:
            nodes = await cluster_api.get_nodes_info(
                nodes=nodes,
                role=role,
                env=env,
                resource=resource,
                statuses=statuses,
                detail=detail,
            )
            result["nodes"] = self._convert_node_dict(nodes)
        self.write(json.dumps(result))

    @web_api("bands", method="get", cache_blocking=True)
    async def get_all_bands(self):
        role_arg = self.get_argument("role", None)
        role = NodeRole(int(role_arg)) if role_arg is not None else None
        watch = bool(int(self.get_argument("watch", "0")))

        statuses_arg = self.get_argument("statuses", None)
        statuses = (
            set(NodeStatus(int(v)) for v in statuses_arg.split(","))
            if statuses_arg
            else None
        )

        cluster_api = await self._get_cluster_api()
        if watch:
            version = self.get_argument("version", "") or None
            if version:
                version = int(version)

            async for version, bands in cluster_api.watch_all_bands(
                role, statuses=statuses, version=version
            ):
                self.write(serialize_serializable((version, bands)))
                break
        else:
            self.write(
                serialize_serializable(
                    await cluster_api.get_all_bands(role, statuses=statuses)
                )
            )

    @web_api("versions", method="get", cache_blocking=True)
    async def get_mars_versions(self):
        cluster_api = await self._get_cluster_api()
        self.write(json.dumps(list(await cluster_api.get_mars_versions())))

    @web_api("pools", method="get", cache_blocking=True)
    async def get_node_pool_configs(self):
        cluster_api = await self._get_cluster_api()
        address = self.get_argument("address", "") or None
        pools = list(await cluster_api.get_node_pool_configs(address))
        # Since logging_conf field cannot be serialized by json,
        # and this field is not used by the front end, it is removed.
        for pool in pools:
            pool.pop("logging_conf", None)
        self.write(json.dumps({"pools": pools}))

    @web_api("stacks", method="get", cache_blocking=True)
    async def get_node_thread_stacks(self):
        cluster_api = await self._get_cluster_api()
        address = self.get_argument("address", "") or None
        stacks = list(await cluster_api.get_node_thread_stacks(address))
        self.write(
            json.dumps(
                {
                    "generate_time": time.time(),
                    "stacks": stacks,
                }
            )
        )

    @web_api("logs", method="get", cache_blocking=True)
    async def fetch_node_log(self):
        cluster_api = await self._get_cluster_api()
        address = self.get_argument("address", "") or None
        # 10MB by default
        size = int(self.get_argument("size", str(10 * 1024 * 1024)))
        offset = 0
        content = await cluster_api.fetch_node_log(size, address=address, offset=offset)
        if size != -1:
            self.write(json.dumps({"content": content}))
        # size == -1 means downloading the current file
        else:
            self.set_header("Content-Type", "application/octet-stream")
            self.set_header("Content-Disposition", "attachment")

            while True:
                if len(content) == 0:  # read to file end
                    await self.finish()
                    break
                else:
                    self.write(content)
                    await self.flush()
                offset = offset + len(content)
                content = await cluster_api.fetch_node_log(
                    size, address=address, offset=offset
                )

    @web_api("workers/scale", method=["patch"], cache_blocking=True)
    async def request_workers(self):
        worker_num = int(self.get_argument(name="worker_num"))
        timeout = self.get_argument(name="timeout", default=None)
        if timeout is not None:
            timeout = int(timeout)
        cluster_api = await self._get_cluster_api()
        addresses = await cluster_api.request_workers(worker_num, timeout=timeout)
        self.write(json.dumps({"workers": addresses}))  # pragma: no cover


web_handlers = {ClusterWebAPIHandler.get_root_pattern(): ClusterWebAPIHandler}


class WebClusterAPI(AbstractClusterAPI, MarsWebAPIClientMixin):
    def __init__(self, address: str, request_rewriter: Callable = None):
        self._address = address.rstrip("/")
        self.request_rewriter = request_rewriter

    @staticmethod
    def _convert_node_dict(node_info_list: Dict[str, Dict]):
        res = {}
        for node_addr, node in node_info_list.items():
            res_dict = node.copy()
            res_dict["status"] = NodeStatus(res_dict["status"])
            res[node_addr] = res_dict
        return res

    async def _get_nodes_info(
        self,
        nodes: List[str] = None,
        role: NodeRole = None,
        env: bool = False,
        resource: bool = False,
        detail: bool = False,
        watch: bool = False,
        statuses: Set[NodeStatus] = None,
        version: Optional[int] = None,
    ):
        statuses_str = (
            ",".join(str(status.value) for status in statuses) if statuses else ""
        )
        args = [
            ("nodes", ",".join(nodes) if nodes else None),
            ("role", role.value if role is not None else None),
            ("env", 1 if env else 0),
            ("resource", 1 if resource else 0),
            ("detail", 1 if detail else 0),
            ("watch", 1 if watch else 0),
            ("statuses", statuses_str),
            ("version", str(version or "")),
        ]
        args_str = "&".join(f"{key}={val}" for key, val in args if val is not None)

        path = f"{self._address}/api/cluster/nodes"
        res = await self._request_url(
            path=path,
            method="POST",
            data=args_str,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        result = json.loads(res.body)
        if watch:
            return result["version"], self._convert_node_dict(result["nodes"])
        else:
            return self._convert_node_dict(result["nodes"])

    async def get_supervisors(self, filter_ready: bool = True) -> List[str]:
        statuses = (
            {NodeStatus.READY}
            if filter_ready
            else {NodeStatus.STARTING, NodeStatus.READY}
        )
        res = await self._get_nodes_info(role=NodeRole.SUPERVISOR, statuses=statuses)
        return list(res.keys())

    @watch_method
    async def watch_supervisors(self, version: Optional[int] = None):
        version, res = await self._get_nodes_info(
            role=NodeRole.SUPERVISOR, watch=True, version=version
        )
        return version, list(res.keys())

    async def get_nodes_info(
        self,
        nodes: List[str] = None,
        role: NodeRole = None,
        env: bool = False,
        resource: bool = False,
        detail: bool = False,
        statuses: Set[NodeStatus] = None,
        exclude_statuses: Set[NodeStatus] = None,
    ):
        statuses = self._calc_statuses(statuses, exclude_statuses)
        return await self._get_nodes_info(
            nodes,
            role=role,
            env=env,
            resource=resource,
            detail=detail,
            watch=False,
            statuses=statuses,
        )

    @watch_method
    async def watch_nodes(
        self,
        role: NodeRole,
        env: bool = False,
        resource: bool = False,
        detail: bool = False,
        statuses: Set[NodeStatus] = None,
        exclude_statuses: Set[NodeStatus] = None,
        version: Optional[int] = None,
    ) -> List[Dict[str, Dict]]:
        statuses = self._calc_statuses(statuses, exclude_statuses)
        return await self._get_nodes_info(
            role=role,
            env=env,
            resource=resource,
            detail=detail,
            watch=True,
            statuses=statuses,
            version=version,
        )

    async def get_all_bands(
        self,
        role: NodeRole = None,
        statuses: Set[NodeStatus] = None,
        exclude_statuses: Set[NodeStatus] = None,
    ) -> Dict[BandType, Resource]:
        statuses = self._calc_statuses(statuses, exclude_statuses)
        statuses_str = (
            ",".join(str(status.value) for status in statuses) if statuses else ""
        )
        params = {}
        if role is not None:  # pragma: no cover
            params["role"] = role.value
        if statuses_str:
            params["statuses"] = statuses_str

        path = f"{self._address}/api/cluster/bands"
        res = await self._request_url("GET", path, params=params)
        return deserialize_serializable(res.body)

    @watch_method
    async def watch_all_bands(
        self,
        role: NodeRole = None,
        statuses: List[NodeStatus] = None,
        exclude_statuses: Set[NodeStatus] = None,
        version: Optional[int] = None,
    ):
        statuses = self._calc_statuses(statuses, exclude_statuses)
        statuses_str = (
            ",".join(str(status.value) for status in statuses) if statuses else ""
        )
        params = dict(watch=1, version=str(version or ""))
        if role is not None:  # pragma: no cover
            params["role"] = role.value
        if statuses_str:
            params["statuses"] = statuses_str

        path = f"{self._address}/api/cluster/bands"
        res = await self._request_url("GET", path, params=params)
        return deserialize_serializable(res.body)

    async def get_mars_versions(self) -> List[str]:
        path = f"{self._address}/api/cluster/versions"
        res = await self._request_url("GET", path)
        return list(json.loads(res.body))

    async def get_node_pool_configs(self, address: str) -> List[Dict]:
        path = f"{self._address}/api/cluster/pools?address={address}"
        res = await self._request_url("GET", path)
        return list(json.loads(res.body)["pools"])

    async def get_node_thread_stacks(self, address: str) -> List[Dict]:
        path = f"{self._address}/api/cluster/stacks?address={address}"
        res = await self._request_url("GET", path)
        return list(json.loads(res.body)["stacks"])

    async def fetch_node_log(
        self, size: int = None, address: str = None, offset: int = 0
    ) -> str:
        path = f"{self._address}/api/cluster/logs?address={address}"
        if size is not None:
            path += f"&&size={size}"
        res = await self._request_url("GET", path)
        if size == -1:
            return res.body.decode(encoding="utf8")
        else:
            return str(json.loads(res.body)["content"])

    async def request_workers(
        self, worker_num: int, timeout: Optional[int] = None
    ) -> List[str]:
        path = f"{self._address}/api/cluster/workers/scale?worker_num={worker_num}"
        if timeout is not None:
            path += f"&&timeout={timeout}"
        res = await self._request_url("PATCH", path, data=b"")
        return json.loads(res.body)["workers"]  # pragma: no cover

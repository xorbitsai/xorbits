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
from abc import ABC, abstractmethod
from typing import Coroutine, List

from .._mars.core.context import Context, get_context
from .._mars.deploy.oscar.session import get_default_session
from .._mars.lib.aio import Isolation, new_isolation
from .._mars.services.cluster import ClusterAPI, NodeRole, WebClusterAPI
from .._mars.utils import implements


class API(ABC):
    """
    This is now for internal usage, APIs could change without notification.
    """

    @classmethod
    def create(cls):
        ctx = get_context()
        if ctx is not None:
            return _MarsContextBasedAPI(ctx)
        else:
            mars_session = get_default_session()
            if mars_session is None:
                raise ValueError("Xorbits is not inited yet, call `xorbits.init()`")
            return _ClientAPI(mars_session.address)

    @abstractmethod
    def list_workers(self) -> List[str]:
        """
        List addresses of workers.

        Returns
        -------
        addresses:
            Addresses of workers
        """


class _MarsContextBasedAPI(API):
    _mars_ctx: Context

    def __init__(self, mars_ctx: Context):
        self._mars_ctx = mars_ctx

    @implements(API.list_workers)
    def list_workers(self) -> List[str]:
        return self._mars_ctx.get_worker_addresses()


class _ClientAPI(API):
    _supervisor_address: str
    _isolation: Isolation
    _mars_cluster_api: ClusterAPI

    def __init__(self, supervisor_addr: str):
        self._supervisor_address = supervisor_addr

        self._inited = False
        # Isolation
        self._isolation = new_isolation("xorbits_api")
        # Mars service APIs
        self._mars_cluster_api = None

    async def _init(self):
        if self._inited:
            return

        if "://" in self._supervisor_address:
            self._mars_cluster_api = WebClusterAPI(self._supervisor_address)
        else:
            self._mars_cluster_api = await ClusterAPI.create(self._supervisor_address)
        self._inited = True

    def _call(self, coro: Coroutine):
        fut = asyncio.run_coroutine_threadsafe(coro, self._isolation.loop)
        return fut.result()

    async def _list_workers(self) -> List[str]:
        await self._init()
        return list(await self._mars_cluster_api.get_nodes_info(role=NodeRole.WORKER))

    @implements(API.list_workers)
    def list_workers(self) -> List[str]:
        return self._call(self._list_workers())

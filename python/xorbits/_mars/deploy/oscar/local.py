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

import asyncio
import logging
import os
import sys
from typing import Dict, List, Optional, Union

import numpy as np
import xoscar as mo
from xoscar.backends.router import Router
from xoscar.metrics import init_metrics

from ...lib.aio import get_isolation
from ...resource import cpu_count, cuda_count, mem_total
from ...services import NodeRole
from ...services.task.execution.api import ExecutionConfig
from ...typing import ClientType, ClusterType
from ..utils import get_third_party_modules_from_config, load_config
from .pool import create_supervisor_actor_pool, create_worker_actor_pool
from .service import start_supervisor, start_worker, stop_supervisor, stop_worker
from .session import AbstractSession, _new_session, ensure_isolation_created

logger = logging.getLogger(__name__)

# The default config file.
DEFAULT_CONFIG_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "config.yml"
)

# the default times to retry subtask.
DEFAULT_SUBTASK_MAX_RETRIES = 3
# the default time to cancel a subtask.
DEFAULT_SUBTASK_CANCEL_TIMEOUT = 5


def _load_config(config: Union[str, Dict] = None):
    return load_config(config, default_config_file=DEFAULT_CONFIG_FILE)


async def new_cluster_in_isolation(
    address: str = "0.0.0.0",
    n_worker: int = 1,
    n_cpu: Union[int, str] = "auto",
    mem_bytes: Union[int, str] = "auto",
    cuda_devices: Union[List[int], str] = "auto",
    subprocess_start_method: str = None,
    backend: str = None,
    config: Union[str, Dict] = None,
    web: bool = True,
    timeout: float = None,
    n_supervisor_process: int = 0,
    numa_external_addr_scheme: str = None,
    numa_enable_internal_addr: bool = None,
    gpu_external_addr_scheme: str = None,
    gpu_enable_internal_addr: bool = None,
    io_external_addr_scheme: Optional[str] = None,
    io_enable_internal_addr: Optional[bool] = None,
    external_addr_scheme: str = None,
    enable_internal_addr: bool = None,
    oscar_extra_conf: dict = None,
    log_config: dict = None,
    storage_config: Optional[dict] = None,
) -> ClientType:
    cluster = LocalCluster(
        address,
        n_worker,
        n_cpu,
        mem_bytes,
        cuda_devices,
        subprocess_start_method,
        backend,
        config,
        web,
        n_supervisor_process,
        numa_external_addr_scheme=numa_external_addr_scheme,
        numa_enable_internal_addr=numa_enable_internal_addr,
        gpu_external_addr_scheme=gpu_external_addr_scheme,
        gpu_enable_internal_addr=gpu_enable_internal_addr,
        io_external_addr_scheme=io_external_addr_scheme,
        io_enable_internal_addr=io_enable_internal_addr,
        external_addr_scheme=external_addr_scheme,
        enable_internal_addr=enable_internal_addr,
        oscar_extra_conf=oscar_extra_conf,
        log_config=log_config,
        storage_config=storage_config,
    )
    await cluster.start()
    return await LocalClient.create(cluster, timeout)


async def new_cluster(
    address: str = "0.0.0.0",
    n_worker: int = 1,
    n_cpu: Union[int, str] = "auto",
    mem_bytes: Union[int, str] = "auto",
    cuda_devices: Union[List[int], str] = "auto",
    subprocess_start_method: str = None,
    backend: str = None,
    config: Union[str, Dict] = None,
    web: bool = True,
    loop: asyncio.AbstractEventLoop = None,
    use_uvloop: Union[bool, str] = "auto",
    n_supervisor_process: int = 0,
    numa_external_addr_scheme: str = None,
    numa_enable_internal_addr: bool = None,
    gpu_external_addr_scheme: str = None,
    gpu_enable_internal_addr: bool = None,
    io_external_addr_scheme: Optional[str] = None,
    io_enable_internal_addr: Optional[bool] = None,
    external_addr_scheme: str = None,
    enable_internal_addr: bool = None,
    oscar_extra_conf: dict = None,
) -> ClientType:
    coro = new_cluster_in_isolation(
        address,
        n_worker=n_worker,
        n_cpu=n_cpu,
        mem_bytes=mem_bytes,
        cuda_devices=cuda_devices,
        subprocess_start_method=subprocess_start_method,
        backend=backend,
        config=config,
        web=web,
        n_supervisor_process=n_supervisor_process,
        numa_external_addr_scheme=numa_external_addr_scheme,
        numa_enable_internal_addr=numa_enable_internal_addr,
        gpu_external_addr_scheme=gpu_external_addr_scheme,
        gpu_enable_internal_addr=gpu_enable_internal_addr,
        io_external_addr_scheme=io_external_addr_scheme,
        io_enable_internal_addr=io_enable_internal_addr,
        external_addr_scheme=external_addr_scheme,
        enable_internal_addr=enable_internal_addr,
        oscar_extra_conf=oscar_extra_conf,
    )
    isolation = ensure_isolation_created(dict(loop=loop, use_uvloop=use_uvloop))
    fut = asyncio.run_coroutine_threadsafe(coro, isolation.loop)
    client = await asyncio.wrap_future(fut)
    client.session.as_default()
    return client


async def stop_cluster(cluster: ClusterType):
    isolation = get_isolation()
    coro = cluster.stop()
    await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(coro, isolation.loop))
    Router.set_instance(None)


class LocalCluster:
    def __init__(
        self: ClusterType,
        address: str = "0.0.0.0",
        n_worker: int = 1,
        n_cpu: Union[int, str] = "auto",
        mem_bytes: Union[int, str] = "auto",
        cuda_devices: Union[List[int], List[List[int]], str] = "auto",
        subprocess_start_method: str = None,
        backend: str = None,
        config: Union[str, Dict] = None,
        web: Union[bool, str] = "auto",
        n_supervisor_process: int = 0,
        numa_external_addr_scheme: str = None,
        numa_enable_internal_addr: bool = None,
        gpu_external_addr_scheme: str = None,
        gpu_enable_internal_addr: bool = None,
        io_external_addr_scheme: Optional[str] = None,
        io_enable_internal_addr: Optional[bool] = None,
        external_addr_scheme: str = None,
        enable_internal_addr: str = None,
        oscar_extra_conf: dict = None,
        log_config: dict = None,
        storage_config: Optional[dict] = None,
    ):
        # auto choose the subprocess_start_method.
        if subprocess_start_method is None:
            subprocess_start_method = (
                "spawn" if sys.platform == "win32" else "forkserver"
            )
        self._address = address
        self._n_worker = n_worker
        self._n_cpu = cpu_count() if n_cpu == "auto" else n_cpu
        self._mem_bytes = mem_total() if mem_bytes == "auto" else mem_bytes
        self._cuda_devices = self._get_cuda_devices(cuda_devices, n_worker)
        self._subprocess_start_method = subprocess_start_method
        self._config = load_config(config, default_config_file=DEFAULT_CONFIG_FILE)
        execution_config = ExecutionConfig.from_config(self._config, backend=backend)
        self._log_config = log_config
        self._backend = execution_config.backend
        self._web = web
        self._n_supervisor_process = n_supervisor_process

        execution_config.merge_from(
            ExecutionConfig.from_params(
                backend=self._backend,
                n_worker=self._n_worker,
                n_cpu=self._n_cpu,
                mem_bytes=self._mem_bytes,
                cuda_devices=self._cuda_devices,
                subtask_cancel_timeout=self._config.get("scheduling", {}).get(
                    "subtask_cancel_timeout", DEFAULT_SUBTASK_CANCEL_TIMEOUT
                ),
                subtask_max_retries=self._config.get("scheduling", {}).get(
                    "subtask_max_retries", DEFAULT_SUBTASK_MAX_RETRIES
                ),
            )
        )

        # process oscar config
        self._process_oscar_config(
            numa_external_addr_scheme=numa_external_addr_scheme,
            numa_enable_internal_addr=numa_enable_internal_addr,
            gpu_external_addr_scheme=gpu_external_addr_scheme,
            gpu_enable_internal_addr=gpu_enable_internal_addr,
            io_external_addr_scheme=io_external_addr_scheme,
            io_enable_internal_addr=io_enable_internal_addr,
            external_addr_scheme=external_addr_scheme,
            enable_internal_addr=enable_internal_addr,
            oscar_extra_conf=oscar_extra_conf,
        )

        # make memory allocation policy more aggressive on local by
        # assuming all the memory is available.
        if self._config.get("scheduling", {}).get("mem_hard_limit", None) is None:
            self._config["scheduling"]["mem_hard_limit"] = None

        if storage_config is not None:
            backend = list(storage_config.keys())[0]
            self._config["storage"]["backends"] = [backend]
            self._config["storage"][backend] = storage_config[backend]

        self._bands_to_resource = execution_config.get_deploy_band_resources()
        self._supervisor_pool = None
        self._worker_pools = []

        self.supervisor_address = None
        self.web_address = None

    def _process_oscar_config(
        self,
        numa_external_addr_scheme: str = None,
        numa_enable_internal_addr: bool = None,
        gpu_external_addr_scheme: str = None,
        gpu_enable_internal_addr: bool = None,
        io_external_addr_scheme: Optional[str] = None,
        io_enable_internal_addr: Optional[bool] = None,
        external_addr_scheme: str = None,
        enable_internal_addr: str = None,
        oscar_extra_conf: dict = None,
    ):
        # process oscar config
        assert "oscar" in self._config
        oscar_config = self._config["oscar"]
        numa_config = oscar_config["numa"]
        numa_external_addr_scheme = (
            numa_external_addr_scheme
            if numa_external_addr_scheme is not None
            else external_addr_scheme
        )
        if numa_external_addr_scheme:
            numa_config["external_addr_scheme"] = numa_external_addr_scheme
        numa_enable_internal_addr = (
            numa_enable_internal_addr
            if numa_enable_internal_addr is not None
            else enable_internal_addr
        )
        if numa_enable_internal_addr is not None:
            numa_config["enable_internal_addr"] = numa_enable_internal_addr
        gpu_config = oscar_config["gpu"]
        gpu_external_addr_scheme = (
            gpu_external_addr_scheme
            if gpu_external_addr_scheme is not None
            else external_addr_scheme
        )
        if gpu_external_addr_scheme:
            gpu_config["external_addr_scheme"] = gpu_external_addr_scheme
        gpu_enable_internal_addr = (
            gpu_enable_internal_addr
            if gpu_enable_internal_addr is not None
            else enable_internal_addr
        )
        if gpu_enable_internal_addr is not None:
            gpu_config["enable_internal_addr"] = gpu_enable_internal_addr

        io_config = oscar_config["io"]
        io_external_addr_scheme = (
            io_external_addr_scheme
            if io_external_addr_scheme is not None
            else external_addr_scheme
        )
        if io_external_addr_scheme is not None:
            io_config["external_addr_scheme"] = io_external_addr_scheme
        io_enable_internal_addr = (
            io_enable_internal_addr
            if io_enable_internal_addr is not None
            else enable_internal_addr
        )
        if io_enable_internal_addr is not None:
            io_config["enable_internal_addr"] = io_enable_internal_addr

        if oscar_extra_conf is not None:
            oscar_config["extra_conf"] = oscar_extra_conf

    @staticmethod
    def _get_cuda_devices(cuda_devices, n_worker):
        if cuda_devices == "auto":
            total = cuda_count()
            all_devices = np.arange(total)
            return [list(arr) for arr in np.array_split(all_devices, n_worker)]

        else:
            if not cuda_devices:
                return [[]] * n_worker
            elif isinstance(cuda_devices[0], int):
                assert n_worker == 1
                return [cuda_devices]
            else:
                assert len(cuda_devices) == n_worker
                return cuda_devices

    @property
    def backend(self):
        return self._backend

    @property
    def external_address(self):
        return self._supervisor_pool.external_address

    async def start(self):
        await self._start_supervisor_pool()
        await self._start_worker_pools()
        # start service
        await self._start_service()

        # init metrics to guarantee metrics use in driver
        metric_configs = self._config.get("metrics", {})
        metric_backend = metric_configs.get("backend")
        init_metrics(metric_backend, config=metric_configs.get(metric_backend))

        if self._web:
            from ...services.web.supervisor import WebActor

            web_actor = await mo.actor_ref(
                WebActor.default_uid(), address=self.supervisor_address
            )
            self.web_address = await web_actor.get_web_address()
            logger.warning("Web service started at %s", self.web_address)

    async def _start_supervisor_pool(self):
        supervisor_modules = get_third_party_modules_from_config(
            self._config, NodeRole.SUPERVISOR
        )
        self._supervisor_pool = await create_supervisor_actor_pool(
            self._address,
            n_process=self._n_supervisor_process,
            modules=supervisor_modules,
            subprocess_start_method=self._subprocess_start_method,
            metrics=self._config.get("metrics", {}),
            web=self._web,
            # passing logging conf to config logging when create pools
            logging_conf=self._log_config,
            oscar_config=self._config.get("oscar"),
        )
        self.supervisor_address = self._supervisor_pool.external_address

    async def _start_worker_pools(self):
        worker_modules = get_third_party_modules_from_config(
            self._config, NodeRole.WORKER
        )
        for band_to_resource, worker_devices in zip(
            self._bands_to_resource, self._cuda_devices
        ):
            worker_pool = await create_worker_actor_pool(
                self._address,
                band_to_resource,
                modules=worker_modules,
                subprocess_start_method=self._subprocess_start_method,
                metrics=self._config.get("metrics", {}),
                cuda_devices=worker_devices,
                web=self._web,
                # passing logging conf to config logging when create pools
                logging_conf=self._log_config,
                oscar_config=self._config.get("oscar"),
            )
            self._worker_pools.append(worker_pool)

    async def _start_service(self):
        self._web = await start_supervisor(
            self.supervisor_address, config=self._config, web=self._web
        )
        for worker_pool, band_to_resource in zip(
            self._worker_pools, self._bands_to_resource
        ):
            await start_worker(
                worker_pool.external_address,
                self.supervisor_address,
                band_to_resource,
                config=self._config,
            )

    async def stop(self):
        from .session import SessionAPI

        # delete all sessions
        session_api = await SessionAPI.create(self._supervisor_pool.external_address)
        await session_api.delete_all_sessions()

        for worker_pool in self._worker_pools:
            await stop_worker(worker_pool.external_address, self._config)
        await stop_supervisor(self._supervisor_pool.external_address, self._config)
        for worker_pool in self._worker_pools:
            await worker_pool.stop()
        await self._supervisor_pool.stop()
        AbstractSession.reset_default()
        Router.set_instance(None)


class LocalClient:
    def __init__(self: ClientType, cluster: ClusterType, session: AbstractSession):
        self._cluster = cluster
        self.session = session

    @classmethod
    async def create(
        cls,
        cluster: LocalCluster,
        timeout: float = None,
    ) -> ClientType:
        session = await _new_session(
            cluster.external_address,
            backend=cluster.backend,
            default=True,
            timeout=timeout,
        )
        client = LocalClient(cluster, session)
        session.client = client
        return client

    @property
    def web_address(self):
        return self._cluster.web_address

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.stop()

    async def stop(self):
        await stop_cluster(self._cluster)

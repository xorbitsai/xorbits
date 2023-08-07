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
import os
import sys
from typing import Dict

import numpy as np
import pandas as pd
import pytest
import xoscar as mo

from ....oscar import create_actor_pool
from ....resource import cuda_count
from ....storage import StorageLevel
from ....tests.core import require_cudf, require_cupy
from ....utils import lazy_import
from ..core import StorageManagerActor
from ..handler import StorageHandlerActor

cupy = lazy_import("cupy")
cudf = lazy_import("cudf")


class MockStorageManagerActor(StorageManagerActor):
    def __init__(
        self, storage_configs: Dict, transfer_block_size: int = None, **kwargs
    ):
        bands = kwargs.pop("bands")
        super().__init__(storage_configs, transfer_block_size, **kwargs)
        self._mock_all_bands = bands

    async def __post_create__(self):
        self._all_bands = list({b for b in self._mock_all_bands})
        await super()._handle_post_create()


@pytest.fixture
async def create_actors(request):
    bands = request.param[0]
    schemes = request.param[1]

    async def start_pool(band: str):
        start_method = (
            os.environ.get("POOL_START_METHOD", "forkserver")
            if sys.platform != "win32"
            else None
        )

        pool = await create_actor_pool(
            "127.0.0.1",
            n_process=2,
            labels=["main", band, "io"],
            subprocess_start_method=start_method,
            external_address_schemes=[None, schemes[0], schemes[1]],
        )
        await pool.start()
        return pool

    worker_pool_1 = await start_pool(bands[0])
    worker_pool_2 = await start_pool(bands[1])

    gpu = bands[0].startswith("gpu")
    xpd = cudf if gpu else pd
    xnp = cupy if gpu else np

    if gpu:
        storage_configs = {"cuda": {}}
    else:
        storage_configs = {"shared_memory": {}}

    manager_ref1 = await mo.create_actor(
        MockStorageManagerActor,
        storage_configs,
        bands=bands,
        uid=StorageManagerActor.default_uid(),
        address=worker_pool_1.external_address,
    )

    manager_ref2 = await mo.create_actor(
        MockStorageManagerActor,
        storage_configs,
        bands=bands,
        uid=StorageManagerActor.default_uid(),
        address=worker_pool_2.external_address,
    )

    yield worker_pool_1.external_address, worker_pool_2.external_address, bands, schemes, gpu, xpd, xnp
    await mo.destroy_actor(manager_ref1)
    await mo.destroy_actor(manager_ref2)
    await worker_pool_1.stop()
    await worker_pool_2.stop()


def _generate_band_scheme():
    gpu_counts = cuda_count()
    gpu_bands = []
    if gpu_counts:
        gpu_bands.extend([(f"gpu-{i}", f"gpu-{i+1}") for i in range(gpu_counts - 1)])
    bands = [("gpu-0", "gpu-0")]
    bands.extend(gpu_bands)
    schemes = [(None, None), ("ucx", "ucx"), (None, "ucx"), ("ucx", None)]
    for band in bands:
        for scheme in schemes:
            yield band, scheme


@require_cudf
@require_cupy
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "create_actors", _generate_band_scheme(), indirect=["create_actors"]
)
async def test_simple_transfer(create_actors):
    worker_address_1, worker_address_2, bands, schemes, gpu, xpd, xnp = create_actors

    session_id = "mock_session"
    if gpu:
        data1 = cupy.random.rand(100, 100)
        data2 = cudf.DataFrame(cupy.random.randint(0, 100, (500, 10)))
    else:
        data1 = np.random.rand(100, 100)
        data2 = pd.DataFrame(np.random.randint(0, 100, (500, 10)))

    storage_handler1 = await mo.actor_ref(
        uid=StorageHandlerActor.gen_uid(bands[0]), address=worker_address_1
    )
    storage_handler2: mo.ActorRefType[StorageHandlerActor] = await mo.actor_ref(
        uid=StorageHandlerActor.gen_uid(bands[1]), address=worker_address_2
    )

    storage_level = StorageLevel.GPU if gpu else StorageLevel.MEMORY
    await storage_handler1.put(session_id, "data_key1", data1, storage_level)
    await storage_handler1.put(session_id, "data_key2", data2, storage_level)
    await storage_handler2.put(session_id, "data_key3", data2, storage_level)

    await storage_handler2.fetch_via_transfer(
        session_id,
        ["data_key1", "data_key2"],
        storage_level,
        (worker_address_1, bands[0]),
        bands[1],
        "raise",
    )

    get_data1 = await storage_handler2.get(session_id, "data_key1")
    xnp.testing.assert_array_equal(data1, get_data1)

    get_data2 = await storage_handler2.get(session_id, "data_key2")
    xpd.testing.assert_frame_equal(data2, get_data2)

    await storage_handler1.fetch_via_transfer(
        session_id,
        ["data_key3"],
        storage_level,
        (worker_address_2, bands[1]),
        bands[0],
        "raise",
    )

    get_data3 = await storage_handler1.get(session_id, "data_key3")
    xpd.testing.assert_frame_equal(data2, get_data3)


@require_cudf
@require_cupy
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "create_actors", _generate_band_scheme(), indirect=["create_actors"]
)
async def test_transfer_same_data(create_actors):
    worker_address_1, worker_address_2, bands, schemes, gpu, xpd, xnp = create_actors

    session_id = "mock_session"
    storage_level = StorageLevel.GPU if gpu else StorageLevel.MEMORY

    data1 = xnp.random.rand(100, 100)
    storage_handler1 = await mo.actor_ref(
        uid=StorageHandlerActor.gen_uid(bands[0]), address=worker_address_1
    )
    storage_handler2 = await mo.actor_ref(
        uid=StorageHandlerActor.gen_uid(bands[1]), address=worker_address_2
    )

    await storage_handler1.put(session_id, "data_key1", data1, storage_level)

    storage_handler2.fetch_via_transfer(
        session_id,
        ["data_key1"],
        storage_level,
        (worker_address_1, bands[0]),
        bands[1],
        "raise",
    )

    # send data to worker2 from worker1
    task1 = asyncio.create_task(
        storage_handler2.fetch_via_transfer(
            session_id,
            ["data_key1"],
            storage_level,
            (worker_address_1, bands[0]),
            bands[1],
            "raise",
        )
    )
    task2 = asyncio.create_task(
        storage_handler2.fetch_via_transfer(
            session_id,
            ["data_key1"],
            storage_level,
            (worker_address_1, bands[0]),
            bands[1],
            "raise",
        )
    )
    await asyncio.gather(task1, task2)
    get_data1 = await storage_handler2.get(session_id, "data_key1")
    xnp.testing.assert_array_equal(data1, get_data1)

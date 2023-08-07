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
import os
import sys
import tempfile
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest
import xoscar as mo
from xoscar.backends.allocate_strategy import IdleLabel
from xoscar.errors import NoIdleSlot

from ....oscar import create_actor_pool
from ....storage import StorageLevel
from ..core import StorageManagerActor, StorageQuotaActor
from ..errors import DataNotExist
from ..handler import StorageHandlerActor
from ..transfer import ReceiverManagerActor, SenderManagerActor

_is_windows = sys.platform.lower().startswith("win")


@pytest.fixture
async def actor_pools():
    async def start_pool():
        start_method = (
            os.environ.get("POOL_START_METHOD", "forkserver")
            if sys.platform != "win32"
            else None
        )

        pool = await create_actor_pool(
            "127.0.0.1",
            n_process=2,
            labels=["main", "numa-0", "io"],
            subprocess_start_method=start_method,
        )
        await pool.start()
        return pool

    worker_pool_1 = await start_pool()
    worker_pool_2 = await start_pool()
    try:
        yield worker_pool_1, worker_pool_2
    finally:
        await worker_pool_1.stop()
        await worker_pool_2.stop()


async def _get_io_address(pool):
    pool_config = (await mo.get_pool_config(pool.external_address)).as_dict()
    return [
        v["external_address"][0]
        for k, v in pool_config["pools"].items()
        if v["label"] == "io"
    ][0]


@pytest.fixture
async def create_actors(actor_pools):
    worker_pool_1, worker_pool_2 = actor_pools

    io1 = await _get_io_address(worker_pool_1)
    io2 = await _get_io_address(worker_pool_2)

    tmp_dir = tempfile.mkdtemp()
    storage_configs = {"shared_memory": {}, "disk": {"root_dirs": f"{tmp_dir}"}}

    manager_ref1 = await mo.create_actor(
        StorageManagerActor,
        storage_configs,
        uid=StorageManagerActor.default_uid(),
        address=worker_pool_1.external_address,
    )

    manager_ref2 = await mo.create_actor(
        StorageManagerActor,
        storage_configs,
        uid=StorageManagerActor.default_uid(),
        address=worker_pool_2.external_address,
    )
    yield worker_pool_1.external_address, worker_pool_2.external_address, io1, io2
    try:
        await mo.destroy_actor(manager_ref1)
        await mo.destroy_actor(manager_ref2)
    except FileNotFoundError:
        pass
    assert not os.path.exists(tmp_dir)


@pytest.fixture
async def create_actors_mock(actor_pools):
    worker_pool_1, worker_pool_2 = actor_pools

    io1 = await _get_io_address(worker_pool_1)
    io2 = await _get_io_address(worker_pool_2)

    tmp_dir = tempfile.mkdtemp()
    storage_configs = {"shared_memory": {}, "disk": {"root_dirs": f"{tmp_dir}"}}

    manager_ref1 = await mo.create_actor(
        MockStorageManagerActor,
        storage_configs,
        uid=MockStorageManagerActor.default_uid(),
        address=worker_pool_1.external_address,
    )

    manager_ref2 = await mo.create_actor(
        MockStorageManagerActor,
        storage_configs,
        uid=MockStorageManagerActor.default_uid(),
        address=worker_pool_2.external_address,
    )
    yield worker_pool_1.external_address, worker_pool_2.external_address, io1, io2
    try:
        await mo.destroy_actor(manager_ref1)
        await mo.destroy_actor(manager_ref2)
    except FileNotFoundError:
        pass
    assert not os.path.exists(tmp_dir)


@pytest.mark.asyncio
async def test_simple_transfer(create_actors):
    worker_address_1, worker_address_2, io1, io2 = create_actors

    data1 = np.random.rand(100, 100)
    data2 = pd.DataFrame(np.random.randint(0, 100, (500, 10)))

    storage_handler1: mo.ActorRefType[StorageHandlerActor] = await mo.actor_ref(
        uid=StorageHandlerActor.gen_uid("numa-0"), address=io1
    )
    storage_handler2: mo.ActorRefType[StorageHandlerActor] = await mo.actor_ref(
        uid=StorageHandlerActor.gen_uid("numa-0"), address=io2
    )

    for level in (StorageLevel.MEMORY, StorageLevel.DISK):
        session_id = f"mock_session_{level}"
        await storage_handler1.put(session_id, "data_key1", data1, level)
        await storage_handler1.put(session_id, "data_key2", data2, level)
        await storage_handler2.put(session_id, "data_key3", data2, level)

        await storage_handler2.fetch_via_transfer(
            session_id,
            ["data_key1", "data_key2"],
            level,
            (io1, "numa-0"),
            "numa-0",
            "raise",
        )

        get_data1 = await storage_handler2.get(session_id, "data_key1")
        np.testing.assert_array_equal(data1, get_data1)

        get_data2 = await storage_handler2.get(session_id, "data_key2")
        pd.testing.assert_frame_equal(data2, get_data2)

        await storage_handler1.fetch_via_transfer(
            session_id, ["data_key3"], level, (io2, "numa-0"), "numa-0", "raise"
        )
        get_data3 = await storage_handler1.get(session_id, "data_key3")
        pd.testing.assert_frame_equal(data2, get_data3)


# test for cancelling happens when writing
class MockReceiverManagerActor(ReceiverManagerActor):
    pass


class MockSenderManagerActor(SenderManagerActor):
    @staticmethod
    async def get_receiver_ref(address: str, band_name: str):
        return await mo.actor_ref(
            address=address, uid=MockReceiverManagerActor.gen_uid(band_name)
        )

    async def _copy_to_receiver(
        self,
        receiver_ref: mo.ActorRefType["MockReceiverManagerActor"],
        local_buffers: List,
        remote_buffers: List,
        session_id: str,
        data_keys: List[str],
        block_size: int,
    ):
        await asyncio.sleep(3)
        await super()._copy_to_receiver(
            receiver_ref,
            local_buffers,
            remote_buffers,
            session_id,
            data_keys,
            block_size,
        )


class MockStorageHandlerActor(StorageHandlerActor):
    async def get_receive_manager_ref(self, band_name: str):
        return await mo.actor_ref(
            address=self.address,
            uid=MockReceiverManagerActor.gen_uid(band_name),
        )

    @staticmethod
    async def get_send_manager_ref(address: str, band: str):
        return await mo.actor_ref(
            address=address, uid=MockSenderManagerActor.gen_uid(band)
        )


class MockStorageManagerActor(StorageManagerActor):
    def __init__(
        self, storage_configs: Dict, transfer_block_size: int = None, **kwargs
    ):
        super().__init__(storage_configs, transfer_block_size, **kwargs)
        self._handler_cls = MockStorageHandlerActor

    async def _create_transfer_actors(self):
        default_band_name = "numa-0"
        sender_strategy = IdleLabel("io", "sender")
        receiver_strategy = IdleLabel("io", "receiver")
        handler_strategy = IdleLabel("io", "handler")
        while True:
            try:
                handler_ref = await mo.create_actor(
                    MockStorageHandlerActor,
                    self._init_params[default_band_name],
                    self._data_manager,
                    self._spill_managers[default_band_name],
                    self._quotas[default_band_name],
                    default_band_name,
                    uid=MockStorageHandlerActor.gen_uid(default_band_name),
                    address=self.address,
                    allocate_strategy=handler_strategy,
                )
                await mo.create_actor(
                    MockSenderManagerActor,
                    data_manager_ref=self._data_manager,
                    storage_handler_ref=handler_ref,
                    uid=MockSenderManagerActor.gen_uid(default_band_name),
                    address=self.address,
                    allocate_strategy=sender_strategy,
                )

                await mo.create_actor(
                    MockReceiverManagerActor,
                    self._quotas[default_band_name],
                    handler_ref,
                    address=self.address,
                    uid=MockReceiverManagerActor.gen_uid(default_band_name),
                    allocate_strategy=receiver_strategy,
                )
            except NoIdleSlot:
                break


@pytest.mark.parametrize(
    "mock_sender_cls, mock_receiver_cls",
    [(MockSenderManagerActor, MockReceiverManagerActor)],
)
@pytest.mark.asyncio
async def test_cancel_transfer(create_actors_mock, mock_sender_cls, mock_receiver_cls):
    _, _, io1, io2 = create_actors_mock

    session_id = "mock_session"
    quota_refs = {
        StorageLevel.MEMORY: await mo.actor_ref(
            StorageQuotaActor,
            StorageLevel.MEMORY,
            5 * 1024 * 1024,
            address=io2,
            uid=StorageQuotaActor.gen_uid("numa-0", StorageLevel.MEMORY),
        )
    }
    storage_handler1 = await mo.actor_ref(
        uid=MockStorageHandlerActor.gen_uid("numa-0"), address=io1
    )
    storage_handler2 = await mo.actor_ref(
        uid=MockStorageHandlerActor.gen_uid("numa-0"), address=io2
    )

    data1 = np.random.rand(10, 10)
    await storage_handler1.put(session_id, "data_key1", data1, StorageLevel.MEMORY)
    data2 = pd.DataFrame(np.random.rand(100, 100))
    await storage_handler1.put(session_id, "data_key2", data2, StorageLevel.MEMORY)

    used_before = (await quota_refs[StorageLevel.MEMORY].get_quota())[1]

    transfer_task = asyncio.create_task(
        storage_handler2.fetch_via_transfer(
            session_id,
            ["data_key1"],
            StorageLevel.MEMORY,
            (io1, "numa-0"),
            "numa-0",
            "raise",
        )
    )

    await asyncio.sleep(1)
    transfer_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await transfer_task

    used = (await quota_refs[StorageLevel.MEMORY].get_quota())[1]
    assert used == used_before

    with pytest.raises(DataNotExist):
        await storage_handler2.get(session_id, "data_key1")

    transfer_task = asyncio.create_task(
        storage_handler2.fetch_via_transfer(
            session_id,
            ["data_key1"],
            StorageLevel.MEMORY,
            (io1, "numa-0"),
            "numa-0",
            "raise",
        )
    )
    await transfer_task
    get_data = await storage_handler2.get(session_id, "data_key1")
    np.testing.assert_array_equal(data1, get_data)

    # cancel when fetch the same data Simultaneously
    if mock_sender_cls is MockSenderManagerActor:
        transfer_task1 = asyncio.create_task(
            storage_handler2.fetch_via_transfer(
                session_id,
                ["data_key1"],
                StorageLevel.MEMORY,
                (io1, "numa-0"),
                "numa-0",
                "raise",
            )
        )
        transfer_task2 = asyncio.create_task(
            storage_handler2.fetch_via_transfer(
                session_id,
                ["data_key2"],
                StorageLevel.MEMORY,
                (io1, "numa-0"),
                "numa-0",
                "raise",
            )
        )
        await asyncio.sleep(1)
        transfer_task1.cancel()
        with pytest.raises(asyncio.CancelledError):
            await transfer_task1
        await transfer_task2
        get_data2 = await storage_handler2.get(session_id, "data_key2")
        pd.testing.assert_frame_equal(get_data2, data2)


@pytest.mark.asyncio
async def test_transfer_same_data(create_actors):
    worker_address_1, worker_address_2, io1, io2 = create_actors

    session_id = "mock_session"
    data1 = np.random.rand(100, 100)
    storage_handler1 = await mo.actor_ref(
        uid=StorageHandlerActor.gen_uid("numa-0"), address=io1
    )
    storage_handler2 = await mo.actor_ref(
        uid=StorageHandlerActor.gen_uid("numa-0"), address=io2
    )

    await storage_handler1.put(session_id, "data_key1", data1, StorageLevel.MEMORY)

    task1 = storage_handler2.fetch_via_transfer(
        session_id,
        ["data_key1"],
        StorageLevel.MEMORY,
        (io1, "numa-0"),
        "numa-0",
        "raise",
    )
    task2 = storage_handler2.fetch_via_transfer(
        session_id,
        ["data_key1"],
        StorageLevel.MEMORY,
        (io1, "numa-0"),
        "numa-0",
        "raise",
    )

    await asyncio.gather(task1, task2)

    get_data = await storage_handler2.get(session_id, "data_key1")
    np.testing.assert_array_equal(get_data, data1)

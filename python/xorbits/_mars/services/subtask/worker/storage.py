import asyncio
import logging
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Type

import xoscar as mo

from ....core import ChunkGraph
from ....typing import BandType
from ..core import Subtask
from ..errors import RunnerStorageDataNotFound

logger = logging.getLogger(__name__)


RunnerStorageRef = mo.ActorRefType["RunnerStorageActor"]


class RunnerStorageActor(mo.Actor):
    _data_storage: Dict[str, Any]
    
    def __init__(
        self,
        band: BandType,
        # worker_address: str,
        slot_id: int,
    ):
        self._band_name = band
        self._slot_id = slot_id
        # self._worker_address = worker_address
        
        self._data_storage = dict()
    
    @classmethod
    def gen_uid(cls, band_name: str, slot_id: int):
        return f"slot_{band_name}_{slot_id}_runner_storage"
    
    async def get_data(
        self, 
        key: str
    ):
        logger.info(
            f"Getting data with key {key} on runner storage with slot id {self._slot_id} and band name {self._band_name}"
        )

        if key not in self._data_storage:
            raise RunnerStorageDataNotFound(
                f"There is no data with key {key}) in Runner Storage {self.uid} at {self.address}, cannot find value. "
            )
        data = yield self._data_storage[key]
        raise mo.Return(data)
    
    async def put_data(
        self, 
        key: str,
        data: Any
    ):
        logger.info(
            f"Putting data with key {key} to runner storage with slot id {self._slot_id} and band name {self._band_name}"
        )
        # Add or update
        self._data_storage[key] = data
    
    async def check(
        self,
    ):
        keys = yield self._data_storage.keys()
        raise mo.Return(keys)
        

# Usage example
async def usage_example():
    # 参考 runner.py 中创建 SubtaskProcessorActor
    try:
        runner_storage_actor: RunnerStorageActor = await mo.create_actor(
            RunnerStorageActor,
            band="band",
            worker_address="worker_address",
            slot_id=0,
            uid=RunnerStorageActor.gen_uid("session_id"), # 应该传什么参
            address="address", # 这是干嘛的
        )
    except mo.ActorAlreadyExist:
        runner_storage_actor: RunnerStorageActor = await mo.actor_ref(
            uid=RunnerStorageActor.gen_uid("session_id"),
            address="address",
        )
    result = await runner_storage_actor.get_data()


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
from ..errors import WorkerStorageDataNotFound

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
        return f"slot_{band_name}_{slot_id}_worker_storage"
    
    async def _get_data(
        self, 
        band: BandType, 
        slot_id: int, 
        data_key: str
    ):
        logger.info(
            f"Getting data with key {data_key} on worker storage with slot id {slot_id} and band name {band}"
        )
        assert band == self._band_name
        assert slot_id == self._slot_id
        
        if data_key not in self._data_storage:
            raise WorkerStorageDataNotFound(
                f"There is no data with key {data_key}) in Worker Storage {self.uid} at {self.address}, cannot find value. "
            )
        data = yield self._data_storage[data_key]
        raise mo.Return(data)
    
    async def _put_data(
        self, 
        band: BandType, 
        slot_id: int, 
        data_key: str, 
        data: Any
    ):
        logger.info(
            f"Putting data with key {data_key} to worker storage with slot id {slot_id} and band name {band}"
        )
        assert band == self._band_name
        assert slot_id == self._slot_id
        # Add or update
        self._data_storage[data_key] = data
        
        

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
    result = await runner_storage_actor._get_data()


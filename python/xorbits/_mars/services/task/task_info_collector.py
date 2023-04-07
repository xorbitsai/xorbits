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

from __future__ import annotations

import asyncio
import functools
import logging
import os
from collections import defaultdict
from typing import Any, Dict, Optional

import xoscar as mo
import yaml

from ...constants import MARS_PROFILING_RESULTS_DIR
from ...core.operand import Fetch, FetchShuffle
from ...lib.aio import AioFileObject, Isolation, alru_cache
from ...lib.filesystem import get_fs, get_scheme, open_file
from ...services.subtask import Subtask, SubtaskGraph
from ...services.task.core import Task
from ...typing import ChunkType, TileableType
from ...utils import calc_data_size

logger = logging.getLogger(__name__)


def collect_on_demand(f):
    @functools.wraps(f)
    async def wrapper(self, *args, **kwargs):
        if await self.collect_task_info():
            await f(self, *args, **kwargs)
        else:
            pass

    return wrapper


class TaskInfoCollector:
    def __init__(
        self,
        session_id: str,
        task_id: str,
        address: str,
        collect_task_info: Optional[bool] = None,
    ):
        self._session_id = session_id
        self._task_id = task_id
        self._address = address
        self._result_chunk_to_subtask = dict()
        self._save_dir = os.path.join(self._session_id, self._task_id)
        self._collect_task_info = collect_task_info
        self._subtask_infos = defaultdict(dict)

    async def collect_task_info(self):
        if self._collect_task_info is None:
            task_api = await self._get_task_api()
            return await task_api.collect_task_info_enabled()
        else:
            return self._collect_task_info

    @collect_on_demand
    async def collect_subtask_operand_structure(
        self,
        task: Task,
        subtask_graph: SubtaskGraph,
        stage_id: str,
    ):
        """
        This method may be called once for each stage ID.
        """
        self._validate_task(task)

        if self._task_id not in self._result_chunk_to_subtask:
            self._result_chunk_to_subtask[self._task_id] = dict()

        op_info = {}
        op_save_path = os.path.join(self._save_dir, f"{stage_id}_stage_operand.yaml")
        subtask_info = {}
        subtask_save_path = os.path.join(
            self._save_dir, f"{stage_id}_stage_subtask.yaml"
        )
        for subtask in subtask_graph.topological_iter():
            chunk_graph = subtask.chunk_graph
            for c in chunk_graph.results:
                self._result_chunk_to_subtask[self._task_id][c.key] = subtask.subtask_id

            visited = set()
            subtask_info = dict()
            subtask_info["pre_subtasks"] = list()
            subtask_info["ops"] = list()
            subtask_info["stage_id"] = stage_id

            for node in chunk_graph.iter_nodes():
                op = node.op
                if isinstance(op, (Fetch, FetchShuffle)):  # pragma: no cover
                    continue
                if op.key in visited:  # pragma: no cover
                    continue

                subtask_info["ops"].append(op.key)
                op_info = dict()
                op_name = type(op).__name__
                op_info["op_name"] = op_name
                if op.stage is not None:  # pragma: no cover
                    op_info["stage"] = op.stage.name
                else:
                    op_info["stage"] = None
                op_info["subtask_id"] = subtask.subtask_id
                op_info["stage_id"] = stage_id
                op_info["predecessors"] = list()
                op_info["inputs"] = dict()

                for input_chunk in op.inputs or []:
                    if input_chunk.key not in visited:
                        op_info["predecessors"].append(input_chunk.key)
                        if (
                            isinstance(input_chunk.op, (Fetch, FetchShuffle))
                            and input_chunk.key
                            in self._result_chunk_to_subtask[self._task_id]
                        ):  # pragma: no cover
                            subtask_info["pre_subtasks"].append(
                                self._result_chunk_to_subtask[self._task_id][
                                    input_chunk.key
                                ]
                            )

                        chunk_dict = dict()
                        if (
                            hasattr(input_chunk, "dtypes")
                            and input_chunk.dtypes is not None
                        ):  # pragma: no cover
                            unique_dtypes = list(set(input_chunk.dtypes.map(str)))
                            chunk_dict["dtypes"] = unique_dtypes
                        elif (
                            hasattr(input_chunk, "dtype")
                            and input_chunk.dtype is not None
                        ):
                            chunk_dict["dtype"] = str(input_chunk.dtype)

                        op_info["inputs"][input_chunk.key] = chunk_dict

                op_info[op.key] = op_info
                visited.add(op.key)

            subtask_info[subtask.subtask_id] = subtask_info

        await self._save_task_info(op_info, op_save_path)
        await self._save_task_info(subtask_info, subtask_save_path)

    @collect_on_demand
    async def collect_result_nodes(
        self, task: Task, subtask_graphs: list[SubtaskGraph]
    ):
        """
        This method will only be called once for each task.
        """
        # TODO: for each stage, would it be better to have an individual file?
        self._validate_task(task)

        result_op_keys = []
        result_subtask_keys = []
        subtask_graph = subtask_graphs[-1]
        for subtask in subtask_graph.topological_iter():
            if len(subtask_graph._successors[subtask]) == 0:
                result_subtask_keys.append(subtask.subtask_id)
                for result in subtask.chunk_graph.results:
                    result_op_keys.append(result.op.key)

        save_path = os.path.join(self._save_dir, "result_nodes.yaml")
        asyncio.create_task(
            self._save_task_info(
                {"op": result_op_keys, "subtask": result_subtask_keys},
                save_path,
            )
        )

    @collect_on_demand
    async def collect_tileable_structure(
        self, task: Task, tileable_to_subtasks: dict[TileableType, list[Subtask]]
    ):
        """
        This method will only be called once for each task.
        """
        self._validate_task(task)
        tileable_dict = dict()
        for tileable, subtasks in tileable_to_subtasks.items():
            tileable_dict[tileable.key] = dict()
            tileable_dict[tileable.key]["subtasks"] = [x.subtask_id for x in subtasks]
            tileable_dict[tileable.key]["name"] = type(tileable.op).__name__
            tileable_dict[tileable.key]["pre_tileables"] = [
                x.key for x in tileable.op.inputs
            ]

        save_path = os.path.join(self._save_dir, "tileables.yaml")
        await self._save_task_info(tileable_dict, save_path)

    @collect_on_demand
    async def append_runtime_subtask_info(
        self,
        subtask: Subtask,
        band: tuple,
        slot_id: int,
        stored_keys: list[str],
        store_sizes: dict[str, int],
        memory_sizes: dict[str, int],
        cost_times: dict[str, tuple],
    ):
        """
        This method will be called once for each subtask.
        """
        self._validate_subtask(subtask)

        subtask_dict = dict()
        subtask_dict["band"] = band
        subtask_dict["slot_id"] = slot_id
        subtask_dict.update(cost_times)
        result_chunks_dict = dict()
        stored_keys = list(set(stored_keys))
        for key in stored_keys:
            chunk_dict = dict()
            chunk_dict["memory_size"] = memory_sizes[key]
            chunk_dict["store_size"] = store_sizes[key]
            if isinstance(key, tuple):
                key = key[0]
            result_chunks_dict[key] = chunk_dict
        subtask_dict["result_chunks"] = result_chunks_dict

        self._subtask_infos[subtask.subtask_id].update(subtask_dict)

    @collect_on_demand
    async def append_runtime_operand_info(
        self,
        subtask: Subtask,
        start: float,
        end: float,
        chunk: ChunkType,
        processor_context,
    ):
        """
        This method will be called once for each chunk.
        """
        self._validate_subtask(subtask)

        op = chunk.op
        if isinstance(op, (Fetch, FetchShuffle)):
            return
        op_info = dict()
        op_key = op.key
        op_info["op_name"] = type(op).__name__
        op_info["subtask_id"] = subtask.subtask_id
        op_info["execute_time"] = (start, end)
        if chunk.key in processor_context:
            op_info["memory_use"] = calc_data_size(processor_context[chunk.key])
            op_info["result_count"] = 1
        else:
            cnt = 0
            total_size = 0
            for k, v in processor_context.items():
                if isinstance(k, tuple) and k[0] == chunk.key:
                    cnt += 1
                    total_size += calc_data_size(v)
            op_info["memory_use"] = total_size
            op_info["result_count"] = cnt

        if "op_info" not in self._subtask_infos[subtask.subtask_id]:
            self._subtask_infos[subtask.subtask_id]["op_info"] = dict()
        self._subtask_infos[subtask.subtask_id]["op_info"][op_key] = op_info

    @collect_on_demand
    async def flush_subtask_info(self, subtask_id: str):
        if subtask_id in self._subtask_infos:
            save_path = os.path.join(self._save_dir, f"{subtask_id}_subtask_info.yaml")
            subtask_info = self._subtask_infos[subtask_id]
            del self._subtask_infos[subtask_id]
            await self._save_task_info(subtask_info, save_path)

    @collect_on_demand
    async def collect_fetch_time(
        self, subtask: Subtask, fetch_start: float, fetch_end: float
    ):
        # TODO: merge into subtask info so that the number of files could be reduced greatly.
        """
        This method will be called once on worker for each subtask.
        """
        self._validate_subtask(subtask)

        save_path = os.path.join(
            self._save_dir, f"{subtask.subtask_id}_subtask_fetch_time.yaml"
        )
        asyncio.create_task(
            self._save_task_info(
                {subtask.subtask_id: (fetch_start, fetch_end)},
                save_path,
            )
        )

    @alru_cache
    async def _get_task_api(self):
        from .api.oscar import TaskAPI

        return await TaskAPI.create(
            session_id=self._session_id, local_address=self._address
        )

    def _validate_task(self, task: Task):
        assert task.session_id == self._session_id
        assert task.task_id == self._task_id

    def _validate_subtask(self, subtask: Subtask):
        assert subtask.session_id == self._session_id
        assert subtask.task_id == self._task_id

    async def _save_task_info(self, task_info: dict, path: str):
        task_api = await self._get_task_api()
        asyncio.create_task(task_api.save_task_info(task_info, path))


class TaskInfoCollectorActor(mo.Actor):
    def __init__(self, profiling_config: Optional[Dict[str, Any]] = None):
        super().__init__()
        if profiling_config is None:
            profiling_config = dict()
        experimental_profiling_config = profiling_config.get("experimental", dict())
        self._collect_task_info_enabled = experimental_profiling_config.get(
            "collect_task_info_enabled", False
        )
        if self._collect_task_info_enabled:
            self._task_info_root_path = experimental_profiling_config.get(
                "task_info_root_path", None
            )
            if self._task_info_root_path is None:
                self._task_info_root_path = (
                    self._get_or_create_default_profiling_results_dir()
                )
            logger.info(f"Task info root path: {self._task_info_root_path}")
            self._task_info_storage_options = experimental_profiling_config.get(
                "task_info_storage_options", {}
            )

            self._scheme = get_scheme(self._task_info_root_path)
            self._fs = get_fs(
                self._task_info_root_path, self._task_info_storage_options
            )
            self._loop = asyncio.new_event_loop()
            self._isolation = Isolation(self._loop)
            self._isolation.start()

    @staticmethod
    def _get_or_create_default_profiling_results_dir():
        os.makedirs(MARS_PROFILING_RESULTS_DIR, exist_ok=True)
        return MARS_PROFILING_RESULTS_DIR

    async def __pre_destroy__(self):
        if self._collect_task_info_enabled:
            self._isolation.stop()

    async def collect_task_info_enabled(self):
        return self._collect_task_info_enabled

    async def save_task_info(self, info: dict, path: str):
        async def save_info():
            try:
                abs_save_path = os.path.join(self._task_info_root_path, path)
                abs_save_dir, _ = os.path.split(abs_save_path)

                # mkdir when saving to a local file.
                if self._scheme == "file" and not self._fs.exists(abs_save_dir):
                    self._fs.mkdir(abs_save_dir, create_parents=True)

                # since append is not available for object storages, create a new file every time.
                async with AioFileObject(open_file(abs_save_path, "w")) as f:
                    s = await asyncio.create_task(asyncio.to_thread(yaml.dump, info))
                    await f.write(s)
            except:  # noqa: E722   # pragma: no cover
                logger.error(f"Failed to save task info", exc_info=True)

        asyncio.run_coroutine_threadsafe(save_info(), self._loop)

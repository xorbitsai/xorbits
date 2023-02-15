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
import tempfile

import yaml

from ... import oscar as mo
from ...constants import MARS_LOG_PATH_KEY
from ...core.operand import Fetch, FetchShuffle
from ...services.subtask import Subtask, SubtaskGraph
from ...services.task.core import Task
from ...typing import ChunkType, TileableType
from ...utils import calc_data_size

logger = logging.getLogger(__name__)


def collect_on_demand(f):
    @functools.wraps(f)
    async def wrapper(self, *args, **kwargs):
        if self.collect_task_info:
            await f(self, *args, **kwargs)
        else:
            pass

    return wrapper


class TaskInfoCollector:
    def __init__(self, address: str, collect_task_info: bool = False):
        self.collect_task_info = collect_task_info
        self.result_chunk_to_subtask = dict()
        self._address = address
        self._task_api = None

    @collect_on_demand
    async def collect_subtask_operand_structure(
        self,
        task: Task,
        subtask_graph: SubtaskGraph,
        stage_id: str,
        trunc_key: int = 5,
    ):
        session_id, task_id = task.session_id, task.task_id
        save_path = os.path.join(session_id, task_id)

        if task_id not in self.result_chunk_to_subtask:
            self.result_chunk_to_subtask[task_id] = dict()

        for subtask in subtask_graph.topological_iter():
            chunk_graph = subtask.chunk_graph
            for c in chunk_graph.results:
                self.result_chunk_to_subtask[task_id][c.key] = subtask.subtask_id

            visited = set()
            subtask_dict = dict()
            subtask_dict["pre_subtasks"] = list()
            subtask_dict["ops"] = list()
            subtask_dict["stage_id"] = stage_id

            for node in chunk_graph.iter_nodes():
                op = node.op
                if isinstance(op, (Fetch, FetchShuffle)):  # pragma: no cover
                    continue
                if op.key in visited:  # pragma: no cover
                    continue

                subtask_dict["ops"].append(op.key)
                op_dict = dict()
                op_name = type(op).__name__
                op_dict["op_name"] = op_name
                if op.stage is not None:  # pragma: no cover
                    op_dict["stage"] = op.stage.name
                else:
                    op_dict["stage"] = None
                op_dict["subtask_id"] = subtask.subtask_id
                op_dict["stage_id"] = stage_id
                op_dict["predecessors"] = list()
                op_dict["inputs"] = dict()

                for input_chunk in op.inputs or []:
                    if input_chunk.key not in visited:
                        op_dict["predecessors"].append(input_chunk.key)
                        if (
                            isinstance(input_chunk.op, (Fetch, FetchShuffle))
                            and input_chunk.key in self.result_chunk_to_subtask[task_id]
                        ):  # pragma: no cover
                            subtask_dict["pre_subtasks"].append(
                                self.result_chunk_to_subtask[task_id][input_chunk.key]
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

                        op_dict["inputs"][input_chunk.key] = chunk_dict

                op_save_path = os.path.join(
                    save_path, f"{stage_id[:trunc_key]}_operand.yaml"
                )
                await self.save_task_info({op.key: op_dict}, op_save_path, session_id)
                visited.add(op.key)

            subtask_save_path = os.path.join(
                save_path, f"{stage_id[:trunc_key]}_subtask.yaml"
            )
            await self.save_task_info(
                {subtask.subtask_id: subtask_dict}, subtask_save_path, session_id
            )

    @collect_on_demand
    async def collect_last_node_info(
        self, task: Task, subtask_graphs: list[SubtaskGraph]
    ):
        last_op_keys = []
        last_subtask_keys = []
        subtask_graph = subtask_graphs[-1]
        for subtask in subtask_graph.topological_iter():
            if len(subtask_graph._successors[subtask]) == 0:
                last_subtask_keys.append(subtask.subtask_id)
                for res in subtask.chunk_graph.results:
                    last_op_keys.append(res.op.key)

        save_path = os.path.join(task.session_id, task.task_id, "last_nodes.yaml")
        await self.save_task_info(
            {"op": last_op_keys, "subtask": last_subtask_keys},
            save_path,
            task.session_id,
        )

    @collect_on_demand
    async def collect_tileable_structure(
        self, task: Task, tileable_to_subtasks: dict[TileableType, list[Subtask]]
    ):
        tileable_dict = dict()
        for tileable, subtasks in tileable_to_subtasks.items():
            tileable_dict[tileable.key] = dict()
            tileable_dict[tileable.key]["subtasks"] = [x.subtask_id for x in subtasks]
            tileable_dict[tileable.key]["name"] = type(tileable.op).__name__
            tileable_dict[tileable.key]["pre_tileables"] = [
                x.key for x in tileable.op.inputs
            ]
        save_path = os.path.join(task.session_id, task.task_id, "tileable.yaml")
        await self.save_task_info(tileable_dict, save_path, task.session_id)

    @collect_on_demand
    async def collect_runtime_subtask_info(
        self,
        subtask: Subtask,
        band: tuple,
        slot_id: int,
        stored_keys: list[str],
        store_sizes: dict[str, int],
        memory_sizes: dict[str, int],
        cost_times: dict[str, tuple],
    ):
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
        save_path = os.path.join(
            subtask.session_id, subtask.task_id, "subtask_runtime.yaml"
        )
        await self.save_task_info(
            {subtask.subtask_id: subtask_dict}, save_path, subtask.session_id
        )

    @collect_on_demand
    async def collect_runtime_operand_info(
        self,
        subtask: Subtask,
        start: float,
        end: float,
        chunk: ChunkType,
        processor_context,
    ):
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
        save_path = os.path.join(
            subtask.session_id, subtask.task_id, "operand_runtime.yaml"
        )
        await self.save_task_info({op_key: op_info}, save_path, subtask.session_id)

    @collect_on_demand
    async def collect_fetch_time(
        self, subtask: Subtask, fetch_start: float, fetch_end: float
    ):
        save_path = os.path.join(subtask.session_id, subtask.task_id, "fetch_time.yaml")
        await self.save_task_info(
            {
                subtask.subtask_id: {
                    "fetch_time": {"start_time": fetch_start, "end_time": fetch_end}
                }
            },
            save_path,
            subtask.session_id,
        )

    async def save_task_info(self, task_info: dict, path: str, session_id: str):
        from .api.oscar import TaskAPI

        if self._task_api is None:
            self._task_api = await TaskAPI.create(
                session_id=session_id, local_address=self._address
            )
        await self._task_api.save_task_info(task_info, path)


class TaskInfoCollectorActor(mo.Actor):
    def __init__(self):
        mars_temp_dir = os.environ.get(MARS_LOG_PATH_KEY)
        if mars_temp_dir is None:
            self.yaml_root_dir = os.path.join(tempfile.tempdir, "mars_task_infos")
        else:
            self.yaml_root_dir = os.path.abspath(
                os.path.join(mars_temp_dir, "../..", "mars_task_infos")
            )
        logger.info(f"Save task info in {self.yaml_root_dir}")

    async def save_task_info(self, task_info: dict, path: str):
        def save_info(task_info_, path_):
            abs_save_path = os.path.join(self.yaml_root_dir, path_)
            abs_save_dir, _ = os.path.split(abs_save_path)
            if not os.path.exists(abs_save_dir):
                os.makedirs(abs_save_dir)

            if os.path.isfile(abs_save_path):
                mode = "a"
            else:
                mode = "w"
            with open(abs_save_path, mode) as f:
                yaml.dump(task_info_, f)

        await asyncio.to_thread(save_info, task_info, path)

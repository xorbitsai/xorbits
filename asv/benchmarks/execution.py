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

import dataclasses
import unittest.mock as mock

from xoscar.serialization import serialize

import xorbits._mars.tensor as mt
from xorbits._mars import new_session
from xorbits._mars.core.graph import (
    TileableGraph,
    TileableGraphBuilder,
    ChunkGraphBuilder,
    ChunkGraph,
)
from xorbits._mars.services.task import new_task_id


def _gen_subtask_chunk_graph(t):
    graph = TileableGraph([t.data])
    next(TileableGraphBuilder(graph).build())
    return next(ChunkGraphBuilder(graph, fuse_enabled=False).build())


@dataclasses.dataclass
class _ASVSubtaskInfo:
    subtask_id: str
    serialized_subtask_chunk_graph: ChunkGraph


class NumExprExecutionSuite:
    """
    Benchmark that times performance of numexpr execution.
    """

    def setup(self):
        self.session = new_session(default=True)
        self.asv_subtasks = []
        for _ in range(100):
            a = mt.arange(1e6)
            b = mt.arange(1e6) * 0.1
            c = mt.sin(a) + mt.arcsinh(a / b)
            subtask_id = new_task_id()
            subtask_chunk_graph = _gen_subtask_chunk_graph(c)
            self.asv_subtasks.append(
                _ASVSubtaskInfo(
                    subtask_id=subtask_id,
                    serialized_subtask_chunk_graph=serialize(subtask_chunk_graph),
                )
            )

            c = a * b - 4.1 * a > 2.5 * b
            subtask_id = new_task_id()
            subtask_chunk_graph = _gen_subtask_chunk_graph(c)
            self.asv_subtasks.append(
                _ASVSubtaskInfo(
                    subtask_id=subtask_id,
                    serialized_subtask_chunk_graph=serialize(subtask_chunk_graph),
                )
            )

    def teardown(self):
        self.session.stop_server()

    def time_numexpr_execution(self):
        for _ in range(50):
            a = mt.arange(1e6)
            b = mt.arange(1e6) * 0.1
            c = mt.sin(a) + mt.arcsinh(a / b)
            c.execute(show_progress=False)
            c = a * b - 4.1 * a > 2.5 * b
            c.execute(show_progress=False)


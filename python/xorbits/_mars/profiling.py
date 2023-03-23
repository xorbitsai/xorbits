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
import copy
import heapq
import json
import logging
import operator
from collections import Counter

from xoscar.profiling import _CallStats
from xoscar.profiling import _ProfilingData as XoscarProfilingData
from xoscar.profiling import _ProfilingOptionDescriptor
from xoscar.profiling import _ProfilingOptions as XoscarProfilingOptions

from .typing import BandType

logger = logging.getLogger(__name__)


class _ProfilingOptions(XoscarProfilingOptions):
    debug_interval_seconds = _ProfilingOptionDescriptor(float, default=None)
    slow_calls_duration_threshold = _ProfilingOptionDescriptor(int, default=1)
    slow_subtasks_duration_threshold = _ProfilingOptionDescriptor(int, default=10)


class _SubtaskStats:
    def __init__(self, options: _ProfilingOptions):
        self._options = options
        self._band_counter = Counter()
        self._slow_subtasks = []

    def collect(self, subtask, band: BandType, duration: float):
        band_address = band[0]
        self._band_counter[band_address] += 1
        if duration < self._options.slow_subtasks_duration_threshold:
            return
        key = (duration, band_address, subtask)
        try:
            if len(self._slow_subtasks) < 10:
                heapq.heappush(self._slow_subtasks, key)
            else:
                heapq.heapreplace(self._slow_subtasks, key)
        except TypeError:
            pass

    def to_dict(self) -> dict:
        band_subtasks = {}
        key = operator.itemgetter(1)
        if len(self._band_counter) > 10:
            items = self._band_counter.items()
            band_subtasks.update(heapq.nlargest(5, items, key=key))
            band_subtasks.update(reversed(heapq.nsmallest(5, items, key=key)))
        else:
            band_subtasks.update(
                sorted(self._band_counter.items(), key=key, reverse=True)
            )
        slow_subtasks = {}
        for duration, band, subtask in sorted(
            self._slow_subtasks, key=operator.itemgetter(0), reverse=True
        ):
            slow_subtasks[f"[{band}]{subtask}"] = duration
        return {"band_subtasks": band_subtasks, "slow_subtasks": slow_subtasks}


class _ProfilingData(XoscarProfilingData):
    def __init__(self):
        super().__init__()
        self._subtask_stats = {}

    def init(self, task_id: str, options=None):
        options = _ProfilingOptions(options)
        logger.info(
            "Init profiling data for task %s with debug interval seconds %s.",
            task_id,
            options.debug_interval_seconds,
        )
        self._data[task_id] = {
            "general": {},
            "serialization": {},
            "most_calls": {},
            "slow_calls": {},
            "band_subtasks": {},
            "slow_subtasks": {},
        }
        self._call_stats[task_id] = _CallStats(options)
        self._subtask_stats[task_id] = _SubtaskStats(options)

        async def _debug_profiling_log():
            while True:
                try:
                    r = self._data.get(task_id, None)
                    if r is None:
                        logger.info("Profiling debug log break.")
                        break
                    r = copy.copy(r)  # shadow copy is enough.
                    r.update(self._call_stats.get(task_id).to_dict())
                    r.update(self._subtask_stats.get(task_id).to_dict())
                    logger.warning("Profiling debug:\n%s", json.dumps(r, indent=4))
                except Exception:
                    logger.exception("Profiling debug log failed.")
                await asyncio.sleep(options.debug_interval_seconds)

        if options.debug_interval_seconds is not None:
            self._debug_task[task_id] = task = asyncio.create_task(
                _debug_profiling_log()
            )
            task.add_done_callback(lambda _: self._debug_task.pop(task_id, None))

    def pop(self, task_id: str):
        logger.info("Pop profiling data of task %s.", task_id)
        debug_task = self._debug_task.pop(task_id, None)
        if debug_task is not None:
            debug_task.cancel()
        r = self._data.pop(task_id, None)
        if r is not None:
            r.update(self._call_stats.pop(task_id).to_dict())
            r.update(self._subtask_stats.pop(task_id).to_dict())
        return r

    def collect_subtask(self, subtask, band: BandType, duration: float):
        if self._subtask_stats:
            stats = self._subtask_stats.get(subtask.task_id)
            if stats is not None:
                stats.collect(subtask, band, duration)


XoscarProfilingData.set_instance(_ProfilingData())
ProfilingData = XoscarProfilingData.get_instance()

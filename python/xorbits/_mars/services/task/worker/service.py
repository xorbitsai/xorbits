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

import xoscar as mo

from ...core import AbstractService
from ..task_info_collector import TaskInfoCollectorActor


class TaskWorkerService(AbstractService):
    async def start(self):
        profiling_config = self._config.get("profiling", dict())
        await mo.create_actor(
            TaskInfoCollectorActor,
            profiling_config,
            uid=TaskInfoCollectorActor.default_uid(),
            address=self._address,
        )

    async def stop(self):
        await mo.destroy_actor(
            mo.create_actor_ref(
                uid=TaskInfoCollectorActor.default_uid(), address=self._address
            )
        )

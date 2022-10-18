# -*- coding: utf-8 -*-
# Copyright 2022 XProbe Inc.
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

from typing import TYPE_CHECKING

from ..adapter.mars import mars_execute

if TYPE_CHECKING:
    from ..adapter.mars import MarsEntity


def execute(mars_entity: "MarsEntity"):
    if need_to_execute(mars_entity):
        mars_execute(mars_entity)


def need_to_execute(mars_entity: "MarsEntity"):
    return (
        hasattr(mars_entity, "_executed_sessions")
        and len(getattr(mars_entity, "_executed_sessions")) == 0
    )

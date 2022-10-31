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

from .adapter import mars_execute
from .data import DataRef


def execute(ref: DataRef):
    if need_to_execute(ref):
        mars_entity = getattr(ref.data, "_mars_entity", None)
        if mars_entity is not None:
            mars_execute(mars_entity)
        else:  # pragma: no cover
            raise NotImplementedError(
                f"Unable to execute an instance of {type(ref).__name__} "
            )


def need_to_execute(ref: DataRef):
    mars_entity = getattr(ref.data, "_mars_entity", None)
    if mars_entity is not None:
        return (
            hasattr(mars_entity, "_executed_sessions")
            and len(getattr(mars_entity, "_executed_sessions")) == 0
        )
    else:  # pragma: no cover
        raise NotImplementedError(
            f"Unable to execute an instance of {type(ref).__name__} "
        )

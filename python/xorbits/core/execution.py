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

from typing import List, Tuple, Union

from .adapter import MarsEntity, mars_execute
from .data import DataRef


def _get_mars_entity(ref: DataRef) -> MarsEntity:
    mars_entity = getattr(ref.data, "_mars_entity", None)
    if mars_entity is not None:
        return mars_entity
    else:  # pragma: no cover
        raise NotImplementedError(
            f"Unable to execute an instance of {type(ref).__name__} "
        )


def run(obj: Union[DataRef, List[DataRef], Tuple[DataRef]]) -> None:
    """
    Manually trigger execution.

    Parameters
    ----------
    obj : DataRef or collection of DataRefs
        DataRef or collection of DataRefs to execute.
    """
    if isinstance(obj, DataRef):
        if need_to_execute(obj):
            mars_execute(_get_mars_entity(obj))
    else:
        refs_to_execute = [_get_mars_entity(ref) for ref in obj if need_to_execute(ref)]
        if refs_to_execute:
            mars_execute(refs_to_execute)


def need_to_execute(ref: DataRef):
    mars_entity = _get_mars_entity(ref)
    return (
        hasattr(mars_entity, "_executed_sessions")
        and len(getattr(mars_entity, "_executed_sessions")) == 0
    )

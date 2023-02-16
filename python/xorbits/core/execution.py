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

import builtins
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


def run(obj: Union[DataRef, List[DataRef], Tuple[DataRef]], **kwargs) -> None:
    """
    Manually trigger execution.

    Parameters
    ----------
    obj : DataRef or collection of DataRefs
        DataRef or collection of DataRefs to execute.
    """
    refs = set(_collect_user_ns_refs())

    if isinstance(obj, DataRef):
        refs.add(obj)
    else:
        refs.union(set(obj))

    mars_tileables = [_get_mars_entity(ref) for ref in refs if need_to_execute(ref)]
    mars_execute(mars_tileables, **kwargs)


def need_to_execute(ref: DataRef):
    mars_entity = _get_mars_entity(ref)
    return (
        hasattr(mars_entity, "_executed_sessions")
        and len(getattr(mars_entity, "_executed_sessions")) == 0
    )


def _collect_user_ns_refs():
    if not _is_interactive() or not _is_ipython_available():
        print("not in interactive env or ipython is not available")
        return []

    ipython = getattr(builtins, "get_ipython")()
    return [v for k, v in ipython.user_ns.items() if isinstance(v, DataRef)]


def _is_interactive():
    import sys

    # See: https://stackoverflow.com/a/64523765/7098025
    return hasattr(sys, "ps1")


def _is_ipython_available():
    return (
        hasattr(builtins, "get_ipython")
        and getattr(builtins, "get_ipython", None) is not None
    )

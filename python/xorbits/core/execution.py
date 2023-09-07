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

import builtins

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


def run(obj: DataRef | list[DataRef] | tuple[DataRef], **kwargs) -> None:
    """
    Manually trigger execution.

    Parameters
    ----------
    obj : DataRef or collection of DataRefs
        DataRef or collection of DataRefs to execute.
    """
    refs = []
    if isinstance(obj, DataRef):
        refs.append(obj)
    else:
        refs.extend(obj)

    refs_to_execute = _collect_executable_user_ns_refs(refs)
    for ref in refs:
        if id(ref) not in refs_to_execute and need_to_execute(ref):
            refs_to_execute[id(ref)] = ref

    mars_tileables = [_get_mars_entity(ref) for ref in refs_to_execute.values()]
    if mars_tileables:
        mars_execute(mars_tileables, **kwargs)


def need_to_execute(ref: DataRef) -> bool:
    mars_entity = _get_mars_entity(ref)
    return (
        hasattr(mars_entity, "_executed_sessions")
        and len(getattr(mars_entity, "_executed_sessions")) == 0
    )


def _is_in_final_results(ref: DataRef, results: list[DataRef]):
    mars_entity = _get_mars_entity(ref)
    result_mars_entities = [_get_mars_entity(result) for result in results]
    for result_mars_entity in result_mars_entities:
        stack = [result_mars_entity]
        while stack:
            e = stack.pop()
            if e.key == mars_entity.key:
                return True
            stack.extend(e.inputs)
    return False


def _collect_executable_user_ns_refs(result_refs: list[DataRef]) -> dict[int, DataRef]:
    """
    Collect DataRefs defined in user's interactive namespace that are able to execute.
    """

    def _is_ipython_output_cache(name: str):
        # _, __, ___ or _<n> where <n> stands for the output line number.
        return name in ["_", "__", "___"] or name.startswith("_") and name[1:].isdigit()

    if not _is_interactive() or not _is_ipython_available():
        return {}

    ipython = getattr(builtins, "get_ipython")()
    return dict(
        (id(v), v)
        for k, v in ipython.user_ns.items()
        if isinstance(v, DataRef)
        and not _is_ipython_output_cache(k)
        and need_to_execute(v)
        and _is_in_final_results(v, result_refs)
    )


def _is_interactive() -> bool:
    import sys

    # See: https://stackoverflow.com/a/64523765/7098025
    return hasattr(sys, "ps1")


def _is_ipython_available() -> bool:
    return (
        hasattr(builtins, "get_ipython")
        and getattr(builtins, "get_ipython", None) is not None
    )

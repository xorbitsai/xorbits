# Copyright 2022-2023 XProbe Inc.
# derived from copyright 1999-2021 Alibaba Group Holding Ltd.
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
import concurrent.futures
import itertools
import json
import logging
import threading
import time
import warnings
from abc import ABC, ABCMeta, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from functools import wraps
from numbers import Integral
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse
from weakref import WeakKeyDictionary, WeakSet, ref

import xoscar as mo
from xoscar.metrics import Metrics

from ...config import options
from ...core import ChunkType, TileableGraph, TileableType, enter_mode
from ...core.operand import Fetch
from ...lib.aio import (
    Isolation,
    alru_cache,
    get_isolation,
    new_isolation,
    stop_isolation,
)
from ...services.cluster import AbstractClusterAPI, ClusterAPI
from ...services.lifecycle import AbstractLifecycleAPI, LifecycleAPI
from ...services.meta import AbstractMetaAPI, MetaAPI
from ...services.session import AbstractSessionAPI, SessionAPI
from ...services.storage import StorageAPI
from ...services.task import AbstractTaskAPI, TaskAPI, TaskResult
from ...services.task.execution.api import Fetcher
from ...services.web import OscarWebAPI
from ...typing import BandType, ClientType
from ...utils import (
    Timer,
    build_fetch,
    classproperty,
    copy_tileables,
    implements,
    merge_chunks,
    merged_chunk_as_tileable_type,
    register_asyncio_task_timeout_detector,
)

logger = logging.getLogger(__name__)


@dataclass
class Progress:
    value: float = 0.0


@dataclass
class Profiling:
    result: dict = None


class TileableWrapper:
    """
    This class gives the Tileable object an additional attribute of whether it needs an attached session or not,
    which is used in the ipython environment to determine whether the Tileable object needs to be re-executed.
    """

    def __init__(self, tileable: TileableType):
        self._data = ref(tileable)
        self._attach_session = True

    @property
    def tileable(self):
        return self._data

    @property
    def attach_session(self) -> bool:
        return self._attach_session

    @attach_session.setter
    def attach_session(self, value):
        self._attach_session = value


class ExecutionInfo:
    def __init__(
        self,
        aio_task: asyncio.Task,
        progress: Progress,
        profiling: Profiling,
        loop: asyncio.AbstractEventLoop,
        to_execute_tileables: List[TileableType],
    ):
        self._aio_task = aio_task
        self._progress = progress
        self._profiling = profiling
        self._loop = loop
        self._to_execute_tileables = to_execute_tileables
        self._future_local = threading.local()

    def _ensure_future(self):
        try:
            self._future_local.future
        except AttributeError:

            async def wait():
                return await self._aio_task

            self._future_local.future = fut = asyncio.run_coroutine_threadsafe(
                wait(), self._loop
            )
            self._future_local.aio_future = asyncio.wrap_future(fut)

    @property
    def loop(self):
        return self._loop

    @property
    def aio_task(self):
        return self._aio_task

    def progress(self) -> float:
        return self._progress.value

    @property
    def to_execute_tileables(self) -> List[TileableType]:
        return self._to_execute_tileables

    def profiling_result(self) -> dict:
        return self._profiling.result

    def result(self, timeout=None):
        self._ensure_future()
        return self._future_local.future.result(timeout=timeout)

    def cancel(self):
        self._aio_task.cancel()

    def __getattr__(self, attr):
        self._ensure_future()
        return getattr(self._future_local.aio_future, attr)

    def __await__(self):
        self._ensure_future()
        return self._future_local.aio_future.__await__()

    def get_future(self):
        self._ensure_future()
        return self._future_local.aio_future


warning_msg = """
No session found, local session \
will be created in background, \
it may take a while before execution. \
If you want to new a local session by yourself, \
run code below:

```
import mars

mars.new_session()
```
"""


class AbstractSession(ABC):
    name = None
    _default = None
    _lock = threading.Lock()

    def __init__(self, address: str, session_id: str):
        self._address = address
        self._session_id = session_id
        self._closed = False

    @property
    def address(self):
        return self._address

    @property
    def session_id(self):
        return self._session_id

    def __eq__(self, other):
        return (
            isinstance(other, AbstractSession)
            and self._address == other.address
            and self._session_id == other.session_id
        )

    def __hash__(self):
        return hash((AbstractSession, self._address, self._session_id))

    def as_default(self) -> "AbstractSession":
        """
        Mark current session as default session.
        """
        AbstractSession._default = self
        return self

    @classmethod
    def reset_default(cls):
        AbstractSession._default = None

    @classproperty
    def default(self):
        return AbstractSession._default


class AbstractAsyncSession(AbstractSession, metaclass=ABCMeta):
    @classmethod
    @abstractmethod
    async def init(
        cls, address: str, session_id: str, new: bool = True, **kwargs
    ) -> "AbstractSession":
        """
        Init a new session.

        Parameters
        ----------
        address : str
            Address.
        session_id : str
            Session ID.
        new : bool
            New a session.
        kwargs

        Returns
        -------
        session
        """

    async def destroy(self):
        """
        Destroy a session.
        """
        self.reset_default()
        self._closed = True

    @abstractmethod
    async def execute(self, *tileables, **kwargs) -> ExecutionInfo:
        """
        Execute tileables.

        Parameters
        ----------
        tileables
            Tileables.
        kwargs
        """

    @abstractmethod
    async def fetch(self, *tileables, **kwargs) -> list:
        """
        Fetch tileables' data.

        Parameters
        ----------
        tileables
            Tileables.

        Returns
        -------
        data
        """

    @abstractmethod
    async def _get_ref_counts(self) -> Dict[str, int]:
        """
        Get all ref counts

        Returns
        -------
        ref_counts
        """

    @abstractmethod
    async def fetch_tileable_op_logs(
        self,
        tileable_op_key: str,
        offsets: Union[Dict[str, List[int]], str, int],
        sizes: Union[Dict[str, List[int]], str, int],
    ) -> Dict:
        """
        Fetch logs given tileable op key.

        Parameters
        ----------
        tileable_op_key : str
            Tileable op key.
        offsets
            Chunk op key to offsets.
        sizes
            Chunk op key to sizes.

        Returns
        -------
        chunk_key_to_logs
        """

    @abstractmethod
    async def get_total_n_cpu(self):
        """
        Get number of cluster cpus.

        Returns
        -------
        number_of_cpu: int
        """

    @abstractmethod
    async def get_cluster_versions(self) -> List[str]:
        """
        Get versions used in current Mars cluster

        Returns
        -------
        version_list : list
            List of versions
        """

    @abstractmethod
    async def get_web_endpoint(self) -> Optional[str]:
        """
        Get web endpoint of current session

        Returns
        -------
        web_endpoint : str
            web endpoint
        """

    @abstractmethod
    async def create_remote_object(
        self, session_id: str, name: str, object_cls, *args, **kwargs
    ):
        """
        Create remote object

        Parameters
        ----------
        session_id : str
            Session ID.
        name : str
        object_cls
        args
        kwargs

        Returns
        -------
        actor_ref
        """

    @abstractmethod
    async def get_remote_object(self, session_id: str, name: str):
        """
        Get remote object.

        Parameters
        ----------
        session_id : str
            Session ID.
        name : str

        Returns
        -------
        actor_ref
        """

    @abstractmethod
    async def destroy_remote_object(self, session_id: str, name: str):
        """
        Destroy remote object.

        Parameters
        ----------
        session_id : str
            Session ID.
        name : str
        """

    async def stop_server(self):
        """
        Stop server.
        """


class AbstractSyncSession(AbstractSession, metaclass=ABCMeta):
    @classmethod
    @abstractmethod
    def init(
        cls,
        address: str,
        session_id: str,
        backend: str = "mars",
        new: bool = True,
        **kwargs,
    ) -> "AbstractSession":
        """
        Init a new session.

        Parameters
        ----------
        address : str
            Address.
        session_id : str
            Session ID.
        backend : str
            Backend.
        new : bool
            New a session.
        kwargs

        Returns
        -------
        session
        """

    @abstractmethod
    def execute(
        self, tileable, *tileables, show_progress: Union[bool, str] = None, **kwargs
    ) -> Union[List[TileableType], TileableType, ExecutionInfo]:
        """
        Execute tileables.

        Parameters
        ----------
        tileable
            Tileable.
        tileables
            Tileables.
        show_progress
            If show progress.
        kwargs

        Returns
        -------
        result
        """

    @abstractmethod
    def fetch(self, *tileables, **kwargs) -> list:
        """
        Fetch tileables.

        Parameters
        ----------
        tileables
            Tileables.
        kwargs

        Returns
        -------
        fetched_data : list
        """

    @abstractmethod
    def fetch_infos(self, *tileables, fields, **kwargs) -> list:
        """
        Fetch infos of tileables.

        Parameters
        ----------
        tileables
            Tileables.
        fields
            List of fields
        kwargs

        Returns
        -------
        fetched_infos : list
        """

    @abstractmethod
    def decref(self, *tileables_keys):
        """
        Decref tileables.

        Parameters
        ----------
        tileables_keys : list
            Tileables' keys
        """

    @abstractmethod
    def incref(self, *tileables_keys):
        """
        Incref tileables.

        Parameters
        ----------
        tileables_keys : list
            Tileables' keys
        """

    @abstractmethod
    def _get_ref_counts(self) -> Dict[str, int]:
        """
        Get all ref counts

        Returns
        -------
        ref_counts
        """

    @abstractmethod
    def fetch_tileable_op_logs(
        self,
        tileable_op_key: str,
        offsets: Union[Dict[str, List[int]], str, int],
        sizes: Union[Dict[str, List[int]], str, int],
    ) -> Dict:
        """
        Fetch logs given tileable op key.

        Parameters
        ----------
        tileable_op_key : str
            Tileable op key.
        offsets
            Chunk op key to offsets.
        sizes
            Chunk op key to sizes.

        Returns
        -------
        chunk_key_to_logs
        """

    @abstractmethod
    def get_total_n_cpu(self):
        """
        Get number of cluster cpus.

        Returns
        -------
        number_of_cpu: int
        """

    @abstractmethod
    def get_cluster_versions(self) -> List[str]:
        """
        Get versions used in current Mars cluster

        Returns
        -------
        version_list : list
            List of versions
        """

    @abstractmethod
    def get_web_endpoint(self) -> Optional[str]:
        """
        Get web endpoint of current session

        Returns
        -------
        web_endpoint : str
            web endpoint
        """

    def fetch_log(
        self,
        tileables: List[TileableType],
        offsets: List[int] = None,
        sizes: List[int] = None,
    ):
        from ...core.custom_log import fetch

        return fetch(tileables, self, offsets=offsets, sizes=sizes)


@dataclass
class ChunkFetchInfo:
    tileable: TileableType
    chunk: ChunkType
    indexes: List[Union[int, slice]]
    data: Any = None


_submitted_tileables = WeakSet()


@enter_mode(build=True, kernel=True)
def gen_submit_tileable_graph(
    session: "AbstractSession",
    result_tileables: List[TileableType],
    warn_duplicated_execution: bool = False,
) -> Tuple[TileableGraph, List[TileableType]]:
    tileable_to_copied = dict()
    indexer = itertools.count()
    result_to_index = {t: i for t, i in zip(result_tileables, indexer)}
    result = list()
    to_execute_tileables = list()
    graph = TileableGraph(result)

    q = list(result_tileables)
    while q:
        tileable = q.pop()
        if tileable in tileable_to_copied:
            continue
        if tileable.cache and tileable not in result_to_index:
            result_to_index[tileable] = next(indexer)
        outputs = tileable.op.outputs
        inputs = tileable.inputs if session not in tileable._executed_sessions else []
        new_inputs = []
        all_inputs_processed = True
        for inp in inputs:
            if inp in tileable_to_copied:
                new_inputs.append(tileable_to_copied[inp])
            elif session in inp._executed_sessions:
                # executed, gen fetch
                fetch_input = build_fetch(inp).data
                tileable_to_copied[inp] = fetch_input
                graph.add_node(fetch_input)
                new_inputs.append(fetch_input)
            else:
                # some input not processed before
                all_inputs_processed = False
                # put back tileable
                q.append(tileable)
                q.append(inp)
                break
        if all_inputs_processed:
            if isinstance(tileable.op, Fetch):
                new_outputs = [tileable]
            elif session in tileable._executed_sessions:
                new_outputs = []
                for out in outputs:
                    fetch_out = tileable_to_copied.get(out, build_fetch(out).data)
                    new_outputs.append(fetch_out)
            else:
                new_outputs = [
                    t.data for t in copy_tileables(outputs, inputs=new_inputs)
                ]
            for out, new_out in zip(outputs, new_outputs):
                tileable_to_copied[out] = new_out
                graph.add_node(new_out)
                for new_inp in new_inputs:
                    graph.add_edge(new_inp, new_out)

    # process results
    result.extend([None] * len(result_to_index))
    for t, i in result_to_index.items():
        result[i] = tileable_to_copied[t]
        to_execute_tileables.append(t)

    if warn_duplicated_execution:
        for n, c in tileable_to_copied.items():
            if not isinstance(c.op, Fetch) and n in _submitted_tileables:
                warnings.warn(
                    f"Tileable {repr(n)} has been submitted before", RuntimeWarning
                )
        # add all nodes into submitted tileables
        _submitted_tileables.update(
            n for n, c in tileable_to_copied.items() if not isinstance(c.op, Fetch)
        )

    return graph, to_execute_tileables


class _IsolatedSession(AbstractAsyncSession):
    def __init__(
        self,
        address: str,
        session_id: str,
        backend: str,
        session_api: AbstractSessionAPI,
        meta_api: AbstractMetaAPI,
        lifecycle_api: AbstractLifecycleAPI,
        task_api: AbstractTaskAPI,
        cluster_api: AbstractClusterAPI,
        web_api: Optional[OscarWebAPI],
        client: ClientType = None,
        timeout: float = None,
        request_rewriter: Callable = None,
    ):
        super().__init__(address, session_id)
        self._backend = backend
        self._session_api = session_api
        self._task_api = task_api
        self._meta_api = meta_api
        self._lifecycle_api = lifecycle_api
        self._cluster_api = cluster_api
        self._web_api = web_api
        self.client = client
        self.timeout = timeout
        self._request_rewriter = request_rewriter

        self._tileable_to_fetch = WeakKeyDictionary()
        self._asyncio_task_timeout_detector_task = (
            register_asyncio_task_timeout_detector()
        )

        # add metrics
        self._tileable_graph_gen_time = Metrics.gauge(
            "mars.tileable_graph_gen_time_secs",
            "Time consuming in seconds to generate a tileable graph",
            ("address", "session_id"),
        )

    @classmethod
    async def _init(
        cls,
        address: str,
        session_id: str,
        backend: str,
        new: bool = True,
        timeout: float = None,
    ):
        session_api = await SessionAPI.create(address)
        if new:
            # create new session
            session_id, session_address = await session_api.create_session(session_id)
        else:
            session_address = await session_api.get_session_address(session_id)
        lifecycle_api = await LifecycleAPI.create(session_id, session_address)
        meta_api = await MetaAPI.create(session_id, session_address)
        task_api = await TaskAPI.create(session_id, session_address)
        cluster_api = await ClusterAPI.create(session_address)
        try:
            web_api = await OscarWebAPI.create(session_address)
        except mo.ActorNotExist:
            web_api = None
        return cls(
            address,
            session_id,
            backend,
            session_api,
            meta_api,
            lifecycle_api,
            task_api,
            cluster_api,
            web_api,
            timeout=timeout,
        )

    @classmethod
    @implements(AbstractAsyncSession.init)
    async def init(
        cls,
        address: str,
        session_id: str,
        backend: str,
        new: bool = True,
        timeout: float = None,
        **kwargs,
    ) -> "AbstractAsyncSession":
        init_local = kwargs.pop("init_local", False)
        request_rewriter = kwargs.pop("request_rewriter", None)
        if init_local:
            from .local import new_cluster_in_isolation

            return (
                await new_cluster_in_isolation(
                    address, timeout=timeout, backend=backend, **kwargs
                )
            ).session

        if kwargs:  # pragma: no cover
            unexpected_keys = ", ".join(list(kwargs.keys()))
            raise TypeError(
                f"Oscar session got unexpected arguments: {unexpected_keys}"
            )

        if urlparse(address).scheme == "http":
            return await _IsolatedWebSession._init(
                address,
                session_id,
                backend,
                new=new,
                timeout=timeout,
                request_rewriter=request_rewriter,
            )
        else:
            return await cls._init(
                address,
                session_id,
                backend,
                new=new,
                timeout=timeout,
            )

    async def _update_progress(self, task_id: str, progress: Progress):
        zero_acc_time = 0
        delay = 0.5
        while True:
            try:
                last_progress_value = progress.value
                progress.value = await self._task_api.get_task_progress(task_id)
                if abs(progress.value - last_progress_value) < 1e-4:
                    # if percentage does not change, we add delay time by 0.5 seconds every time
                    zero_acc_time = min(5, zero_acc_time + 0.5)
                    delay = zero_acc_time
                else:
                    # percentage changes, we use percentage speed to calc progress time
                    zero_acc_time = 0
                    speed = abs(progress.value - last_progress_value) / delay
                    # one percent for one second
                    delay = 0.01 / speed
                delay = max(0.5, min(delay, 5.0))
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                break

    async def _run_in_background(
        self,
        tileables: list,
        task_id: str,
        progress: Progress,
        profiling: Profiling,
    ):
        from ...dataframe.core import DataFrameData

        with enter_mode(build=True, kernel=True):
            exec_tileables = [t.tileable() for t in tileables]
            # wait for task to finish
            cancelled = False
            progress_task = asyncio.create_task(
                self._update_progress(task_id, progress)
            )
            start_time = time.time()
            task_result: Optional[TaskResult] = None
            try:
                if self.timeout is None:
                    check_interval = 30
                else:
                    elapsed = time.time() - start_time
                    check_interval = min(self.timeout - elapsed, 30)

                while True:
                    task_result = await self._task_api.wait_task(
                        task_id, timeout=check_interval
                    )
                    if task_result is not None:
                        break
                    elif (
                        self.timeout is not None
                        and time.time() - start_time > self.timeout
                    ):
                        raise TimeoutError(
                            f"Task({task_id}) running time > {self.timeout}"
                        )
            except asyncio.CancelledError:
                # cancelled
                cancelled = True
                await self._task_api.cancel_task(task_id)
            finally:
                progress_task.cancel()
                if task_result is not None:
                    progress.value = 1.0
                else:
                    # not finished, set progress
                    progress.value = await self._task_api.get_task_progress(task_id)
            if task_result is not None:
                profiling.result = task_result.profiling
                if task_result.profiling:
                    logger.warning(
                        "Profile task %s execution result:\n%s",
                        task_id,
                        json.dumps(task_result.profiling, indent=4),
                    )
                if task_result.error:
                    raise task_result.error.with_traceback(task_result.traceback)
            if cancelled:
                return
            fetch_tileables = await self._task_api.get_fetch_tileables(task_id)
            assert len(exec_tileables) == len(fetch_tileables)

            re_execution_indexes = []
            for i, (tileable, fetch_tileable) in enumerate(
                zip(exec_tileables, fetch_tileables)
            ):
                tileable_shape = tileable.params.get("shape", None)
                fetch_tileable_shape = fetch_tileable.params.get("shape", None)
                # The shape inconsistency is usually due to column pruning,
                # which in ipython can't be fetched directly and needs to be recalculated.
                if (
                    tileable_shape is not None
                    and fetch_tileable_shape is not None
                    and isinstance(tileable, DataFrameData)
                    and len(tileable_shape) == 2
                    and 0
                    < tileable_shape[1]
                    != fetch_tileable_shape[1]
                    > 0  # `>0` condition is for possible `nan`
                ):
                    tileable._executed_sessions.clear()
                    re_execution_indexes.append(i)
                else:
                    self._tileable_to_fetch[tileable] = fetch_tileable
                    # update meta, e.g. unknown shape
                    tileable.params = fetch_tileable.params

            for idx in re_execution_indexes:
                tileables[idx].attach_session = False

    async def execute(self, *tileables, **kwargs) -> ExecutionInfo:
        if self._closed:
            raise RuntimeError("Session closed already")
        fuse_enabled: bool = kwargs.pop("fuse_enabled", None)
        extra_config: dict = kwargs.pop("extra_config", None)
        warn_duplicated_execution: bool = kwargs.pop("warn_duplicated_execution", False)
        if kwargs:  # pragma: no cover
            raise TypeError(f"run got unexpected key arguments {list(kwargs)!r}")

        tileables = [
            tileable.data if hasattr(tileable, "data") else tileable
            for tileable in tileables
        ]

        # build tileable graph
        with Timer() as timer:
            tileable_graph, to_execute_tileables = gen_submit_tileable_graph(
                self, tileables, warn_duplicated_execution=warn_duplicated_execution
            )

        logger.info(
            "Time consuming to generate a tileable graph is %ss with address %s, session id %s",
            timer.duration,
            self.address,
            self._session_id,
        )
        self._tileable_graph_gen_time.record(
            timer.duration, {"address": self.address, "session_id": self._session_id}
        )

        # submit task
        task_id = await self._task_api.submit_tileable_graph(
            tileable_graph,
            fuse_enabled=fuse_enabled,
            extra_config=extra_config,
        )

        progress = Progress()
        profiling = Profiling()

        tileable_wrappers = [TileableWrapper(t) for t in to_execute_tileables]
        # create asyncio.Task
        aio_task = asyncio.create_task(
            self._run_in_background(tileable_wrappers, task_id, progress, profiling)
        )
        return ExecutionInfo(
            aio_task,
            progress,
            profiling,
            asyncio.get_running_loop(),
            tileable_wrappers,
        )

    def _get_to_fetch_tileable(
        self, tileable: TileableType
    ) -> Tuple[TileableType, List[Union[slice, Integral]]]:
        from ...dataframe.indexing.iloc import (
            DataFrameIlocGetItem,
            IndexIlocGetItem,
            SeriesIlocGetItem,
        )
        from ...tensor.indexing import TensorIndex

        slice_op_types = (
            TensorIndex,
            DataFrameIlocGetItem,
            SeriesIlocGetItem,
            IndexIlocGetItem,
        )

        if hasattr(tileable, "data"):
            tileable = tileable.data

        indexes = None
        while tileable not in self._tileable_to_fetch:
            # if tileable's op is slice, try to check input
            if isinstance(tileable.op, slice_op_types):
                indexes = tileable.op.indexes
                tileable = tileable.inputs[0]
                if not all(isinstance(index, (slice, Integral)) for index in indexes):
                    raise ValueError("Only support fetch data slices")
            elif isinstance(tileable.op, Fetch):
                break
            else:
                raise ValueError(f"Cannot fetch unexecuted tileable: {tileable!r}")

        if isinstance(tileable.op, Fetch):
            return tileable, indexes
        else:
            return self._tileable_to_fetch[tileable], indexes

    @classmethod
    def _calc_chunk_indexes(
        cls, fetch_tileable: TileableType, indexes: List[Union[slice, Integral]]
    ) -> Dict[ChunkType, List[Union[slice, int]]]:
        from ...tensor.utils import slice_split

        axis_to_slices = {
            axis: slice_split(ind, fetch_tileable.nsplits[axis])
            for axis, ind in enumerate(indexes)
        }
        result = dict()
        for chunk_index in itertools.product(
            *[v.keys() for v in axis_to_slices.values()]
        ):
            # slice_obj: use tuple, since numpy complains
            #
            # FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use
            # `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array
            # index, `arr[np.array(seq)]`, which will result either in an error or a different result.
            slice_obj = [
                axis_to_slices[axis][chunk_idx]
                for axis, chunk_idx in enumerate(chunk_index)
            ]
            chunk = fetch_tileable.cix[chunk_index]
            result[chunk] = slice_obj
        return result

    def _process_result(self, tileable, result):  # pylint: disable=no-self-use
        return result

    @alru_cache(cache_exceptions=False)
    async def _get_storage_api(self, band: BandType):
        if urlparse(self.address).scheme == "http":
            from ...services.storage.api import WebStorageAPI

            storage_api = WebStorageAPI(
                self._session_id, self.address, band[1], self._request_rewriter
            )
        else:
            storage_api = await StorageAPI.create(self._session_id, band[0], band[1])
        return storage_api

    async def fetch(self, *tileables, **kwargs) -> list:
        fetcher = Fetcher.create(
            self._backend, get_storage_api=self._get_storage_api, **kwargs
        )

        with enter_mode(build=True):
            chunks = []
            get_chunk_metas = []
            fetch_infos_list = []
            for tileable in tileables:
                fetch_tileable, indexes = self._get_to_fetch_tileable(tileable)
                chunk_to_slice = None
                if indexes is not None:
                    chunk_to_slice = self._calc_chunk_indexes(fetch_tileable, indexes)
                fetch_infos = []
                for chunk in fetch_tileable.chunks:
                    if indexes and chunk not in chunk_to_slice:
                        continue
                    chunks.append(chunk)
                    get_chunk_metas.append(
                        self._meta_api.get_chunk_meta.delay(
                            chunk.key,
                            fields=fetcher.required_meta_keys,
                        )
                    )
                    indexes = (
                        chunk_to_slice[chunk] if chunk_to_slice is not None else None
                    )
                    fetch_infos.append(
                        ChunkFetchInfo(tileable=tileable, chunk=chunk, indexes=indexes)
                    )
                fetch_infos_list.append(fetch_infos)

            chunk_metas = await self._meta_api.get_chunk_meta.batch(*get_chunk_metas)
            for chunk, meta, fetch_info in zip(
                chunks, chunk_metas, itertools.chain(*fetch_infos_list)
            ):
                await fetcher.append(chunk.key, meta, fetch_info.indexes)
            fetched_data = await fetcher.get()
            for fetch_info, data in zip(
                itertools.chain(*fetch_infos_list), fetched_data
            ):
                fetch_info.data = data

            result = []
            for tileable, fetch_infos in zip(tileables, fetch_infos_list):
                index_to_data = [
                    (fetch_info.chunk.index, fetch_info.data)
                    for fetch_info in fetch_infos
                ]
                merged = merge_chunks(index_to_data)
                merged = merged_chunk_as_tileable_type(merged, tileable)
                result.append(self._process_result(tileable, merged))
            return result

    async def fetch_infos(self, *tileables, fields, **kwargs) -> list:
        available_fields = {
            "data_key",
            "object_id",
            "object_refs",
            "level",
            "memory_size",
            "store_size",
            "bands",
        }
        if fields is None:
            fields = available_fields
        else:
            for field_name in fields:
                if field_name not in available_fields:  # pragma: no cover
                    raise TypeError(
                        f"`fetch_infos` got unexpected field name: {field_name}"
                    )
            fields = set(fields)

        if kwargs:  # pragma: no cover
            unexpected_keys = ", ".join(list(kwargs.keys()))
            raise TypeError(f"`fetch` got unexpected arguments: {unexpected_keys}")
        # following fields needs to access storage API to get the meta.
        _need_query_storage_fields = {"level", "memory_size", "store_size"}
        _need_query_storage = bool(_need_query_storage_fields & fields)
        with enter_mode(build=True):
            chunk_to_bands, fetch_infos_list, result = await self._query_meta_service(
                tileables, fields, _need_query_storage
            )
            if not _need_query_storage:
                assert result is not None
                return result
            storage_api_to_gets = defaultdict(list)
            storage_api_to_fetch_infos = defaultdict(list)
            for fetch_info in itertools.chain(*fetch_infos_list):
                chunk = fetch_info.chunk
                bands = chunk_to_bands[chunk]
                storage_api = await self._get_storage_api(bands[0])
                storage_api_to_gets[storage_api].append(
                    storage_api.get_infos.delay(chunk.key)
                )
                storage_api_to_fetch_infos[storage_api].append(fetch_info)
            for storage_api in storage_api_to_gets:
                fetched_data = await storage_api.get_infos.batch(
                    *storage_api_to_gets[storage_api]
                )
                infos = storage_api_to_fetch_infos[storage_api]
                for info, data in zip(infos, fetched_data):
                    info.data = data

            result = []
            for fetch_infos in fetch_infos_list:
                fetched = defaultdict(list)
                for fetch_info in fetch_infos:
                    bands = chunk_to_bands[fetch_info.chunk]
                    # Currently there's only one item in the returned List from storage_api.get_infos()
                    data = fetch_info.data[0]
                    if "data_key" in fields:
                        fetched["data_key"].append(fetch_info.chunk.key)
                    if "object_id" in fields:
                        fetched["object_id"].append(data.object_id)
                    if "level" in fields:
                        fetched["level"].append(data.level)
                    if "memory_size" in fields:
                        fetched["memory_size"].append(data.memory_size)
                    if "store_size" in fields:
                        fetched["store_size"].append(data.store_size)
                    # data.band misses ip info, e.g. 'numa-0'
                    # while band doesn't, e.g. (address0, 'numa-0')
                    if "bands" in fields:
                        fetched["bands"].append(bands)
                result.append(fetched)

            return result

    async def _query_meta_service(self, tileables, fields, query_storage):
        chunks = []
        get_chunk_metas = []
        fetch_infos_list = []
        for tileable in tileables:
            fetch_tileable, _ = self._get_to_fetch_tileable(tileable)
            fetch_infos = []
            for chunk in fetch_tileable.chunks:
                chunks.append(chunk)
                get_chunk_metas.append(
                    self._meta_api.get_chunk_meta.delay(
                        chunk.key,
                        fields=["bands"] if query_storage else fields - {"data_key"},
                    )
                )
                fetch_infos.append(
                    ChunkFetchInfo(tileable=tileable, chunk=chunk, indexes=None)
                )
            fetch_infos_list.append(fetch_infos)
        chunk_metas = await self._meta_api.get_chunk_meta.batch(*get_chunk_metas)
        if not query_storage:
            result = []
            chunk_to_meta = dict(zip(chunks, chunk_metas))
            for fetch_infos in fetch_infos_list:
                fetched = defaultdict(list)
                for fetch_info in fetch_infos:
                    if "data_key" in fields:
                        fetched["data_key"].append(fetch_info.chunk.key)
                    for field in fields - {"data_key"}:
                        fetched[field].append(chunk_to_meta[fetch_info.chunk][field])
                result.append(fetched)
            return {}, fetch_infos_list, result
        chunk_to_bands = {
            chunk: meta["bands"] for chunk, meta in zip(chunks, chunk_metas)
        }
        return chunk_to_bands, fetch_infos_list, None

    async def decref(self, *tileable_keys):
        logger.debug("Decref tileables on client: %s", tileable_keys)
        return await self._lifecycle_api.decref_tileables(list(tileable_keys))

    async def incref(self, *tileable_keys):
        logger.debug("Incref tileables on client: %s", tileable_keys)
        return await self._lifecycle_api.incref_tileables(list(tileable_keys))

    async def _get_ref_counts(self) -> Dict[str, int]:
        return await self._lifecycle_api.get_all_chunk_ref_counts()

    async def fetch_tileable_op_logs(
        self,
        tileable_op_key: str,
        offsets: Union[Dict[str, List[int]], str, int],
        sizes: Union[Dict[str, List[int]], str, int],
    ) -> Dict:
        return await self._session_api.fetch_tileable_op_logs(
            self.session_id, tileable_op_key, offsets, sizes
        )

    async def get_total_n_cpu(self):
        all_bands = await self._cluster_api.get_all_bands()
        n_cpu = 0
        for band, resource in all_bands.items():
            _, band_name = band
            if band_name.startswith("numa-"):
                n_cpu += resource.num_cpus
        return n_cpu

    async def get_cluster_versions(self) -> List[str]:
        return list(await self._cluster_api.get_mars_versions())

    async def get_web_endpoint(self) -> Optional[str]:
        if self._web_api is None:
            return None
        return await self._web_api.get_web_address()

    async def destroy(self):
        await super().destroy()
        await self._session_api.delete_session(self._session_id)
        self._tileable_to_fetch.clear()
        if self._asyncio_task_timeout_detector_task:  # pragma: no cover
            self._asyncio_task_timeout_detector_task.cancel()

    async def create_remote_object(
        self, session_id: str, name: str, object_cls, *args, **kwargs
    ):
        return await self._session_api.create_remote_object(
            session_id, name, object_cls, *args, **kwargs
        )

    async def get_remote_object(self, session_id: str, name: str):
        return await self._session_api.get_remote_object(session_id, name)

    async def destroy_remote_object(self, session_id: str, name: str):
        return await self._session_api.destroy_remote_object(session_id, name)

    async def stop_server(self):
        if self.client:
            await self.client.stop()


class _IsolatedWebSession(_IsolatedSession):
    @classmethod
    async def _init(
        cls,
        address: str,
        session_id: str,
        backend: str,
        new: bool = True,
        timeout: float = None,
        request_rewriter: Callable = None,
    ):
        from ...services.cluster import WebClusterAPI
        from ...services.lifecycle import WebLifecycleAPI
        from ...services.meta import WebMetaAPI
        from ...services.session import WebSessionAPI
        from ...services.task import WebTaskAPI

        session_api = WebSessionAPI(address, request_rewriter)
        if new:
            # create new session
            session_id, _ = await session_api.create_session(session_id)
        lifecycle_api = WebLifecycleAPI(session_id, address, request_rewriter)
        meta_api = WebMetaAPI(session_id, address, request_rewriter)
        task_api = WebTaskAPI(session_id, address, request_rewriter)
        cluster_api = WebClusterAPI(address, request_rewriter)

        return cls(
            address,
            session_id,
            backend,
            session_api,
            meta_api,
            lifecycle_api,
            task_api,
            cluster_api,
            None,
            timeout=timeout,
            request_rewriter=request_rewriter,
        )

    async def get_web_endpoint(self) -> Optional[str]:
        return self.address


def _delegate_to_isolated_session(func: Union[Callable, Coroutine]):
    if asyncio.iscoroutinefunction(func):

        @wraps(func)
        async def inner(session: "AsyncSession", *args, **kwargs):
            coro = getattr(session._isolated_session, func.__name__)(*args, **kwargs)
            fut = asyncio.run_coroutine_threadsafe(coro, session._loop)
            return await asyncio.wrap_future(fut)

    else:

        @wraps(func)
        def inner(session: "SyncSession", *args, **kwargs):
            coro = getattr(session._isolated_session, func.__name__)(*args, **kwargs)
            fut = asyncio.run_coroutine_threadsafe(coro, session._loop)
            return fut.result()

    return inner


class AsyncSession(AbstractAsyncSession):
    def __init__(
        self,
        address: str,
        session_id: str,
        isolated_session: _IsolatedSession,
        isolation: Isolation,
    ):
        super().__init__(address, session_id)

        self._isolated_session = _get_isolated_session(isolated_session)
        self._isolation = isolation
        self._loop = isolation.loop

    @classmethod
    def from_isolated_session(
        cls, isolated_session: _IsolatedSession
    ) -> "AsyncSession":
        return cls(
            isolated_session.address,
            isolated_session.session_id,
            isolated_session,
            get_isolation(),
        )

    @property
    def client(self):
        return self._isolated_session.client

    @client.setter
    def client(self, client: ClientType):
        self._isolated_session.client = client

    @classmethod
    @implements(AbstractAsyncSession.init)
    async def init(
        cls,
        address: str,
        session_id: str,
        backend: str = "mars",
        new: bool = True,
        **kwargs,
    ) -> "AbstractSession":
        isolation = ensure_isolation_created(kwargs)
        coro = _IsolatedSession.init(address, session_id, backend, new=new, **kwargs)
        fut = asyncio.run_coroutine_threadsafe(coro, isolation.loop)
        isolated_session = await asyncio.wrap_future(fut)
        return AsyncSession(
            address, isolated_session.session_id, isolated_session, isolation
        )

    def as_default(self) -> AbstractSession:
        AbstractSession._default = self._isolated_session
        return self

    @implements(AbstractAsyncSession.destroy)
    async def destroy(self):
        coro = self._isolated_session.destroy()
        await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(coro, self._loop))
        self.reset_default()

    @implements(AbstractAsyncSession.execute)
    @_delegate_to_isolated_session
    async def execute(self, *tileables, **kwargs) -> ExecutionInfo:
        pass  # pragma: no cover

    @implements(AbstractAsyncSession.fetch)
    async def fetch(self, *tileables, **kwargs) -> list:
        coro = _fetch(*tileables, session=self._isolated_session, **kwargs)
        return await asyncio.wrap_future(
            asyncio.run_coroutine_threadsafe(coro, self._loop)
        )

    @implements(AbstractAsyncSession._get_ref_counts)
    @_delegate_to_isolated_session
    async def _get_ref_counts(self) -> Dict[str, int]:
        pass  # pragma: no cover

    @implements(AbstractAsyncSession.fetch_tileable_op_logs)
    @_delegate_to_isolated_session
    async def fetch_tileable_op_logs(
        self,
        tileable_op_key: str,
        offsets: Union[Dict[str, List[int]], str, int],
        sizes: Union[Dict[str, List[int]], str, int],
    ) -> Dict:
        pass  # pragma: no cover

    @implements(AbstractAsyncSession.get_total_n_cpu)
    @_delegate_to_isolated_session
    async def get_total_n_cpu(self):
        pass  # pragma: no cover

    @implements(AbstractAsyncSession.get_cluster_versions)
    @_delegate_to_isolated_session
    async def get_cluster_versions(self) -> List[str]:
        pass  # pragma: no cover

    @implements(AbstractAsyncSession.create_remote_object)
    @_delegate_to_isolated_session
    async def create_remote_object(
        self, session_id: str, name: str, object_cls, *args, **kwargs
    ):
        pass  # pragma: no cover

    @implements(AbstractAsyncSession.get_remote_object)
    @_delegate_to_isolated_session
    async def get_remote_object(self, session_id: str, name: str):
        pass  # pragma: no cover

    @implements(AbstractAsyncSession.destroy_remote_object)
    @_delegate_to_isolated_session
    async def destroy_remote_object(self, session_id: str, name: str):
        pass  # pragma: no cover

    @implements(AbstractAsyncSession.get_web_endpoint)
    @_delegate_to_isolated_session
    async def get_web_endpoint(self) -> Optional[str]:
        pass  # pragma: no cover

    @implements(AbstractAsyncSession.stop_server)
    async def stop_server(self):
        coro = self._isolated_session.stop_server()
        await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(coro, self._loop))
        stop_isolation()


class ProgressBar:
    def __init__(self, show_progress):
        if not show_progress:
            self.progress_bar = None
        else:
            try:
                from tqdm.auto import tqdm
            except ImportError:
                if show_progress != "auto":  # pragma: no cover
                    raise ImportError("tqdm is required to show progress")
                else:
                    self.progress_bar = None
            else:
                self.progress_bar = tqdm(
                    total=100,
                    bar_format="{l_bar}{bar}| {n:6.2f}/{total_fmt} "
                    "[{elapsed}<{remaining}, {rate_fmt}{postfix}]",
                )

        self.last_progress: float = 0.0

    @property
    def show_progress(self) -> bool:
        return self.progress_bar is not None

    def __enter__(self):
        self.progress_bar.__enter__()

    def __exit__(self, *_):
        self.progress_bar.__exit__(*_)

    def update(self, progress: float):
        progress = min(progress, 100)
        last_progress = self.last_progress
        if self.progress_bar:
            incr = max(progress - last_progress, 0)
            self.progress_bar.update(incr)
        self.last_progress = max(last_progress, progress)


class SyncSession(AbstractSyncSession):
    _execution_pool = concurrent.futures.ThreadPoolExecutor(1)

    def __init__(
        self,
        address: str,
        session_id: str,
        isolated_session: _IsolatedSession,
        isolation: Isolation,
    ):
        super().__init__(address, session_id)

        self._isolated_session = _get_isolated_session(isolated_session)
        self._isolation = isolation
        self._loop = isolation.loop

    @classmethod
    def from_isolated_session(cls, isolated_session: _IsolatedSession) -> "SyncSession":
        return cls(
            isolated_session.address,
            isolated_session.session_id,
            isolated_session,
            get_isolation(),
        )

    @classmethod
    def init(
        cls,
        address: str,
        session_id: str,
        backend: str = "mars",
        new: bool = True,
        **kwargs,
    ) -> "AbstractSession":
        isolation = ensure_isolation_created(kwargs)
        coro = _IsolatedSession.init(address, session_id, backend, new=new, **kwargs)
        fut = asyncio.run_coroutine_threadsafe(coro, isolation.loop)
        isolated_session = fut.result()
        return SyncSession(
            address, isolated_session.session_id, isolated_session, isolation
        )

    def as_default(self) -> AbstractSession:
        AbstractSession._default = self._isolated_session
        return self

    @property
    def _session(self):
        return self._isolated_session

    def _new_cancel_event(self):
        async def new_event():
            return asyncio.Event()

        return asyncio.run_coroutine_threadsafe(new_event(), self._loop).result()

    @implements(AbstractSyncSession.execute)
    def execute(
        self,
        tileable,
        *tileables,
        show_progress: Union[bool, str] = None,
        warn_duplicated_execution: bool = None,
        **kwargs,
    ) -> Union[List[TileableType], TileableType, ExecutionInfo]:
        wait = kwargs.get("wait", True)
        if show_progress is None:
            show_progress = options.show_progress
        if warn_duplicated_execution is None:
            warn_duplicated_execution = options.warn_duplicated_execution
        to_execute_tileables = []
        for t in (tileable,) + tileables:
            to_execute_tileables.extend(t.op.outputs)

        cancelled = kwargs.get("cancelled")
        if cancelled is None:
            cancelled = kwargs["cancelled"] = self._new_cancel_event()

        coro = _execute(
            *set(to_execute_tileables),
            session=self._isolated_session,
            show_progress=show_progress,
            warn_duplicated_execution=warn_duplicated_execution,
            **kwargs,
        )
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            execution_info: ExecutionInfo = fut.result(
                timeout=self._isolated_session.timeout
            )
        except KeyboardInterrupt:  # pragma: no cover
            logger.warning("Cancelling running task")
            cancelled.set()
            fut.result()
            logger.warning("Cancel finished")

        if wait:
            return tileable if len(tileables) == 0 else [tileable] + list(tileables)
        else:
            aio_task = execution_info.aio_task

            async def run():
                await aio_task
                return tileable if len(tileables) == 0 else [tileable] + list(tileables)

            async def driver():
                return asyncio.create_task(run())

            new_aio_task = asyncio.run_coroutine_threadsafe(
                driver(), execution_info.loop
            ).result()
            new_execution_info = ExecutionInfo(
                new_aio_task,
                execution_info._progress,
                execution_info._profiling,
                execution_info.loop,
                to_execute_tileables,
            )
            return new_execution_info

    @implements(AbstractSyncSession.fetch)
    def fetch(self, *tileables, **kwargs) -> list:
        coro = _fetch(*tileables, session=self._isolated_session, **kwargs)
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    @implements(AbstractSyncSession.fetch_infos)
    def fetch_infos(self, *tileables, fields, **kwargs) -> list:
        coro = _fetch_infos(
            *tileables, fields=fields, session=self._isolated_session, **kwargs
        )
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    @implements(AbstractSyncSession.decref)
    @_delegate_to_isolated_session
    def decref(self, *tileables_keys):
        pass  # pragma: no cover

    @implements(AbstractSyncSession.incref)
    @_delegate_to_isolated_session
    def incref(self, *tileables_keys):
        pass  # pragma: no cover

    @implements(AbstractSyncSession._get_ref_counts)
    @_delegate_to_isolated_session
    def _get_ref_counts(self) -> Dict[str, int]:
        pass  # pragma: no cover

    @implements(AbstractSyncSession.fetch_tileable_op_logs)
    @_delegate_to_isolated_session
    def fetch_tileable_op_logs(
        self,
        tileable_op_key: str,
        offsets: Union[Dict[str, List[int]], str, int],
        sizes: Union[Dict[str, List[int]], str, int],
    ) -> Dict:
        pass  # pragma: no cover

    @implements(AbstractSyncSession.get_total_n_cpu)
    @_delegate_to_isolated_session
    def get_total_n_cpu(self):
        pass  # pragma: no cover

    @implements(AbstractSyncSession.get_web_endpoint)
    @_delegate_to_isolated_session
    def get_web_endpoint(self) -> Optional[str]:
        pass  # pragma: no cover

    @implements(AbstractSyncSession.get_cluster_versions)
    @_delegate_to_isolated_session
    def get_cluster_versions(self) -> List[str]:
        pass  # pragma: no cover

    def destroy(self):
        coro = self._isolated_session.destroy()
        asyncio.run_coroutine_threadsafe(coro, self._loop)
        self.reset_default()

    def stop_server(self, isolation=True):
        try:
            coro = self._isolated_session.stop_server()
            future = asyncio.run_coroutine_threadsafe(coro, self._loop)
            future.result(timeout=5)
        finally:
            self.reset_default()
            if isolation:
                stop_isolation()

    def close(self):
        self.destroy()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


async def _execute_with_progress(
    execution_info: ExecutionInfo,
    progress_bar: ProgressBar,
    progress_update_interval: Union[int, float],
    cancelled: asyncio.Event,
):
    with progress_bar:
        while not cancelled.is_set():
            done, _pending = await asyncio.wait(
                [execution_info.get_future()], timeout=progress_update_interval
            )
            if not done:
                if not cancelled.is_set() and execution_info.progress() is not None:
                    progress_bar.update(execution_info.progress() * 100)
            else:
                # done
                if not cancelled.is_set():
                    progress_bar.update(100)
                break


async def _execute(
    *tileables: Tuple[TileableType],
    session: _IsolatedSession = None,
    wait: bool = True,
    show_progress: Union[bool, str] = "auto",
    progress_update_interval: Union[int, float] = 1,
    cancelled: asyncio.Event = None,
    **kwargs,
):
    execution_info = await session.execute(*tileables, **kwargs)

    def _attach_session(future: asyncio.Future):
        if future.exception() is None:
            for t in execution_info.to_execute_tileables:
                if t.attach_session:
                    t.tileable()._attach_session(session)

    execution_info.add_done_callback(_attach_session)
    cancelled = cancelled or asyncio.Event()

    if wait:
        progress_bar = ProgressBar(show_progress)
        if progress_bar.show_progress:
            await _execute_with_progress(
                execution_info, progress_bar, progress_update_interval, cancelled
            )
        else:
            exec_task = asyncio.ensure_future(execution_info)
            cancel_task = asyncio.ensure_future(cancelled.wait())
            await asyncio.wait(
                [exec_task, cancel_task], return_when=asyncio.FIRST_COMPLETED
            )
        if cancelled.is_set():
            execution_info.remove_done_callback(_attach_session)
            execution_info.cancel()
        else:
            # set cancelled to avoid wait task leak
            cancelled.set()
        await execution_info
    else:
        return execution_info


def execute(
    tileable: TileableType,
    *tileables: Tuple[TileableType],
    session: SyncSession = None,
    wait: bool = True,
    new_session_kwargs: dict = None,
    show_progress: Union[bool, str] = None,
    progress_update_interval=1,
    **kwargs,
):
    if isinstance(tileable, (tuple, list)) and len(tileables) == 0:
        tileable, tileables = tileable[0], tileable[1:]
    if session is None:
        session = get_default_or_create(**(new_session_kwargs or dict()))
    session = _ensure_sync(session)
    return session.execute(
        tileable,
        *tileables,
        wait=wait,
        show_progress=show_progress,
        progress_update_interval=progress_update_interval,
        **kwargs,
    )


async def _fetch(
    tileable: TileableType,
    *tileables: Tuple[TileableType],
    session: _IsolatedSession = None,
    **kwargs,
):
    if isinstance(tileable, tuple) and len(tileables) == 0:
        tileable, tileables = tileable[0], tileable[1:]
    session = _get_isolated_session(session)
    data = await session.fetch(tileable, *tileables, **kwargs)
    return data[0] if len(tileables) == 0 else data


async def _fetch_infos(
    tileable: TileableType,
    *tileables: Tuple[TileableType],
    session: _IsolatedSession = None,
    fields: List[str] = None,
    **kwargs,
):
    if isinstance(tileable, tuple) and len(tileables) == 0:
        tileable, tileables = tileable[0], tileable[1:]
    session = _get_isolated_session(session)
    data = await session.fetch_infos(tileable, *tileables, fields=fields, **kwargs)
    return data[0] if len(tileables) == 0 else data


def fetch(
    tileable: TileableType,
    *tileables: Tuple[TileableType],
    session: SyncSession = None,
    **kwargs,
):
    if isinstance(tileable, (tuple, list)) and len(tileables) == 0:
        tileable, tileables = tileable[0], tileable[1:]
    if session is None:
        session = get_default_session()
        if session is None:  # pragma: no cover
            raise ValueError("No session found")

    session = _ensure_sync(session)
    return session.fetch(tileable, *tileables, **kwargs)


def fetch_infos(
    tileable: TileableType,
    *tileables: Tuple[TileableType],
    fields: List[str],
    session: SyncSession = None,
    **kwargs,
):
    if isinstance(tileable, tuple) and len(tileables) == 0:
        tileable, tileables = tileable[0], tileable[1:]
    if session is None:
        session = get_default_session()
        if session is None:  # pragma: no cover
            raise ValueError("No session found")
    session = _ensure_sync(session)
    return session.fetch_infos(tileable, *tileables, fields=fields, **kwargs)


def fetch_log(*tileables: TileableType, session: SyncSession = None, **kwargs):
    if len(tileables) == 1 and isinstance(tileables[0], (list, tuple)):
        tileables = tileables[0]
    if session is None:
        session = get_default_session()
        if session is None:  # pragma: no cover
            raise ValueError("No session found")
    session = _ensure_sync(session)
    return session.fetch_log(list(tileables), **kwargs)


def ensure_isolation_created(kwargs):
    loop = kwargs.pop("loop", None)
    use_uvloop = kwargs.pop("use_uvloop", "auto")

    try:
        return get_isolation()
    except KeyError:
        if loop is None:
            if not use_uvloop:
                loop = asyncio.new_event_loop()
            else:
                try:
                    import uvloop

                    loop = uvloop.new_event_loop()
                except ImportError:
                    if use_uvloop == "auto":
                        loop = asyncio.new_event_loop()
                    else:  # pragma: no cover
                        raise
        return new_isolation(loop=loop)


async def _new_session(
    address: str,
    session_id: str = None,
    backend: str = "mars",
    default: bool = False,
    **kwargs,
) -> AbstractSession:
    session = await AsyncSession.init(
        address, session_id=session_id, backend=backend, new=True, **kwargs
    )
    if default:
        session.as_default()
    return session


def new_session(
    address: str = None,
    session_id: str = None,
    backend: str = "mars",
    default: bool = True,
    new: bool = True,
    **kwargs,
) -> AbstractSession:
    ensure_isolation_created(kwargs)

    if address is None:
        address = "127.0.0.1"
        if "init_local" not in kwargs:
            kwargs["init_local"] = True

    session = SyncSession.init(
        address, session_id=session_id, backend=backend, new=new, **kwargs
    )
    if default:
        session.as_default()
    return session


def get_default_session() -> Optional[SyncSession]:
    if AbstractSession.default is None:
        return
    return SyncSession.from_isolated_session(AbstractSession.default)


def clear_default_session():
    AbstractSession.reset_default()


def get_default_async_session() -> Optional[AsyncSession]:
    if AbstractSession.default is None:
        return
    return AsyncSession.from_isolated_session(AbstractSession.default)


def get_default_or_create(**kwargs):
    with AbstractSession._lock:
        session = AbstractSession.default
        if session is None:
            # no session attached, try to create one
            warnings.warn(warning_msg)
            session = new_session("127.0.0.1", init_local=True, **kwargs)
            session.as_default()
    if isinstance(session, _IsolatedSession):
        session = SyncSession.from_isolated_session(session)
    return _ensure_sync(session)


def stop_server():
    if AbstractSession.default:
        SyncSession.from_isolated_session(AbstractSession.default).stop_server()


def _get_isolated_session(session: AbstractSession) -> _IsolatedSession:
    if hasattr(session, "_isolated_session"):
        return session._isolated_session
    return session


def _ensure_sync(session: AbstractSession) -> SyncSession:
    if isinstance(session, SyncSession):
        return session
    isolated_session = _get_isolated_session(session)
    return SyncSession.from_isolated_session(isolated_session)

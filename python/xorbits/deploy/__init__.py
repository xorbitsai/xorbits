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

import atexit
import os
from typing import Any, Dict, List, Optional, Union

from .._mars.utils import no_default
from ..core.adapter import mars_new_session, session


def init(
    address: Optional[str] = None,
    init_local: bool = no_default,
    session_id: Optional[str] = None,
    timeout: Optional[float] = None,
    n_worker: int = 1,
    n_cpu: Union[int, str] = "auto",
    mem_bytes: Union[int, str] = "auto",
    cuda_devices: Union[List[int], List[List[int]], str] = "auto",
    web: Union[bool, str] = "auto",
    new: bool = True,
    storage_config: Optional[Dict] = None,
    **kwargs,
) -> None:
    """
    Init Xorbits runtime locally or connect to an Xorbits cluster.

    Parameters
    ----------
    address: str, optional
        - if None which is default, address will be "127.0.0.1", a local runtime will be initialized
        - if specify an address for creating a new local runtime, specify like ``<ip>:<port>``
        - if connect to a Xorbits cluster address, e.g. ``http://<supervisor_ip>:<supervisor_web_port>``
    init_local: bool, no default value
        Indicates if creating a new local runtime.

        - If has initialized, ``init_local`` cannot be True, it will skip creating,
        - When address is None and not initialized, ``init_local`` will be True,
        - Otherwise, if it's not specified, False will be set.
    session_id: str, optional
        Session ID, if not specified, a new ID will be auto generated.
    timeout: float
        Timeout about creating a new runtime or connecting to an existing cluster.
    n_worker: int, optional
        How many workers to start when creating a local runtime.

        .. note::

          Take effect only when ``init_local`` is True

    n_cpu: int, str
        Number of CPUs, if ``auto``, the number of cores will be specified.

        .. note::

          Take effect only when ``init_local`` is True

    mem_bytes: int, str
        Memory to use, in bytes, if ``auto``, total memory bytes will be specified.

        .. note::

          Take effect only when ``init_local`` is True

    cuda_devices: list of int, list of list
        - when ``auto`` which is default, all visible GPU devices will be used
        - When n_worker is 1, list of int can be specified, means the device indexes to use
        - When n_worker > 1, list of list can be specified for each worker.

        .. note::

          Take effect only when ``init_local`` is True

    web: bool, str
        If creating a web UI.

        .. note::

          Take effect only when ``init_local`` is True

    new: bool

        If creating a new session when connecting to an existing cluster.

        .. note::

          Take effect only when ``init_local`` is False

    storage_config: dict, optional
        storage backend and its configuration when init a new local cluster.
        Using ``shared_memory`` storage backend by default.
        Currently, support `shared_memory`` and ``mmap`` two options.

        .. note::

          Take effect only when ``init_local`` is True
    """
    default_session = session.get_default_session()
    if init_local is True and default_session is not None:
        raise ValueError(
            f"`init_local` cannot be True if has initialized,"
            f"call `shutdown` before init."
        )
    if storage_config is not None:
        backends = list(storage_config.keys())
        if len(backends) > 1 or len(backends) == 0:
            raise ValueError("Only support one storage backend.")
        backend = backends[0]
        if backend not in ["shared_memory", "mmap", "disk"]:
            raise ValueError(
                "Only support one of these storage backends: `shared_memory`, `mmap`."
            )

    if init_local is no_default:
        # if address not specified and has not initialized,
        # force to initialize a local runtime.
        # otherwise when init_local not specified, set to False
        init_local = True if (address is None and default_session is None) else False

    kw: Dict[str, Any] = dict(
        address=address,
        init_local=init_local,
        session_id=session_id,
        timeout=timeout,
        new=new,
    )
    if address is None and default_session is not None:
        # if has initialized, no need to new session
        if new:  # pragma: no branch
            session.get_default_session().as_default()
        return
    if init_local:
        kw.update(
            dict(
                n_worker=n_worker,
                n_cpu=n_cpu,
                mem_bytes=mem_bytes,
                cuda_devices=cuda_devices,
                web=web,
                storage_config=storage_config,
            )
        )
    kw.update(kwargs)
    mars_new_session(**kw)


def shutdown(**kw) -> None:
    """
    Shutdown current local runtime.
    """
    sess = session.get_default_session()
    if sess:  # pragma: no branch
        # When connecting to an existing cluster by xorbits.init,
        # stop_server will not do anything,
        # so we need to delete the session first to release resources
        sess.destroy()
        sess.stop_server(**kw)


def safe_shutdown(**kw) -> None:  # pragma: no cover
    """
    Shutdown current local runtime, and ignore all the errors.
    """
    try:
        shutdown(**kw)
    except:  # noqa: E722  # pylint: disable=bare-except
        pass


# shutdown when python process exit
atexit.register(safe_shutdown)


__all__ = [
    "init",
    "shutdown",
]

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

from typing import List, Optional, Union

from .._mars.utils import no_default
from ..core.adapter import mars_new_session, mars_stop_server


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
    **kwargs
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

        - When address is None, ``init_local`` will be True,
        - Otherwise, if it's not specified, False will be set.
    session_id: str, optional
        Session ID, if not specified, a new ID will be auto generated.
    timeout: float
        Timeout about creating a new runtime or connecting to an exising cluster.
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
    """
    if init_local is no_default:
        # if address not specified, force to initialize a local runtime
        # otherwise when init_local not specified, set to False
        init_local = True if address is None else False

    kw = dict(
        address=address,
        init_local=init_local,
        session_id=session_id,
        timeout=timeout,
        n_worker=n_worker,
        n_cpu=n_cpu,
        mem_bytes=mem_bytes,
        cuda_devices=cuda_devices,
        web=web,
        new=new,
    )
    kw.update(kwargs)
    mars_new_session(**kw)


def shutdown() -> None:
    """
    Shutdown current local runtime.
    """
    mars_stop_server()


__all__ = [
    "init",
    "shutdown",
]

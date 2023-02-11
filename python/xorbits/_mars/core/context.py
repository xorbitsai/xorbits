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

import os
from abc import ABC, abstractmethod
from typing import Dict, List

from ..storage.base import StorageLevel
from ..typing import BandType, SessionType
from ..utils import classproperty


class Context(ABC):
    """
    Context that providing API that can be
    used inside `tile` and `execute`.
    """

    all_contexts = []

    def __init__(
        self,
        session_id: str = None,
        supervisor_address: str = None,
        worker_address: str = None,
        local_address: str = None,
        band: BandType = None,
    ):
        if session_id is None:
            # try to get session id from environment
            session_id = os.environ.get("MARS_SESSION_ID")
            if session_id is None:
                raise ValueError("session_id should be provided to create a context")
        if supervisor_address is None:
            # try to get supervisor address from environment
            supervisor_address = os.environ.get("MARS_SUPERVISOR_ADDRESS")
            if supervisor_address is None:
                raise ValueError(
                    "supervisor_address should be provided to create a context"
                )

        self.session_id = session_id
        self.supervisor_address = supervisor_address
        self.worker_address = worker_address
        self.local_address = local_address
        self.band = band

    @abstractmethod
    def get_current_session(self) -> SessionType:
        """
        Get current session

        Returns
        -------
        session
        """

    @abstractmethod
    def get_local_host_ip(self) -> str:
        """
        Get local worker's host ip

        Returns
        -------
        host_ip : str
        """

    @abstractmethod
    def get_supervisor_addresses(self) -> List[str]:
        """
        Get supervisor addresses.

        Returns
        -------
        supervisor_addresses : list
        """

    @abstractmethod
    def get_worker_addresses(self) -> List[str]:
        """
        Get worker addresses.

        Returns
        -------
        worker_addresses : list
        """

    @abstractmethod
    def get_worker_bands(self) -> List[BandType]:
        """
        Get worker bands.

        Returns
        -------
        worker_bands : list
        """

    @abstractmethod
    def get_total_n_cpu(self) -> int:
        """
        Get number of cpus.

        Returns
        -------
        number_of_cpu: int
        """

    @abstractmethod
    def get_slots(self) -> int:
        """
        Get num of slots of current band

        Returns
        -------
        number_of_bands: int
        """

    @abstractmethod
    def get_chunks_result(self, data_keys: List[str], fetch_only: bool = False) -> List:
        """
        Get result of chunks.

        Parameters
        ----------
        data_keys : list
            Data keys.
        fetch_only : bool
            If fetch_only, only fetch data but not return.

        Returns
        -------
        results : list
            Result of chunks if not fetch_only, else return None
        """

    @abstractmethod
    def get_chunks_meta(
        self, data_keys: List[str], fields: List[str] = None, error="raise"
    ) -> List[Dict]:
        """
        Get meta of chunks.

        Parameters
        ----------
        data_keys : list
            Data keys.
        fields : list
            Fields to filter.
        error : str
            raise, ignore

        Returns
        -------
        meta_list : list
            Meta list.
        """

    @abstractmethod
    def get_storage_info(self, address: str, level: StorageLevel):
        """
        Get the customized storage backend info of requested storage backend level at given worker.

        Parameters
        ----------
        address: str
            The worker address.
        level: StorageLevel
            The storage level to fetch the backend info.

        Returns
        -------
        info: dict
            Customized storage backend info dict of all workers. The key is
            worker address, the value is the backend info dict.
        """

    @abstractmethod
    def create_remote_object(self, name: str, object_cls, *args, **kwargs):
        """
        Create remote object.

        Parameters
        ----------
        name : str
            Object name.
        object_cls
            Object class.
        args
        kwargs

        Returns
        -------
        ref
        """

    @abstractmethod
    def get_remote_object(self, name: str):
        """
        Get remote object

        Parameters
        ----------
        name : str
            Object name.

        Returns
        -------
        ref
        """

    @abstractmethod
    def destroy_remote_object(self, name: str):
        """
        Destroy remote object.

        Parameters
        ----------
        name : str
            Object name.
        """

    @abstractmethod
    def register_custom_log_path(
        self,
        session_id: str,
        tileable_op_key: str,
        chunk_op_key: str,
        worker_address: str,
        log_path: str,
    ):
        """
        Register custom log path.

        Parameters
        ----------
        session_id : str
            Session ID.
        tileable_op_key : str
            Key of tileable's op.
        chunk_op_key : str
            Kye of chunk's op.
        worker_address : str
            Worker address.
        log_path : str
            Log path.
        """

    def new_custom_log_dir(self) -> str:
        """
        New custom log dir.

        Returns
        -------
        custom_log_dir : str
            Custom log dir.
        """

    def set_running_operand_key(self, session_id: str, op_key: str):
        """
        Set key of running operand.

        Parameters
        ----------
        session_id : str
        op_key : str
        """

    def set_progress(self, progress: float):
        """
        Set progress of running operand.

        Parameters
        ----------
        progress : float
        """

    def __enter__(self):
        Context.all_contexts.append(self)

    def __exit__(self, *_):
        Context.all_contexts.pop()

    @classproperty
    def current(cls):
        return cls.all_contexts[-1] if cls.all_contexts else None


def set_context(context: Context):
    Context.all_contexts.append(context)


def get_context() -> Context:
    return Context.current

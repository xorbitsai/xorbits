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

import logging
import os
from typing import Optional

import xoscar as mo

from ...constants import DEFAULT_MARS_LOG_FILE_NAME, MARS_LOG_DIR_KEY

logger = logging.getLogger(__name__)


class FileLoggerActor(mo.Actor):
    """
    Read log file path from env (source from yaml config) for each node (including supervisor and all the workers).
    Expose interface for web frontend to fetch log content.
    """

    def __init__(self):
        super().__init__()
        log_dir = os.environ.get(MARS_LOG_DIR_KEY)
        self._log_file_path: Optional[str] = None
        if log_dir is None:
            # normally, this env var is set when a supervisor or worker pool is created, but some
            # test cases just use an actor pool for convenience, in which case the env var
            # MARS_LOG_DIR is None.
            logger.warning("Log directory is not set")
        else:
            self._log_file_path = os.path.join(log_dir, DEFAULT_MARS_LOG_FILE_NAME)

    def fetch_logs(self, size: int, offset: int) -> str:
        """
        Externally exposed interface.

        Parameters
        ----------
        size
        offset

        Returns
        -------

        """
        if self._log_file_path is None or not os.path.exists(self._log_file_path):
            return ""

        if size != -1:
            content = self._get_n_bytes_tail_file(size)
        else:
            content = self._get_n_bytes_from_pos(10 * 1024 * 1024, offset)
        return content

    def _get_n_bytes_tail_file(self, bytes_num: int) -> str:
        """
        Read last n bytes of file.

        Parameters
        ----------
        bytes_num: the bytes to read. -1 means read the whole file.

        Returns
        -------

        """
        f_size = os.stat(self._log_file_path).st_size
        target = f_size - bytes_num if f_size > bytes_num else 0
        with open(self._log_file_path) as f:
            f.seek(target)
            if target == 0:
                res = f.read()
            else:
                f.readline()
                res = f.read()

        return res

    def _get_n_bytes_from_pos(self, size: int, offset: int) -> str:
        """
        Read n bytes from a position.
        Parameters
        ----------
        size
        offset

        Returns
        -------

        """
        with open(self._log_file_path) as f:
            f.seek(offset)
            res = f.read(size)
        return res

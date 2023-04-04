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

import os

from xoscar.utils import get_current_user

MARS_TEMP_DIR = f"/tmp/{get_current_user()}/xorbits"
MARS_TEMP_DIR_WIN = f"C:\\Temp\\{get_current_user()}\\xorbits"

# logging.
MARS_LOG_DIR_KEY = "MARS_LOG_DIR"
DEFAULT_MARS_LOG_DIR = os.path.join(MARS_TEMP_DIR, "logs")
DEFAULT_MARS_LOG_DIR_WIN = os.path.join(MARS_TEMP_DIR_WIN, "logs")
DEFAULT_MARS_LOG_FILE_NAME = "xorbits.log"
DEFAULT_MARS_LOG_MAX_BYTES = 100 * 1024 * 1024
DEFAULT_MARS_LOG_BACKUP_COUNT = 30

# profiling.
MARS_PROFILING_RESULTS_DIR = os.path.join(MARS_TEMP_DIR, "profiling")
MARS_PROFILING_RESULTS_DIR_WIN = os.path.join(MARS_TEMP_DIR_WIN, "profiling")

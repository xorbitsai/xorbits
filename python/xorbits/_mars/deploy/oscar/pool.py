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
import configparser
import logging
import os
import sys
import tempfile
from typing import Dict, List, Optional, Tuple

from ... import oscar as mo
from ...constants import MARS_LOG_PATH_KEY, MARS_LOG_PREFIX, MARS_TMP_DIR_PREFIX
from ...resource import Resource, cuda_count

logger = logging.getLogger(__name__)


def _need_suspend_sigint() -> bool:
    try:
        from IPython import get_ipython

        return get_ipython() is not None
    except ImportError:
        return False


def _get_root_logger_level_and_format() -> Tuple[str, Optional[str]]:
    root = logging.getLogger()
    level = logging.getLevelName(root.getEffectiveLevel())
    if level.startswith("WARN"):
        level = "WARN"
    handler = root.handlers[0] if root.handlers else None
    fmt = handler.formatter._fmt if handler else None
    return level, fmt


def _parse_file_logging_config(
    file_path: str,
    log_path: str,
    level: Optional[str],
    formatter: Optional[str] = None,
    from_cmd: bool = False,
) -> configparser.RawConfigParser:
    """
    If env is ipython (from_cmd=False), the log level and format on the web follow our default configuration file,
    and the level and format on the console use the user's configuration (logging.basicConfig) or keep the default.

    If env is cmd (from_cmd=True, e.g. user invokes `python -m mars.worker`),
    the log level and format on the web and console follow user's config (--log-level and --log-format)
    or our default configuration file.
    """
    config = configparser.RawConfigParser()
    config.read(file_path)
    logger_sections = [
        "logger_root",
        "logger_main",
        "logger_deploy",
        "logger_oscar",
        "logger_services",
        "logger_dataframe",
        "logger_learn",
        "logger_tensor",
        "handler_stream_handler",
        "handler_file_handler",
    ]
    all_sections = config.sections()
    for section in logger_sections:
        if level and section in all_sections:
            config[section]["level"] = level.upper()

    if "handler_file_handler" in config:
        if sys.platform.startswith("win"):
            log_path = log_path.replace("\\", "/")
        config["handler_file_handler"]["args"] = rf"('{log_path}',)"
    if formatter:
        format_section = "formatter_formatter"
        config[format_section]["format"] = formatter

    stream_handler_sec = "handler_stream_handler"
    file_handler_sec = "handler_file_handler"
    root_sec = "logger_root"
    # If not from cmd (like ipython) and user uses its own config file,
    # need to judge that whether handler_stream_handler section is in the config.
    if not from_cmd and stream_handler_sec in all_sections:
        # console and web log keeps the default config as root logger
        root_level, root_fmt = _get_root_logger_level_and_format()
        config[file_handler_sec]["level"] = root_level or "WARN"
        config[stream_handler_sec]["level"] = root_level or "WARN"
        config[root_sec]["level"] = root_level or "WARN"
        if root_fmt:
            config.add_section("formatter_console")
            config["formatter_console"]["format"] = root_fmt
            config["formatters"]["keys"] += ",console"
            config[stream_handler_sec]["formatter"] = "console"
    return config


def _config_logging(**kwargs) -> Optional[configparser.RawConfigParser]:
    web: bool = kwargs.get("web", True)
    # web=False usually means it is a test environment.
    if not web:
        return
    if kwargs.get("logging_conf", None) is None:
        return
    config = kwargs["logging_conf"]
    from_cmd = config.get("from_cmd", False)
    log_dir = config.get("log_dir", None)
    log_conf_file = config.get("file", None)
    level = config.get("level", None)
    formatter = config.get("formatter", None)
    logging_config_path = log_conf_file or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "file-logging.conf"
    )
    # default config, then create a temp file
    if (os.environ.get(MARS_LOG_PATH_KEY, None)) is None or (
        not os.path.exists(os.environ[MARS_LOG_PATH_KEY])
    ):
        if log_dir is None:
            mars_tmp_dir = tempfile.mkdtemp(prefix=MARS_TMP_DIR_PREFIX)
        else:
            mars_tmp_dir = os.path.join(log_dir, MARS_TMP_DIR_PREFIX)
            os.makedirs(mars_tmp_dir, exist_ok=True)
        _, file_path = tempfile.mkstemp(prefix=MARS_LOG_PREFIX, dir=mars_tmp_dir)
        os.environ[MARS_LOG_PATH_KEY] = file_path
        logging_conf = _parse_file_logging_config(
            logging_config_path, file_path, level, formatter, from_cmd
        )
        # bind user's level and format when using default log conf
        logging.config.fileConfig(
            logging_conf,
            disable_existing_loggers=False,
        )
        logger.debug("Use logging config file at %s", logging_config_path)
        return logging_conf
    else:
        logging_conf = _parse_file_logging_config(
            logging_config_path,
            os.environ[MARS_LOG_PATH_KEY],
            level,
            formatter,
            from_cmd,
        )
        logging.config.fileConfig(
            logging_conf,
            os.environ[MARS_LOG_PATH_KEY],
            disable_existing_loggers=False,
        )
        logger.debug("Use logging config file at %s", logging_config_path)
        return logging_conf


async def create_supervisor_actor_pool(
    address: str,
    n_process: int,
    modules: List[str] = None,
    ports: List[int] = None,
    subprocess_start_method: str = None,
    oscar_config: dict = None,
    **kwargs,
):
    logging_conf = _config_logging(**kwargs)
    kwargs["logging_conf"] = logging_conf
    if oscar_config:
        numa_config = oscar_config.get("numa", dict())
        numa_external_address_scheme = numa_config.get("external_addr_scheme", None)
        numa_enable_internal_address = numa_config.get("enable_internal_addr", True)
        external_address_schemes = [numa_external_address_scheme] * (n_process + 1)
        enable_internal_addresses = [numa_enable_internal_address] * (n_process + 1)
        extra_conf = oscar_config["extra_conf"]
    else:
        external_address_schemes = enable_internal_addresses = extra_conf = None
    return await mo.create_actor_pool(
        address,
        n_process=n_process,
        ports=ports,
        external_address_schemes=external_address_schemes,
        enable_internal_addresses=enable_internal_addresses,
        modules=modules,
        subprocess_start_method=subprocess_start_method,
        suspend_sigint=_need_suspend_sigint(),
        extra_conf=extra_conf,
        **kwargs,
    )


async def create_worker_actor_pool(
    address: str,
    band_to_resource: Dict[str, Resource],
    n_io_process: int = 1,
    modules: List[str] = None,
    ports: List[int] = None,
    cuda_devices: List[int] = None,
    subprocess_start_method: str = None,
    oscar_config: dict = None,
    **kwargs,
):
    logging_conf = _config_logging(**kwargs)
    kwargs["logging_conf"] = logging_conf
    # TODO: support NUMA when ready
    n_process = sum(
        int(resource.num_cpus) or int(resource.num_gpus)
        for resource in band_to_resource.values()
    )
    envs = []
    labels = ["main"]

    oscar_config = oscar_config or dict()
    numa_config = oscar_config.get("numa", dict())
    numa_external_address_scheme = numa_config.get("external_addr_scheme")
    numa_enable_internal_address = numa_config.get("enable_internal_addr")
    gpu_config = oscar_config.get("gpu", dict())
    gpu_external_address_scheme = gpu_config.get("external_addr_scheme")
    gpu_enable_internal_address = gpu_config.get("enable_internal_addr")
    extra_conf = oscar_config.get("extra_conf", dict())

    if cuda_devices is None:  # pragma: no cover
        env_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if not env_devices:
            cuda_devices = list(range(cuda_count()))
        else:
            cuda_devices = [int(i) for i in env_devices.split(",")]

    external_address_schemes = [numa_external_address_scheme]
    enable_internal_addresses = [numa_enable_internal_address]
    i_gpu = iter(sorted(cuda_devices))
    for band, resource in band_to_resource.items():
        if band.startswith("gpu"):
            idx = str(next(i_gpu))
            envs.append({"CUDA_VISIBLE_DEVICES": idx})
            labels.append(f"gpu-{idx}")
            external_address_schemes.append(gpu_external_address_scheme)
            enable_internal_addresses.append(gpu_enable_internal_address)
        else:
            assert band.startswith("numa")
            num_cpus = int(resource.num_cpus)
            if cuda_devices:
                # if has cuda device, disable all cuda devices for numa processes
                envs.extend([{"CUDA_VISIBLE_DEVICES": "-1"} for _ in range(num_cpus)])
            labels.extend([band] * num_cpus)
            external_address_schemes.extend(
                [numa_external_address_scheme for _ in range(num_cpus)]
            )
            enable_internal_addresses.extend(
                [numa_enable_internal_address for _ in range(num_cpus)]
            )

    return await mo.create_actor_pool(
        address,
        n_process=n_process,
        ports=ports,
        n_io_process=n_io_process,
        labels=labels,
        envs=envs,
        modules=modules,
        subprocess_start_method=subprocess_start_method,
        suspend_sigint=_need_suspend_sigint(),
        external_address_schemes=external_address_schemes,
        enable_internal_addresses=enable_internal_addresses,
        extra_conf=extra_conf,
        **kwargs,
    )

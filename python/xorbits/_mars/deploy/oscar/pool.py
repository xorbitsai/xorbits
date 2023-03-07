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

import configparser
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from ....utils import get_default_logging_config_file_absolute
from ... import oscar as mo
from ...constants import (
    DEFAULT_MARS_LOG_BACKUP_COUNT,
    DEFAULT_MARS_LOG_DIR,
    DEFAULT_MARS_LOG_DIR_WIN,
    DEFAULT_MARS_LOG_FILE_NAME,
    DEFAULT_MARS_LOG_MAX_BYTES,
    MARS_LOG_DIR_KEY,
)
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


def _apply_log_level(level: Optional[str], config: configparser.RawConfigParser):
    if level is not None:
        loggers = config["loggers"]["keys"].split(",")
        for lg in loggers:
            lg = lg.strip()
            config[f"logger_{lg}"]["level"] = level.upper()
    return config


def _apply_log_format(fmt: Optional[str], config: configparser.RawConfigParser):
    if fmt is not None:
        formatters = config["formatters"]["keys"].split(",")
        for formatter in formatters:
            formatter = formatter.strip()
            config[f"formatter_{formatter}"]["format"] = fmt
    return config


def _apply_log_dir(
    log_dir: str,
    is_default_logging_config: bool,
    is_default_log_dir: bool,
    config: configparser.RawConfigParser,
):
    if is_default_logging_config:
        assert "handler_file_handler" in config
        if sys.platform.startswith("win"):
            log_dir = log_dir.replace("\\", "/")
        log_file_path = os.path.join(log_dir, DEFAULT_MARS_LOG_FILE_NAME)
        config["handler_file_handler"]["args"] = (
            rf"('{log_file_path}', 'a', "
            rf"{DEFAULT_MARS_LOG_MAX_BYTES}, "
            rf"{DEFAULT_MARS_LOG_BACKUP_COUNT})"
        )
    elif not is_default_log_dir:
        # TODO: don't have a perfect way to handle this situation.
        raise ValueError(
            "Unable to change the log directory when using a user-defined logging"
            " configuration file."
        )
    return config


def _parse_file_logging_config(
    logging_conf: Dict[str, Any]
) -> configparser.RawConfigParser:
    """
    Parse the file logging config and apply logging configurations with higher priority.

    For interactive environments (from_cmd=False), the log level and format on the web follow our
    default configuration file, while the level and format on the console follow the user's
    configuration (logging.basicConfig) or keep the default.

    If env is cmd (from_cmd=True, e.g. user invokes `python -m mars.worker`),
    the log level and format on the web and console follow user's config.
    """
    from_cmd = logging_conf.get("from_cmd", False)
    log_level = logging_conf.get("level", None)
    log_fmt = logging_conf.get("format", None)
    log_dir = _get_log_dir(logging_conf.get("log_dir", None))
    log_config_path = _get_log_config_path(logging_conf.get("log_config", None))
    is_default_logging_config = logging_conf.get("log_config", None) is None
    is_default_log_dir = logging_conf.get("log_dir", None) is None

    # for FileLoggerActor(s).
    os.environ[MARS_LOG_DIR_KEY] = log_dir
    logger.info("Logging configuration file %s", log_config_path)
    logger.info("Logging directory %s", log_dir)

    config = configparser.RawConfigParser()
    config.read(log_config_path)
    _apply_log_format(log_fmt, config)
    _apply_log_level(log_level, config)
    _apply_log_dir(log_dir, is_default_logging_config, is_default_log_dir, config)

    # optimize logs for local runtimes (like IPython).
    if not from_cmd and is_default_logging_config:
        # console outputs follows user's configuration.
        root_level, root_fmt = _get_root_logger_level_and_format()

        assert "handler_stream_handler" in config
        config["handler_stream_handler"]["level"] = root_level or log_level or "WARN"
        if root_fmt:
            assert "formatter_console" not in config
            config.add_section("formatter_console")
            config["formatter_console"]["format"] = root_fmt
            config["formatters"]["keys"] += ",console"
            config["handler_stream_handler"]["formatter"] = "console"
    return config


def _get_or_create_default_log_dir() -> str:
    if sys.platform.startswith("win"):
        log_dir = DEFAULT_MARS_LOG_DIR_WIN
    else:
        log_dir = DEFAULT_MARS_LOG_DIR

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def _get_log_dir(log_dir: Optional[str]) -> str:
    if log_dir is None:
        log_dir = _get_or_create_default_log_dir()
    if not os.path.exists(log_dir):
        raise RuntimeError(f"Log directory does not exist: {log_dir}")

    log_dir = os.path.join(log_dir, str(time.time_ns()))
    os.mkdir(log_dir)
    return log_dir


def _get_default_logging_config_path() -> str:
    return str(get_default_logging_config_file_absolute())


def _get_log_config_path(log_config: Optional[str]) -> str:
    if log_config is None:
        log_config = _get_default_logging_config_path()
    if not os.path.exists(log_config):
        raise RuntimeError(f"Logging configuration file does not exist: {log_config}")
    return log_config


def _config_logging(**kwargs) -> Optional[configparser.RawConfigParser]:
    if kwargs.get("logging_conf", None) is None:
        return

    parsed_logging_conf = _parse_file_logging_config(kwargs["logging_conf"])

    logging.config.fileConfig(
        parsed_logging_conf,
        disable_existing_loggers=False,
    )

    return parsed_logging_conf


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

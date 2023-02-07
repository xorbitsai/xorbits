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

import asyncio
import concurrent.futures as futures
import functools
import logging
import os
import weakref
from typing import Any, Callable, Coroutine, Dict, Tuple, Type, List

import cloudpickle
import numpy as np

from ....utils import lazy_import, implements, classproperty
from ....lib.nvutils import get_index_and_uuid, get_cuda_context
from ....serialization import deserialize
from ....serialization.aio import AioSerializer, get_header_length, BUFFER_SIZES_NAME
from .base import Channel, ChannelType, Server, Client
from .core import register_client, register_server
from .errors import ChannelClosed

ucp = lazy_import("ucp")
numba_cuda = lazy_import("numba.cuda")
rmm = lazy_import("rmm")

_warning_suffix = (
    "This is often the result of a CUDA-enabled library calling a CUDA runtime function before "
    "spawning worker processes. Please make sure any such function calls don't happen "
    "at import time or in the global scope of a program."
)


logger = logging.getLogger(__name__)


def synchronize_stream(stream: int = 0):
    ctx = numba_cuda.current_context()
    cu_stream = numba_cuda.driver.drvapi.cu_stream(stream)
    stream = numba_cuda.driver.Stream(ctx, cu_stream, None)
    stream.synchronize()


class UCXInitializer:
    _inited = False

    @staticmethod
    def _get_options(ucx_config: dict) -> Tuple[dict, dict]:
        """
        Get options and envs from ucx options in oscar config
        """
        options = dict()
        envs = dict()

        # if any of the flags are set, as long as they are not Null/None,
        # we assume we should configure basic TLS settings for UCX, otherwise we
        # leave UCX to its default configuration
        if any(ucx_config.get(name) for name in ["tcp", "nvlink", "infiniband"]):
            if ucx_config.get("rdmacm"):  # pragma: no cover
                tls = "tcp"
                tls_priority = "rdmacm"
            else:
                tls = "tcp"
                tls_priority = "tcp"

            # CUDA COPY can optionally be used with ucx -- we rely on the user
            # to define when messages will include CUDA objects.  Note:
            # defining only the Infiniband flag will not enable cuda_copy
            if any(
                ucx_config.get(name) for name in ["nvlink", "cuda-copy"]
            ):  # pragma: no cover
                tls += ",cuda_copy"

            if ucx_config.get("infiniband"):  # pragma: no cover
                tls = "rc," + tls
            if ucx_config.get("nvlink"):  # pragma: no cover
                tls += ",cuda_ipc"

            options["TLS"] = tls
            options["SOCKADDR_TLS_PRIORITY"] = tls_priority
        elif "UCX_TLS" in os.environ:  # pragma: no cover
            options["TLS"] = os.environ["UCX_TLS"]

        for k, v in ucx_config.get("environment", dict()).items():  # pragma: no cover
            # {"some-name": value} is translated to {"UCX_SOME_NAME": value}
            key = f'UCX_{"_".join(s.upper() for s in k.split("-"))}'
            opt_key = key[4:]
            if opt_key in options:
                logger.warning(
                    f"Ignoring {k}={v} (key={key}) in ucx.environment, "
                    f"preferring {opt_key}={options[opt_key]} "
                    "from high level options"
                )
            elif key in os.environ:
                # This is only info because setting UCX configuration via
                # environment variables is a reasonably common approach
                logger.info(
                    f"Ignoring {k}={v} (key={key}) in ucx.environment, "
                    f"preferring {key}={os.environ[key]} from external environment"
                )
            else:
                envs[key] = v

        return options, envs

    @staticmethod
    def init(ucx_config: dict):
        if UCXInitializer._inited:
            return

        options, envs = UCXInitializer._get_options(ucx_config)

        # We ensure the CUDA context is created before initializing UCX. This can't
        # be safely handled externally because communications start before
        # preload scripts run.
        # Precedence:
        # 1. external environment
        # 2. ucx_config (high level settings passed to ucp.init)
        # 3. ucx_environment (low level settings equivalent to environment variables)
        ucx_tls = os.environ.get("UCX_TLS", options.get("TLS", envs.get("UCX_TLS", "")))
        if (
            ucx_config.get("create-cuda-contex") is True
            # This is not foolproof, if UCX_TLS=all we might require CUDA
            # depending on configuration of UCX, but this is better than
            # nothing
            or ("cuda" in ucx_tls and "^cuda" not in ucx_tls)
        ):
            if numba_cuda is None:  # pragma: no cover
                raise ImportError(
                    "CUDA support with UCX requires Numba for context management"
                )

            pre_existing_cuda_context = get_cuda_context()
            if pre_existing_cuda_context.has_context:
                dev = pre_existing_cuda_context.device_info
                logger.warning(
                    f"A CUDA context for device {dev.device_index} ({str(dev.uuid)}) "
                    f"already exists on process ID {os.getpid()}. {_warning_suffix}"
                )

            numba_cuda.current_context()

            cuda_context_created = get_cuda_context()
            cuda_visible_device = get_index_and_uuid(
                os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
            )
            if (
                cuda_context_created.has_context
                and cuda_context_created.device_info.uuid != cuda_visible_device.uuid
            ):  # pragma: no cover
                cuda_context_created_dev = cuda_context_created.device_info
                logger.warning(
                    f"Worker with process ID {os.getpid()} should have a CUDA context assigned to device "
                    f"{cuda_visible_device.device_index} ({str(cuda_visible_device.uuid)}), "
                    f"but instead the CUDA context is on device {cuda_context_created_dev.device_index} "
                    f"({str(cuda_context_created_dev.uuid)}). {_warning_suffix}"
                )

        original_environ = os.environ
        new_environ = os.environ.copy()
        new_environ.update(envs)
        os.environ = new_environ
        try:
            ucp.init(
                options=options, env_takes_precedence=True, blocking_progress_mode=False
            )
        finally:
            os.environ = original_environ

        UCXInitializer._inited = True

    @staticmethod
    def reset():
        ucp.reset()
        UCXInitializer._inited = False


class UCXChannel(Channel):
    __slots__ = (
        "ucp_endpoint",
        "_closed",
        "_has_close_callback",
        "_send_lock",
        "_recv_lock",
        "__weakref__",
    )

    name = "ucx"

    def __init__(
        self,
        ucp_endpoint: "ucp.Endpoint",
        local_address: str = None,
        dest_address: str = None,
        compression: int = None,
    ):
        super().__init__(
            local_address=local_address,
            dest_address=dest_address,
            compression=compression,
        )
        self.ucp_endpoint = ucp_endpoint

        self._send_lock = asyncio.Lock()
        self._recv_lock = asyncio.Lock()

        # When the UCX endpoint closes or errors the registered callback
        # is called.
        if hasattr(self.ucp_endpoint, "set_close_callback"):
            ref = weakref.ref(self)
            self.ucp_endpoint.set_close_callback(
                functools.partial(UCXChannel._close_channel, ref)
            )
            self._closed = False
            self._has_close_callback = True
        else:  # pragma: no cover
            self._has_close_callback = False

    @staticmethod
    def _close_channel(channel_ref: weakref.ReferenceType):
        channel = channel_ref()
        if channel is not None:
            channel._closed = True

    @property
    @implements(Channel.type)
    def type(self) -> ChannelType:
        return ChannelType.remote

    @implements(Channel.send)
    async def send(self, message: Any):
        if self.closed:
            raise ChannelClosed("UCX Endpoint is closed, unable to send message")

        compress = self.compression or 0
        serializer = AioSerializer(message, compress=compress)
        buffers = await serializer.run()
        try:
            # It is necessary to first synchronize the default stream before start
            # sending We synchronize the default stream because UCX is not
            # stream-ordered and syncing the default stream will wait for other
            # non-blocking CUDA streams. Note this is only sufficient if the memory
            # being sent is not currently in use on non-blocking CUDA streams.
            if any(hasattr(buf, "__cuda_array_interface__") for buf in buffers):
                # has GPU buffer
                synchronize_stream(0)

            async with self._send_lock:
                for buffer in buffers:
                    if buffer.nbytes if hasattr(buffer, "nbytes") else len(buffer) > 0:
                        await self.ucp_endpoint.send(buffer)
        except ucp.exceptions.UCXBaseException:  # pragma: no cover
            self.abort()
            raise ChannelClosed("While writing, the connection was closed")

    @implements(Channel.recv)
    async def recv(self):
        async with self._recv_lock:
            try:
                info_buffer = np.empty(11, dtype="u1").data
                await self.ucp_endpoint.recv(info_buffer)
                head_length = get_header_length(info_buffer)
                header_buffer = np.empty(head_length, dtype="u1").data
                await self.ucp_endpoint.recv(header_buffer)
                header = cloudpickle.loads(header_buffer)

                is_cuda_buffers = header[0].get("is_cuda_buffers")
                buffer_sizes = header[0].pop(BUFFER_SIZES_NAME)

                buffers = []
                for is_cuda_buffer, buf_size in zip(is_cuda_buffers, buffer_sizes):
                    if buf_size == 0:  # pragma: no cover
                        buffers.append(bytes())
                    elif is_cuda_buffer:
                        cuda_buffer = rmm.DeviceBuffer(size=buf_size)
                        await self.ucp_endpoint.recv(cuda_buffer)
                        buffers.append(cuda_buffer)
                    else:
                        buffer = np.empty(buf_size, dtype="u1").data
                        await self.ucp_endpoint.recv(buffer)
                        buffers.append(buffer)
            except BaseException as e:
                if not self._closed:
                    # In addition to UCX exceptions, may be CancelledError or another
                    # "low-level" exception. The only safe thing to do is to abort.
                    self.abort()
                    raise ChannelClosed(
                        f"Connection closed by writer.\nInner exception: {e!r}"
                    ) from e
                else:
                    raise EOFError("Server closed already")
        return deserialize(header, buffers)

    def abort(self):
        self._closed = True
        if self.ucp_endpoint is not None:
            self.ucp_endpoint.abort()
            self.ucp_endpoint = None

    @implements(Channel.close)
    async def close(self):
        self._closed = True
        if self.ucp_endpoint is not None:
            await self.ucp_endpoint.close()
            # abort
            self.ucp_endpoint.abort()
            self.ucp_endpoint = None

    @property
    @implements(Channel.closed)
    def closed(self):
        if self._has_close_callback is None:  # pragma: no cover
            # The self._closed flag is separate from the endpoint's lifetime, even when
            # the endpoint has closed or errored, there may be messages on its buffer
            # still to be received, even though sending is not possible anymore.
            return self._closed
        else:
            return self.ucp_endpoint is None


@register_server
class UCXServer(Server):
    __slots__ = "host", "port", "_ucp_listener", "_channels", "_closed"

    scheme = "ucx"

    _ucp_listener: "ucp.Listener"
    _channels: List[UCXChannel]

    def __init__(
        self,
        host: str,
        port: int,
        ucp_listener: "ucp.Listener",
        channel_handler: Callable[[Channel], Coroutine] = None,
    ):
        super().__init__(f"{UCXServer.scheme}://{host}:{port}", channel_handler)
        self.host = host
        self.port = port
        self._ucp_listener = ucp_listener
        self._channels = []
        self._closed = asyncio.Event()

    @classproperty
    @implements(Server.client_type)
    def client_type(self) -> Type["Client"]:
        return UCXClient

    @property
    @implements(Server.channel_type)
    def channel_type(self) -> ChannelType:
        return ChannelType.remote

    @staticmethod
    async def create(config: Dict) -> "Server":
        config = config.copy()
        if "address" in config:
            address = config.pop("address")
            prefix = f"{UCXServer.scheme}://"
            if address.startswith(prefix):
                address = address[len(prefix) :]
            host, port = address.split(":", 1)
            port = int(port)
        else:
            host = config.pop("host")
            port = int(config.pop("port"))
        handle_channel = config.pop("handle_channel")

        # init
        UCXInitializer.init(config.get("ucx", dict()))

        async def serve_forever(client_ucp_endpoint: "ucp.Endpoint"):
            try:
                await server.on_connected(
                    client_ucp_endpoint, local_address=server.address
                )
            except ChannelClosed:  # pragma: no cover
                logger.debug("Connection closed before handshake completed")
                return

        ucp_listener = ucp.create_listener(serve_forever, port=port)

        # get port of the ucp listener if not specified
        if not port:
            port = ucp_listener.port

        server = UCXServer(host, port, ucp_listener, channel_handler=handle_channel)
        return server

    @classmethod
    def parse_config(cls, config: dict) -> dict:
        return config

    @implements(Server.start)
    async def start(self):
        pass

    @implements(Server.join)
    async def join(self, timeout=None):
        wait_coro = self._closed.wait()
        try:
            await asyncio.wait_for(wait_coro, timeout=timeout)
        except (futures.TimeoutError, asyncio.TimeoutError):
            pass

    @implements(Server.on_connected)
    async def on_connected(self, *args, **kwargs):
        (ucp_endpoint,) = args
        local_address = kwargs.pop("local_address", None)
        dest_address = kwargs.pop("dest_address", None)
        if kwargs:  # pragma: no cover
            raise TypeError(
                f"{type(self).__name__} got unexpected "
                f'arguments: {",".join(kwargs)}'
            )
        channel = UCXChannel(
            ucp_endpoint, local_address=local_address, dest_address=dest_address
        )
        self._channels.append(channel)
        # handle over channel to some handlers
        await self.channel_handler(channel)

    @implements(Server.stop)
    async def stop(self):
        self._ucp_listener.close()
        # close all channels
        await asyncio.gather(
            *(channel.close() for channel in self._channels if not channel.closed)
        )
        self._ucp_listener = None
        self._closed.set()

    @property
    @implements(Server.stopped)
    def stopped(self) -> bool:
        return self._ucp_listener is None


@register_client
class UCXClient(Client):
    __slots__ = ()

    scheme = UCXServer.scheme

    @classmethod
    def parse_config(cls, config: dict) -> dict:
        return config

    @staticmethod
    @implements(Client.connect)
    async def connect(
        dest_address: str, local_address: str = None, **kwargs
    ) -> "Client":
        prefix = f"{UCXClient.scheme}://"
        if dest_address.startswith(prefix):
            dest_address = dest_address[len(prefix) :]
        host, port = dest_address.split(":", 1)
        port = int(port)
        kwargs = kwargs.copy()
        ucx_config = kwargs.pop("config", dict()).get("ucx", dict())
        UCXInitializer.init(ucx_config)

        try:
            ucp_endpoint = await ucp.create_endpoint(host, port)
        except ucp.exceptions.UCXBaseException:  # pragma: no cover
            raise ChannelClosed("Connection closed before handshake completed")
        channel = UCXChannel(
            ucp_endpoint, local_address=local_address, dest_address=dest_address
        )
        return UCXClient(local_address, dest_address, channel)

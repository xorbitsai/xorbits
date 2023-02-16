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
import multiprocessing
import sys
from typing import Dict, List, Tuple, Type, Union

import numpy as np
import pandas as pd
import pytest

from .....lib.aio import AioEvent
from .....tests.core import require_cudf, require_cupy
from .....utils import get_next_port, lazy_import
from .. import (
    Channel,
    DummyChannel,
    DummyClient,
    DummyServer,
    Server,
    SocketChannel,
    SocketClient,
    SocketServer,
    UCXServer,
    UnixSocketClient,
    UnixSocketServer,
    get_client_type,
)
from ..ucx import UCXInitializer

test_data = np.random.RandomState(0).rand(10, 10)
port = get_next_port()
cupy = lazy_import("cupy")
cudf = lazy_import("cudf")
ucp = lazy_import("ucp")


def gen_params():
    # server_type, config, con
    params: List[Tuple[Type[Server], Dict, str]] = [
        (SocketServer, dict(host="127.0.0.1", port=port), f"127.0.0.1:{port}"),
    ]
    if sys.platform != "win32":
        params.append((UnixSocketServer, dict(process_index="0"), f"unixsocket:///0"))
    if ucp is not None:
        ucp_port = get_next_port()
        # test ucx
        params.append(
            (UCXServer, dict(host="127.0.0.1", port=ucp_port), f"127.0.0.1:{ucp_port}")
        )
    return params


params = gen_params()
local_params = gen_params().copy()
local_params.append((DummyServer, dict(), "dummy://0"))


@pytest.mark.parametrize("server_type, config, con", local_params)
@pytest.mark.asyncio
async def test_comm(server_type, config, con):
    async def check_data(chan: Union[SocketChannel, DummyChannel]):
        np.testing.assert_array_equal(test_data, await chan.recv())
        await chan.send("success")

    config = config.copy()
    config["handle_channel"] = check_data

    # create server
    server = await server_type.create(config)
    await server.start()
    assert isinstance(server.info, dict)

    # create client
    client = await server_type.client_type.connect(con)
    assert isinstance(client.info, dict)
    assert isinstance(client.channel.info, dict)
    await client.send(test_data)

    assert "success" == await client.recv()

    await client.close()
    assert client.closed

    # create client2
    async with await server_type.client_type.connect(con) as client2:
        assert not client2.closed
    assert client2.closed

    await server.join(0.001)
    await server.stop()

    assert server.stopped

    if server_type is UCXServer:
        UCXInitializer.reset()
        # skip create server on same port for ucx
        return

    async with await server_type.create(config) as server2:
        assert not server2.stopped
    assert server2.stopped


def _wrap_test(server_started_event, conf, tp):
    async def _test():
        async def check_data(chan: SocketChannel):
            np.testing.assert_array_equal(test_data, await chan.recv())
            await chan.send("success")

        nonlocal conf
        conf = conf.copy()
        conf["handle_channel"] = check_data

        # create server
        server = await tp.create(conf)
        await server.start()
        server_started_event.set()
        await server.join()

    asyncio.run(_test())


@pytest.mark.asyncio
@pytest.mark.parametrize("server_type, config, con", params)
async def test_multiprocess_comm(server_type, config, con):
    if server_type is UCXServer:
        UCXInitializer.reset()

    server_started = multiprocessing.Event()

    p = multiprocessing.Process(
        target=_wrap_test, args=(server_started, config, server_type)
    )
    p.daemon = True
    p.start()

    try:
        await AioEvent(server_started).wait()

        # create client
        client = await server_type.client_type.connect(con)
        await client.channel.send(test_data)

        assert "success" == await client.recv()

        await client.close()
        assert client.closed
    finally:
        p.kill()


cupy_data = np.arange(100).reshape((10, 10))
cudf_data = pd.DataFrame({"col1": np.arange(10), "col2": [f"s{i}" for i in range(10)]})


def _wrap_cuda_test(server_started_event, conf, tp):
    async def _test():
        async def check_data(chan: Channel):
            import cupy

            r = await chan.recv()

            if isinstance(r, cupy.ndarray):
                np.testing.assert_array_equal(cupy.asnumpy(r), cupy_data)
            else:
                pd.testing.assert_frame_equal(r.to_pandas(), cudf_data)
            await chan.send("success")

        conf["handle_channel"] = check_data

        # create server
        server = await tp.create(conf)
        await server.start()
        server_started_event.set()
        await server.join()

    asyncio.run(_test())


@require_cupy
@require_cudf
@pytest.mark.parametrize("server_type", [SocketServer, UCXServer])
@pytest.mark.asyncio
async def test_multiprocess_cuda_comm(server_type):
    mp_ctx = multiprocessing.get_context("spawn")

    server_started = mp_ctx.Event()
    port = get_next_port()
    p = mp_ctx.Process(
        target=_wrap_cuda_test,
        args=(server_started, dict(host="127.0.0.1", port=port), server_type),
    )
    p.daemon = True
    p.start()

    await AioEvent(server_started).wait()

    # create client
    client = await server_type.client_type.connect(f"127.0.0.1:{port}")

    await client.channel.send(cupy.asarray(cupy_data))
    assert "success" == await client.recv()

    client = await server_type.client_type.connect(f"127.0.0.1:{port}")

    await client.channel.send(cudf.DataFrame(cudf_data))
    assert "success" == await client.recv()

    await client.close()


def test_get_client_type():
    assert issubclass(get_client_type("127.0.0.1"), SocketClient)
    assert issubclass(get_client_type("unixsocket:///1"), UnixSocketClient)
    assert issubclass(get_client_type("dummy://"), DummyClient)

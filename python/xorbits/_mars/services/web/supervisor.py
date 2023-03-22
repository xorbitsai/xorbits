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

import importlib
import logging
import os

import xoscar as mo
from tornado import web

from ...utils import get_next_port
from ..core import AbstractService

logger = logging.getLogger(__name__)


class WebActor(mo.Actor):
    def __init__(self, config):
        super().__init__()
        self._config = config
        self._web_server = None
        self._web_app = None

        extra_mod_names = self._config.get("extra_discovery_modules") or []
        web_handlers = self._config.get("web_handlers", {})
        for mod_name in extra_mod_names:
            try:
                web_mod = importlib.import_module(mod_name)
                web_handlers.update(getattr(web_mod, "web_handlers", {}))
            except ImportError:  # pragma: no cover
                pass

    async def __post_create__(self):
        from .handlers import handlers, static_handlers

        supervisor_addr = self.address

        host = self._config.get("host") or "0.0.0.0"
        port = self._config.get("port") or get_next_port()
        self._web_address = f"http://{host}:{port}"
        handlers.update(self._config.get("web_handlers", {}))
        web_handlers = []
        for p, h in handlers.items():
            web_handlers.append((p, h, {"supervisor_addr": supervisor_addr}))
        web_handlers.extend([(*[p], *v) for p, v in static_handlers.items()])

        retrial = 5
        while retrial:
            try:
                if port is None:
                    port = get_next_port()

                # For debugging tornado, use debug=True to enable hot deploy
                self._web_app = web.Application(web_handlers)
                self._web_server = self._web_app.listen(port, host)
                logger.info("Mars Web started at %s:%d", host, port)
                break
            except OSError:  # pragma: no cover
                if port is not None:
                    raise
                retrial -= 1
                if retrial == 0:
                    raise

    async def __pre_destroy__(self):
        if self._web_server is not None:
            self._web_server.stop()

    def get_web_address(self):
        web_address = self._web_address
        if os.name == "nt":
            web_address = web_address.replace("0.0.0.0", "127.0.0.1")
        return web_address


class WebSupervisorService(AbstractService):
    """
    Web service on supervisor.

    Service Configuration
    ---------------------
    {
        "web": {
            "host": "<web host>",
            "port": "<web port>",
            "web_handlers": [
                <web_handlers>,
            ],
            "extra_discovery_modules": [
                "path.to.modules",
            ]
        }
    }
    """

    async def start(self):
        await mo.create_actor(
            WebActor,
            config=self._config.get("web", {}),
            uid=WebActor.default_uid(),
            address=self._address,
        )

    async def stop(self):
        await mo.destroy_actor(
            mo.create_actor_ref(uid=WebActor.default_uid(), address=self._address)
        )

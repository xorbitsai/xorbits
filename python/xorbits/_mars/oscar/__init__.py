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

from typing import TypeVar, Union

# import aio to ensure patch enabled for Python 3.6
from ..lib import aio

del aio

from . import debug
from .api import (
    Actor,
    StatelessActor,
    actor_ref,
    create_actor,
    create_actor_pool,
    destroy_actor,
    get_pool_config,
    has_actor,
    kill_actor,
    setup_cluster,
    wait_actor_pool_recovered,
)

# make sure methods are registered
from .backends import allocate_strategy, mars, ray, test
from .backends.pool import MainActorPoolType
from .batch import extensible
from .core import ActorRef
from .debug import DebugOptions, get_debug_options, set_debug_options
from .errors import (
    ActorAlreadyExist,
    ActorNotExist,
    Return,
    SendMessageFailed,
    ServerClosed,
)
from .utils import create_actor_ref

del mars, ray, test

_T = TypeVar("_T")
ActorRefType = Union[ActorRef, _T]

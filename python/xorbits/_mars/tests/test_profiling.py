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

import dataclasses

from xoscar.backends.message import SendMessage

from ..profiling import _CallStats, _ProfilingOptions, _SubtaskStats


def test_collect():
    options = _ProfilingOptions(
        {"slow_calls_duration_threshold": 0, "slow_subtasks_duration_threshold": 0}
    )

    # Test collect message with incomparable arguments.
    from xoscar.core import ActorRef

    fake_actor_ref = ActorRef("def", b"uid")
    fake_message1 = SendMessage(b"abc", fake_actor_ref, ["name", {}])
    fake_message2 = SendMessage(b"abc", fake_actor_ref, ["name", 1])

    cs = _CallStats(options)
    cs.collect(fake_message1, 1.0)
    cs.collect(fake_message2, 1.0)

    @dataclasses.dataclass
    class _FakeSubtask:
        extra_config: dict

    # Test collect subtask with incomparable arguments.
    band = ("1.2.3.4", "numa-0")
    subtask1 = _FakeSubtask({})
    subtask2 = _FakeSubtask(None)
    ss = _SubtaskStats(options)
    ss.collect(subtask1, band, 1.0)
    ss.collect(subtask2, band, 1.0)

    # Test call stats order.
    cs = _CallStats(options)
    for i in range(20):
        fake_message = SendMessage(
            f"{i}".encode(), fake_actor_ref, ["name", True, (i,), {}]
        )
        cs.collect(fake_message, i)
    d = cs.to_dict()
    assert list(d["most_calls"].values())[0] == 20
    assert list(d["slow_calls"].values()) == list(reversed(range(10, 20)))

    # Test subtask stats order.
    ss = _SubtaskStats(options)
    counter = 0
    for i in range(20):
        for j in range(i):
            fake_message = _FakeSubtask(counter)
            ss.collect(fake_message, (str(j), "numa-0"), counter)
            counter += 1
    d = ss.to_dict()
    assert list(d["band_subtasks"].values()) == [19, 18, 17, 16, 15, 5, 4, 3, 2, 1]
    assert list(d["slow_subtasks"].values()) == list(
        reversed(range(counter - 10, counter))
    )

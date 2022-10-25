# -*- coding: utf-8 -*-
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

from . import execution


class Data:
    pass


class DataRef:
    def __init__(self, data: "Data"):
        self._data = data

    def __getattr__(self, item):
        return getattr(self._data, item)

    def __str__(self):
        execution.execute(self)
        return self._data.__str__()

    def __repr__(self):
        execution.execute(self)
        return self._data.__repr__()

    # TODO: How to define the execution conditions that works for mars, pandas and xorbits?

    @property
    def data(self):
        return self._data

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

from .._mars.core.operand.objects import ObjectOperand, ObjectOperandMixin


class DataOperand(ObjectOperand):
    def __init__(self, output_types=None, *args, **kwargs):
        super().__init__(_output_types=output_types, *args, **kwargs)


class DataOperandMixin(ObjectOperandMixin):
    pass

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

import itertools
import numpy as np
import pandas as pd

from ...._mars.core.entity import OutputType
from ...._mars.dataframe.utils import parse_index
from ...operand import DataOperand, DataOperandMixin


class HuggingfaceToDataframe(DataOperand, DataOperandMixin):
    def __call__(self, inp):
        self.output_types = [OutputType.dataframe]
        params = inp.params.copy()
        # dtypes is None trigger auto execution.
        if inp.dtypes is not None:
            params["index_value"] = parse_index(pd.RangeIndex(-1))
            params["columns_value"] = parse_index(inp.dtypes.index, store_data=True)
        return self.new_tileable([inp], **params)

    @classmethod
    def tile(cls, op: "HuggingfaceToDataframe"):
        all_chunks = itertools.chain(*(inp.chunks for inp in op.inputs))
        chunks = []
        out = op.outputs[0]
        for index, c in enumerate(all_chunks):
            chunk_op = op.copy().reset_key()
            new_c = chunk_op.new_chunk(
                [c],
                index=(index, 0),
                shape=(np.nan, np.nan),
                columns_value=out.columns_value,
                index_value=parse_index(pd.RangeIndex(-1)),
                dtypes=out.dtypes,
            )
            chunks.append(new_c)
        return op.copy().new_tileable(
            op.inputs,
            chunks=chunks,
            nsplits=((np.nan,) * len(chunks), (np.nan,)),
            **out.params
        )

    @classmethod
    def execute(cls, ctx, op: "HuggingfaceToDataframe"):
        ds = ctx[op.inputs[0].key]
        out = op.outputs[0]
        df = ds.to_pandas()
        ctx[out.key] = df


def to_dataframe(dataset):
    return HuggingfaceToDataframe()(dataset)

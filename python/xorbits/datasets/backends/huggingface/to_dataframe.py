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

import numpy as np
import pandas as pd

from ...._mars.core.entity import OutputType
from ...._mars.dataframe.utils import parse_index
from ...._mars.serialization.serializables import AnyField
from ...operand import DataOperand, DataOperandMixin


class HuggingfaceToDataframe(DataOperand, DataOperandMixin):
    types_mapper = AnyField("types_mapper")

    def __call__(self, dataset):
        params = dataset.params.copy()
        # dtypes is None trigger auto execution.
        if dataset.dtypes is not None:
            params["index_value"] = parse_index(pd.RangeIndex(-1))
            params["columns_value"] = parse_index(dataset.dtypes.index, store_data=True)
        return self.new_tileable([dataset], **params)

    @classmethod
    def tile(cls, op: "HuggingfaceToDataframe"):
        assert len(op.inputs) == 1
        inp = op.inputs[0]
        out = op.outputs[0]
        chunks = []
        for index, c in enumerate(inp.chunks):
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
        if op.types_mapper is None:
            df = ds.to_pandas()
        else:
            df = cls._to_pandas(ds, op.types_mapper)
        ctx[out.key] = df

    @classmethod
    def _to_pandas(cls, ds, types_mapper):
        from datasets.features.features import pandas_types_mapper
        from datasets.formatting import query_table

        def _types_mapper(_dtype):
            mapped = pandas_types_mapper(_dtype)
            if mapped is None:
                return types_mapper(_dtype)

        return query_table(
            table=ds._data,
            key=slice(0, len(ds)),
            indices=ds._indices if ds._indices is not None else None,
        ).to_pandas(types_mapper=_types_mapper)


def to_dataframe(dataset, types_mapper=None):
    return HuggingfaceToDataframe(
        output_types=[OutputType.dataframe], types_mapper=types_mapper
    )(dataset)

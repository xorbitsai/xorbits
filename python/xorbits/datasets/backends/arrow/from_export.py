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
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Union

import fsspec
import pyarrow as pa

from ...._mars.core.entity import OutputType
from ...._mars.serialization.serializables import (
    AnyField,
    ListField,
    StringField,
    TupleField,
)
from ...iterable_dataset import IterableDataset
from ...operand import DataOperand, DataOperandMixin


class FromExport(DataOperand, DataOperandMixin):
    path = StringField("path")
    groups = ListField("groups")
    index = TupleField("index")
    iterable_dataset = AnyField("iterable_dataset")

    def __call__(self):
        ids = self.iterable_dataset
        return self.new_tileable(
            [], dtypes=ids.schema.empty_table().to_pandas().dtypes, shape=ids.shape
        )

    @classmethod
    def tile(cls, op: "FromExport"):
        assert len(op.inputs) == 0
        out = op.outputs[0]
        ids: IterableDataset = op.iterable_dataset
        op.iterable_dataset = None

        chunks = []
        default_group = ids.group_infos()[0]
        splits = default_group.index[1]
        for cdx, index in enumerate(default_group.index[0]):
            chunk_op = op.copy().reset_key()
            chunk_op.index = index.as_py()
            c = chunk_op.new_chunk(
                inputs=[], index=(cdx, 0), shape=(splits[cdx].as_py(), ids.shape[1])
            )
            chunks.append(c)
        return op.copy().new_tileable(
            op.inputs,
            chunks=chunks,
            nsplits=(splits.to_pylist(), (ids.shape[1],)),
            **out.params,
        )

    @classmethod
    def execute(cls, ctx, op: "FromExport"):
        arrow_file_paths = [
            os.path.join(
                op.path, name, IterableDataset._FILE_NAME_FORMATTER.format(*op.index)
            )
            for name in op.groups
        ]

        def _load_arrow_table(filepath):
            # TODO(codingl2k1): mmap if local.
            with fsspec.open(filepath, "rb") as f:
                with pa.ipc.RecordBatchStreamReader(f) as reader:
                    return reader.read_all()

        futures = []
        with ThreadPoolExecutor(
            thread_name_prefix=IterableDataset._get_infos.__qualname__
        ) as executor:
            for arrow_file in arrow_file_paths:
                futures.append(executor.submit(_load_arrow_table, arrow_file))

        arrow_tables = [fut.result() for fut in futures]
        result_table = arrow_tables[0]
        # TODO(codingl2k1): Better way to concat table columns.
        for table in arrow_tables[1:]:
            for idx, col in enumerate(table.itercolumns()):
                result_table = result_table.append_column(table.field(idx), col)
        ctx[op.outputs[0].key] = result_table


def from_export(
    path: Union[str, os.PathLike],
    storage_options: Optional[dict] = None,
    version: Optional[str] = None,
):
    """Create a dataset from exported Dataset.

    Parameters
    ----------
    path: str
        The export path.
    storage_options: dict
        Key/value pairs to be passed on to the caching file-system backend, if any.
    version: str
        The dataset version.

    Returns
    -------
        Dataset
    """
    ids = IterableDataset(path=path, storage_options=storage_options, version=version)
    op = FromExport(
        output_types=[OutputType.arrow_dataset],
        iterable_dataset=ids,
        path=ids.path,
        groups=ids.groups,
    )
    return op().to_dataset()

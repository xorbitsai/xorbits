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

import hashlib
from functools import partial
from typing import Union

import numpy as np
import pandas as pd

from .._mars import opcodes
from .._mars.core import recursive_tile
from .._mars.core.context import Context
from .._mars.core.entity import OutputType
from .._mars.core.operand import ObjectOperand, ObjectOperandMixin, OperandStage
from .._mars.dataframe.operands import DataFrameOperand, DataFrameOperandMixin
from .._mars.dataframe.utils import build_concatenated_rows_frame
from .._mars.serialization.serializables import AnyField
from .._mars.utils import CUnionFind as UnionFind
from ..core.adapter import from_mars, to_mars
from .utils import MERSENNE_PRIME, minhash_embed_func, optimal_param


class DataFrameUnionFind(ObjectOperand, ObjectOperandMixin):
    _output_type_ = OutputType.object
    union_find = AnyField("union_find", default=None)

    @classmethod
    def execute(cls, ctx: Union[dict, Context], op: "DataFrameUnionFind"):
        if op.stage == OperandStage.map:
            clusters = ctx[op.inputs[0].key]
            out = op.outputs[0]

            for cluster in clusters:
                if len(cluster) <= 1:
                    continue

                idx = min(cluster)
                for x in cluster:
                    op.union_find.union_(x, idx)

            ctx[out.key] = op.union_find
        elif op.stage == OperandStage.reduce:
            out = op.outputs[0]

            for i in range(len(op.inputs)):
                op.union_find.union_uf(ctx[op.inputs[i].key])
            ctx[out.key] = op.union_find


class DataFrameDedup(DataFrameOperand, DataFrameOperandMixin):
    _op_type = opcodes.DEDUP

    func = AnyField("func")

    @classmethod
    def execute(cls, ctx: Union[dict, Context], op: "DataFrameDedup"):
        input_data = ctx[op.inputs[0].key]
        uf = ctx[op.inputs[1].key]
        out = op.outputs[0]

        ctx[out.key] = input_data[
            input_data["__dedup_id"].map(lambda x: uf.find(x) == x)
        ].drop(columns="__dedup_id")

    @classmethod
    def tile(cls, op: "DataFrameDedup"):
        in_df = build_concatenated_rows_frame(op.inputs[0])
        out_df = op.outputs[0]

        def gen_id_column(df):
            from xoscar._utils import new_random_id

            df["__dedup_id"] = [new_random_id(16) for _ in range(len(df))]

            return df

        new_dtypes = in_df.dtypes.copy()
        new_dtypes["__dedup_id"] = "str"

        in_df_with_id = in_df.map_chunk(
            gen_id_column, output_type="dataframe", dtypes=new_dtypes
        )

        in_df_with_id = yield from recursive_tile(in_df_with_id)
        yield in_df_with_id.chunks

        embedded = in_df_with_id.apply(
            op.func,
            axis=1,
            output_type="dataframe",
            dtypes=pd.Series(["object", "bytes"], index=["__signatures", "__id"]),
        )

        clusters = (
            embedded.explode("__signatures")
            .groupby("__signatures", sort=False)["__id"]
            .apply(set)
        )
        tiled_clusters = yield from recursive_tile(clusters)

        # union find stage
        chunks = []
        for c in tiled_clusters.chunks:
            new_op = DataFrameUnionFind(union_find=UnionFind())
            new_op.stage = OperandStage.map
            chunks.append(
                new_op.new_chunk(
                    [c],
                    index=c.index,
                )
            )

        combine_size = 4
        while len(chunks) > combine_size:  # pragma: no cover
            new_chunks = []
            for i in range(0, len(chunks), combine_size):
                chks = chunks[i : i + combine_size]
                if len(chks) == 1:
                    chk = chks[0]
                else:
                    union_op = DataFrameUnionFind(union_find=UnionFind())
                    union_op.stage = OperandStage.reduce
                    for j, c in enumerate(chks):
                        c._index = (j, 0)
                    chk = union_op.new_chunk(chks)
                new_chunks.append(chk)
            chunks = new_chunks

        new_op = DataFrameUnionFind(union_find=UnionFind())
        new_op.stage = OperandStage.reduce
        union_chunk = new_op.new_chunk(chunks, index=(0,))
        union_chunk.is_broadcaster = True

        # dedup stage
        dedup_chunks = []
        for c in in_df_with_id.chunks:
            new_shape = c.shape

            new_op = op.copy().reset_key()

            dedup_chunks.append(
                new_op.new_chunk(
                    [c, union_chunk],
                    shape=(np.nan, new_shape[1] - 1),
                    index=c.index,
                    dtypes=out_df.dtypes,
                    index_value=c.index_value,
                    columns_value=out_df.columns_value,
                )
            )

        new_nsplits = tuple(chunk.shape[0] for chunk in dedup_chunks), (
            dedup_chunks[0].shape[1],
        )

        new_op = op.copy()
        kw = out_df.params.copy()
        kw.update(dict(chunks=dedup_chunks, nsplits=new_nsplits))

        return new_op.new_tileables(op.inputs, **kw)

    def __call__(self, df: pd.DataFrame):
        return self.new_dataframe([df])


def dedup(
    df: pd.DataFrame,
    col: str,
    method: str = "minhash",
    **kws,
) -> pd.DataFrame:
    """
    Applies deduplication on a DataFrame based on the chosen method.

    This function provides two methods for deduplication: exact matching and MinHash-based.
    The exact matching uses md5 hashing for deduplication, while the MinHash-based method
    utilizes MinHash and MinHashLSH for identifying and removing duplicates based on Jaccard similarity.
    For the MinHash-based method, it operates by generating hash values for a specified column
    of the DataFrame, computing similarity between these hash values, and then removing the rows
    that are determined to be duplicates according to a provided Jaccard similarity threshold.

    Parameters
    ----------
    df: pd.DataFrame,
        The DataFrame to deduplicate.

    col : str
        The column of the DataFrame on which to calculate hash values.

    method : str, default "minhash"
        The method for deduplication. Options include 'exact' and 'minhash'.

    Additional Parameters for MinHash method
    ----------------------------------------
    threshold : float, default 0.7
        The Jaccard similarity threshold to use in the MinHashLSH.

    num_perm : int, default 128
        The number of permutations to use in the MinHash.

    min_length : int, default 5
        The minimum number of tokens to use in the MinHash. Texts shorter than
        this value will be filtered out.

    ngrams : int, default 5
        The size of ngram to use in the MinHash.

    seed : int, default 42
        The seed for the random number generator.

    Returns
    -------
    DataFrame
        The DataFrame after applying the chosen deduplication method.

    Notes
    -----
    The 'exact' method performs deduplication by hashing each entry in the specified column with md5
    and removing duplicates.

    The 'minhash' method uses a combination of MinHash and MinHashLSH for efficient calculation of
    Jaccard similarity and identification of duplicates. This process involves hashing text to a
    finite set of integers (hash values), and then comparing these hash values to find duplicates.

    The optimal parameters for the number of bands `B` and rows `R` per band
    are automatically calculated based on the provided similarity threshold and
    number of permutations, to balance the trade-off between precision and recall.

    Examples
    --------
    >>> from xorbits.experimental import dedup
    >>> words = list("abcdefghijklmnopqrstuvwxyz")
    >>> df = pd.DataFrame(
    ...     {
    ...         "text": [
    ...             " ".join(["".join(np.random.choice(words, 5)) for i in range(50)])
    ...             for _ in np.arange(10)
    ...         ]
    ...         * 2,
    ...     }
    ... )
    >>> res = dedup(df, col="text", method="exact") # for 'exact' method
    >>> res = dedup(df, col="text", method="minhash", threshold=0.8, num_perm=128, min_length=5, ngrams=5, seed=42) # for 'minhash' method
    """

    if method not in ["exact", "minhash"]:
        raise ValueError("method must be one of 'exact' or 'minhash'")

    # Check if the DataFrame contains the text column
    if col not in df.dtypes.index:
        raise ValueError(f"{col} column not found in the DataFrame")

    if method == "exact":
        df = to_mars(df)

        df["__exact"] = df[col].apply(
            lambda x: hashlib.md5(x.encode("utf-8")).hexdigest()
        )

        df.drop_duplicates(subset=["__exact"], inplace=True).drop(
            columns=["__exact"], inplace=True
        )

        return from_mars(df)

    if method == "minhash":
        threshold = kws.pop("threshold", 0.7)
        num_perm = kws.pop("num_perm", 128)
        min_length = kws.pop("min_length", 5)
        ngrams = kws.pop("ngrams", 5)
        seed = kws.pop("seed", 42)

        # Check the threshold type and range
        if not isinstance(threshold, (float, int)) or not 0 <= threshold <= 1:
            raise ValueError(
                f"Expected 'threshold' to be a float between 0 and 1, got {threshold}"
            )

        # Check the num_perm, min_length, ngram and seed type and value
        for var, var_name in [
            (num_perm, "num_perm"),
            (min_length, "min_length"),
            (ngrams, "ngrams"),
            (seed, "seed"),
        ]:
            if not isinstance(var, int) or var <= 0:
                raise ValueError(
                    f"Expected '{var_name}' to be a positive integer, got {var}"
                )

        B, R = optimal_param(threshold, num_perm)

        HASH_RANGES = [(i * R, (i + 1) * R) for i in range(B)]

        RNG = np.random.RandomState(seed)

        PERMUTATIONS = np.array(
            [
                (
                    RNG.randint(1, MERSENNE_PRIME, dtype=np.uint64),
                    RNG.randint(0, MERSENNE_PRIME, dtype=np.uint64),
                )
                for _ in range(num_perm)
            ],
            dtype=np.uint64,
        ).T

        func = partial(
            minhash_embed_func,
            text=col,
            num_perm=num_perm,
            hashranges=HASH_RANGES,
            ngram_size=ngrams,
            min_length=min_length,
            permutations=PERMUTATIONS,
        )

        op = DataFrameDedup(func=func)

        return from_mars(op(to_mars(df)))

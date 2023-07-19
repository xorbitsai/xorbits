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
import re
from functools import partial
from itertools import tee
from typing import Iterator, List, Set, Text, Tuple, Union

import numpy as np
import pandas as pd
from scipy.integrate import quad as integrate

from ... import opcodes
from ...core import recursive_tile
from ...core.context import Context
from ...core.entity import OutputType
from ...core.operand import ObjectOperand, ObjectOperandMixin, OperandStage
from ...serialization.serializables import AnyField
from ...utils import CUnionFind as UnionFind
from ..operands import DataFrameOperand, DataFrameOperandMixin
from ..utils import build_concatenated_rows_frame

NON_ALPHA = re.compile("\\W", re.UNICODE)
MAX_HASH = np.uint64((1 << 32) - 1)
MERSENNE_PRIME = np.uint64((1 << 61) - 1)


def ngrams(
    sequence: List[Text], n: int, min_length: int = 5
) -> Iterator[Tuple[str, str]]:
    """
    Return the ngrams generated from a sequence of items, as an iterator.

    This is copied from https://github.com/ChenghaoMou/text-dedup.

    Parameters
    ----------
    sequence : List[Text]
        The sequence of items.
    n : int
        The length of each ngram.
    min_length : int, optional
        The minimum length of each ngram, by default 5

    Returns
    -------
    iterator
        The ngrams.

    Examples
    --------
    >>> list(ngrams(["a", "b", "c", "d"], 2, min_length=1))
    [('a', 'b'), ('b', 'c'), ('c', 'd')]
    >>> list(ngrams(["a", "b", "c", "d"], 2, min_length=5))
    []
    >>> list(ngrams(["a", "b"], 3, min_length=1))
    [('a', 'b')]
    """
    if len(sequence) < min_length:  # pragma: no cover
        return []
    if len(sequence) < n:  # pragma: no cover
        return [tuple(sequence)]
    iterables = tee(iter(sequence), n)
    for i, sub_iterable in enumerate(iterables):
        for _ in range(i):
            next(sub_iterable, None)
    return zip(*iterables)


def sha1_hash(data: bytes, d: int = 32) -> int:
    """
    Generate a d-bit hash value from the given data.

    This is copied from https://github.com/ChenghaoMou/text-dedup.

    Parameters
    ----------
    data : bytes
        The data to be hashed.
    d : int
        The number of bits of the hash value.

    Returns
    -------
    int
        The hash value.

    Examples
    --------
    >>> sha1_hash(b"hello world", 32)
    896314922
    >>> sha1_hash(b"hello world", 64)
    13028719972609469994
    >>> sha1_hash(b"hello world", 128)
    310522945683037930239412421226792791594
    """
    return int.from_bytes(hashlib.sha1(data).digest()[: d // 8], byteorder="little")


def optimal_param(
    threshold: float,
    num_perm: int,
    false_positive_weight: float = 0.5,
    false_negative_weight: float = 0.5,
) -> Tuple[int, int]:
    """
    Compute the optimal `MinHashLSH` parameter that minimizes the weighted sum
    of probabilities of false positive and false negative, taken from datasketch.

    You can also refer to the interactive demo at https://huggingface.co/spaces/bigcode/near-deduplication.
    This is copied from https://github.com/ChenghaoMou/text-dedup.

    Parameters
    ----------
    threshold : float
        The threshold for similarity.
    num_perm : int
        The number of permutations.
    false_positive_weight : float
        The weight of false positive.
    false_negative_weight : float
        The weight of false negative.

    Returns
    -------
    Tuple[int, int]
        The optimal `b` (bands) and `r` (rows) parameters.

    Examples
    --------
    >>> optimal_param(0.75, 256)
    (21, 12)
    >>> optimal_param(0.75, 256, 0.1, 0.9)
    (28, 9)
    """

    def false_positive_area(threshold: float, b: int, r: int):
        """Source: `datasketch.lsh`"""

        def proba(s):
            return 1 - (1 - s ** float(r)) ** float(b)

        a, _ = integrate(proba, 0.0, threshold)
        return a

    def false_negative_area(threshold: float, b: int, r: int):
        """Source: `datasketch.lsh`"""

        def proba(s):
            return 1 - (1 - (1 - s ** float(r)) ** float(b))

        a, _ = integrate(proba, threshold, 1.0)
        return a

    min_error = float("inf")
    opt = (0, 0)
    for b in range(1, num_perm + 1):
        max_r = int(num_perm / b)
        for r in range(1, max_r + 1):
            fp = false_positive_area(threshold, b, r)
            fn = false_negative_area(threshold, b, r)
            error = fp * false_positive_weight + fn * false_negative_weight
            if error < min_error:
                min_error = error
                opt = (b, r)
    return opt


def embed_func(
    row: pd.Series,
    *,
    text: str,
    num_perm: int,
    ngram_size: int,
    min_length: int,
    hashranges: List[Tuple[int, int]],
    permutations: np.ndarray,
) -> pd.Series:
    """
    Calculate hash values for the content.

    This is a modified version of https://github.com/ChenghaoMou/text-dedup.

    Parameters
    ----------
    row : pd.Series
        The row content to be embedded.
    text : str
        The text column of the columns.
    num_perm : int
        The number of permutations.
    ngram_size : int
        The size of n-grams.
    min_length : int
        The minimum length of the document in terms of tokens.
    hashranges : List[Tuple[int, int]]
        The ranges of hash values.
    permutations : np.ndarray
        The permutations for the minhash.

    Returns
    -------
    Dict[str, Any]
        The hash values in each range and the index.

    Examples
    --------
    >>> row = pd.Series({"text": "hello world", "__dedup_id": 0})
    >>> text = "text"
    >>> num_perm = 250
    >>> ngram_size = 1
    >>> hashranges = [(i, i + 25) for i in range(0, 250, 25)]
    >>> PERMUTATIONS = np.array(
    ...     [
    ...         (
    ...             RNG.randint(1, MERSENNE_PRIME, dtype=np.uint64),
    ...             RNG.randint(0, MERSENNE_PRIME, dtype=np.uint64),
    ...         )
    ...         for _ in range(num_perm)
    ...     ],
    ...     dtype=np.uint64,
    ... ).T
    >>> res = embed_func(content, idx, num_perm=num_perm, ngram_size=ngram_size, min_length=0, hashranges=hashranges, permutations=PERMUTATIONS)
    >>> len(res["__signatures"])
    10
    """
    content, idx = row[text], row["__dedup_id"]

    a, b = permutations

    masks: np.ndarray = np.full(shape=num_perm, dtype=np.uint64, fill_value=MAX_HASH)
    tokens: Set[str] = {
        " ".join(t) for t in ngrams(NON_ALPHA.split(content), ngram_size, min_length)
    }

    hashvalues: np.ndarray = np.array(
        [sha1_hash(token.lower().encode("utf-8")) for token in tokens], dtype=np.uint64
    )

    permuted_hashvalues = np.bitwise_and(
        ((hashvalues * np.tile(a, (len(hashvalues), 1)).T).T + b) % MERSENNE_PRIME,
        MAX_HASH,
    )
    hashvalues = np.vstack([permuted_hashvalues, masks]).min(axis=0)

    Hs = [
        (i, bytes(hashvalues[start:end].byteswap().data))
        for i, (start, end) in enumerate(hashranges)
    ]
    return pd.Series({"__signatures": Hs, "__id": idx})


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


def df_dedup(
    df: pd.DataFrame,
    col: str,
    threshold: float = 0.7,
    num_perm: int = 128,
    min_length: int = 5,
    ngram: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Applies MinHash-based deduplication on a DataFrame.

    This function utilizes MinHash and MinHashLSH for identifying and removing
    duplicates based on Jaccard similarity. It operates by generating hash
    values for a specified column of the DataFrame, computing similarity
    between these hash values, and then removing the rows that are determined
    to be duplicates according to a provided Jaccard similarity threshold.

    Parameters
    ----------
    col : str
        The column of the DataFrame on which to calculate hash values.

    threshold : float, default 0.7
        The Jaccard similarity threshold to use in the MinHashLSH.

    num_perm : int, default 128
        The number of permutations to use in the MinHash.

    min_length : int, default 5
        The minimum number of tokens to use in the MinHash. Texts shorter than
        this value will be filtered out.

    ngram : int, default None
        The size of ngram to use in the MinHash.

    seed : int, default 42
        The seed for the random number generator.

    Returns
    -------
    DataFrame
        The DataFrame after applying MinHash-based deduplication.

    Notes
    -----
    The deduplication process uses a combination of MinHash and MinHashLSH for
    efficient calculation of Jaccard similarity and identification of duplicates.
    This process involves hashing text to a finite set of integers (hash values),
    and then comparing these hash values to find duplicates.

    The optimal parameters for the number of bands `B` and rows `R` per band
    are automatically calculated based on the provided similarity threshold and
    number of permutations, to balance the trade-off between precision and recall.

    Examples
    --------
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
    >>> res = df.dedup(col="text")
    >>> res.execute()

    """

    # Check the threshold type and range
    if not isinstance(threshold, (float, int)) or not 0 <= threshold <= 1:
        raise ValueError(
            f"Expected 'threshold' to be a float between 0 and 1, got {threshold}"
        )

    # Check the num_perm, min_length, ngram and seed type and value
    for var, var_name in [
        (num_perm, "num_perm"),
        (min_length, "min_length"),
        (ngram, "ngram"),
        (seed, "seed"),
    ]:
        if not isinstance(var, int) or var <= 0:
            raise ValueError(
                f"Expected '{var_name}' to be a positive integer, got {var}"
            )

    # Check if the DataFrame contains the text column
    if col not in df.dtypes.index:
        raise ValueError(f"{col} column not found in the DataFrame")

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
        embed_func,
        text=col,
        num_perm=num_perm,
        hashranges=HASH_RANGES,
        ngram_size=ngram,
        min_length=min_length,
        permutations=PERMUTATIONS,
    )

    op = DataFrameDedup(func=func)

    return op(df)

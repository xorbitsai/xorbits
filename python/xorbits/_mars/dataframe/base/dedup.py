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
import uuid
from functools import partial
from itertools import tee
from typing import Any, List, Set, Text, Tuple, Union

import cloudpickle
import numpy as np
import pandas as pd
from scipy.integrate import quad as integrate

from ... import opcodes
from ...core import recursive_tile
from ...core.context import Context, get_context
from ...core.operand import OperandStage
from ...serialization.serializables import AnyField, StringField
from ..operands import DataFrameOperand, DataFrameOperandMixin
from ..utils import build_concatenated_rows_frame, parse_index

NON_ALPHA = re.compile("[^A-Za-z_0-9]")
MAX_HASH = np.uint64((1 << 32) - 1)
MERSENNE_PRIME = np.uint64((1 << 61) - 1)


class UnionFind:
    """
    A data structure for maintaining disjoint sets. This helps build connected components for given duplicate pairs.

    Examples
    --------
    >>> uf = UnionFind()
    >>> uf.union(1, 2)
    >>> uf.union(2, 3)
    >>> uf.union(4, 5)
    >>> uf.find(1)
    1
    >>> uf.find(2)
    1
    >>> uf.find(3)
    1
    >>> uf.find(4)
    4
    >>> uf.find(5)
    4
    """

    def __init__(self):
        self.parent = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            return x

        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])

        return self.parent[x]

    def union(self, x, y):
        px = self.find(x)
        py = self.find(y)
        self.parent[px] = self.parent[py] = min(px, py)

    def get_self(self):
        return self

    def set_parent(self, parent):
        self.parent = parent

    def union_uf(self, uf: "UnionFind"):
        for x in uf.parent:
            if x not in self.parent:
                self.parent[x] = uf.parent[x]
            else:
                self.union(self.find(x), uf.find(x))


def ngrams(sequence: List[Text], n: int, min_length: int = 5):
    """
    Return the ngrams generated from a sequence of items, as an iterator.

    This is a modified version of nltk.util.ngrams.

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
    if len(sequence) < min_length:
        return []
    if len(sequence) < n:
        return [tuple(sequence)]
    iterables = tee(iter(sequence), n)
    for i, sub_iterable in enumerate(iterables):
        for _ in range(i):
            next(sub_iterable, None)
    return zip(*iterables)


def sha1_hash(data: bytes, d: int = 32) -> int:
    """
    Generate a d-bit hash value from the given data.

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


def embed_func(
    row: pd.Series,
    *,
    text: str,
    id: str,
    num_perm: int,
    ngram_size: int,
    min_length: int,
    hashranges: List[Tuple[int, int]],
    permutations: np.ndarray,
) -> pd.Series:
    """
    Calculate hash values for the content.

    Parameters
    ----------
    content : str
        The content to be embedded.
    idx : int
        The index of the content.
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
    >>> content = "hello world"
    >>> idx = 0
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
    >>> len(res["__signatures__"])
    10
    >>> res["__id__"]
    0
    """
    content, idx = row[text], row[id]
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
    return pd.Series({"__signatures__": Hs, "__id__": idx})


def optimal_param(
    threshold: float,
    num_perm: int,
    false_positive_weight: float = 0.5,
    false_negative_weight: float = 0.5,
):
    """
    Compute the optimal `MinHashLSH` parameter that minimizes the weighted sum
    of probabilities of false positive and false negative, taken from datasketch.

    You can also refer to the interactive demo at https://huggingface.co/spaces/bigcode/near-deduplication.

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


class DataFrameDedup(DataFrameOperand, DataFrameOperandMixin):
    _op_type = opcodes.DEDUP

    func = AnyField("func")
    union_find_name = StringField("union_find_name")

    def _load_func(self):
        if isinstance(self.func, bytes):
            return cloudpickle.loads(self.func)
        else:
            return self.func

    @classmethod
    def execute_uf(cls, ctx: Union[dict, Context], op: "DataFrameDedup"):
        uf = UnionFind()
        clusters = ctx[op.inputs[0].key]
        out = op.outputs[0]
        union_find = ctx.get_remote_object(op.union_find_name)

        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            idx = min(cluster)
            for x in cluster:
                uf.union(x, idx)

        union_find.union_uf(uf)
        ctx[out.key] = pd.DataFrame()

    @classmethod
    def execute(cls, ctx: Union[dict, Context], op: "DataFrameDedup"):
        if op.stage == OperandStage.map:
            cls.execute_uf(ctx, op)
        else:
            input_data = ctx[op.inputs[0].key]
            out = op.outputs[0]
            union_find = ctx.get_remote_object(op.union_find_name)
            uf = union_find.get_self()
            ctx[out.key] = input_data[input_data["id"].map(lambda x: uf.find(x) == x)]

    @classmethod
    def tile(cls, op: "DataFrameDedup"):
        in_df = build_concatenated_rows_frame(op.inputs[0])
        out_df = op.outputs[0]

        ctx = get_context()
        union_find_name = str(uuid.uuid4())
        ctx.create_remote_object(union_find_name, UnionFind)

        embedded = in_df.apply(
            op._load_func(),
            axis=1,
            output_type="dataframe",
            dtypes=pd.Series(["object", "int"], index=["__signatures__", "__id__"]),
        )

        clusters = (
            embedded.explode("__signatures__")
            .groupby("__signatures__", sort=False)["__id__"]
            .apply(set)
        )

        tiled_clusters = yield from recursive_tile(clusters)

        # union find stage
        chunks = []
        for c in tiled_clusters.chunks:
            new_op = op.copy().reset_key()
            new_op.union_find_name = union_find_name
            new_op.stage = OperandStage.map
            chunks.append(
                new_op.new_chunk(
                    [c],
                    index_value=parse_index(pd.RangeIndex(-1)),
                    columns_value=parse_index(pd.RangeIndex(-1)),
                    dtypes=object,
                    index=c.index,
                )
            )

        # dedup stage
        dedup_chunks = []
        for c in in_df.chunks:
            new_shape = c.shape

            new_index_value, new_columns_value = c.index_value, c.columns_value

            new_dtypes = out_df.dtypes

            new_op = op.copy().reset_key()
            new_op.union_find_name = union_find_name
            pure_depends = [False] + [True] * len(chunks)
            new_op._pure_depends = pure_depends
            dedup_chunks.append(
                new_op.new_chunk(
                    [c] + chunks,
                    shape=tuple((np.nan, new_shape[1])),
                    index=c.index,
                    dtypes=new_dtypes,
                    index_value=new_index_value,
                    columns_value=new_columns_value,
                )
            )

        yield dedup_chunks
        ctx.destroy_remote_object(union_find_name)

        new_nsplits = tuple(chunk.shape[0] for chunk in dedup_chunks), (
            dedup_chunks[0].shape[1],
        )

        new_op = op.copy()
        kw = out_df.params.copy()
        kw.update(dict(chunks=dedup_chunks, nsplits=new_nsplits))
        return new_op.new_tileables(op.inputs, **kw)

    def __call__(self, df: pd.DataFrame):
        params = {"shape": (np.nan, df.shape[1])}
        return self.new_dataframe([df], **params)


def df_dedup(
    df: pd.DataFrame,
    text: Any = "text",
    id: Any = "id",
    threshold: float = 0.7,
    num_perm: int = 256,
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
    text : str, default 'text'
        The column of the DataFrame on which to calculate hash values.

    id : str, default 'id'
        The column to store the index. If df has no such column, it will be generated.

    threshold : float, default 0.7
        The Jaccard similarity threshold to use in the MinHashLSH.

    num_perm : int, default 256
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
    ...         "id": np.arange(20),
    ...         "text": [
    ...             " ".join(["".join(np.random.choice(words, 5)) for i in range(50)])
    ...             for _ in np.arange(10)
    ...         ]
    ...         * 2,
    ...     }
    ... )
    >>> res = df.dedup()
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
    if text not in df.dtypes.index:
        raise ValueError(f"{text} column not found in the DataFrame")

    if id not in df.dtypes.index:
        df[id] = range(0, len(df))

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
        text=text,
        id=id,
        num_perm=num_perm,
        hashranges=HASH_RANGES,
        ngram_size=ngram,
        min_length=min_length,
        permutations=PERMUTATIONS,
    )

    op = DataFrameDedup(func=func)

    return op(df)

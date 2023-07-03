import numpy as np
import pandas as pd

from ... import opcodes
from ...core import OutputType
from ...serialization.serializables import Int64Field, ListField, StringField
from ..core import IndexValue
from ..merge.concat import DataFrameConcat
from ..operands import DataFrameOperand, DataFrameOperandMixin
from ..utils import build_concatenated_rows_frame, parse_index


class DataFrameNLargest(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = opcodes.NLARGEST

    largestOrSmallest = StringField("largestOrSmallest", default=None)
    n = Int64Field("n", default=0)
    columns = ListField("columns", default=None)
    keep = StringField("keep", default="first")

    def __init__(self, output_types=None, **kw):
        super().__init__(_output_types=output_types, **kw)

    @classmethod
    def _tile_dataframe(cls, op: "DataFrameNLargest"):
        df = op.inputs[0]
        df = build_concatenated_rows_frame(df)
        input_chunks = df.chunks

        if op.n >= input_chunks[0].shape[0]:
            out_chunks = input_chunks
        else:
            out_chunks = []
            for chunk in input_chunks:
                chunk_op = op.copy().reset_key()
                chunk_params = chunk.params
                if op.keep == "all":
                    chunk_params["shape"] = (np.nan, input_chunks[0].shape[1])
                else:
                    n = max(chunk.shape[0], op.n)
                    chunk_params["shape"] = (n, input_chunks[0].shape[1])
                chunk_params["index_value"] = parse_index(
                    df.chunks[0].index_value.to_pandas()[:0]
                )
                out_chunks.append(chunk_op.new_chunk([chunk], kws=[chunk_params]))
        op_concat = DataFrameConcat(
            axis=0,
            output_types=[OutputType.dataframe],
        )
        if op.keep == "all":
            shape_concat = (np.nan, input_chunks[0].shape[1])
        else:
            shape_concat = (
                sum(c.shape[0] for c in out_chunks),
                input_chunks[0].shape[1],
            )
        chunk_concat = op_concat.new_chunk(
            out_chunks,
            shape=shape_concat,
            index_value=parse_index(df.chunks[0].index_value.to_pandas()[:0]),
        )
        final_op = op.copy().reset_key()
        chunk_params = input_chunks[0].params
        chunk_params["index_value"] = parse_index(
            df.chunks[0].index_value.to_pandas()[:0]
        )
        if op.keep == "all":
            chunk_params["shape"] = (np.nan, input_chunks[0].shape[1])
        else:
            chunk_params["shape"] = (op.n, input_chunks[0].shape[1])
        c = final_op.new_chunk([chunk_concat], kws=[chunk_params])
        new_op = op.copy()
        kws = op.outputs[0].params.copy()
        if op.keep == "all":
            kws["nsplits"] = ((np.nan,), (op.outputs[0].shape[1],))
        else:
            kws["nsplits"] = ((op.n,), (op.outputs[0].shape[1],))
        kws["chunks"] = [c]
        return new_op.new_dataframes(op.inputs, **kws)

    @classmethod
    def _tile_series(cls, op: "DataFrameNLargest"):
        inp = op.inputs[0]
        if op.n >= inp.chunks[0].shape[0]:
            out_chunks = inp.chunks
        else:
            out_chunks = []
            for chunk in inp.chunks:
                chunk_op = op.copy().reset_key()
                chunk_params = chunk.params
                if op.keep == "all":
                    chunk_params["shape"] = (np.nan,)
                else:
                    n = max(chunk.shape[0], op.n)
                    chunk_params["shape"] = (n,)
                chunk_params["index_value"] = parse_index(
                    chunk.index_value.to_pandas()[:0]
                )
                out_chunks.append(chunk_op.new_chunk([chunk], kws=[chunk_params]))

        op_concat = DataFrameConcat(
            axis=0,
            output_types=[OutputType.dataframe],
        )
        if op.keep == "all":
            shape_concat = (np.nan,)
        else:
            shape_concat = (sum(c.shape[0] for c in out_chunks),)
        chunk_concat = op_concat.new_chunk(
            out_chunks,
            shape=shape_concat,
            index_value=parse_index(inp.chunks[0].index_value.to_pandas()[:0]),
        )
        final_op = op.copy().reset_key()
        chunk_params = inp.chunks[0].params
        chunk_params["index_value"] = parse_index(
            inp.chunks[0].index_value.to_pandas()[:0]
        )
        if op.keep == "all":
            chunk_params["shape"] = (np.nan,)
        else:
            chunk_params["shape"] = (op.n,)
        c = final_op.new_chunk([chunk_concat], kws=[chunk_params])
        new_op = op.copy()
        params = op.outputs[0].params
        params["chunks"] = [c]
        if op.keep == "all":
            params["nsplits"] = ((np.nan,),)
        else:
            params["nsplits"] = ((op.n,),)
        return new_op.new_seriess(op.inputs, kws=[params])

    @classmethod
    def tile(cls, op: "DataFrameNLargest"):
        inp = op.inputs[0]
        if inp.ndim == 2:
            return cls._tile_dataframe(op)
        else:
            return cls._tile_series(op)

    @classmethod
    def execute(cls, ctx, op: "DataFrameNLargest"):
        in_data = ctx[op.inputs[0].key]
        if in_data.ndim == 2:
            if op.largestOrSmallest == "largest":
                result = in_data.nlargest(n=op.n, columns=op.columns, keep=op.keep)
            elif op.largestOrSmallest == "smallest":
                result = in_data.nsmallest(n=op.n, columns=op.columns, keep=op.keep)
        else:
            if op.largestOrSmallest == "largest":
                result = in_data.nlargest(n=op.n, keep=op.keep)
            elif op.largestOrSmallest == "smallest":
                result = in_data.nsmallest(n=op.n, keep=op.keep)
        ctx[op.outputs[0].key] = result

    def __call__(self, a):
        if self.n > a.shape[0]:
            self.n = a.shape[0]
        if a.ndim == 2:
            if self.keep == "all":
                return self.new_dataframe(
                    [a],
                    shape=(np.nan, a.shape[1]),
                    dtypes=a.dtypes,
                    columns_value=a.columns_value,
                )
            else:
                return self.new_dataframe(
                    [a],
                    shape=(self.n, a.shape[1]),
                    dtypes=a.dtypes,
                    columns_value=a.columns_value,
                )
        else:
            if isinstance(a.index_value.value, IndexValue.RangeIndex):
                index_value = parse_index(pd.Index([], dtype=np.int64))
            else:
                index_value = a.index_value
            if self.keep == "all":
                return self.new_series(
                    [a],
                    shape=(np.nan,),
                    dtype=a.dtype,
                    index_value=index_value,
                    name=a.name,
                )
            else:
                return self.new_series(
                    [a],
                    shape=(self.n,),
                    dtype=a.dtype,
                    index_value=index_value,
                    name=a.name,
                )


def dataframe_nlargest(df, n, columns, keep="first"):
    """
    Return the first n rows ordered by columns in descending order.

    Parameters
    ----------
    df : Mars DataFrame
         Input dataframe.
    n :  int
         Number of rows to return.
    columns :  label or list of labels
         Column label(s) to order by.
    keep{‘first’, ‘last’, ‘all’}, default ‘first’
        Where there are duplicate values:
            first : prioritize the first occurrence(s)
            last : prioritize the last occurrence(s)
            all : do not drop any duplicates, even it means selecting more than n items.

    Returns
    -------
    sorted_obj : Mars DataFrame
        The first n rows ordered by the given columns in descending order.

    Examples
    --------
    >>> import mars.dataframe as md
    >>> df = md.DataFrame({
    ...     'col1': ['A', 'A', 'B', 'E', 'D', 'C'],
    ...     'col2': [2, 1, 9, 8, 7, 4],
    ...     'col3': [0, 1, 9, 4, 2, 3],
    ... })
    >>> df.execute()
        col1 col2 col3
    0   A    2    0
    1   A    1    1
    2   B    9    9
    3   E    8    2
    4   D    8    4
    5   C    4    3

    Choose the first 3 rows ordered by col2

    >>> df.nlargest(3,"col2").execute()
        col1 col2 col3
    2    B     9     9
    3    E     8     2
    4    D     8     4

    Choose the first 3 rows ordered by multiple columns

    >>> df.nlargest(3,['col2', 'col3']).execute()
        col1 col2 col3
    2    B     9     9
    4    E     8     4
    3    D     8     2
    """
    if keep not in ["last", "first", "all"]:
        raise ValueError(f'''keep must be either "first", "last" or "all"''')
    if isinstance(columns, str):
        columns = [columns]
    elif isinstance(columns, list):
        columns = columns
    else:
        raise KeyError(columns)

    op = DataFrameNLargest(
        largestOrSmallest="largest",
        n=n,
        columns=columns,
        keep=keep,
        output_types=[OutputType.dataframe],
        gpu=df.op.is_gpu(),
    )
    nlargest_df = op(df)
    return nlargest_df


def series_nlargest(series, n, keep="first"):
    """
    Return the largest n elements.

    Parameters
    ----------
    df : Mars Series
         Input Series.
    n :  int
         Number of rows to return.
    keep{‘first’, ‘last’, ‘all’}, default ‘first’
        Where there are duplicate values:
            first : prioritize the first occurrence(s)
            last : prioritize the last occurrence(s)
            all : do not drop any duplicates, even it means selecting more than n items.

    Returns
    -------
    sorted_obj : Mars Series
        The n largest values in the Series, sorted in decreasing order.

    Examples
    --------
    >>> import mars.dataframe as md
    >>> raw = pd.Series([8, 1, 3, 10, 5])
    >>> df.execute()
    0     8.0
    1     1.0
    2     3.0
    3     10.0
    4     5.0
    dtype: float64

    Choose the largest 3 rows

    >>> df.nlargest(3).execute()
    3     10.0
    0     8.0
    4     5.0
    dtype: float64
    """
    if keep not in ["last", "first", "all"]:
        raise ValueError(f'''keep must be either "first", "last" or "all"''')
    op = DataFrameNLargest(
        largestOrSmallest="largest",
        n=n,
        keep=keep,
        output_types=[OutputType.series],
        gpu=series.op.is_gpu(),
    )
    nlargest_df = op(series)
    return nlargest_df

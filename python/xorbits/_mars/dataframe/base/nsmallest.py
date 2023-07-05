from ...core import OutputType
from .nlargest import DataFrameNLargest


def dataframe_nsmallest(df, n, columns, keep="first"):
    """
    Return the first n rows ordered by columns in ascending order.

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
        The first n rows ordered by the given columns in ascending order.

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

    >>> df.nsmallest(3,"col2").execute()
        col1 col2 col3
    1   A    1    1
    0   A    2    0
    5   C    4    3
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
        largestOrSmallest="smallest",
        n=n,
        columns=columns,
        keep=keep,
        output_types=[OutputType.dataframe],
        gpu=df.op.is_gpu(),
    )
    nlargest_df = op(df)
    return nlargest_df


def series_nsmallest(series, n, keep="first"):
    """
    Return the smallest n elements.

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
        The n smallest values in the Series, sorted in ascreasing order.

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

    Choose the smallest 3 rows

    >>> df.nsmallest(3).execute()
    1     1.0
    2     3.0
    4     5.0
    dtype: float64
    """
    if keep not in ["last", "first", "all"]:
        raise ValueError(f'''keep must be either "first", "last" or "all"''')
    op = DataFrameNLargest(
        largestOrSmallest="smallest",
        n=n,
        keep=keep,
        output_types=[OutputType.series],
        gpu=series.op.is_gpu(),
    )
    nlargest_df = op(series)
    return nlargest_df

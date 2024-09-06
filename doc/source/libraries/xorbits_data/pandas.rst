.. _10min_pandas:

====================================
10 minutes to :code:`xorbits.pandas`
====================================

.. currentmodule:: xorbits.pandas

This is a short introduction to :code:`xorbits.pandas` which is originated from pandas' quickstart.

Customarily, we import and init as follows:

:: 

   >>> import xorbits
   >>> import xorbits.numpy as np
   >>> import xorbits.pandas as pd
   >>> xorbits.init()

Object creation
---------------

Creating a :class:`Series` by passing a list of values, letting it create a default integer index:

::

   >>> s = pd.Series([1, 3, 5, np.nan, 6, 8])
   >>> s
   0    1.0
   1    3.0
   2    5.0
   3    NaN
   4    6.0
   5    8.0
   dtype: float64

Creating a :class:`DataFrame` by passing an array, with a datetime index and labeled columns:

::

   >>> dates = pd.date_range('20130101', periods=6)
   >>> dates
   DatetimeIndex(['2013-01-01', '2013-01-02', '2013-01-03', '2013-01-04',
                  '2013-01-05', '2013-01-06'],
               dtype='datetime64[ns]', freq='D')
   >>> df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
   >>> df
                     A         B         C         D
   2013-01-01  0.411902  1.709468 -0.213158  0.821644
   2013-01-02 -0.721910 -1.677311 -1.570986 -0.621969
   2013-01-03  0.421083 -0.750191  0.269751 -2.799289
   2013-01-04 -1.329158  1.274036  2.442691 -0.409725
   2013-01-05  0.689205 -1.501951  0.363000  0.401498
   2013-01-06  0.426947 -0.469598 -1.295293 -1.435165

Creating a :class:`DataFrame` by passing a dict of objects that can be converted to series-like.

::

   >>> df2 = pd.DataFrame({'A': 1.,
                           'B': pd.Timestamp('20130102'),
                           'C': pd.Series(1, index=list(range(4)), dtype='float32'),
                           'D': np.array([3] * 4, dtype='int32'),
                           'E': 'foo'})
   >>> df2
      A          B    C  D    E
   0  1.0 2013-01-02  1.0  3  foo
   1  1.0 2013-01-02  1.0  3  foo
   2  1.0 2013-01-02  1.0  3  foo
   3  1.0 2013-01-02  1.0  3  foo

The columns of the resulting :class:`DataFrame` have different dtypes.

::

   >>> df2.dtypes
   A          float64
   B    datetime64[s]
   C          float32
   D            int32
   E           object
   dtype: object


Viewing data
------------

Here is how to view the top and bottom rows of the frame:

::

   >>> df.head()
                     A         B         C         D
   2013-01-01  0.411902  1.709468 -0.213158  0.821644
   2013-01-02 -0.721910 -1.677311 -1.570986 -0.621969
   2013-01-03  0.421083 -0.750191  0.269751 -2.799289
   2013-01-04 -1.329158  1.274036  2.442691 -0.409725
   2013-01-05  0.689205 -1.501951  0.363000  0.401498
   >>> df.tail(3)
                     A         B         C         D
   2013-01-04 -1.329158  1.274036  2.442691 -0.409725
   2013-01-05  0.689205 -1.501951  0.363000  0.401498
   2013-01-06  0.426947 -0.469598 -1.295293 -1.435165

Display the index, columns:

::

   >>> df.index
   DatetimeIndex(['2013-01-01', '2013-01-02', '2013-01-03', '2013-01-04',
                  '2013-01-05', '2013-01-06'],
               dtype='datetime64[ns]', freq='D')
   >>> df.columns
   Index(['A', 'B', 'C', 'D'], dtype='object')


:meth:`DataFrame.to_numpy` gives a ndarray representation of the underlying data. Note that this
can be an expensive operation when your :class:`DataFrame` has columns with different data types,
which comes down to a fundamental difference between DataFrame and ndarray: **ndarrays have one
dtype for the entire ndarray, while DataFrames have one dtype per column**. When you call
:meth:`DataFrame.to_numpy`, :code:`xorbits.pandas` will find the ndarray dtype that can hold *all*
of the dtypes in the DataFrame. This may end up being ``object``, which requires casting every
value to a Python object.

For ``df``, our :class:`DataFrame` of all floating-point values,
:meth:`DataFrame.to_numpy` is fast and doesn't require copying data.

::

   >>> df.to_numpy()
   array([[ 0.41190169,  1.70946816, -0.21315821,  0.82164367],
         [-0.72191001, -1.67731119, -1.57098611, -0.62196894],
         [ 0.42108334, -0.75019064,  0.26975121, -2.79928919],
         [-1.32915794,  1.2740364 ,  2.44269141, -0.40972548],
         [ 0.68920499, -1.50195139,  0.36299995,  0.40149762],
         [ 0.42694729, -0.46959787, -1.29529258, -1.43516459]])

For ``df2``, the :class:`DataFrame` with multiple dtypes, :meth:`DataFrame.to_numpy` is relatively
expensive.

::

   >>> df2.to_numpy()
   array([[1.0, Timestamp('2013-01-02 00:00:00'), 1.0, 3, 'foo'],
         [1.0, Timestamp('2013-01-02 00:00:00'), 1.0, 3, 'foo'],
         [1.0, Timestamp('2013-01-02 00:00:00'), 1.0, 3, 'foo'],
         [1.0, Timestamp('2013-01-02 00:00:00'), 1.0, 3, 'foo']],
         dtype=object)

.. note::

   :meth:`DataFrame.to_numpy` does *not* include the index or column
   labels in the output.

:func:`~DataFrame.describe` shows a quick statistic summary of your data:

::

   >>> df.describe()
               A         B         C         D
   count  6.000000  6.000000  6.000000  6.000000
   mean  -0.016988 -0.235924 -0.000666 -0.673834
   std    0.811215  1.418734  1.439617  1.308619
   min   -1.329158 -1.677311 -1.570986 -2.799289
   25%   -0.438457 -1.314011 -1.024759 -1.231866
   50%    0.416493 -0.609894  0.028296 -0.515847
   75%    0.425481  0.838128  0.339688  0.198692
   max    0.689205  1.709468  2.442691  0.821644

Sorting by an axis:

::

   >>> df.sort_index(axis=1, ascending=False)
                      D         C         B         A
   2013-01-01  0.821644 -0.213158  1.709468  0.411902
   2013-01-02 -0.621969 -1.570986 -1.677311 -0.721910
   2013-01-03 -2.799289  0.269751 -0.750191  0.421083
   2013-01-04 -0.409725  2.442691  1.274036 -1.329158
   2013-01-05  0.401498  0.363000 -1.501951  0.689205
   2013-01-06 -1.435165 -1.295293 -0.469598  0.426947

Sorting by values:

::

   >>> df.sort_values(by='B')
                      A         B         C         D
   2013-01-02 -0.721910 -1.677311 -1.570986 -0.621969
   2013-01-05  0.689205 -1.501951  0.363000  0.401498
   2013-01-03  0.421083 -0.750191  0.269751 -2.799289
   2013-01-06  0.426947 -0.469598 -1.295293 -1.435165
   2013-01-04 -1.329158  1.274036  2.442691 -0.409725
   2013-01-01  0.411902  1.709468 -0.213158  0.821644

Selection
---------

.. note::

   While standard Python expressions for selecting and setting are
   intuitive and come in handy for interactive work, for production code, we
   recommend the optimized :code:`xorbits.pandas` data access methods, ``.at``, ``.iat``,
   ``.loc`` and ``.iloc``.


Getting
~~~~~~~

Selecting a single column, which yields a :class:`Series`, equivalent to ``df.A``:

::

   >>> df['A']
   2013-01-01    0.411902
   2013-01-02   -0.721910
   2013-01-03    0.421083
   2013-01-04   -1.329158
   2013-01-05    0.689205
   2013-01-06    0.426947
   Freq: D, Name: A, dtype: float64

Selecting via ``[]``, which slices the rows:

::

   >>> df[0:3]
                      A         B         C         D
   2013-01-01  0.411902  1.709468 -0.213158  0.821644
   2013-01-02 -0.721910 -1.677311 -1.570986 -0.621969
   2013-01-03  0.421083 -0.750191  0.269751 -2.799289
   >>> df['20130102':'20130104']
                      A         B         C         D
   2013-01-02 -0.721910 -1.677311 -1.570986 -0.621969
   2013-01-03  0.421083 -0.750191  0.269751 -2.799289
   2013-01-04 -1.329158  1.274036  2.442691 -0.409725

Selection by label
~~~~~~~~~~~~~~~~~~

For getting a cross section using a label:

::

   >>> df.loc['20130101']
   A    0.411902
   B    1.709468
   C   -0.213158
   D    0.821644
   Name: 2013-01-01 00:00:00, dtype: float64

Selecting on a multi-axis by label:

::

   >>> df.loc[:, ['A', 'B']]
                      A         B
   2013-01-01  0.411902  1.709468
   2013-01-02 -0.721910 -1.677311
   2013-01-03  0.421083 -0.750191
   2013-01-04 -1.329158  1.274036
   2013-01-05  0.689205 -1.501951
   2013-01-06  0.426947 -0.469598

Showing label slicing, both endpoints are *included*:

::

   >>> df.loc['20130102':'20130104', ['A', 'B']]
                      A         B
   2013-01-02 -0.721910 -1.677311
   2013-01-03  0.421083 -0.750191
   2013-01-04 -1.329158  1.274036

Reduction in the dimensions of the returned object:

::

   >>> df.loc['20130102', ['A', 'B']]
   A   -0.721910
   B   -1.677311
   Name: 2013-01-02 00:00:00, dtype: float64

For getting a scalar value:

::

   >>> df.loc['20130101', 'A']
   0.41190169091385387

For getting fast access to a scalar (equivalent to the prior method):

::

   >>> df.at['20130101', 'A']
   0.41190169091385387

Selection by position
~~~~~~~~~~~~~~~~~~~~~

Select via the position of the passed integers:

::

   >>> df.iloc[3]
   A   -1.329158
   B    1.274036
   C    2.442691
   D   -0.409725
   Name: 2013-01-04 00:00:00, dtype: float64

By integer slices, acting similar to python:

::

   >>> df.iloc[3:5, 0:2]
                      A         B
   2013-01-04 -1.329158  1.274036
   2013-01-05  0.689205 -1.501951

By lists of integer position locations, similar to the python style:

::

   >>> df.iloc[[1, 2, 4], [0, 2]]
                      A         C
   2013-01-02 -0.721910 -1.570986
   2013-01-03  0.421083  0.269751
   2013-01-05  0.689205  0.363000

For slicing rows explicitly:

::

   >>> df.iloc[1:3, :]
                      A         B         C         D
   2013-01-02 -0.721910 -1.677311 -1.570986 -0.621969
   2013-01-03  0.421083 -0.750191  0.269751 -2.799289

For slicing columns explicitly:

::

   >>> df.iloc[:, 1:3]
                      B         C
   2013-01-01  1.709468 -0.213158
   2013-01-02 -1.677311 -1.570986
   2013-01-03 -0.750191  0.269751
   2013-01-04  1.274036  2.442691
   2013-01-05 -1.501951  0.363000
   2013-01-06 -0.469598 -1.295293

For getting a value explicitly:

::

   >>> df.iloc[1, 1]
   -1.6773111933012679

For getting fast access to a scalar (equivalent to the prior method):

::

   >>> df.iat[1, 1]
   -1.6773111933012679

Boolean indexing
~~~~~~~~~~~~~~~~

Using a single column's values to select data.

::

   >>> df[df['A'] > 0]
                      A         B         C         D
   2013-01-01  0.411902  1.709468 -0.213158  0.821644
   2013-01-03  0.421083 -0.750191  0.269751 -2.799289
   2013-01-05  0.689205 -1.501951  0.363000  0.401498
   2013-01-06  0.426947 -0.469598 -1.295293 -1.435165

Selecting values from a DataFrame where a boolean condition is met.

::

   >>> df[df > 0]
                      A         B         C         D
   2013-01-01  0.411902  1.709468       NaN  0.821644
   2013-01-02       NaN       NaN       NaN       NaN
   2013-01-03  0.421083       NaN  0.269751       NaN
   2013-01-04       NaN  1.274036  2.442691       NaN
   2013-01-05  0.689205       NaN  0.363000  0.401498
   2013-01-06  0.426947       NaN       NaN       NaN


Operations
----------

Stats
~~~~~

Operations in general *exclude* missing data.

Performing a descriptive statistic:

::

   >>> df.mean()
   A   -0.016988
   B   -0.235924
   C   -0.000666
   D   -0.673834
   dtype: float64


Same operation on the other axis:

::

   >>> df.mean(1)
   2013-01-01    0.682464
   2013-01-02   -1.148044
   2013-01-03   -0.714661
   2013-01-04    0.494461
   2013-01-05   -0.012062
   2013-01-06   -0.693277
   Freq: D, dtype: float64


Operating with objects that have different dimensionality and need alignment. In addition,
:code:`xorbits.pandas` automatically broadcasts along the specified dimension.

::

   >>> s = pd.Series([1, 3, 5, np.nan, 6, 8], index=dates).shift(2)
   >>> s
   2013-01-01    NaN
   2013-01-02    NaN
   2013-01-03    1.0
   2013-01-04    3.0
   2013-01-05    5.0
   2013-01-06    NaN
   Freq: D, dtype: float64
   >>> df.sub(s, axis='index')
                      A         B         C         D
   2013-01-01       NaN       NaN       NaN       NaN
   2013-01-02       NaN       NaN       NaN       NaN
   2013-01-03 -0.578917 -1.750191 -0.730249 -3.799289
   2013-01-04 -4.329158 -1.725964 -0.557309 -3.409725
   2013-01-05 -4.310795 -6.501951 -4.637000 -4.598502
   2013-01-06       NaN       NaN       NaN       NaN


Apply
~~~~~

Applying functions to the data:

::

   >>> df.apply(lambda x: x.max() - x.min())
   A    2.018363
   B    3.386779
   C    4.013678
   D    3.620933
   dtype: float64

String Methods
~~~~~~~~~~~~~~

Series is equipped with a set of string processing methods in the `str`
attribute that make it easy to operate on each element of the array, as in the
code snippet below. Note that pattern-matching in `str` generally uses `regular
expressions <https://docs.python.org/3/library/re.html>`__ by default (and in
some cases always uses them).

::

   >>> s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])
   >>> s.str.lower()
   0       a
   1       b
   2       c
   3    aaba
   4    baca
   5     NaN
   6    caba
   7     dog
   8     cat
   dtype: object

Merge
-----

Concat
~~~~~~

:code:`xorbits.pandas` provides various facilities for easily combining together Series and
DataFrame objects with various kinds of set logic for the indexes
and relational algebra functionality in the case of join / merge-type
operations.

Concatenating :code:`xorbits.pandas` objects together with :func:`concat`:

::

   >>> df = pd.DataFrame(np.random.randn(10, 4))
   >>> df
             0         1         2         3
   0 -0.495508  0.903802  2.152979  1.098698
   1 -0.327001 -0.586382  1.999350 -1.056401
   2  0.341923 -0.024582  0.439198  0.662602
   3 -1.896886  0.181549  0.119640 -1.426697
   4 -2.407668 -0.780552 -1.301063  0.510010
   5 -0.350738 -0.147771 -0.566869 -2.414299
   6 -1.994935 -0.486425 -0.531758  1.624540
   7 -0.358207 -0.884470  1.257721  0.587503
   8 -0.945414 -1.055967  1.334790  0.817954
   9  1.116094 -0.664818 -0.298791  0.042105

   >>> # break it into pieces
   >>> pieces = [df[:3], df[3:7], df[7:]]

   >>> pd.concat(pieces)
             0         1         2         3
   0 -0.495508  0.903802  2.152979  1.098698
   1 -0.327001 -0.586382  1.999350 -1.056401
   2  0.341923 -0.024582  0.439198  0.662602
   3 -1.896886  0.181549  0.119640 -1.426697
   4 -2.407668 -0.780552 -1.301063  0.510010
   5 -0.350738 -0.147771 -0.566869 -2.414299
   6 -1.994935 -0.486425 -0.531758  1.624540
   7 -0.358207 -0.884470  1.257721  0.587503
   8 -0.945414 -1.055967  1.334790  0.817954
   9  1.116094 -0.664818 -0.298791  0.042105

.. note::
   Adding a column to a :class:`DataFrame` is relatively fast. However, adding
   a row requires a copy, and may be expensive. We recommend passing a
   pre-built list of records to the :class:`DataFrame` constructor instead
   of building a :class:`DataFrame` by iteratively appending records to it.

Join
~~~~

SQL style merges.

::

   >>> left = pd.DataFrame({'key': ['foo', 'foo'], 'lval': [1, 2]})
   >>> right = pd.DataFrame({'key': ['foo', 'foo'], 'rval': [4, 5]})
   >>> left
      key  lval
   0  foo     1
   1  foo     2
   >>> right
      key  rval
   0  foo     4
   1  foo     5
   >>> pd.merge(left, right, on='key')
      key  lval  rval
   0  foo     1     4
   1  foo     1     5
   2  foo     2     4
   3  foo     2     5

Another example that can be given is:

::

   >>> left = pd.DataFrame({'key': ['foo', 'bar'], 'lval': [1, 2]})
   >>> right = pd.DataFrame({'key': ['foo', 'bar'], 'rval': [4, 5]})
   >>> left
      key  lval
   0  foo     1
   1  bar     2
   >>> right
      key  rval
   0  foo     4
   1  bar     5
   >>> pd.merge(left, right, on='key')
      key  lval  rval
   0  foo     1     4
   1  bar     2     5


Grouping
--------

By "group by" we are referring to a process involving one or more of the
following steps:

 - **Splitting** the data into groups based on some criteria
 - **Applying** a function to each group independently
 - **Combining** the results into a data structure


::

   >>> df = pd.DataFrame({'A': ['foo', 'bar', 'foo', 'bar',
                                 'foo', 'bar', 'foo', 'foo'],
                          'B': ['one', 'one', 'two', 'three',
                                 'two', 'two', 'one', 'three'],
                          'C': np.random.randn(8),
                          'D': np.random.randn(8)})
   >>> df
        A      B         C         D
   0  foo    one -0.473456  1.016378
   1  bar    one  0.373591  0.480215
   2  foo    two -0.538622 -0.490436
   3  bar  three -1.833243 -1.471246
   4  foo    two -0.083388  1.389476
   5  bar    two  0.874384  2.006862
   6  foo    one -0.968538 -1.703000
   7  foo  three -1.840837  0.066493

Grouping and then applying the :meth:`~xorbits.pandas.groupby.DataFrameGroupBy.sum` function to
the resulting groups.

::

   >>> df.groupby('A').sum()
                        B         C         D
   A                                         
   bar        onethreetwo -0.585268  1.015831
   foo  onetwotwoonethree -3.904840  0.278910

Grouping by multiple columns forms a hierarchical index, and again we can
apply the `sum` function.

::

   >>> df.groupby(['A', 'B']).sum()
                  C         D
   A   B                        
   bar one    0.373591  0.480215
       three -1.833243 -1.471246
       two    0.874384  2.006862
   foo one   -1.441994 -0.686622
       three -1.840837  0.066493
       two   -0.622010  0.899039

Plotting
--------

We use the standard convention for referencing the matplotlib API:

::

   >>> import matplotlib.pyplot as plt
   >>> plt.close('all')

::

   >>> ts = pd.Series(np.random.randn(1000),
                      index=pd.date_range('1/1/2000', periods=1000))
   >>> ts = ts.cumsum()

   >>> @savefig series_plot_basic.png
   >>> ts.plot()

On a DataFrame, the :meth:`~DataFrame.plot` method is a convenience to plot all
of the columns with labels:

::

   >>> df = pd.DataFrame(np.random.randn(1000, 4), index=ts.index,
                         columns=['A', 'B', 'C', 'D'])
   >>> df = df.cumsum()

   >>> plt.figure()
   >>> df.plot()
   >>> @savefig frame_plot_basic.png
   >>> plt.legend(loc='best')

Getting data in/out
-------------------

CSV
~~~

Writing to a csv file.

::

   >>> df.to_csv('foo.csv')
   Empty DataFrame
   Columns: []
   Index: []

Reading from a csv file.

::

   >>> pd.read_csv('foo.csv')
        Unnamed: 0         A          B         C          D
   0    2000-01-01  0.385646   1.201584 -1.701511  -0.693112
   1    2000-01-02  0.331648  -0.203431 -1.030354  -0.045550
   2    2000-01-03  0.112350   0.024239 -0.690759  -1.354678
   3    2000-01-04 -0.492772  -1.407550  0.535260  -0.030373
   4    2000-01-05 -0.557673   0.116826  2.127525  -0.835155
   ..          ...       ...        ...       ...        ...
   995  2002-09-22  6.795263  15.514409 -8.909048 -43.613612
   996  2002-09-23  5.241447  15.386009 -9.248272 -43.035980
   997  2002-09-24  2.541217  14.514584 -9.051257 -43.824801
   998  2002-09-25  1.450811  14.913616 -9.681888 -42.579596
   999  2002-09-26  1.895067  16.139412 -8.192430 -42.140289

   [1000 rows x 5 columns]

::

   >>> import os
   >>> os.remove('foo.csv')
   >>> xorbits.shutdown()

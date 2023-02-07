.. _10min_pandas:

====================================
10 minutes to :code:`xorbits.pandas`
====================================

.. currentmodule:: xorbits.pandas

This is a short introduction to :code:`xorbits.pandas` which is originated from pandas' quickstart.

Customarily, we import and init as follows:

.. ipython:: python

   import xorbits
   import xorbits.numpy as np
   import xorbits.pandas as pd
   xorbits.init()

Object creation
---------------

Creating a :class:`Series` by passing a list of values, letting it create a default integer index:

.. ipython:: python
   :okwarning:

   s = pd.Series([1, 3, 5, np.nan, 6, 8])
   s

Creating a :class:`DataFrame` by passing an array, with a datetime index and labeled columns:

.. ipython:: python

   dates = pd.date_range('20130101', periods=6)
   dates
   df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
   df

Creating a :class:`DataFrame` by passing a dict of objects that can be converted to series-like.

.. ipython:: python

   df2 = pd.DataFrame({'A': 1.,
                       'B': pd.Timestamp('20130102'),
                       'C': pd.Series(1, index=list(range(4)), dtype='float32'),
                       'D': np.array([3] * 4, dtype='int32'),
                       'E': 'foo'})
   df2

The columns of the resulting :class:`DataFrame` have different dtypes.

.. ipython:: python

   df2.dtypes


Viewing data
------------

Here is how to view the top and bottom rows of the frame:

.. ipython:: python

   df.head()
   df.tail(3)

Display the index, columns:

.. ipython:: python

   df.index
   df.columns


:meth:`DataFrame.to_numpy` gives a ndarray representation of the underlying data. Note that this
can be an expensive operation when your :class:`DataFrame` has columns with different data types,
which comes down to a fundamental difference between DataFrame and ndarray: **ndarrays have one
dtype for the entire ndarray, while DataFrames have one dtype per column**. When you call
:meth:`DataFrame.to_numpy`, :code:`xorbits.pandas` will find the ndarray dtype that can hold *all*
of the dtypes in the DataFrame. This may end up being ``object``, which requires casting every
value to a Python object.

For ``df``, our :class:`DataFrame` of all floating-point values,
:meth:`DataFrame.to_numpy` is fast and doesn't require copying data.

.. ipython:: python

   df.to_numpy()

For ``df2``, the :class:`DataFrame` with multiple dtypes, :meth:`DataFrame.to_numpy` is relatively
expensive.

.. ipython:: python

   df2.to_numpy()

.. note::

   :meth:`DataFrame.to_numpy` does *not* include the index or column
   labels in the output.

:func:`~DataFrame.describe` shows a quick statistic summary of your data:

.. ipython:: python

   df.describe()

Sorting by an axis:

.. ipython:: python

   df.sort_index(axis=1, ascending=False)

Sorting by values:

.. ipython:: python

   df.sort_values(by='B')

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

.. ipython:: python

   df['A']

Selecting via ``[]``, which slices the rows:

.. ipython:: python
   :okwarning:

   df[0:3]
   df['20130102':'20130104']

Selection by label
~~~~~~~~~~~~~~~~~~

For getting a cross section using a label:

.. ipython:: python

   df.loc['20130101']

Selecting on a multi-axis by label:

.. ipython:: python

   df.loc[:, ['A', 'B']]

Showing label slicing, both endpoints are *included*:

.. ipython:: python
   :okwarning:

   df.loc['20130102':'20130104', ['A', 'B']]

Reduction in the dimensions of the returned object:

.. ipython:: python

   df.loc['20130102', ['A', 'B']]

For getting a scalar value:

.. ipython:: python

   df.loc['20130101', 'A']

For getting fast access to a scalar (equivalent to the prior method):

.. ipython:: python

   df.at['20130101', 'A']

Selection by position
~~~~~~~~~~~~~~~~~~~~~

Select via the position of the passed integers:

.. ipython:: python

   df.iloc[3]

By integer slices, acting similar to python:

.. ipython:: python

   df.iloc[3:5, 0:2]

By lists of integer position locations, similar to the python style:

.. ipython:: python

   df.iloc[[1, 2, 4], [0, 2]]

For slicing rows explicitly:

.. ipython:: python

   df.iloc[1:3, :]

For slicing columns explicitly:

.. ipython:: python

   df.iloc[:, 1:3]

For getting a value explicitly:

.. ipython:: python

   df.iloc[1, 1]

For getting fast access to a scalar (equivalent to the prior method):

.. ipython:: python

   df.iat[1, 1]

Boolean indexing
~~~~~~~~~~~~~~~~

Using a single column's values to select data.

.. ipython:: python

   df[df['A'] > 0]

Selecting values from a DataFrame where a boolean condition is met.

.. ipython:: python

   df[df > 0]


Operations
----------

Stats
~~~~~

Operations in general *exclude* missing data.

Performing a descriptive statistic:

.. ipython:: python

   df.mean()

Same operation on the other axis:

.. ipython:: python

   df.mean(1)


Operating with objects that have different dimensionality and need alignment. In addition,
:code:`xorbits.pandas` automatically broadcasts along the specified dimension.

.. ipython:: python

   s = pd.Series([1, 3, 5, np.nan, 6, 8], index=dates).shift(2)
   s
   df.sub(s, axis='index')


Apply
~~~~~

Applying functions to the data:

.. ipython:: python

   df.apply(lambda x: x.max() - x.min())

String Methods
~~~~~~~~~~~~~~

Series is equipped with a set of string processing methods in the `str`
attribute that make it easy to operate on each element of the array, as in the
code snippet below. Note that pattern-matching in `str` generally uses `regular
expressions <https://docs.python.org/3/library/re.html>`__ by default (and in
some cases always uses them).

.. ipython:: python

   s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])
   s.str.lower()

Merge
-----

Concat
~~~~~~

:code:`xorbits.pandas` provides various facilities for easily combining together Series and
DataFrame objects with various kinds of set logic for the indexes
and relational algebra functionality in the case of join / merge-type
operations.

Concatenating :code:`xorbits.pandas` objects together with :func:`concat`:

.. ipython:: python

   df = pd.DataFrame(np.random.randn(10, 4))
   df

   # break it into pieces
   pieces = [df[:3], df[3:7], df[7:]]

   pd.concat(pieces)

.. note::
   Adding a column to a :class:`DataFrame` is relatively fast. However, adding
   a row requires a copy, and may be expensive. We recommend passing a
   pre-built list of records to the :class:`DataFrame` constructor instead
   of building a :class:`DataFrame` by iteratively appending records to it.

Join
~~~~

SQL style merges.

.. ipython:: python

   left = pd.DataFrame({'key': ['foo', 'foo'], 'lval': [1, 2]})
   right = pd.DataFrame({'key': ['foo', 'foo'], 'rval': [4, 5]})
   left
   right
   pd.merge(left, right, on='key')

Another example that can be given is:

.. ipython:: python

   left = pd.DataFrame({'key': ['foo', 'bar'], 'lval': [1, 2]})
   right = pd.DataFrame({'key': ['foo', 'bar'], 'rval': [4, 5]})
   left
   right
   pd.merge(left, right, on='key')

Grouping
--------

By "group by" we are referring to a process involving one or more of the
following steps:

 - **Splitting** the data into groups based on some criteria
 - **Applying** a function to each group independently
 - **Combining** the results into a data structure


.. ipython:: python

   df = pd.DataFrame({'A': ['foo', 'bar', 'foo', 'bar',
                            'foo', 'bar', 'foo', 'foo'],
                      'B': ['one', 'one', 'two', 'three',
                            'two', 'two', 'one', 'three'],
                      'C': np.random.randn(8),
                      'D': np.random.randn(8)})
   df

Grouping and then applying the :meth:`~xorbits.pandas.groupby.DataFrameGroupBy.sum` function to
the resulting groups.

.. ipython:: python
   :okwarning:

   df.groupby('A').sum()

Grouping by multiple columns forms a hierarchical index, and again we can
apply the `sum` function.

.. ipython:: python

   df.groupby(['A', 'B']).sum()

Plotting
--------

We use the standard convention for referencing the matplotlib API:

.. ipython:: python

   import matplotlib.pyplot as plt
   plt.close('all')

.. ipython:: python

   ts = pd.Series(np.random.randn(1000),
                  index=pd.date_range('1/1/2000', periods=1000))
   ts = ts.cumsum()

   @savefig series_plot_basic.png
   ts.plot()

On a DataFrame, the :meth:`~DataFrame.plot` method is a convenience to plot all
of the columns with labels:

.. ipython:: python

   df = pd.DataFrame(np.random.randn(1000, 4), index=ts.index,
                     columns=['A', 'B', 'C', 'D'])
   df = df.cumsum()

   plt.figure()
   df.plot()
   @savefig frame_plot_basic.png
   plt.legend(loc='best')

Getting data in/out
-------------------

CSV
~~~

Writing to a csv file.

.. ipython:: python

   df.to_csv('foo.csv')

Reading from a csv file.

.. ipython:: python

   pd.read_csv('foo.csv')

.. ipython:: python
   :suppress:

   import os
   os.remove('foo.csv')
   xorbits.shutdown()

.. _api.groupby:

=======
GroupBy
=======
.. currentmodule:: xorbits.pandas.groupby

GroupBy objects are returned by groupby calls: :func:`xorbits.pandas.DataFrame.groupby`,
:func:`xorbits.pandas.Series.groupby`, etc.

Indexing, iteration
-------------------
.. autosummary::
   :toctree: generated/

   DataFrameGroupBy.__iter__
   SeriesGroupBy.__iter__
   DataFrameGroupBy.groups
   SeriesGroupBy.groups
   DataFrameGroupBy.indices
   SeriesGroupBy.indices
   DataFrameGroupBy.get_group
   SeriesGroupBy.get_group

.. currentmodule:: xorbits.pandas

.. autosummary::
   :toctree: generated/
   :template: autosummary/class_without_autosummary.rst

   Grouper

Function application helper
---------------------------
.. autosummary::
   :toctree: generated/

   NamedAgg

.. currentmodule:: xorbits.pandas.groupby

Function application
--------------------
.. autosummary::
   :toctree: generated/

   SeriesGroupBy.apply
   DataFrameGroupBy.apply
   SeriesGroupBy.agg
   DataFrameGroupBy.agg
   SeriesGroupBy.aggregate
   DataFrameGroupBy.aggregate
   SeriesGroupBy.transform
   DataFrameGroupBy.transform
   SeriesGroupBy.pipe
   DataFrameGroupBy.pipe
   DataFrameGroupBy.filter
   SeriesGroupBy.filter

``DataFrameGroupBy`` computations / descriptive stats
-----------------------------------------------------
.. autosummary::
   :toctree: generated/

   DataFrameGroupBy.all
   DataFrameGroupBy.any
   DataFrameGroupBy.backfill
   DataFrameGroupBy.bfill
   DataFrameGroupBy.corr
   DataFrameGroupBy.corrwith
   DataFrameGroupBy.count
   DataFrameGroupBy.cov
   DataFrameGroupBy.cumcount
   DataFrameGroupBy.cummax
   DataFrameGroupBy.cummin
   DataFrameGroupBy.cumprod
   DataFrameGroupBy.cumsum
   DataFrameGroupBy.describe
   DataFrameGroupBy.diff
   DataFrameGroupBy.ffill
   DataFrameGroupBy.fillna
   DataFrameGroupBy.first
   DataFrameGroupBy.head
   DataFrameGroupBy.idxmax
   DataFrameGroupBy.idxmin
   DataFrameGroupBy.last
   DataFrameGroupBy.mad
   DataFrameGroupBy.max
   DataFrameGroupBy.mean
   DataFrameGroupBy.median
   DataFrameGroupBy.min
   DataFrameGroupBy.ngroup
   DataFrameGroupBy.nth
   DataFrameGroupBy.nunique
   DataFrameGroupBy.ohlc
   DataFrameGroupBy.pad
   DataFrameGroupBy.pct_change
   DataFrameGroupBy.prod
   DataFrameGroupBy.quantile
   DataFrameGroupBy.rank
   DataFrameGroupBy.resample
   DataFrameGroupBy.sample
   DataFrameGroupBy.sem
   DataFrameGroupBy.shift
   DataFrameGroupBy.size
   DataFrameGroupBy.skew
   DataFrameGroupBy.std
   DataFrameGroupBy.sum
   DataFrameGroupBy.var
   DataFrameGroupBy.tail
   DataFrameGroupBy.take
   DataFrameGroupBy.tshift
   DataFrameGroupBy.value_counts

``SeriesGroupBy`` computations / descriptive stats
--------------------------------------------------
.. autosummary::
   :toctree: generated/

   SeriesGroupBy.all
   SeriesGroupBy.any
   SeriesGroupBy.backfill
   SeriesGroupBy.bfill
   SeriesGroupBy.corr
   SeriesGroupBy.count
   SeriesGroupBy.cov
   SeriesGroupBy.cumcount
   SeriesGroupBy.cummax
   SeriesGroupBy.cummin
   SeriesGroupBy.cumprod
   SeriesGroupBy.cumsum
   SeriesGroupBy.describe
   SeriesGroupBy.diff
   SeriesGroupBy.ffill
   SeriesGroupBy.fillna
   SeriesGroupBy.first
   SeriesGroupBy.head
   SeriesGroupBy.last
   SeriesGroupBy.idxmax
   SeriesGroupBy.idxmin
   SeriesGroupBy.is_monotonic_increasing
   SeriesGroupBy.is_monotonic_decreasing
   SeriesGroupBy.mad
   SeriesGroupBy.max
   SeriesGroupBy.mean
   SeriesGroupBy.median
   SeriesGroupBy.min
   SeriesGroupBy.ngroup
   SeriesGroupBy.nlargest
   SeriesGroupBy.nsmallest
   SeriesGroupBy.nth
   SeriesGroupBy.nunique
   SeriesGroupBy.unique
   SeriesGroupBy.ohlc
   SeriesGroupBy.pad
   SeriesGroupBy.pct_change
   SeriesGroupBy.prod
   SeriesGroupBy.quantile
   SeriesGroupBy.rank
   SeriesGroupBy.resample
   SeriesGroupBy.sample
   SeriesGroupBy.sem
   SeriesGroupBy.shift
   SeriesGroupBy.size
   SeriesGroupBy.skew
   SeriesGroupBy.std
   SeriesGroupBy.sum
   SeriesGroupBy.var
   SeriesGroupBy.tail
   SeriesGroupBy.take
   SeriesGroupBy.tshift
   SeriesGroupBy.value_counts

Plotting and visualization
--------------------------
.. autosummary::
   :toctree: generated/

   DataFrameGroupBy.boxplot
   DataFrameGroupBy.hist
   SeriesGroupBy.hist
   DataFrameGroupBy.plot
   SeriesGroupBy.plot

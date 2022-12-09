.. _api.general_functions:

=================
General functions
=================
.. currentmodule:: xorbits.pandas

Data manipulations
~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   melt
   pivot
   pivot_table
   crosstab
   cut
   qcut
   merge
   merge_ordered
   merge_asof
   concat
   get_dummies
   from_dummies
   factorize
   unique
   lreshape
   wide_to_long

Top-level missing data
~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   isna
   isnull
   notna
   notnull

Top-level dealing with numeric data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   to_numeric

Top-level dealing with datetimelike data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   to_datetime
   to_timedelta
   date_range
   bdate_range
   period_range
   timedelta_range
   infer_freq

Top-level dealing with Interval data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   interval_range

Top-level evaluation
~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   eval

Hashing
~~~~~~~
.. autosummary::
   :toctree: generated/

   util.hash_array
   util.hash_pandas_object

Importing from other DataFrame libraries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   api.interchange.from_dataframe

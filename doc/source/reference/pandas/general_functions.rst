.. _api.general_functions:

=================
General functions
=================

The following table lists both implemented and not implemented methods. If you have need
of an operation that is listed as not implemented, feel free to open an issue on the
`GitHub repository`_, or give a thumbs up to already created issues. Contributions are
also welcome!

The following table is structured as follows: The first column contains the method name.
The second column contains link to a description of corresponding pandas method.
The third column is a flag for whether or not there is an implementation in Xorbits
for the method in the left column. ``Y`` stands for yes, ``N`` stands for no, ``P`` standsfor partial 
(meaning some parameters may not be supported yet), and ``D`` stands for default to pandas.

Data manipulations
~~~~~~~~~~~~~~~~~~

+--------------------+------------------+------------------------+----------------------------------+
| ``xorbits.pandas`` | ``pandas``       | Implemented? (Y/N/P/D) | Notes for Current implementation |
+--------------------+------------------+------------------------+----------------------------------+
| ``melt``           | `melt`_          | Y                      |                                  |
+--------------------+------------------+------------------------+----------------------------------+
| ``pivot``          | `pivot`_         | Y                      |                                  |
+--------------------+------------------+------------------------+----------------------------------+
| ``pivot_table``    | `pivot_table`_   | Y                      |                                  |
+--------------------+------------------+------------------------+----------------------------------+
| ``crosstab``       | `crosstab`_      | Y                      |                                  |
+--------------------+------------------+------------------------+----------------------------------+
| ``cut``            | `cut`_           | Y                      |                                  |
+--------------------+------------------+------------------------+----------------------------------+
| ``qcut``           | `qcut`_          | Y                      |                                  |
+--------------------+------------------+------------------------+----------------------------------+
| ``merge``          | `merge`_         | Y                      |                                  |
+--------------------+------------------+------------------------+----------------------------------+
| ``merge_ordered``  | `merge_ordered`_ | Y                      |                                  |
+--------------------+------------------+------------------------+----------------------------------+
| ``merge_asof``     | `merge_asof`_    | Y                      |                                  |
+--------------------+------------------+------------------------+----------------------------------+
| ``concat``         | `concat`_        | Y                      |                                  |
+--------------------+------------------+------------------------+----------------------------------+
| ``get_dummies``    | `get_dummies`_   | Y                      |                                  |
+--------------------+------------------+------------------------+----------------------------------+
| ``from_dummies``   | `from_dummies`_  | Y                      |                                  |
+--------------------+------------------+------------------------+----------------------------------+
| ``factorize``      | `factorize`_     | Y                      |                                  |
+--------------------+------------------+------------------------+----------------------------------+
| ``unique``         | `unique`_        | Y                      |                                  |
+--------------------+------------------+------------------------+----------------------------------+
| ``lreshape``       | `lreshape`_      | Y                      |                                  |
+--------------------+------------------+------------------------+----------------------------------+
| ``wide_to_long``   | `wide_to_long`_  | Y                      |                                  |
+--------------------+------------------+------------------------+----------------------------------+

Top-level missing data
~~~~~~~~~~~~~~~~~~~~~~

+--------------------+------------+------------------------+----------------------------------+
| ``xorbits.pandas`` | ``pandas`` | Implemented? (Y/N/P/D) | Notes for Current implementation |
+--------------------+------------+------------------------+----------------------------------+
| ``isna``           | `isna`_    | Y                      |                                  |
+--------------------+------------+------------------------+----------------------------------+
| ``isnull``         | `isnull`_  | Y                      |                                  |
+--------------------+------------+------------------------+----------------------------------+
| ``notna``          | `notna`_   | Y                      |                                  |
+--------------------+------------+------------------------+----------------------------------+
| ``notnull``        | `notnull`_ | Y                      |                                  |
+--------------------+------------+------------------------+----------------------------------+

Top-level dealing with numeric data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+--------------------+---------------+------------------------+----------------------------------+
| ``xorbits.pandas`` | ``pandas``    | Implemented? (Y/N/P/D) | Notes for Current implementation |
+--------------------+---------------+------------------------+----------------------------------+
| ``to_numeric``     | `to_numeric`_ | Y                      |                                  |
+--------------------+---------------+------------------------+----------------------------------+

Top-level dealing with datetimelike data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+---------------------+--------------------+------------------------+----------------------------------+
| ``xorbits.pandas``  | ``pandas``         | Implemented? (Y/N/P/D) | Notes for Current implementation |
+---------------------+--------------------+------------------------+----------------------------------+
| ``to_datetime``     | `to_datetime`_     | Y                      |                                  |
+---------------------+--------------------+------------------------+----------------------------------+
| ``to_timedelta``    | `to_timedelta`_    | Y                      |                                  |
+---------------------+--------------------+------------------------+----------------------------------+
| ``date_range``      | `date_range`_      | Y                      |                                  |
+---------------------+--------------------+------------------------+----------------------------------+
| ``bdate_range``     | `bdate_range`_     | Y                      |                                  |
+---------------------+--------------------+------------------------+----------------------------------+
| ``period_range``    | `period_range`_    | Y                      |                                  |
+---------------------+--------------------+------------------------+----------------------------------+
| ``timedelta_range`` | `timedelta_range`_ | Y                      |                                  |
+---------------------+--------------------+------------------------+----------------------------------+
| ``infer_freq``      | `infer_freq`_      | Y                      |                                  |
+---------------------+--------------------+------------------------+----------------------------------+

Top-level dealing with Interval data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+--------------------+-------------------+------------------------+----------------------------------+
| ``xorbits.pandas`` | ``pandas``        | Implemented? (Y/N/P/D) | Notes for Current implementation |
+--------------------+-------------------+------------------------+----------------------------------+
| ``interval_range`` | `interval_range`_ | Y                      |                                  |
+--------------------+-------------------+------------------------+----------------------------------+

Top-level evaluation
~~~~~~~~~~~~~~~~~~~~

+--------------------+------------+------------------------+----------------------------------+
| ``xorbits.pandas`` | ``pandas`` | Implemented? (Y/N/P/D) | Notes for Current implementation |
+--------------------+------------+------------------------+----------------------------------+
| ``eval``           | `eval`_    | Y                      |                                  |
+--------------------+------------+------------------------+----------------------------------+

Hashing
~~~~~~~

+-----------------------------+----------------------------+------------------------+----------------------------------+
| ``xorbits.pandas``          | ``pandas``                 | Implemented? (Y/N/P/D) | Notes for Current implementation |
+-----------------------------+----------------------------+------------------------+----------------------------------+
| ``util.hash_array``         | `util.hash_array`_         | Y                      |                                  |
+-----------------------------+----------------------------+------------------------+----------------------------------+
| ``util.hash_pandas_object`` | `util.hash_pandas_object`_ | Y                      |                                  |
+-----------------------------+----------------------------+------------------------+----------------------------------+

Importing from other DataFrame libraries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+------------------------------------+-----------------------------------+------------------------+----------------------------------+
| ``xorbits.pandas``                 | ``pandas``                        | Implemented? (Y/N/P/D) | Notes for Current implementation |
+------------------------------------+-----------------------------------+------------------------+----------------------------------+
| ``api.interchange.from_dataframe`` | `api.interchange.from_dataframe`_ | Y                      |                                  |
+------------------------------------+-----------------------------------+------------------------+----------------------------------+

.. _`GitHub repository`: https://github.com/xorbitsai/xorbits/issues
.. _`melt`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.melt.html
.. _`pivot`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.pivot.html
.. _`pivot_table`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.pivot_table.html
.. _`crosstab`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.crosstab.html
.. _`cut`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.cut.html
.. _`qcut`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.qcut.html
.. _`merge`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.merge.html
.. _`merge_ordered`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.merge_ordered.html
.. _`merge_asof`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.merge_asof.html
.. _`concat`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.concat.html
.. _`get_dummies`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_dummies.html
.. _`from_dummies`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.from_dummies.html
.. _`factorize`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.factorize.html
.. _`unique`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.unique.html
.. _`lreshape`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.lreshape.html
.. _`wide_to_long`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.wide_to_long.html
.. _`isna`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.isna.html
.. _`isnull`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.isnull.html
.. _`notna`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.notna.html
.. _`notnull`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.notnull.html
.. _`to_numeric`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_numeric.html
.. _`to_datetime`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_datetime.html
.. _`to_timedelta`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_timedelta.html
.. _`date_range`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.date_range.html
.. _`bdate_range`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.bdate_range.html
.. _`period_range`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.period_range.html
.. _`timedelta_range`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.timedelta_range.html
.. _`infer_freq`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.infer_freq.html
.. _`interval_range`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.interval_range.html
.. _`eval`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.eval.html
.. _`util.hash_array`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.util.hash_array.html
.. _`util.hash_pandas_object`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.util.hash_pandas_object.html
.. _`api.interchange.from_dataframe`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.api.interchange.from_dataframe.html

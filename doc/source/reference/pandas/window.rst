.. _api.window:

======
Window
======

The following table lists both implemented and not implemented methods. If you have need
of an operation that is listed as not implemented, feel free to open an issue on the
`GitHub repository`_, or give a thumbs up to already created issues. Contributions are
also welcome!

The following table is structured as follows: The first column contains the method name.
The second column contains link to a description of corresponding pandas method.
The third column is a flag for whether or not there is an implementation in Xorbits
for the method in the left column. ``Y`` stands for yes, ``N`` stands for no, ``P`` standsfor partial 
(meaning some parameters may not be supported yet), and ``D`` stands for default to pandas.

Rolling objects are returned by ``.rolling`` calls: :func:`xorbits.pandas.DataFrame.rolling`,
:func:`xorbits.pandas.Series.rolling`, etc. Expanding objects are returned by ``.expanding`` calls:
:func:`xorbits.pandas.DataFrame.expanding`, :func:`xorbits.pandas.Series.expanding`, etc.
ExponentialMovingWindow objects are returned by ``.ewm`` calls: :func:`xorbits.pandas.DataFrame.ewm`
, :func:`xorbits.pandas.Series.ewm`, etc.

.. _api.functions_rolling:

Rolling window functions
------------------------

+---------------------------+------------------------+------------------------+----------------------------------+
| ``xorbits.pandas.window`` | ``pandas.core.window`` | Implemented? (Y/N/P/D) | Notes for Current implementation |
+---------------------------+------------------------+------------------------+----------------------------------+
| ``Rolling.count``         | `Rolling.count`_       | Y                      |                                  |
+---------------------------+------------------------+------------------------+----------------------------------+
| ``Rolling.sum``           | `Rolling.sum`_         | Y                      |                                  |
+---------------------------+------------------------+------------------------+----------------------------------+
| ``Rolling.mean``          | `Rolling.mean`_        | Y                      |                                  |
+---------------------------+------------------------+------------------------+----------------------------------+
| ``Rolling.median``        | `Rolling.median`_      | Y                      |                                  |
+---------------------------+------------------------+------------------------+----------------------------------+
| ``Rolling.var``           | `Rolling.var`_         | Y                      |                                  |
+---------------------------+------------------------+------------------------+----------------------------------+
| ``Rolling.std``           | `Rolling.std`_         | Y                      |                                  |
+---------------------------+------------------------+------------------------+----------------------------------+
| ``Rolling.min``           | `Rolling.min`_         | Y                      |                                  |
+---------------------------+------------------------+------------------------+----------------------------------+
| ``Rolling.max``           | `Rolling.max`_         | Y                      |                                  |
+---------------------------+------------------------+------------------------+----------------------------------+
| ``Rolling.corr``          | `Rolling.corr`_        | Y                      |                                  |
+---------------------------+------------------------+------------------------+----------------------------------+
| ``Rolling.cov``           | `Rolling.cov`_         | Y                      |                                  |
+---------------------------+------------------------+------------------------+----------------------------------+
| ``Rolling.skew``          | `Rolling.skew`_        | Y                      |                                  |
+---------------------------+------------------------+------------------------+----------------------------------+
| ``Rolling.kurt``          | `Rolling.kurt`_        | Y                      |                                  |
+---------------------------+------------------------+------------------------+----------------------------------+
| ``Rolling.apply``         | `Rolling.apply`_       | Y                      |                                  |
+---------------------------+------------------------+------------------------+----------------------------------+
| ``Rolling.aggregate``     | `Rolling.aggregate`_   | Y                      |                                  |
+---------------------------+------------------------+------------------------+----------------------------------+
| ``Rolling.quantile``      | `Rolling.quantile`_    | Y                      |                                  |
+---------------------------+------------------------+------------------------+----------------------------------+
| ``Rolling.sem``           | `Rolling.sem`_         | Y                      |                                  |
+---------------------------+------------------------+------------------------+----------------------------------+
| ``Rolling.rank``          | `Rolling.rank`_        | Y                      |                                  |
+---------------------------+------------------------+------------------------+----------------------------------+

.. _api.functions_window:

Weighted window functions
-------------------------

+---------------------------+------------------------+------------------------+----------------------------------+
| ``xorbits.pandas.window`` | ``pandas.core.window`` | Implemented? (Y/N/P/D) | Notes for Current implementation |
+---------------------------+------------------------+------------------------+----------------------------------+
| ``Window.mean``           | `Window.mean`_         | Y                      |                                  |
+---------------------------+------------------------+------------------------+----------------------------------+
| ``Window.sum``            | `Window.sum`_          | Y                      |                                  |
+---------------------------+------------------------+------------------------+----------------------------------+
| ``Window.var``            | `Window.var`_          | Y                      |                                  |
+---------------------------+------------------------+------------------------+----------------------------------+
| ``Window.std``            | `Window.std`_          | Y                      |                                  |
+---------------------------+------------------------+------------------------+----------------------------------+

.. _api.functions_expanding:

Expanding window functions
--------------------------

+---------------------------+----------------------------------+------------------------+----------------------------------+
| ``xorbits.pandas.window`` | ``pandas.core.window.expanding`` | Implemented? (Y/N/P/D) | Notes for Current implementation |
+---------------------------+----------------------------------+------------------------+----------------------------------+
| ``Expanding.count``       | `Expanding.count`_               | Y                      |                                  |
+---------------------------+----------------------------------+------------------------+----------------------------------+
| ``Expanding.sum``         | `Expanding.sum`_                 | Y                      |                                  |
+---------------------------+----------------------------------+------------------------+----------------------------------+
| ``Expanding.mean``        | `Expanding.mean`_                | Y                      |                                  |
+---------------------------+----------------------------------+------------------------+----------------------------------+
| ``Expanding.median``      | `Expanding.median`_              | Y                      |                                  |
+---------------------------+----------------------------------+------------------------+----------------------------------+
| ``Expanding.var``         | `Expanding.var`_                 | Y                      |                                  |
+---------------------------+----------------------------------+------------------------+----------------------------------+
| ``Expanding.std``         | `Expanding.std`_                 | Y                      |                                  |
+---------------------------+----------------------------------+------------------------+----------------------------------+
| ``Expanding.min``         | `Expanding.min`_                 | Y                      |                                  |
+---------------------------+----------------------------------+------------------------+----------------------------------+
| ``Expanding.max``         | `Expanding.max`_                 | Y                      |                                  |
+---------------------------+----------------------------------+------------------------+----------------------------------+
| ``Expanding.corr``        | `Expanding.corr`_                | Y                      |                                  |
+---------------------------+----------------------------------+------------------------+----------------------------------+
| ``Expanding.cov``         | `Expanding.cov`_                 | Y                      |                                  |
+---------------------------+----------------------------------+------------------------+----------------------------------+
| ``Expanding.skew``        | `Expanding.skew`_                | Y                      |                                  |
+---------------------------+----------------------------------+------------------------+----------------------------------+
| ``Expanding.kurt``        | `Expanding.kurt`_                | Y                      |                                  |
+---------------------------+----------------------------------+------------------------+----------------------------------+
| ``Expanding.apply``       | `Expanding.apply`_               | Y                      |                                  |
+---------------------------+----------------------------------+------------------------+----------------------------------+
| ``Expanding.aggregate``   | `Expanding.aggregate`_           | Y                      |                                  |
+---------------------------+----------------------------------+------------------------+----------------------------------+
| ``Expanding.quantile``    | `Expanding.quantile`_            | Y                      |                                  |
+---------------------------+----------------------------------+------------------------+----------------------------------+
| ``Expanding.sem``         | `Expanding.sem`_                 | Y                      |                                  |
+---------------------------+----------------------------------+------------------------+----------------------------------+
| ``Expanding.rank``        | `Expanding.rank`_                | Y                      |                                  |
+---------------------------+----------------------------------+------------------------+----------------------------------+

.. _api.functions_ewm:

Exponentially-weighted window functions
---------------------------------------
.. currentmodule:: xorbits.pandas.window

+----------------------------------+---------------------------------+------------------------+----------------------------------+
| ``xorbits.pandas.window``        | ``pandas.core.window.ewm``      | Implemented? (Y/N/P/D) | Notes for Current implementation |
+----------------------------------+---------------------------------+------------------------+----------------------------------+
| ``ExponentialMovingWindow.mean`` | `ExponentialMovingWindow.mean`_ | Y                      |                                  |
+----------------------------------+---------------------------------+------------------------+----------------------------------+
| ``ExponentialMovingWindow.sum``  | `ExponentialMovingWindow.sum`_  | Y                      |                                  |
+----------------------------------+---------------------------------+------------------------+----------------------------------+
| ``ExponentialMovingWindow.std``  | `ExponentialMovingWindow.std`_  | Y                      |                                  |
+----------------------------------+---------------------------------+------------------------+----------------------------------+
| ``ExponentialMovingWindow.var``  | `ExponentialMovingWindow.var`_  | Y                      |                                  |
+----------------------------------+---------------------------------+------------------------+----------------------------------+
| ``ExponentialMovingWindow.corr`` | `ExponentialMovingWindow.corr`_ | Y                      |                                  |
+----------------------------------+---------------------------------+------------------------+----------------------------------+
| ``ExponentialMovingWindow.cov``  | `ExponentialMovingWindow.cov`_  | Y                      |                                  |
+----------------------------------+---------------------------------+------------------------+----------------------------------+

.. _api.indexers_window:

Window indexer
--------------

Base class for defining custom window boundaries.

+----------------------------------------------+---------------------------------------------+------------------------+----------------------------------+
| ``xorbits.pandas``                           | ``pandas``                                  | Implemented? (Y/N/P/D) | Notes for Current implementation |
+----------------------------------------------+---------------------------------------------+------------------------+----------------------------------+
| ``api.indexers.BaseIndexer``                 | `api.indexers.BaseIndexer`_                 | Y                      |                                  |
+----------------------------------------------+---------------------------------------------+------------------------+----------------------------------+
| ``api.indexers.FixedForwardWindowIndexer``   | `api.indexers.FixedForwardWindowIndexer`_   | Y                      |                                  |
+----------------------------------------------+---------------------------------------------+------------------------+----------------------------------+
| ``api.indexers.VariableOffsetWindowIndexer`` | `api.indexers.VariableOffsetWindowIndexer`_ | Y                      |                                  |
+----------------------------------------------+---------------------------------------------+------------------------+----------------------------------+

.. _`GitHub repository`: https://github.com/xorbitsai/xorbits/issues
.. _`Rolling.count`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.window.Rolling.count.html
.. _`Rolling.sum`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.window.Rolling.sum.html
.. _`Rolling.mean`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.window.Rolling.mean.html
.. _`Rolling.median`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.window.Rolling.median.html
.. _`Rolling.var`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.window.Rolling.var.html
.. _`Rolling.std`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.window.Rolling.std.html
.. _`Rolling.min`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.window.Rolling.min.html
.. _`Rolling.max`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.window.Rolling.max.html
.. _`Rolling.corr`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.window.Rolling.corr.html
.. _`Rolling.cov`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.window.Rolling.cov.html
.. _`Rolling.skew`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.window.Rolling.skew.html
.. _`Rolling.kurt`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.window.Rolling.kurt.html
.. _`Rolling.apply`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.window.Rolling.apply.html
.. _`Rolling.aggregate`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.window.Rolling.aggregate.html
.. _`Rolling.quantile`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.window.Rolling.quantile.html
.. _`Rolling.sem`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.window.Rolling.sem.html
.. _`Rolling.rank`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.window.Rolling.rank.html
.. _`Window.mean`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Window.mean.html
.. _`Window.sum`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Window.sum.html
.. _`Window.var`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Window.var.html
.. _`Window.std`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Window.std.html
.. _`Expanding.count`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.window.expanding.Expanding.count.html
.. _`Expanding.sum`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.window.expanding.Expanding.sum.html
.. _`Expanding.mean`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.window.expanding.Expanding.mean.html
.. _`Expanding.median`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.window.expanding.Expanding.median.html
.. _`Expanding.var`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.window.expanding.Expanding.var.html
.. _`Expanding.std`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.window.expanding.Expanding.std.html
.. _`Expanding.min`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.window.expanding.Expanding.min.html
.. _`Expanding.max`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.window.expanding.Expanding.max.html
.. _`Expanding.corr`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.window.expanding.Expanding.corr.html
.. _`Expanding.cov`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.window.expanding.Expanding.cov.html
.. _`Expanding.skew`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.window.expanding.Expanding.skew.html
.. _`Expanding.kurt`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.window.expanding.Expanding.kurt.html
.. _`Expanding.apply`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.window.expanding.Expanding.apply.html
.. _`Expanding.aggregate`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.window.expanding.Expanding.aggregate.html
.. _`Expanding.quantile`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.window.expanding.Expanding.quantile.html
.. _`Expanding.sem`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.window.expanding.Expanding.sem.html
.. _`Expanding.rank`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.window.expanding.Expanding.rank.html
.. _`ExponentialMovingWindow.mean`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.window.ewm.ExponentialMovingWindow.mean.html
.. _`ExponentialMovingWindow.sum`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.window.ewm.ExponentialMovingWindow.sum.html
.. _`ExponentialMovingWindow.std`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.window.ewm.ExponentialMovingWindow.std.html
.. _`ExponentialMovingWindow.var`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.window.ewm.ExponentialMovingWindow.var.html
.. _`ExponentialMovingWindow.corr`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.window.ewm.ExponentialMovingWindow.corr.html
.. _`ExponentialMovingWindow.cov`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.window.ewm.ExponentialMovingWindow.cov.html
.. _`api.indexers.BaseIndexer`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.api.indexers.BaseIndexer.html
.. _`api.indexers.FixedForwardWindowIndexer`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.api.indexers.FixedForwardWindowIndexer.html
.. _`api.indexers.VariableOffsetWindowIndexer`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.api.indexers.VariableOffsetWindowIndexer.html

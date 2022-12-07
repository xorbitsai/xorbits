.. _api.window:

======
Window
======

Rolling objects are returned by ``.rolling`` calls: :func:`xorbits.pandas.DataFrame.rolling`, :func:`xorbits.pandas.Series.rolling`, etc.
Expanding objects are returned by ``.expanding`` calls: :func:`xorbits.pandas.DataFrame.expanding`, :func:`xorbits.pandas.Series.expanding`, etc.
ExponentialMovingWindow objects are returned by ``.ewm`` calls: :func:`xorbits.pandas.DataFrame.ewm`, :func:`xorbits.pandas.Series.ewm`, etc.

.. _api.functions_rolling:

Rolling window functions
------------------------
.. currentmodule:: xorbits.pandas.window

.. autosummary::
   :toctree: api/

   Rolling.count
   Rolling.sum
   Rolling.mean
   Rolling.median
   Rolling.var
   Rolling.std
   Rolling.min
   Rolling.max
   Rolling.corr
   Rolling.cov
   Rolling.skew
   Rolling.kurt
   Rolling.apply
   Rolling.aggregate
   Rolling.quantile
   Rolling.sem
   Rolling.rank

.. _api.functions_expanding:

Expanding window functions
--------------------------
.. currentmodule:: xorbits.pandas.window

.. autosummary::
   :toctree: api/

   Expanding.count
   Expanding.sum
   Expanding.mean
   Expanding.median
   Expanding.var
   Expanding.std
   Expanding.min
   Expanding.max
   Expanding.corr
   Expanding.cov
   Expanding.skew
   Expanding.kurt
   Expanding.apply
   Expanding.aggregate
   Expanding.quantile
   Expanding.sem
   Expanding.rank

.. _api.functions_ewm:

Exponentially-weighted window functions
---------------------------------------
.. currentmodule:: xorbits.pandas.window

.. autosummary::
   :toctree: api/

   ExponentialMovingWindow.mean
   ExponentialMovingWindow.sum
   ExponentialMovingWindow.std
   ExponentialMovingWindow.var
   ExponentialMovingWindow.corr
   ExponentialMovingWindow.cov

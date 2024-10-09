.. _api.xgboost_sklearn:

================
Scikit-Learn API
================

The following table lists both implemented and not implemented methods. If you have need
of an operation that is listed as not implemented, feel free to open an issue on the
`GitHub repository`_, or give a thumbs up to already created issues. Contributions are
also welcome!

The following table is structured as follows: The first column contains the method name.
The second column contains link to a description of corresponding xgboost method.
The third column is a flag for whether or not there is an implementation in Xorbits
for the method in the left column. ``Y`` stands for yes, ``N`` stands for no, ``P`` standsfor partial 
(meaning some parameters may not be supported yet), and ``D`` stands for default to xgboost.

XGBRegressor
============

+------------------------------------------+-----------------------------------------+------------------------+----------------------------------+
| ``xorbits.xgboost``                      | ``xgboost``                             | Implemented? (Y/N/P/D) | Notes for Current implementation |
+------------------------------------------+-----------------------------------------+------------------------+----------------------------------+
| ``XGBRegressor.apply``                   | `XGBRegressor.apply`_                   | Y                      |                                  |
+------------------------------------------+-----------------------------------------+------------------------+----------------------------------+
| ``XGBRegressor.evals_result``            | `XGBRegressor.evals_result`_            | Y                      |                                  |
+------------------------------------------+-----------------------------------------+------------------------+----------------------------------+
| ``XGBRegressor.fit``                     | `XGBRegressor.fit`_                     | Y                      |                                  |
+------------------------------------------+-----------------------------------------+------------------------+----------------------------------+
| ``XGBRegressor.get_booster``             | `XGBRegressor.get_booster`_             | Y                      |                                  |
+------------------------------------------+-----------------------------------------+------------------------+----------------------------------+
| ``XGBRegressor.get_num_boosting_rounds`` | `XGBRegressor.get_num_boosting_rounds`_ | Y                      |                                  |
+------------------------------------------+-----------------------------------------+------------------------+----------------------------------+
| ``XGBRegressor.get_params``              | `XGBRegressor.get_params`_              | Y                      |                                  |
+------------------------------------------+-----------------------------------------+------------------------+----------------------------------+
| ``XGBRegressor.get_xgb_params``          | `XGBRegressor.get_xgb_params`_          | Y                      |                                  |
+------------------------------------------+-----------------------------------------+------------------------+----------------------------------+
| ``XGBRegressor.load_model``              | `XGBRegressor.load_model`_              | Y                      |                                  |
+------------------------------------------+-----------------------------------------+------------------------+----------------------------------+
| ``XGBRegressor.predict``                 | `XGBRegressor.predict`_                 | Y                      |                                  |
+------------------------------------------+-----------------------------------------+------------------------+----------------------------------+
| ``XGBRegressor.save_model``              | `XGBRegressor.save_model`_              | Y                      |                                  |
+------------------------------------------+-----------------------------------------+------------------------+----------------------------------+
| ``XGBRegressor.set_params``              | `XGBRegressor.set_params`_              | Y                      |                                  |
+------------------------------------------+-----------------------------------------+------------------------+----------------------------------+

XGBClassifier
=============

+-------------------------------------------+------------------------------------------+------------------------+----------------------------------+
| ``xorbits.xgboost``                       | ``xgboost``                              | Implemented? (Y/N/P/D) | Notes for Current implementation |
+-------------------------------------------+------------------------------------------+------------------------+----------------------------------+
| ``XGBClassifier.apply``                   | `XGBClassifier.apply`_                   | Y                      |                                  |
+-------------------------------------------+------------------------------------------+------------------------+----------------------------------+
| ``XGBClassifier.evals_result``            | `XGBClassifier.evals_result`_            | Y                      |                                  |
+-------------------------------------------+------------------------------------------+------------------------+----------------------------------+
| ``XGBClassifier.fit``                     | `XGBClassifier.fit`_                     | Y                      |                                  |
+-------------------------------------------+------------------------------------------+------------------------+----------------------------------+
| ``XGBClassifier.get_booster``             | `XGBClassifier.get_booster`_             | Y                      |                                  |
+-------------------------------------------+------------------------------------------+------------------------+----------------------------------+
| ``XGBClassifier.get_num_boosting_rounds`` | `XGBClassifier.get_num_boosting_rounds`_ | Y                      |                                  |
+-------------------------------------------+------------------------------------------+------------------------+----------------------------------+
| ``XGBClassifier.get_params``              | `XGBClassifier.get_params`_              | Y                      |                                  |
+-------------------------------------------+------------------------------------------+------------------------+----------------------------------+
| ``XGBClassifier.get_xgb_params``          | `XGBClassifier.get_xgb_params`_          | Y                      |                                  |
+-------------------------------------------+------------------------------------------+------------------------+----------------------------------+
| ``XGBClassifier.load_model``              | `XGBClassifier.load_model`_              | Y                      |                                  |
+-------------------------------------------+------------------------------------------+------------------------+----------------------------------+
| ``XGBClassifier.predict``                 | `XGBClassifier.predict`_                 | Y                      |                                  |
+-------------------------------------------+------------------------------------------+------------------------+----------------------------------+
| ``XGBClassifier.predict_proba``           | `XGBClassifier.predict_proba`_           | Y                      |                                  |
+-------------------------------------------+------------------------------------------+------------------------+----------------------------------+
| ``XGBClassifier.save_model``              | `XGBClassifier.save_model`_              | Y                      |                                  |
+-------------------------------------------+------------------------------------------+------------------------+----------------------------------+
| ``XGBClassifier.score``                   | `XGBClassifier.score`_                   | Y                      |                                  |
+-------------------------------------------+------------------------------------------+------------------------+----------------------------------+
| ``XGBClassifier.set_params``              | `XGBClassifier.set_params`_              | Y                      |                                  |
+-------------------------------------------+------------------------------------------+------------------------+----------------------------------+

.. _`GitHub repository`: https://github.com/xorbitsai/xorbits/issues
.. _`XGBRegressor.apply`: https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRegressor.apply
.. _`XGBRegressor.evals_result`: https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRegressor.evals_result
.. _`XGBRegressor.fit`: https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRegressor.fit
.. _`XGBRegressor.get_booster`: https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRegressor.get_booster
.. _`XGBRegressor.get_num_boosting_rounds`: https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRegressor.get_num_boosting_rounds
.. _`XGBRegressor.get_params`: https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRegressor.get_params
.. _`XGBRegressor.get_xgb_params`: https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRegressor.get_xgb_params
.. _`XGBRegressor.load_model`: https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRegressor.load_model
.. _`XGBRegressor.predict`: https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRegressor.predict
.. _`XGBRegressor.save_model`: https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRegressor.save_model
.. _`XGBRegressor.set_params`: https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRegressor.set_params
.. _`XGBClassifier.apply`: https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier.apply
.. _`XGBClassifier.evals_result`: https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier.evals_result
.. _`XGBClassifier.fit`: https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier.fit
.. _`XGBClassifier.get_booster`: https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier.get_booster
.. _`XGBClassifier.get_num_boosting_rounds`: https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier.get_num_boosting_rounds
.. _`XGBClassifier.get_params`: https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier.get_params
.. _`XGBClassifier.get_xgb_params`: https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier.get_xgb_params
.. _`XGBClassifier.load_model`: https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier.load_model
.. _`XGBClassifier.predict`: https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier.predict
.. _`XGBClassifier.predict_proba`: https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier.predict_proba
.. _`XGBClassifier.save_model`: https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier.save_model
.. _`XGBClassifier.score`: https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier.score
.. _`XGBClassifier.set_params`: https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier.set_params

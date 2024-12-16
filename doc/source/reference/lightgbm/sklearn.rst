.. _api.lightgbm_sklearn:

================
Scikit-Learn API
================

The following table lists both implemented and not implemented methods. If you have need
of an operation that is listed as not implemented, feel free to open an issue on the
`GitHub repository`_, or give a thumbs up to already created issues. Contributions are
also welcome!

The following table is structured as follows: The first column contains the method name.
The second column contains link to a description of corresponding lightgbm method.
The third column is a flag for whether or not there is an implementation in Xorbits for
the method in the left column. ``Y`` stands for yes, ``N`` stands for no, ``P`` stands
for partial (meaning some parameters may not be supported yet), and ``D`` stands for
default to lightgbm.

LGBMClassifier
==============

+----------------------------------------------+---------------------------------------------+------------------------+----------------------------------+
| ``xorbits.lightgbm``                         | ``lightgbm``                                | Implemented? (Y/N/P/D) | Notes for Current implementation |
+----------------------------------------------+---------------------------------------------+------------------------+----------------------------------+
| ``LGBMClassifier.fit``                       | `LGBMClassifier.fit`_                       | Y                      |                                  |
+----------------------------------------------+---------------------------------------------+------------------------+----------------------------------+
| ``LGBMClassifier.get_metadata_routing``      | `LGBMClassifier.get_metadata_routing`_      | Y                      |                                  |
+----------------------------------------------+---------------------------------------------+------------------------+----------------------------------+
| ``LGBMClassifier.get_params``                | `LGBMClassifier.get_params`_                | Y                      |                                  |
+----------------------------------------------+---------------------------------------------+------------------------+----------------------------------+
| ``LGBMClassifier.load_model``                | `LGBMClassifier.load_model`_                | Y                      |                                  |
+----------------------------------------------+---------------------------------------------+------------------------+----------------------------------+
| ``LGBMClassifier.predict``                   | `LGBMClassifier.predict`_                   | Y                      |                                  |
+----------------------------------------------+---------------------------------------------+------------------------+----------------------------------+
| ``LGBMClassifier.predict_proba``             | `LGBMClassifier.predict_proba`_             | Y                      |                                  |
+----------------------------------------------+---------------------------------------------+------------------------+----------------------------------+
| ``LGBMClassifier.score``                     | `LGBMClassifier.score`_                     | Y                      |                                  |
+----------------------------------------------+---------------------------------------------+------------------------+----------------------------------+
| ``LGBMClassifier.set_fit_request``           | `LGBMClassifier.set_fit_request`_           | Y                      |                                  |
+----------------------------------------------+---------------------------------------------+------------------------+----------------------------------+
| ``LGBMClassifier.set_params``                | `LGBMClassifier.set_params`_                | Y                      |                                  |
+----------------------------------------------+---------------------------------------------+------------------------+----------------------------------+
| ``LGBMClassifier.set_predict_proba_request`` | `LGBMClassifier.set_predict_proba_request`_ | Y                      |                                  |
+----------------------------------------------+---------------------------------------------+------------------------+----------------------------------+
| ``LGBMClassifier.set_predict_request``       | `LGBMClassifier.set_predict_request`_       | Y                      |                                  |
+----------------------------------------------+---------------------------------------------+------------------------+----------------------------------+
| ``LGBMClassifier.set_score_request``         | `LGBMClassifier.set_score_request`_         | Y                      |                                  |
+----------------------------------------------+---------------------------------------------+------------------------+----------------------------------+
| ``LGBMClassifier.to_local``                  | `LGBMClassifier.to_local`_                  | Y                      |                                  |
+----------------------------------------------+---------------------------------------------+------------------------+----------------------------------+

LGBMRegressor
=============

+----------------------------------------+---------------------------------------+------------------------+----------------------------------+
| ``xorbits.lightgbm``                   | ``lightgbm``                          | Implemented? (Y/N/P/D) | Notes for Current implementation |
+----------------------------------------+---------------------------------------+------------------------+----------------------------------+
| ``LGBMRegressor.fit``                  | `LGBMRegressor.fit`_                  | Y                      |                                  |
+----------------------------------------+---------------------------------------+------------------------+----------------------------------+
| ``LGBMRegressor.get_metadata_routing`` | `LGBMRegressor.get_metadata_routing`_ | Y                      |                                  |
+----------------------------------------+---------------------------------------+------------------------+----------------------------------+
| ``LGBMRegressor.get_params``           | `LGBMRegressor.get_params`_           | Y                      |                                  |
+----------------------------------------+---------------------------------------+------------------------+----------------------------------+
| ``LGBMRegressor.load_model``           | `LGBMRegressor.load_model`_           | Y                      |                                  |
+----------------------------------------+---------------------------------------+------------------------+----------------------------------+
| ``LGBMRegressor.predict``              | `LGBMRegressor.predict`_              | Y                      |                                  |
+----------------------------------------+---------------------------------------+------------------------+----------------------------------+
| ``LGBMRegressor.predict_proba``        | `LGBMRegressor.predict_proba`_        | Y                      |                                  |
+----------------------------------------+---------------------------------------+------------------------+----------------------------------+
| ``LGBMRegressor.score``                | `LGBMRegressor.score`_                | Y                      |                                  |
+----------------------------------------+---------------------------------------+------------------------+----------------------------------+
| ``LGBMRegressor.set_fit_request``      | `LGBMRegressor.set_fit_request`_      | Y                      |                                  |
+----------------------------------------+---------------------------------------+------------------------+----------------------------------+
| ``LGBMRegressor.set_params``           | `LGBMRegressor.set_params`_           | Y                      |                                  |
+----------------------------------------+---------------------------------------+------------------------+----------------------------------+
| ``LGBMRegressor.set_predict_request``  | `LGBMRegressor.set_predict_request`_  | Y                      |                                  |
+----------------------------------------+---------------------------------------+------------------------+----------------------------------+
| ``LGBMRegressor.set_score_request``    | `LGBMRegressor.set_score_request`_    | Y                      |                                  |
+----------------------------------------+---------------------------------------+------------------------+----------------------------------+
| ``LGBMRegressor.to_local``             | `LGBMRegressor.to_local`_             | Y                      |                                  |
+----------------------------------------+---------------------------------------+------------------------+----------------------------------+

LGBMRanker
==========

+-------------------------------------+------------------------------------+------------------------+----------------------------------+
| ``xorbits.lightgbm``                | ``lightgbm``                       | Implemented? (Y/N/P/D) | Notes for Current implementation |
+-------------------------------------+------------------------------------+------------------------+----------------------------------+
| ``LGBMRanker.fit``                  | `LGBMRanker.fit`_                  | Y                      |                                  |
+-------------------------------------+------------------------------------+------------------------+----------------------------------+
| ``LGBMRanker.get_metadata_routing`` | `LGBMRanker.get_metadata_routing`_ | Y                      |                                  |
+-------------------------------------+------------------------------------+------------------------+----------------------------------+
| ``LGBMRanker.get_params``           | `LGBMRanker.get_params`_           | Y                      |                                  |
+-------------------------------------+------------------------------------+------------------------+----------------------------------+
| ``LGBMRanker.load_model``           | `LGBMRanker.load_model`_           | Y                      |                                  |
+-------------------------------------+------------------------------------+------------------------+----------------------------------+
| ``LGBMRanker.predict``              | `LGBMRanker.predict`_              | Y                      |                                  |
+-------------------------------------+------------------------------------+------------------------+----------------------------------+
| ``LGBMRanker.predict_proba``        | `LGBMRanker.predict_proba`_        | Y                      |                                  |
+-------------------------------------+------------------------------------+------------------------+----------------------------------+
| ``LGBMRanker.set_fit_request``      | `LGBMRanker.set_fit_request`_      | Y                      |                                  |
+-------------------------------------+------------------------------------+------------------------+----------------------------------+
| ``LGBMRanker.set_params``           | `LGBMRanker.set_params`_           | Y                      |                                  |
+-------------------------------------+------------------------------------+------------------------+----------------------------------+
| ``LGBMRanker.set_predict_request``  | `LGBMRanker.set_predict_request`_  | Y                      |                                  |
+-------------------------------------+------------------------------------+------------------------+----------------------------------+
| ``LGBMRanker.to_local``             | `LGBMRanker.to_local`_             | Y                      |                                  |
+-------------------------------------+------------------------------------+------------------------+----------------------------------+

.. _`GitHub repository`: https://github.com/xorbitsai/xorbits/issues
.. _`LGBMClassifier.fit`: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html#lightgbm.LGBMClassifier.fit
.. _`LGBMClassifier.get_metadata_routing`: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html#lightgbm.LGBMClassifier.get_metadata_routing
.. _`LGBMClassifier.get_params`: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html#lightgbm.LGBMClassifier.get_params
.. _`LGBMClassifier.load_model`: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html#lightgbm.LGBMClassifier.load_model
.. _`LGBMClassifier.predict`: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html#lightgbm.LGBMClassifier.predict
.. _`LGBMClassifier.predict_proba`: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html#lightgbm.LGBMClassifier.predict_proba
.. _`LGBMClassifier.score`: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html#lightgbm.LGBMClassifier.score
.. _`LGBMClassifier.set_fit_request`: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html#lightgbm.LGBMClassifier.set_fit_request
.. _`LGBMClassifier.set_params`: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html#lightgbm.LGBMClassifier.set_params
.. _`LGBMClassifier.set_predict_proba_request`: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html#lightgbm.LGBMClassifier.set_predict_proba_request
.. _`LGBMClassifier.set_predict_request`: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html#lightgbm.LGBMClassifier.set_predict_request
.. _`LGBMClassifier.set_score_request`: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html#lightgbm.LGBMClassifier.set_score_request
.. _`LGBMClassifier.to_local`: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html#lightgbm.LGBMClassifier.to_local
.. _`LGBMRegressor.fit`: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html#lightgbm.LGBMRegressor.fit
.. _`LGBMRegressor.get_params`: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html#lightgbm.LGBMRegressor.get_params
.. _`LGBMRegressor.get_metadata_routing`: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html#lightgbm.LGBMRegressor.get_metadata_routing
.. _`LGBMRegressor.load_model`: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html#lightgbm.LGBMRegressor.load_model
.. _`LGBMRegressor.predict`: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html#lightgbm.LGBMRegressor.predict
.. _`LGBMRegressor.predict_proba`: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html#lightgbm.LGBMRegressor.predict_proba
.. _`LGBMRegressor.score`: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html#lightgbm.LGBMRegressor.score
.. _`LGBMRegressor.set_fit_request`: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html#lightgbm.LGBMRegressor.set_fit_request
.. _`LGBMRegressor.set_params`: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html#lightgbm.LGBMRegressor.set_params
.. _`LGBMRegressor.set_predict_request`: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html#lightgbm.LGBMRegressor.set_predict_request
.. _`LGBMRegressor.set_score_request`: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html#lightgbm.LGBMRegressor.set_score_request
.. _`LGBMRegressor.to_local`: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html#lightgbm.LGBMRegressor.to_local
.. _`LGBMRanker.fit`: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRanker.html#lightgbm.LGBMRanker.fit
.. _`LGBMRanker.get_metadata_routing`: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRanker.html#lightgbm.LGBMRanker.get_metadata_routing
.. _`LGBMRanker.get_params`: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRanker.html#lightgbm.LGBMRanker.get_params
.. _`LGBMRanker.load_model`: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRanker.html#lightgbm.LGBMRanker.load_model
.. _`LGBMRanker.predict`: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRanker.html#lightgbm.LGBMRanker.predict
.. _`LGBMRanker.predict_proba`: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRanker.html#lightgbm.LGBMRanker.predict_proba
.. _`LGBMRanker.set_fit_request`: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRanker.html#lightgbm.LGBMRanker.set_fit_request
.. _`LGBMRanker.set_params`: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRanker.html#lightgbm.LGBMRanker.set_params
.. _`LGBMRanker.set_predict_request`: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRanker.html#lightgbm.LGBMRanker.set_predict_request
.. _`LGBMRanker.to_local`: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRanker.html#lightgbm.LGBMRanker.to_local

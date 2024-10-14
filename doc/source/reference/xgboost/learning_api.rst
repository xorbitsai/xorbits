.. _api.xgboost_learning_api:

============
Learning API
============

The following table lists both implemented and not implemented methods. If you have need
of an operation that is listed as not implemented, feel free to open an issue on the
`GitHub repository`_, or give a thumbs up to already created issues. Contributions are
also welcome!

The following table is structured as follows: The first column contains the method name.
The second column contains link to a description of corresponding xgboost method.
The third column is a flag for whether or not there is an implementation in Xorbits
for the method in the left column. ``Y`` stands for yes, ``N`` stands for no, ``P`` standsfor partial 
(meaning some parameters may not be supported yet), and ``D`` stands for default to xgboost.

+---------------------+-------------+------------------------+----------------------------------+
| ``xorbits.xgboost`` | ``xgboost`` | Implemented? (Y/N/P/D) | Notes for Current implementation |
+---------------------+-------------+------------------------+----------------------------------+
| ``train``           | `train`_    | Y                      |                                  |
+---------------------+-------------+------------------------+----------------------------------+

.. Originally, `predict` was included,
.. but it is not consistent with the interface of xgboost.

.. _`GitHub repository`: https://github.com/xorbitsai/xorbits/issues
.. _`train`: https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.train

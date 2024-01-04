.. _10min_lightgbm:

======================================
10 minutes to :code:`xorbits.lightgbm`
======================================

.. currentmodule:: xorbits.lightgbm

This is a short introduction to :code:`xorbits.lightgbm` which is originated from LightGBM's quickstart.

Let's take :code:`LGBMRegressor` as an example and explain how to build a regression model, find the relationship between independent variables (features) and the dependent variable (target), and make predictions based on these relationships.

Customarily, we import and init as follows:

::

    >>> import xorbits
    >>> import xorbits.numpy as np
    >>> from xorbits.lightgbm import LGBMRegressor
    >>> from xorbits.sklearn.model_selection import train_test_split
    >>> xorbits.init()


Model Creation
--------------
First, we build a :code:`LGBMRegressor` model and define its parameters.

This model has many adjustable hyperparameters that allow you to configure parameters such as tree depth, the number of leaf nodes, learning rate, and more to optimize the model's performance.

::

    >>> lgbm_regressor = LGBMRegressor(learning_rate=0.05, n_estimators=100)


:code:`.get_params` method returns a dictionary containing all the parameter names of the model along with their corresponding values. You can inspect these values to understand the current configuration of the model.

Inspect the parameters of the LightGBM regressor.

::

    >>> paras=lgbm_regressor.get_params()
    >>> paras
    {'boosting_type': 'gbdt', 'class_weight': None, 'colsample_bytree': 1.0, 
    'importance_type': 'split', 'learning_rate': 0.05, 'max_depth': -1, 
    'min_child_samples': 20, 'min_child_weight': 0.001, 'min_split_gain': 0.0, 
    'n_estimators': 100, 'n_jobs': None, 'num_leaves': 31, 'objective': None, 
    'random_state': None, 'reg_alpha': 0.0, 'reg_lambda': 0.0, 'subsample': 1.0, 
    'subsample_for_bin': 200000, 'subsample_freq': 0}


Set/modify parameters.

:code:`.set_params` method allows you to dynamically modify the parameter settings of a machine learning model by specifying parameter names and their corresponding values, without the need to recreate the model object.

::

    >>> lgbm_regressor.set_params(learning_rate=0.1, n_estimators=100)
    >>> lgbm_regressor.get_params()
    {'boosting_type': 'gbdt', 'class_weight': None, 'colsample_bytree': 1.0, 
    'importance_type': 'split', 'learning_rate': 0.1, 'max_depth': -1, 
    'min_child_samples': 20, 'min_child_weight': 0.001, 'min_split_gain': 0.0, 
    'n_estimators': 100, 'n_jobs': None, 'num_leaves': 31, 'objective': None, 
    'random_state': None, 'reg_alpha': 0.0, 'reg_lambda': 0.0, 'subsample': 1.0, 
    'subsample_for_bin': 200000, 'subsample_freq': 0}


Data Preparation
----------------
We can use real data as input. For the sake of simplicity, we will use randomly generated x and y data as an example.

::

    >>> x = np.random.rand(100)
    >>> y_regression = 2 * x + 1 + 0.1 * np.random.randn(100)
    >>> x=x.reshape(-1, 1)


In order to train the model, we split the dataset into a training set and a test set.

::

    >>> X_train, X_test, y_train, y_test = train_test_split(x, y_regression, test_size=0.2)


Model Training
--------------
The :code:`.fit` method takes the training data (independent variable x and dependent variable y) and fits the model to the data.

The model adjusts its parameters to minimize the error between the predicted values and the actual observations.

::

    >>> lgbm_regressor.fit(X_train, y_train)


Model Prediction
----------------

Once you have trained a model, you can use the :code:`.predict` method to apply that model to new data and generate predictions for the new data.

::

    >>> y_pred = lgbm_regressor.predict(X_test)
    >>> y_pred
    array([1.81201326, 2.64419095, 1.56956112, 1.56956112, 1.22159593,
       2.64419095, 2.64419095, 2.11885373, 1.81201326, 1.22159593,
       2.64419095, 1.22159593, 1.56956112, 2.64419095, 2.64419095,
       2.64419095, 1.90073406, 2.64419095, 1.81201326, 1.81201326])


Model Evaluation
----------------

:code:`.score` is typically used to assess the performance of a machine learning model.

In regression problems, the :code:`.score` method usually returns the coefficient of determination (R-squared) score, which represents the model's ability to explain the variability in the dependent variable.

Calculate the model's estimated accuracy on the test set.

::

    >>> accuracy = lgbm_regressor.score(X_test, y_test)
    >>> accuracy
    0.9323625100248668

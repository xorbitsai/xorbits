.. _10min_xgboost:

=====================================
10 minutes to :code:`xorbits.xgboost`
=====================================

.. currentmodule:: xorbits.xgboost

This is a short introduction to :code:`xorbits.xgboost` which is originated from XGBoost's quickstart.
This quick start tutorial shows snippets for you to quickly try out XGBoost on the demo dataset on a binary classification task.

Customarily, we import and init as follows:

.. ipython:: python

    import xorbits.pandas as pd
    import xorbits.datasets as xdatasets
    from xorbits.xgboost import XGBClassifier
    from xorbits.sklearn.model_selection import train_test_split

Model Creation
--------------
First, we build a :code:`XGBClassifier` model and define its parameters.

This model implements the Gradient Boosting Decision Tree algorithm, improving model performance by training multiple decision trees. Gradient Boosting Trees is an ensemble learning method that builds a powerful model by combining multiple weak learners, typically decision trees.

.. ipython:: python

    bst = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic')

Data Preparation
----------------
We import the scikit-learn/iris dataset from Hugging Face as the input data for our model.

.. ipython:: python

    dataset = xdatasets.from_huggingface("scikit-learn/iris", split="train")
    iris_df = dataset.to_dataframe()

    iris_df['Species'] = iris_df['Species'].replace({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})
    X = iris_df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
    y = iris_df['Species']

In order to train the model, we split the dataset into a training set and a test set.

.. ipython:: python

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

Model Training
--------------
The :code:`.fit` method takes the training data (independent variable x and dependent variable y) and fits the model to the data.

The model adjusts its parameters to minimize the error between the predicted values and the actual observations.

.. ipython:: python

    bst.fit(X_train, y_train)

Model Prediction
----------------
Once you have trained a model, you can use the :code:`.predict` method to apply that model to new data and generate predictions for the new data.

.. ipython:: python

    preds = bst.predict(X_test)
    preds

Model Evaluation
----------------
:code:`.score` is typically used to assess the performance of a machine learning model.

In regression problems, the :code:`.score` method usually returns the coefficient of determination (R-squared) score, which represents the model's ability to explain the variability in the dependent variable.

Calculate the model's estimated accuracy on the test set.

.. ipython:: python

    accuracy = bst.score(X_test, y_test)
    accuracy

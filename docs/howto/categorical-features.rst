======================
Categorical Features
======================

This guide explains how to handle categorical features with boosters.

Overview
--------

boosters supports categorical features through encoding. The recommended 
approaches depend on your use case:

+---------------------+--------------------------------------------+
| Approach            | When to Use                                |
+=====================+============================================+
| Label encoding      | Ordinal categories, tree-based models      |
+---------------------+--------------------------------------------+
| One-hot encoding    | Few categories, linear models              |
+---------------------+--------------------------------------------+
| Target encoding     | High cardinality, when leakage is managed  |
+---------------------+--------------------------------------------+

Label Encoding
--------------

Simple and effective for GBDT:

.. code-block:: python

   from sklearn.preprocessing import LabelEncoder
   from boosters.sklearn import GBDTClassifier

   # Encode string categories to integers
   encoder = LabelEncoder()
   X[:, 0] = encoder.fit_transform(X[:, 0])

   model = GBDTClassifier()
   model.fit(X, y)

.. note::

   GBDT handles label-encoded categories well because trees can learn 
   arbitrary splits on the encoded values. The ordering doesn't matter.

One-Hot Encoding
----------------

For GBLinear or when you have few categories:

.. code-block:: python

   from sklearn.preprocessing import OneHotEncoder
   from boosters.sklearn import GBLinearRegressor

   # One-hot encode
   encoder = OneHotEncoder(sparse_output=False, drop='first')
   X_encoded = encoder.fit_transform(X_categorical)

   model = GBLinearRegressor()
   model.fit(X_encoded, y)

⚠️ **Warning**: One-hot encoding creates one feature per category level. 
For high-cardinality features (many unique values), this is inefficient.

High-Cardinality Categories
---------------------------

For features with many unique values (e.g., user IDs, zip codes), use 
target encoding:

.. code-block:: python

   from sklearn.preprocessing import TargetEncoder
   from boosters.sklearn import GBDTRegressor

   encoder = TargetEncoder()
   X_encoded = encoder.fit_transform(X, y)

   model = GBDTRegressor()
   model.fit(X_encoded, y)

Target encoding replaces each category with the mean target value for that 
category. Use cross-validation to prevent leakage.

Pandas Integration
------------------

With pandas DataFrames:

.. code-block:: python

   import pandas as pd
   from sklearn.compose import ColumnTransformer
   from sklearn.preprocessing import LabelEncoder, OneHotEncoder
   from sklearn.pipeline import Pipeline
   from boosters.sklearn import GBDTClassifier

   # Define column types
   categorical_cols = ['color', 'size']
   numeric_cols = ['weight', 'height']

   # Create preprocessing pipeline
   preprocessor = ColumnTransformer([
       ('cat', OneHotEncoder(drop='first'), categorical_cols),
       ('num', 'passthrough', numeric_cols),
   ])

   pipeline = Pipeline([
       ('preprocess', preprocessor),
       ('model', GBDTClassifier())
   ])

   pipeline.fit(X_df, y)

Mixed Feature Types
-------------------

For datasets with both numeric and categorical features:

.. code-block:: python

   from sklearn.compose import ColumnTransformer
   from sklearn.preprocessing import StandardScaler, LabelEncoder
   from boosters.sklearn import GBDTRegressor

   preprocessor = ColumnTransformer([
       ('num', StandardScaler(), numeric_columns),
       ('cat', OrdinalEncoder(), categorical_columns),
   ])

   # Note: StandardScaler is optional for GBDT
   # but can improve convergence for GBLinear

Best Practices
--------------

1. **GBDT**: Use label encoding (ordinal). Trees don't need one-hot.
2. **GBLinear**: Use one-hot encoding for low cardinality, target encoding for high.
3. **Unknown categories**: Handle with ``handle_unknown='ignore'`` in encoders.
4. **Feature names**: Keep track of original feature names for interpretation.

.. code-block:: python

   # Handle unknown categories at prediction time
   encoder = OneHotEncoder(handle_unknown='ignore')

See Also
--------

- :doc:`missing-values` — Handling missing values
- :doc:`/explanations/gbdt` — How GBDT works with features

=======================
Handling Missing Values
=======================

boosters handles missing values natively, without requiring imputation. This 
guide explains how missing values work and best practices.

How Missing Values Work in GBDT
-------------------------------

During tree construction, boosters learns the optimal direction for missing 
values at each split:

1. At each candidate split, compute gain for both directions:
   - Send missing values left
   - Send missing values right

2. Choose the direction that maximizes split gain

3. Store this "default direction" in the tree node

During inference, when a feature value is missing (NaN), the sample follows 
the learned default direction.

Marking Missing Values
----------------------

NumPy NaN
^^^^^^^^^

Use ``np.nan`` for missing values:

.. code-block:: python

   import numpy as np
   from boosters.sklearn import GBDTRegressor

   X = np.array([
       [1.0, 2.0, np.nan],
       [3.0, np.nan, 5.0],
       [np.nan, 4.0, 6.0],
   ])
   y = np.array([1.0, 2.0, 3.0])

   model = GBDTRegressor()
   model.fit(X, y)  # Missing values handled automatically

Pandas NA
^^^^^^^^^

Pandas DataFrames work seamlessly:

.. code-block:: python

   import pandas as pd
   from boosters.sklearn import GBDTClassifier

   df = pd.DataFrame({
       'feature_a': [1.0, 2.0, None],
       'feature_b': [None, 3.0, 4.0],
       'target': [0, 1, 1]
   })

   X = df[['feature_a', 'feature_b']]
   y = df['target']

   model = GBDTClassifier()
   model.fit(X, y)

Best Practices
--------------

Don't Impute
^^^^^^^^^^^^

Avoid imputation when using boosters:

.. code-block:: python

   # ❌ Don't do this
   from sklearn.impute import SimpleImputer
   X_imputed = SimpleImputer().fit_transform(X)
   model.fit(X_imputed, y)

   # ✅ Do this instead
   model.fit(X, y)  # Let boosters learn optimal handling

Why? Imputation loses information. The fact that a value is missing is often 
informative (e.g., "user didn't answer this question").

Missing Indicator Features
^^^^^^^^^^^^^^^^^^^^^^^^^^

For additional predictive power, add explicit missing indicators:

.. code-block:: python

   import numpy as np

   # Add binary indicator columns
   X_missing = np.isnan(X).astype(float)
   X_combined = np.hstack([X, X_missing])

   model.fit(X_combined, y)

This lets the model learn different patterns based on missingness.

GBLinear and Missing Values
---------------------------

GBLinear handles missing values differently:

- Missing values are treated as zero in the dot product
- This is equivalent to mean imputation if features are centered

For GBLinear, consider explicit imputation:

.. code-block:: python

   from sklearn.impute import SimpleImputer
   from boosters.sklearn import GBLinearRegressor

   # Impute for linear models
   imputer = SimpleImputer(strategy='median')
   X_imputed = imputer.fit_transform(X)

   model = GBLinearRegressor()
   model.fit(X_imputed, y)

Checking Missing Values
-----------------------

Before training, check the proportion of missing values:

.. code-block:: python

   import numpy as np

   # Per-feature missing rate
   missing_rate = np.isnan(X).mean(axis=0)
   for i, rate in enumerate(missing_rate):
       if rate > 0:
           print(f"Feature {i}: {rate:.1%} missing")

   # If a feature is mostly missing, consider dropping it
   high_missing = missing_rate > 0.9
   X_filtered = X[:, ~high_missing]

See Also
--------

- :doc:`categorical-features` — Handling categorical data
- :doc:`/explanations/gbdt` — How GBDT learns split directions

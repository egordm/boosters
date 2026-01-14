=================
Python Quickstart
=================

This guide gets you training your first boosters model in under 5 minutes.

Prerequisites
-------------

- boosters installed (see :doc:`installation`)
- Basic Python and machine learning knowledge

Which Model Should I Use?
-------------------------

boosters offers three model types, each optimized for different use cases:

.. list-table::
   :header-rows: 1
   :widths: 20 30 25 25

   * - Model
     - Best For
     - Strengths
     - Considerations
   * - **GBDT** (default)
     - General-purpose ML
     - High accuracy, handles complex patterns
     - Cannot extrapolate beyond training data
   * - **GBDT + Linear Leaves**
     - Time series, extrapolation
     - Can extrapolate trends, good for forecasting
     - Needs local linear relationships
   * - **GBLinear**
     - Sparse/linear data
     - Fast, interpretable, memory efficient
     - Limited to linear relationships

**Quick decision guide:**

- 🎯 **Start with GBDT** — Works well for most problems
- 📈 **Need to predict beyond training range?** → Try Linear Leaves
- ⚡ **High-dimensional sparse data?** → Consider GBLinear
- 🔍 **Need feature coefficients?** → Use GBLinear

For detailed comparisons, see :doc:`/rfcs/0010-gblinear` (linear models) and
:doc:`/rfcs/0011-linear-leaves` (linear tree leaves).

Basic GBDT Training
-------------------

The simplest way to use boosters is through the sklearn-compatible interface:

.. code-block:: python

   from boosters.sklearn import GBDTRegressor
   from sklearn.datasets import make_regression
   from sklearn.model_selection import train_test_split

   # Generate sample data
   X, y = make_regression(n_samples=1000, n_features=10, random_state=42)
   X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

   # Train a model
   model = GBDTRegressor(n_estimators=100, max_depth=6, learning_rate=0.1)
   model.fit(X_train, y_train)

   # Make predictions
   predictions = model.predict(X_test)

   # Evaluate
   print(f"R² score: {model.score(X_test, y_test):.4f}")

Using the Core API
------------------

For more control, use the core ``GBDTModel`` API:

.. code-block:: python

   import numpy as np
   from boosters import Dataset, GBDTModel, GBDTConfig, Objective

   # Prepare data
   X_train = np.random.randn(1000, 10).astype(np.float32)
   y_train = np.random.randn(1000).astype(np.float32)

   # Create dataset
   train_data = Dataset(X_train, y_train)

   # Configure model
   config = GBDTConfig(
       n_estimators=100,
       max_depth=6,
       learning_rate=0.1,
       objective=Objective.squared(),
   )

   # Train
   model = GBDTModel.train(config, train_data)

   # Predict
   predictions = model.predict(X_train)

Classification
--------------

For classification tasks, use ``GBDTClassifier``:

.. code-block:: python

   from boosters.sklearn import GBDTClassifier
   from sklearn.datasets import make_classification

   X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
   X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

   model = GBDTClassifier(n_estimators=100, max_depth=6)
   model.fit(X_train, y_train)

   # Get probabilities
   probas = model.predict_proba(X_test)
   
   # Get class predictions
   predictions = model.predict(X_test)
   print(f"Accuracy: {model.score(X_test, y_test):.4f}")

GBLinear (Linear Boosting)
--------------------------

For high-dimensional or sparse data with linear relationships:

.. code-block:: python

   from boosters.sklearn import GBLinearRegressor

   # GBLinear is especially good for sparse/linear data
   model = GBLinearRegressor(n_estimators=100, learning_rate=0.5)
   model.fit(X_train, y_train)

   # Access learned coefficients
   print(f"Coefficients: {model.coef_}")
   print(f"Intercept: {model.intercept_}")

See :doc:`/tutorials/06-gblinear-sparse` for a complete tutorial.

GBDT with Linear Leaves
-----------------------

Enable linear leaves for better extrapolation:

.. code-block:: python

   from boosters.sklearn import GBDTRegressor

   # Enable linear leaves for extrapolation capability
   model = GBDTRegressor(
       n_estimators=50,
       max_depth=4,
       linear_leaves=True,  # Key parameter!
       linear_l2=0.01,
   )
   model.fit(X_train, y_train)

See :doc:`/tutorials/10-linear-trees` for when and how to use linear leaves.

Next Steps
----------

- :doc:`/tutorials/index` — Hands-on tutorials for common tasks
- :doc:`/user-guide/hyperparameters` — Understanding hyperparameters
- :doc:`/api/index` — Complete API reference
- :doc:`/rfcs/0008-gbdt-training` — GBDT training algorithm details

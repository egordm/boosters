==============
Early Stopping
==============

Early stopping prevents overfitting by monitoring validation performance 
and stopping training when the model stops improving.

Basic Usage
-----------

**sklearn API:**

.. code-block:: python

   from boosters.sklearn import GBDTRegressor

   model = GBDTRegressor(
       n_estimators=1000,          # Maximum trees
       early_stopping_rounds=10,   # Stop after 10 rounds without improvement
   )

   # Provide validation data via eval_set
   model.fit(
       X_train, y_train,
       eval_set=[(X_val, y_val)],
   )

   # Check how many trees were actually trained
   print(f"Trained {model.n_trees_} trees (max was 1000)")

**Core API:**

.. code-block:: python

   import boosters as bst

   config = bst.GBDTConfig(
       n_estimators=1000,
       early_stopping_rounds=10,
       objective=bst.Objective.squared(),
       metric=bst.Metric.rmse(),
   )

   train_data = bst.Dataset(X_train, y_train)
   val_data = bst.Dataset(X_val, y_val)

   model = bst.GBDTModel.train(
       train_data,
       evals=[(val_data, "validation")],
       config=config,
   )

How Early Stopping Works
------------------------

1. **Train** — Add trees one by one
2. **Evaluate** — Compute metric on validation set after each tree
3. **Check** — If no improvement for ``early_stopping_rounds``, stop
4. **Return** — Use the best model (not the last one)

::

   Iteration 1:  Train RMSE: 0.85  Val RMSE: 0.90
   Iteration 2:  Train RMSE: 0.72  Val RMSE: 0.82  ← improved
   Iteration 3:  Train RMSE: 0.65  Val RMSE: 0.78  ← improved
   ...
   Iteration 50: Train RMSE: 0.12  Val RMSE: 0.75  ← best so far
   Iteration 51: Train RMSE: 0.11  Val RMSE: 0.76  ← no improvement (1/10)
   Iteration 52: Train RMSE: 0.10  Val RMSE: 0.77  ← no improvement (2/10)
   ...
   Iteration 60: Train RMSE: 0.05  Val RMSE: 0.82  ← no improvement (10/10)
   → Stopping! Returning model from iteration 50.

Choosing early_stopping_rounds
------------------------------

.. list-table::
   :widths: 20 40 40
   :header-rows: 1

   * - Value
     - Behavior
     - When to Use
   * - **5-10**
     - Aggressive stopping
     - Quick experiments, small datasets
   * - **10-20**
     - Balanced (recommended)
     - Most use cases
   * - **50-100**
     - Conservative stopping
     - Large datasets, slow learning rates

**Rule of thumb**: Use higher values with lower learning rates.

Validation Set Strategies
-------------------------

**Hold-out validation** (simple, fast):

.. code-block:: python

   from sklearn.model_selection import train_test_split

   X_train, X_val, y_train, y_val = train_test_split(
       X, y, test_size=0.2, random_state=42
   )

**Time-based split** (for time series):

.. code-block:: python

   # Use last 20% as validation
   split_idx = int(len(X) * 0.8)
   X_train, X_val = X[:split_idx], X[split_idx:]
   y_train, y_val = y[:split_idx], y[split_idx:]

**K-fold with early stopping** (use with caution):

.. code-block:: python

   from sklearn.model_selection import KFold

   kf = KFold(n_splits=5, shuffle=True, random_state=42)
   models = []

   for train_idx, val_idx in kf.split(X):
       model = GBDTRegressor(n_estimators=1000, early_stopping_rounds=10)
       model.fit(
           X[train_idx], y[train_idx],
           eval_set=[(X[val_idx], y[val_idx])],
       )
       models.append(model)

Monitoring Multiple Metrics
---------------------------

You can monitor multiple metrics, but only one is used for early stopping:

.. code-block:: python

   config = bst.GBDTConfig(
       n_estimators=1000,
       early_stopping_rounds=10,
       metric=bst.Metric.rmse(),  # This one controls early stopping
   )

Common Pitfalls
---------------

**Don't use test set for early stopping:**

.. code-block:: python

   # ❌ Wrong: Using test set for early stopping
   model.fit(X_train, y_train, eval_set=[(X_test, y_test)])

   # ✅ Right: Use a separate validation set
   X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
   model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

**Don't forget to set n_estimators high enough:**

.. code-block:: python

   # ❌ Wrong: Low n_estimators limits potential
   model = GBDTRegressor(n_estimators=50, early_stopping_rounds=10)

   # ✅ Right: High n_estimators, let early stopping decide
   model = GBDTRegressor(n_estimators=1000, early_stopping_rounds=10)

See Also
--------

- :doc:`/tutorials/05-early-stopping` — Early stopping tutorial with visualizations
- :doc:`hyperparameters` — Understanding all hyperparameters

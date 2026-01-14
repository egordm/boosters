======================
Debugging Performance
======================

This guide helps you diagnose and fix performance issues with boosters 
models.

Identifying the Problem
-----------------------

First, determine whether the issue is:

1. **Model quality** — predictions are inaccurate
2. **Training speed** — fitting takes too long
3. **Inference speed** — prediction is too slow
4. **Memory usage** — out of memory errors

Model Quality Issues
--------------------

Underfitting
^^^^^^^^^^^^

Symptoms: Training and validation metrics are both poor.

Solutions:

.. code-block:: python

   # Increase model capacity
   model = GBDTRegressor(
       n_estimators=500,       # More trees
       max_depth=8,            # Deeper trees
       learning_rate=0.1,      # Standard learning rate
       min_child_weight=1,     # Lower regularization
   )

Overfitting
^^^^^^^^^^^

Symptoms: Training metrics are good but validation metrics are poor.

Solutions:

.. code-block:: python

   # Increase regularization
   model = GBDTRegressor(
       n_estimators=100,
       max_depth=4,            # Shallower trees
       learning_rate=0.05,     # Lower learning rate
       reg_lambda=10.0,        # More L2 regularization
       subsample=0.8,          # Row subsampling
       colsample_bytree=0.8,   # Column subsampling
       min_child_weight=10,    # Larger minimum leaf
   )

Training Speed Issues
---------------------

Profiling Training
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import time
   from boosters.sklearn import GBDTRegressor

   # Time training
   start = time.perf_counter()
   model = GBDTRegressor(n_estimators=100)
   model.fit(X_train, y_train)
   elapsed = time.perf_counter() - start
   print(f"Training: {elapsed:.2f}s ({len(X_train)/elapsed:.0f} samples/sec)")

Speed Improvements
^^^^^^^^^^^^^^^^^^

1. **Reduce tree depth**: Each depth level doubles work

   .. code-block:: python

      model = GBDTRegressor(max_depth=4)  # vs default 6

2. **Enable subsampling**: Trade accuracy for speed

   .. code-block:: python

      model = GBDTRegressor(subsample=0.5)

3. **Reduce number of bins**: Fewer histogram bins

   .. code-block:: python

      model = GBDTRegressor(max_bin=128)  # vs default 256

4. **Use early stopping**: Stop when validation metric plateaus

   .. code-block:: python

      model.fit(
          X_train, y_train,
          eval_set=[(X_val, y_val)],
          early_stopping_rounds=10
      )

Inference Speed Issues
----------------------

Profiling Inference
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import time
   import numpy as np

   # Warmup
   _ = model.predict(X_test[:100])

   # Batch prediction
   start = time.perf_counter()
   predictions = model.predict(X_test)
   elapsed = time.perf_counter() - start
   print(f"Batch: {elapsed*1000:.2f}ms ({len(X_test)/elapsed:.0f} samples/sec)")

   # Single-row prediction
   n_samples = 1000
   start = time.perf_counter()
   for i in range(n_samples):
       _ = model.predict(X_test[i:i+1])
   elapsed = time.perf_counter() - start
   print(f"Single: {elapsed/n_samples*1000:.3f}ms per sample")

Speed Improvements
^^^^^^^^^^^^^^^^^^

1. **Batch predictions**: Always predict multiple rows at once

   .. code-block:: python

      # ❌ Slow: one at a time
      for x in X:
          model.predict(x.reshape(1, -1))

      # ✅ Fast: all at once
      model.predict(X)

2. **Fewer trees**: Trade accuracy for speed

   .. code-block:: python

      # Train with early stopping to find minimum needed
      model = GBDTRegressor(n_estimators=1000)
      model.fit(X, y, eval_set=[(X_val, y_val)], early_stopping_rounds=10)
      print(f"Best iteration: {model.best_iteration_}")

3. **Consider GBLinear**: If relationships are mostly linear

   .. code-block:: python

      from boosters.sklearn import GBLinearRegressor
      # Much faster inference: O(features) vs O(trees × depth)

Memory Issues
-------------

Reducing Memory Usage
^^^^^^^^^^^^^^^^^^^^^

1. **Use appropriate dtypes**: float32 instead of float64

   .. code-block:: python

      X = X.astype(np.float32)

2. **Reduce max_bin**: Fewer bins = less histogram memory

   .. code-block:: python

      model = GBDTRegressor(max_bin=128)

3. **Process in chunks**: For very large datasets

   .. code-block:: python

      # If your data doesn't fit in memory
      chunk_size = 100000
      for i in range(0, len(X), chunk_size):
          predictions[i:i+chunk_size] = model.predict(X[i:i+chunk_size])

Systematic Debugging
--------------------

Use this checklist:

1. **Check data**
   
   - Are there duplicate rows?
   - Are features on reasonable scales?
   - Are there features with near-zero variance?

2. **Check hyperparameters**
   
   - Start with defaults, then tune
   - Use cross-validation

3. **Compare with baseline**
   
   - Train a simple model (e.g., linear regression)
   - If boosters is much worse, data might have issues

4. **Visualize learning curves**

   .. code-block:: python

      from sklearn.model_selection import learning_curve
      
      train_sizes, train_scores, val_scores = learning_curve(
          model, X, y, cv=5, n_jobs=-1
      )
      # Plot to see underfitting/overfitting

See Also
--------

- :doc:`/explanations/hyperparameters` — Parameter tuning
- :doc:`/explanations/benchmarks` — Performance characteristics
- :doc:`production-deployment` — Optimizing for production

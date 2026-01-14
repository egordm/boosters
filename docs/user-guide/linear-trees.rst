============
Linear Trees
============

Linear trees (also called "trees with linear leaves") combine decision tree 
structure with linear regression in the leaves, offering a middle ground 
between GBDT and GBLinear.

.. note::

   Linear trees are an advanced feature. Start with standard GBDT for most 
   use cases. Consider linear trees when you need both:
   
   - Non-linear feature interactions (like GBDT)
   - Smooth predictions within regions (like linear models)

What Are Linear Trees?
----------------------

Standard GBDT uses **constant leaf values** — each leaf predicts a single number:

::

   Tree (standard GBDT):
         [feature_1 < 0.5]
            /        \
       leaf=2.3    leaf=4.1

Linear trees use **linear models in leaves** — each leaf fits a small linear 
regression on the samples that reach it:

::

   Tree (linear leaves):
         [feature_1 < 0.5]
            /        \
     y = 0.5*x₁ + 1.2    y = -0.3*x₂ + 2.1

This allows smooth, continuous predictions within each leaf region while 
still capturing non-linear interactions via the tree splits.

When to Use Linear Trees
------------------------

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - ✅ Good Fit
     - ❌ Not Ideal
   * - Linear relationships within local regions
     - Purely non-linear relationships
   * - Need smooth prediction surfaces
     - Categorical-heavy data
   * - Few trees with high accuracy needed
     - Very high-dimensional data
   * - Interpretability matters
     - Training speed is critical

**Common use cases:**

- Time series with trend components
- Physical/engineering models with known linear structure
- Tabular data with mix of linear and non-linear effects

Usage
-----

**Core API:**

.. code-block:: python

   import boosters as bst

   config = bst.GBDTConfig(
       n_estimators=50,
       max_depth=4,
       linear_leaves=True,     # Enable linear leaves
       linear_l2=0.01,         # L2 regularization for linear models
       objective=bst.Objective.squared(),
   )

   train_data = bst.Dataset(X_train, y_train)
   model = bst.GBDTModel.train(train_data, config=config)
   
   # Make predictions
   test_data = bst.Dataset(X_test)
   predictions = model.predict(test_data)

Key Parameters
--------------

``linear_leaves`` (default: False)
   Enable linear regression in tree leaves.

``max_depth`` (recommended: 3-5)
   Shallower trees work well with linear leaves since each leaf captures 
   more linear structure.

``n_estimators`` (recommended: 20-100)
   Fewer trees are typically needed since each tree is more expressive.

``linear_l2`` (default: 0.01)
   L2 regularization for the linear models in leaves. Increase if you see 
   overfitting or unstable predictions.

Performance Considerations
--------------------------

**Training time:**

Linear trees are slower to train than standard GBDT because each leaf 
requires fitting a linear regression. However, you typically need fewer 
trees, which can offset this cost.

**Prediction time:**

Prediction is slightly slower due to the linear computation in each leaf, 
but the difference is usually negligible.

**Memory:**

Each leaf stores a weight vector (one per feature used in that leaf) instead 
of a single value, increasing model size.

Example: Regression with Trend
------------------------------

.. code-block:: python

   import numpy as np
   import boosters as bst
   from sklearn.metrics import r2_score

   # Data with local linear trends
   np.random.seed(42)
   X = np.random.randn(1000, 5)
   
   # Target has both non-linear structure and local linear trends
   y = (
       np.sin(X[:, 0]) * 2 +           # Non-linear component
       X[:, 1] * X[:, 2] +              # Interaction
       0.5 * X[:, 3] +                  # Linear trend
       np.random.randn(1000) * 0.1     # Noise
   )

   # Split data
   train_data = bst.Dataset(X[:800], y[:800])
   test_data = bst.Dataset(X[800:])
   y_test = y[800:]

   # Standard GBDT
   config_standard = bst.GBDTConfig(n_estimators=100, max_depth=6)
   model_standard = bst.GBDTModel.train(train_data, config=config_standard)
   pred_standard = model_standard.predict(test_data).flatten()

   # Linear trees
   config_linear = bst.GBDTConfig(
       n_estimators=50, max_depth=4, 
       linear_leaves=True, linear_l2=0.01
   )
   model_linear = bst.GBDTModel.train(train_data, config=config_linear)
   pred_linear = model_linear.predict(test_data).flatten()

   print(f"Standard GBDT R²: {r2_score(y_test, pred_standard):.4f}")
   print(f"Linear Trees R²:  {r2_score(y_test, pred_linear):.4f}")

Comparison with Other Methods
-----------------------------

.. list-table::
   :widths: 25 25 25 25
   :header-rows: 1

   * - Method
     - Non-linear
     - Smooth
     - Speed
   * - GBDT (constant leaves)
     - ✅ Yes
     - ❌ Step-wise
     - ⚡ Fast
   * - GBLinear
     - ❌ No
     - ✅ Yes
     - ⚡ Fast
   * - Linear Trees
     - ✅ Yes
     - ✅ Yes
     - 🐢 Slower
   * - Neural Networks
     - ✅ Yes
     - ✅ Yes
     - 🐢 Slower

See Also
--------

- :doc:`/tutorials/10-linear-trees` — Hands-on tutorial with examples
- :doc:`gblinear` — Pure linear gradient boosting
- :doc:`/rfcs/0011-linear-leaves` — Design document with algorithm details
- :doc:`/research/gradient-boosting` — Background on gradient boosting theory

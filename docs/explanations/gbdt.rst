====
GBDT
====

**Gradient Boosted Decision Trees (GBDT)** uses decision trees as the weak 
learners in gradient boosting. This is the default and most powerful model 
type in boosters.

How GBDT Works
--------------

At each boosting iteration:

1. Compute gradients and Hessians for all training samples
2. Build a decision tree that minimizes the weighted sum of squared residuals
3. Each leaf contains a weight :math:`w` that minimizes the objective
4. Add the tree to the ensemble with learning rate shrinkage

Tree Structure
--------------

Each tree partitions the feature space recursively:

::

   Root Node
   ├── Feature[3] < 0.5 → Left Node
   │   ├── Feature[1] < 2.3 → Leaf (weight: 0.42)
   │   └── Feature[1] ≥ 2.3 → Leaf (weight: -0.18)
   └── Feature[3] ≥ 0.5 → Right Node
       ├── Feature[7] < 1.0 → Leaf (weight: 0.31)
       └── Feature[7] ≥ 1.0 → Leaf (weight: -0.05)

Split Finding
^^^^^^^^^^^^^

For each candidate split, we compute the gain:

.. math::

   \text{Gain} = \frac{1}{2}\left[
   \frac{G_L^2}{H_L + \lambda} + 
   \frac{G_R^2}{H_R + \lambda} - 
   \frac{(G_L + G_R)^2}{H_L + H_R + \lambda}
   \right] - \gamma

where:

- :math:`G_L, G_R` are the sum of gradients for left/right children
- :math:`H_L, H_R` are the sum of Hessians for left/right children
- :math:`\lambda` is L2 regularization
- :math:`\gamma` is the minimum split loss (complexity penalty)

When to Use GBDT
----------------

GBDT is ideal for:

- **Tabular data**: Structured data with mixed feature types
- **Non-linear relationships**: Trees naturally capture interactions
- **Feature interactions**: Automatic discovery of complex patterns
- **Moderate-sized data**: Hundreds to millions of samples

Advantages
^^^^^^^^^^

- Handles missing values natively
- No feature scaling required
- Captures non-linear patterns and interactions
- Robust to outliers
- Built-in feature importance

Disadvantages
^^^^^^^^^^^^^

- Can overfit on small datasets
- Slower inference than linear models
- Less interpretable than linear models
- Struggles with extrapolation

Key Hyperparameters
-------------------

+-------------------+------------------+-------------------------------------+
| Parameter         | Default          | Effect                              |
+===================+==================+=====================================+
| ``max_depth``     | 6                | Maximum tree depth                  |
+-------------------+------------------+-------------------------------------+
| ``n_estimators``  | 100              | Number of trees                     |
+-------------------+------------------+-------------------------------------+
| ``learning_rate`` | 0.3              | Shrinkage per tree                  |
+-------------------+------------------+-------------------------------------+
| ``reg_lambda``    | 1.0              | L2 regularization                   |
+-------------------+------------------+-------------------------------------+
| ``subsample``     | 1.0              | Row sampling ratio                  |
+-------------------+------------------+-------------------------------------+

See :doc:`hyperparameters` for the complete reference.

Example
-------

.. code-block:: python

   from boosters.sklearn import GBDTRegressor

   model = GBDTRegressor(
       n_estimators=100,
       max_depth=6,
       learning_rate=0.1,
       reg_lambda=1.0,
   )
   model.fit(X_train, y_train)
   predictions = model.predict(X_test)

See Also
--------

- :doc:`gradient-boosting` — Theory overview
- :doc:`gblinear` — Alternative: linear boosting
- :doc:`hyperparameters` — Complete parameter guide

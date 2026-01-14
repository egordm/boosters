===============
Hyperparameters
===============

This guide explains all hyperparameters in boosters, their effects, and 
tuning recommendations.

General Parameters
------------------

These apply to both GBDT and GBLinear:

``n_estimators`` (default: 100)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Number of boosting iterations (trees for GBDT, rounds for GBLinear).

- **Too low**: Underfitting, model doesn't learn patterns
- **Too high**: Overfitting, diminishing returns, slower training
- **Typical range**: 50–1000
- **Tuning**: Use early stopping to find optimal value

``learning_rate`` (default: 0.3)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Shrinkage applied to each weak learner. Lower values require more iterations 
but often generalize better.

.. math::

   F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)

- **Too low**: Need many iterations, slow training
- **Too high**: Overfitting, unstable training
- **Typical range**: 0.01–0.3
- **Rule of thumb**: Lower learning rate + more trees = better generalization

``objective`` (default: "reg:squarederror")
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Loss function to optimize. See :doc:`/research/classification-objectives` for details.

- ``reg:squarederror``: Regression (MSE)
- ``binary:logistic``: Binary classification (log loss)
- ``multi:softmax``: Multiclass classification
- ``reg:quantile``: Quantile regression

``seed`` (default: None)
^^^^^^^^^^^^^^^^^^^^^^^^

Random seed for reproducibility. Affects:

- Feature subsampling
- Row subsampling
- Initial tree structure (for some implementations)

GBDT-Specific Parameters
------------------------

Tree Structure
^^^^^^^^^^^^^^

``max_depth`` (default: 6)
""""""""""""""""""""""""""

Maximum depth of each tree.

- **Deeper trees**: More complex patterns, higher overfitting risk
- **Shallower trees**: Simpler patterns, needs more trees
- **Typical range**: 3–10
- **Note**: Computation is O(2^depth), so deep trees are expensive

``max_leaves`` (default: None)
""""""""""""""""""""""""""""""

Maximum number of leaf nodes. Alternative to max_depth for controlling 
tree complexity.

- If set, overrides max_depth
- Allows asymmetric trees
- **Typical range**: 16–256

``min_child_weight`` (default: 1.0)
"""""""""""""""""""""""""""""""""""

Minimum sum of Hessians required in a child node. Regularization parameter 
that prevents splits creating too-small leaf nodes.

- **Higher values**: More conservative, prevents overfitting
- **For regression**: Similar to minimum samples per leaf
- **For classification**: Accounts for class imbalance
- **Typical range**: 1–10

``min_split_loss`` (gamma) (default: 0.0)
"""""""""""""""""""""""""""""""""""""""""

Minimum loss reduction required to make a split. Pruning parameter.

.. math::

   \text{Split if Gain} > \gamma

- **Higher values**: Fewer splits, simpler trees
- **Typical range**: 0–5

Regularization
^^^^^^^^^^^^^^

``reg_lambda`` (default: 1.0)
"""""""""""""""""""""""""""""

L2 regularization on leaf weights.

.. math::

   w^* = -\frac{\sum g_i}{\sum H_i + \lambda}

- Prevents extreme leaf weights
- **Typical range**: 0–10

``reg_alpha`` (default: 0.0)
""""""""""""""""""""""""""""

L1 regularization on leaf weights. Promotes sparse solutions.

- Useful when many features are irrelevant
- **Typical range**: 0–1

Subsampling
^^^^^^^^^^^

``subsample`` (default: 1.0)
""""""""""""""""""""""""""""

Row subsampling ratio. Each tree sees a random subset of training data.

- **Lower values**: More regularization, faster training
- Reduces overfitting through stochastic effects
- **Typical range**: 0.5–1.0

``colsample_bytree`` (default: 1.0)
"""""""""""""""""""""""""""""""""""

Column subsampling ratio per tree.

- Each tree considers a random subset of features
- **Typical range**: 0.5–1.0

``colsample_bylevel`` (default: 1.0)
""""""""""""""""""""""""""""""""""""

Column subsampling ratio per tree level.

- More aggressive feature regularization
- **Typical range**: 0.5–1.0

GBLinear-Specific Parameters
----------------------------

``feature_selector`` (default: "cyclic")
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Strategy for selecting features to update:

+-------------+------------------------------------------------------------+
| Selector    | When to Use                                                |
+=============+============================================================+
| ``cyclic``  | Default, deterministic, good for most cases                |
+-------------+------------------------------------------------------------+
| ``shuffle`` | Breaks feature correlations, often better generalization   |
+-------------+------------------------------------------------------------+
| ``greedy``  | Sparse data, when few features are relevant                |
+-------------+------------------------------------------------------------+
| ``thrifty`` | High dimensions where greedy is too slow                   |
+-------------+------------------------------------------------------------+

``top_k`` (default: 0)
^^^^^^^^^^^^^^^^^^^^^^

For greedy/thrifty selectors: number of top features to consider.

- 0 means use all features
- Useful for very high dimensional data

Tuning Strategy
---------------

Start Simple
^^^^^^^^^^^^

1. Use defaults
2. Set learning_rate=0.1, n_estimators=1000 with early stopping
3. Let early stopping find the right number of trees

Tune Tree Structure
^^^^^^^^^^^^^^^^^^^

- Start with max_depth=6
- If overfitting: reduce max_depth, increase min_child_weight
- If underfitting: increase max_depth, reduce regularization

Add Regularization
^^^^^^^^^^^^^^^^^^

- Add subsampling (subsample=0.8, colsample_bytree=0.8)
- Increase reg_lambda if still overfitting

Cross-Validation
^^^^^^^^^^^^^^^^

Use cross-validation to evaluate hyperparameter choices:

.. code-block:: python

   from sklearn.model_selection import cross_val_score
   from boosters.sklearn import GBDTRegressor

   model = GBDTRegressor(
       n_estimators=100,
       max_depth=6,
       learning_rate=0.1,
   )
   scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error")
   print(f"RMSE: {(-scores.mean()) ** 0.5:.4f} ± {scores.std() ** 0.5:.4f}")

See Also
--------

- :doc:`/research/gradient-boosting` — Theory overview
- :doc:`gblinear` — GBLinear details
- :doc:`/research/classification-objectives` — Loss functions and metrics

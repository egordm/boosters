========
GBLinear
========

**GBLinear** uses linear models as the weak learners instead of decision trees. 
This produces a final linear model through iterative coordinate descent.

How GBLinear Works
------------------

At each boosting iteration:

1. Compute gradients and Hessians for all training samples
2. Update one or more feature weights using coordinate descent
3. The update minimizes the second-order approximation of the loss
4. Apply learning rate shrinkage

The final model is linear:

.. math::

   \hat{y} = w_0 + \sum_{j=1}^{p} w_j x_j

where :math:`w_0` is the bias and :math:`w_j` are feature weights.

Weight Updates
--------------

For a single feature :math:`j`, the optimal weight update is:

.. math::

   \Delta w_j = -\frac{\sum_i g_i x_{ij}}{\sum_i H_i x_{ij}^2 + \lambda}

where:

- :math:`g_i` is the gradient for sample :math:`i`
- :math:`H_i` is the Hessian for sample :math:`i`
- :math:`x_{ij}` is the feature value
- :math:`\lambda` is L2 regularization

Feature Selectors
-----------------

GBLinear supports different strategies for selecting which features to update:

+-------------+------------------------------------------------------------+
| Selector    | Description                                                |
+=============+============================================================+
| ``cyclic``  | Cycle through features in order (deterministic)            |
+-------------+------------------------------------------------------------+
| ``shuffle`` | Random feature order each round (breaks correlations)      |
+-------------+------------------------------------------------------------+
| ``greedy``  | Pick feature with largest gradient (sparse data)           |
+-------------+------------------------------------------------------------+
| ``thrifty`` | Approximate greedy (faster for high dimensions)            |
+-------------+------------------------------------------------------------+

When to Use GBLinear
--------------------

GBLinear is ideal for:

- **High-dimensional sparse data**: Text, click-through prediction
- **Linear relationships**: When the true relationship is mostly linear
- **Fast inference**: Linear prediction is O(features)
- **Interpretability**: Feature weights directly show importance

Advantages
^^^^^^^^^^

- Very fast inference
- Memory efficient for sparse data
- Easily interpretable (linear coefficients)
- Good for high-dimensional data
- L1/L2 regularization built-in

Disadvantages
^^^^^^^^^^^^^

- Cannot capture non-linear patterns
- Cannot capture feature interactions
- Assumes linear separability for classification

GBDT vs GBLinear
----------------

+------------------+----------------------------------+---------------------------+
| Aspect           | GBDT                             | GBLinear                  |
+==================+==================================+===========================+
| Relationships    | Non-linear, interactions         | Linear only               |
+------------------+----------------------------------+---------------------------+
| Inference speed  | O(trees × depth)                 | O(features)               |
+------------------+----------------------------------+---------------------------+
| Sparse data      | OK                               | Excellent                 |
+------------------+----------------------------------+---------------------------+
| Interpretability | Feature importance               | Direct coefficients       |
+------------------+----------------------------------+---------------------------+
| Best for         | Tabular data                     | High-dim linear data      |
+------------------+----------------------------------+---------------------------+

Key Hyperparameters
-------------------

+----------------------+------------+-------------------------------------+
| Parameter            | Default    | Effect                              |
+======================+============+=====================================+
| ``n_estimators``     | 100        | Number of boosting rounds           |
+----------------------+------------+-------------------------------------+
| ``learning_rate``    | 0.3        | Step size for weight updates        |
+----------------------+------------+-------------------------------------+
| ``reg_lambda``       | 0.0        | L2 regularization                   |
+----------------------+------------+-------------------------------------+
| ``reg_alpha``        | 0.0        | L1 regularization                   |
+----------------------+------------+-------------------------------------+
| ``feature_selector`` | ``cyclic`` | How features are selected           |
+----------------------+------------+-------------------------------------+

Example
-------

.. code-block:: python

   from boosters.sklearn import GBLinearRegressor

   model = GBLinearRegressor(
       n_estimators=100,
       learning_rate=0.5,
       reg_lambda=0.1,
       feature_selector="shuffle",
   )
   model.fit(X_train, y_train)
   predictions = model.predict(X_test)

   # Access linear coefficients
   print("Weights:", model.coef_)
   print("Bias:", model.intercept_)

See Also
--------

- :doc:`gradient-boosting` — Theory overview
- :doc:`gbdt` — Alternative: tree boosting
- :doc:`hyperparameters` — Complete parameter guide

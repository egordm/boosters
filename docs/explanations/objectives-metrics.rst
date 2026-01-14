======================
Objectives and Metrics
======================

This page documents all objective functions (loss) and evaluation metrics 
available in boosters, including their mathematical definitions.

Objectives
----------

Objectives define the loss function to minimize during training. They 
provide gradients and Hessians for second-order optimization.

Regression Objectives
^^^^^^^^^^^^^^^^^^^^^

``reg:squarederror`` (default)
""""""""""""""""""""""""""""""

Mean Squared Error (MSE) for regression.

.. math::

   L = \frac{1}{2}(y - \hat{y})^2

.. math::

   g = \hat{y} - y, \quad H = 1

- Use for: Continuous target, no outliers
- Output: Unbounded real value

``reg:absoluteerror``
"""""""""""""""""""""

Mean Absolute Error (MAE) for robust regression.

.. math::

   L = |y - \hat{y}|

.. math::

   g = \text{sign}(\hat{y} - y), \quad H = 0

- Use for: Regression with outliers
- Output: Unbounded real value
- Note: Hessian is zero, uses gradient descent

``reg:quantile``
""""""""""""""""

Quantile regression for predicting percentiles.

.. math::

   L = \begin{cases}
   \alpha (y - \hat{y}) & \text{if } y \geq \hat{y} \\
   (1-\alpha) (\hat{y} - y) & \text{otherwise}
   \end{cases}

- Use for: Prediction intervals, risk quantiles
- Parameter: ``quantile_alpha`` (default: 0.5 = median)

``reg:pseudohuber``
"""""""""""""""""""

Pseudo-Huber loss, smooth approximation of Huber loss.

.. math::

   L = \delta^2 \left(\sqrt{1 + \left(\frac{y - \hat{y}}{\delta}\right)^2} - 1\right)

- Use for: Robust regression (smooth L1/L2 transition)
- Parameter: ``huber_delta`` (default: 1.0)

Classification Objectives
^^^^^^^^^^^^^^^^^^^^^^^^^

``binary:logistic``
"""""""""""""""""""

Binary cross-entropy (log loss) for binary classification.

.. math::

   L = -y \log(\sigma) - (1-y) \log(1-\sigma)

where :math:`\sigma = \frac{1}{1 + e^{-\hat{y}}}` is the sigmoid.

.. math::

   g = \sigma - y, \quad H = \sigma(1 - \sigma)

- Use for: Binary classification
- Output: Log-odds (apply sigmoid for probability)

``multi:softmax``
"""""""""""""""""

Multiclass classification using softmax.

.. math::

   L = -\sum_{k=1}^{K} y_k \log(p_k)

where :math:`p_k = \frac{e^{\hat{y}_k}}{\sum_j e^{\hat{y}_j}}`.

- Use for: Multiclass classification
- Output: One prediction per class (raw scores)

Metrics
-------

Metrics are used for evaluation (early stopping, cross-validation).

Regression Metrics
^^^^^^^^^^^^^^^^^^

+------------------+-----------------------------------------------------+
| Metric           | Description                                         |
+==================+=====================================================+
| ``rmse``         | Root Mean Squared Error                             |
+------------------+-----------------------------------------------------+
| ``mae``          | Mean Absolute Error                                 |
+------------------+-----------------------------------------------------+
| ``mse``          | Mean Squared Error                                  |
+------------------+-----------------------------------------------------+
| ``mape``         | Mean Absolute Percentage Error                      |
+------------------+-----------------------------------------------------+
| ``rmsle``        | Root Mean Squared Log Error                         |
+------------------+-----------------------------------------------------+

Classification Metrics
^^^^^^^^^^^^^^^^^^^^^^

+------------------+-----------------------------------------------------+
| Metric           | Description                                         |
+==================+=====================================================+
| ``logloss``      | Logistic loss (binary cross-entropy)                |
+------------------+-----------------------------------------------------+
| ``error``        | Binary classification error rate                    |
+------------------+-----------------------------------------------------+
| ``auc``          | Area Under ROC Curve                                |
+------------------+-----------------------------------------------------+
| ``aucpr``        | Area Under Precision-Recall Curve                   |
+------------------+-----------------------------------------------------+
| ``mlogloss``     | Multiclass log loss                                 |
+------------------+-----------------------------------------------------+
| ``merror``       | Multiclass error rate                               |
+------------------+-----------------------------------------------------+

Custom Objectives
-----------------

You can define custom objectives by implementing gradient and Hessian:

.. code-block:: python

   import numpy as np

   def custom_squared_log_error(y_true, y_pred):
       """Squared log error with gradient and Hessian."""
       # Ensure positive predictions
       y_pred = np.maximum(y_pred, 1e-6)
       
       residual = np.log(y_pred + 1) - np.log(y_true + 1)
       grad = residual / (y_pred + 1)
       hess = (1 - residual) / (y_pred + 1) ** 2
       
       return grad, hess

See :doc:`/howto/custom-objectives` for detailed examples.

Choosing an Objective
---------------------

+-----------------------+----------------------------------------------+
| Task                  | Recommended Objective                        |
+=======================+==============================================+
| Regression            | ``reg:squarederror``                         |
+-----------------------+----------------------------------------------+
| Robust regression     | ``reg:pseudohuber`` or ``reg:absoluteerror`` |
+-----------------------+----------------------------------------------+
| Quantile prediction   | ``reg:quantile``                             |
+-----------------------+----------------------------------------------+
| Binary classification | ``binary:logistic``                          |
+-----------------------+----------------------------------------------+
| Multiclass            | ``multi:softmax``                            |
+-----------------------+----------------------------------------------+

See Also
--------

- :doc:`gradient-boosting` — How objectives drive optimization
- :doc:`hyperparameters` — Complete parameter reference
- :doc:`/howto/custom-objectives` — Implementing custom objectives

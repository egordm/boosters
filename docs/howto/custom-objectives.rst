=================
Custom Objectives
=================

This guide explains how to implement custom objective functions for gradient 
boosting with boosters.

Overview
--------

A custom objective provides:

1. **Gradient** (first derivative of loss w.r.t. prediction)
2. **Hessian** (second derivative of loss w.r.t. prediction)

These are used by boosters for second-order optimization.

Basic Structure
---------------

A custom objective is a callable that returns gradients and Hessians:

.. code-block:: python

   def custom_objective(y_true, y_pred):
       """
       Compute gradients and Hessians.
       
       Parameters
       ----------
       y_true : array-like of shape (n_samples,)
           True target values.
       y_pred : array-like of shape (n_samples,)
           Predicted values (raw scores, not transformed).
       
       Returns
       -------
       grad : array of shape (n_samples,)
           Gradient of loss w.r.t. y_pred.
       hess : array of shape (n_samples,)
           Hessian of loss w.r.t. y_pred (second derivative).
       """
       grad = ...  # Compute gradient
       hess = ...  # Compute Hessian
       return grad, hess

Example: Focal Loss
-------------------

Focal loss for imbalanced classification:

.. math::

   L = -\alpha (1-p)^\gamma y \log(p) - (1-\alpha) p^\gamma (1-y) \log(1-p)

.. code-block:: python

   import numpy as np

   def focal_loss(gamma=2.0, alpha=0.25):
       """Create a focal loss objective."""
       
       def objective(y_true, y_pred):
           # Sigmoid transformation
           p = 1 / (1 + np.exp(-y_pred))
           
           # Focal weights
           pt = np.where(y_true == 1, p, 1 - p)
           focal_weight = (1 - pt) ** gamma
           
           # Gradient of cross-entropy
           grad = p - y_true
           
           # Apply focal weighting
           grad = alpha * focal_weight * grad
           
           # Hessian approximation
           hess = alpha * focal_weight * p * (1 - p)
           hess = np.maximum(hess, 1e-6)  # Numerical stability
           
           return grad, hess
       
       return objective

   # Usage
   from boosters.sklearn import GBDTClassifier
   
   model = GBDTClassifier(objective=focal_loss(gamma=2.0, alpha=0.25))
   model.fit(X_train, y_train)

Example: Huber Loss
-------------------

Smooth transition between L1 and L2:

.. code-block:: python

   import numpy as np

   def huber_loss(delta=1.0):
       """Create a Huber loss objective."""
       
       def objective(y_true, y_pred):
           residual = y_pred - y_true
           abs_residual = np.abs(residual)
           
           # L2 region (|r| < delta)
           l2_mask = abs_residual < delta
           
           # Gradient
           grad = np.where(l2_mask, residual, delta * np.sign(residual))
           
           # Hessian
           hess = np.where(l2_mask, 1.0, 0.0)
           hess = np.maximum(hess, 1e-6)  # Ensure positive
           
           return grad, hess
       
       return objective

Example: Weighted MSE
---------------------

For sample-weighted regression:

.. code-block:: python

   import numpy as np

   def weighted_mse(sample_weights):
       """MSE with sample-specific weights."""
       
       def objective(y_true, y_pred):
           residual = y_pred - y_true
           
           grad = sample_weights * residual
           hess = sample_weights.copy()
           
           return grad, hess
       
       return objective

   # Usage with higher weight on recent samples
   weights = np.exp(np.arange(len(y_train)) / len(y_train))
   model = GBDTRegressor(objective=weighted_mse(weights))

Numerical Considerations
------------------------

Hessian Stability
^^^^^^^^^^^^^^^^^

Always ensure Hessians are positive:

.. code-block:: python

   hess = np.maximum(hess, 1e-6)

Why? Negative or zero Hessians can cause numerical issues in leaf weight 
computation.

Gradient Clipping
^^^^^^^^^^^^^^^^^

For extreme predictions, clip gradients:

.. code-block:: python

   grad = np.clip(grad, -10.0, 10.0)

Numerical Overflow
^^^^^^^^^^^^^^^^^^

Watch for overflow in exponentials:

.. code-block:: python

   # Instead of exp(x) which can overflow:
   p = 1 / (1 + np.exp(-np.clip(y_pred, -500, 500)))

Custom Metrics
--------------

For evaluation and early stopping, define a custom metric:

.. code-block:: python

   def custom_metric(y_true, y_pred):
       """Compute custom evaluation metric."""
       # Return (metric_name, metric_value, higher_is_better)
       return "custom", float(np.mean((y_true - y_pred) ** 2)), False

Testing Your Objective
----------------------

Verify gradients numerically:

.. code-block:: python

   import numpy as np

   def check_gradient(objective, y_true, y_pred, eps=1e-5):
       """Verify gradients using finite differences."""
       grad, hess = objective(y_true, y_pred)
       
       # Numerical gradient
       y_pred_plus = y_pred + eps
       y_pred_minus = y_pred - eps
       
       loss_plus = compute_loss(y_true, y_pred_plus)
       loss_minus = compute_loss(y_true, y_pred_minus)
       
       numerical_grad = (loss_plus - loss_minus) / (2 * eps)
       
       print(f"Analytical: {grad[:5]}")
       print(f"Numerical:  {numerical_grad[:5]}")
       print(f"Max diff:   {np.max(np.abs(grad - numerical_grad))}")

See Also
--------

- :doc:`/explanations/objectives-metrics` — Built-in objectives
- :doc:`/explanations/gradient-boosting` — How objectives drive optimization

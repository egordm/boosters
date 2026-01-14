=================
Gradient Boosting
=================

This page explains the mathematics behind gradient boosting, the foundation of 
both GBDT (tree-based) and GBLinear (linear) models in boosters.

Overview
--------

Gradient boosting is an ensemble machine learning technique that builds a 
strong predictive model by combining many weak learners (typically decision 
trees or linear models) in an additive fashion.

The Additive Model
------------------

At iteration :math:`m`, the model prediction is:

.. math::

   F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)

where:

- :math:`F_{m-1}(x)` is the prediction from the previous iteration
- :math:`\eta` is the learning rate (shrinkage)
- :math:`h_m(x)` is the new weak learner fitted at iteration :math:`m`

The weak learner :math:`h_m` is chosen to minimize:

.. math::

   h_m = \arg\min_h \sum_i L(y_i, F_{m-1}(x_i) + h(x_i))

Second-Order Optimization
-------------------------

boosters uses second-order (Newton-Raphson) optimization, which uses both 
gradient and Hessian information for faster convergence.

For each sample :math:`i`, we compute:

- **Gradient**: :math:`g_i = \frac{\partial L(y_i, \hat{y}_i)}{\partial \hat{y}_i}`
- **Hessian**: :math:`H_i = \frac{\partial^2 L(y_i, \hat{y}_i)}{\partial \hat{y}_i^2}`

The optimal leaf weight is then:

.. math::

   w^* = -\frac{\sum_{i \in \text{leaf}} g_i}{\sum_{i \in \text{leaf}} H_i + \lambda}

where :math:`\lambda` is the L2 regularization parameter.

Why Second-Order?
^^^^^^^^^^^^^^^^^

First-order methods (gradient descent) only use gradient information. 
Second-order methods use curvature (Hessian) information to:

1. **Converge faster**: Fewer iterations needed
2. **Handle non-quadratic losses**: Better performance on logistic, quantile, etc.
3. **Natural regularization**: Hessian acts as adaptive regularization

Ensemble Visualization
----------------------

The final prediction is the sum of all weak learners:

::

   Prediction = Base Score + η·Tree₁(x) + η·Tree₂(x) + ... + η·Treeₙ(x)

Each tree corrects the residual errors of the ensemble so far.

Loss Functions
--------------

Gradient boosting works with any differentiable loss function. Common choices:

- **Squared Error** (regression): :math:`L = \frac{1}{2}(y - \hat{y})^2`
- **Logistic** (binary classification): :math:`L = -y\log(\sigma) - (1-y)\log(1-\sigma)`
- **Softmax** (multiclass): :math:`L = -\sum_k y_k \log(p_k)`

See :doc:`objectives-metrics` for the complete list with gradients and Hessians.

Regularization
--------------

Boosting is prone to overfitting. boosters provides several regularization 
mechanisms:

1. **Learning rate** (:math:`\eta`): Shrinks contribution of each tree
2. **L2 regularization** (:math:`\lambda`): Penalizes large leaf weights
3. **L1 regularization** (:math:`\alpha`): Promotes sparse leaf weights
4. **Tree constraints**: max_depth, min_child_weight, min_split_loss
5. **Subsampling**: Row and column sampling

See :doc:`hyperparameters` for tuning guidance.

Further Reading
---------------

**Foundational Papers**:

- Friedman, J.H. (2001). `"Greedy Function Approximation: A Gradient Boosting Machine" 
  <https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-function-approximation-A-gradient-boosting-machine/10.1214/aos/1013203451.full>`_
  — The original gradient boosting paper

- Chen, T. & Guestrin, C. (2016). `"XGBoost: A Scalable Tree Boosting System" 
  <https://arxiv.org/abs/1603.02754>`_
  — Introduced second-order optimization and regularization

- Ke, G. et al. (2017). `"LightGBM: A Highly Efficient Gradient Boosting Decision Tree"
  <https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree>`_
  — Histogram-based training and GOSS sampling

Next Steps
----------

- :doc:`gbdt` — Tree-based gradient boosting details
- :doc:`gblinear` — Linear gradient boosting
- :doc:`hyperparameters` — Parameter tuning guide

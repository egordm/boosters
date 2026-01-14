========
Glossary
========

.. glossary::
   :sorted:

   Bagging
      Bootstrap aggregating. Training multiple models on random subsets of 
      data and averaging predictions. Contrast with :term:`boosting`.

   Base score
      Initial prediction before any trees are added. Often the mean of the 
      target (regression) or log-odds of the positive class (classification).

   Binning
      Discretizing continuous features into bins for histogram-based 
      training. Reduces memory and speeds up split finding.

   Boosting
      Ensemble technique that trains weak learners sequentially, with each 
      learner correcting errors of the previous ones.

   Colsample
      Column (feature) subsampling. Randomly selecting a subset of features 
      for each tree or tree level.

   Coordinate descent
      Optimization algorithm that updates one variable at a time while 
      holding others fixed. Used in :term:`GBLinear`.

   Ensemble
      Collection of models whose predictions are combined. Gradient boosting 
      creates an additive ensemble.

   Feature importance
      Measure of how much a feature contributes to predictions. Often 
      computed as total gain from splits on that feature.

   GBDT
      Gradient Boosted Decision Trees. Uses decision trees as weak learners.
      See :doc:`/explanations/gbdt`.

   GBLinear
      Gradient boosting with linear models as weak learners. Produces a 
      final linear model. See :doc:`/explanations/gblinear`.

   Gradient
      First derivative of the loss function with respect to predictions. 
      Points in the direction of steepest increase.

   Hessian
      Second derivative of the loss function. Used in second-order 
      optimization to determine step size.

   Histogram
      Data structure for efficient split finding. Accumulates gradient/Hessian 
      sums per feature bin.

   Hyperparameter
      Configuration parameter set before training (e.g., learning rate, 
      max depth). Contrast with model parameters learned during training.

   L1 regularization
      Regularization that penalizes the sum of absolute parameter values. 
      Promotes sparsity. Also called Lasso.

   L2 regularization
      Regularization that penalizes the sum of squared parameter values. 
      Prevents large weights. Also called Ridge.

   Leaf
      Terminal node in a decision tree. Contains a weight (prediction value).

   Learning rate
      Shrinkage factor applied to each weak learner. Lower values require 
      more iterations but often generalize better. Symbol: η (eta).

   Loss function
      Function measuring prediction error. Also called objective function. 
      Gradient boosting minimizes the loss.

   Max depth
      Maximum depth of trees in the ensemble. Deeper trees can capture 
      more complex patterns but may overfit.

   Min child weight
      Minimum sum of Hessians required in a leaf. Regularization parameter 
      preventing splits that create too-small leaves.

   Missing value
      Absent or unknown feature value, typically represented as NaN. 
      GBDT handles these natively.

   Objective
      The loss function being optimized. Defines gradients and Hessians 
      for training.

   Overfitting
      Model learns training data too well, including noise, and performs 
      poorly on new data.

   Regularization
      Techniques to prevent overfitting by constraining model complexity. 
      Includes L1, L2, subsampling, and tree constraints.

   Shrinkage
      Reducing the contribution of each weak learner. Same as 
      :term:`learning rate`.

   Split
      Division of data at a tree node based on a feature threshold. 
      Samples go left or right based on the condition.

   Split gain
      Improvement in the objective from making a split. Used to select 
      the best split at each node.

   Subsample
      Row subsampling. Randomly selecting a subset of training samples 
      for each tree.

   Underfitting
      Model is too simple to capture patterns in the data. Both training 
      and validation metrics are poor.

   Weak learner
      Individual model in an ensemble. In gradient boosting, each tree 
      or linear model is a weak learner.

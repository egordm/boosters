==============
Classification
==============

boosters supports both binary and multiclass classification tasks.

Binary Classification
---------------------

For binary classification (two classes, 0/1), use the logistic objective:

**sklearn API:**

.. code-block:: python

   from boosters.sklearn import GBDTClassifier

   # Binary classification automatically uses logistic objective
   clf = GBDTClassifier(n_estimators=100, max_depth=6)
   clf.fit(X_train, y_train)

   # Predict class labels
   predictions = clf.predict(X_test)

   # Predict probabilities
   probabilities = clf.predict_proba(X_test)
   # Returns array of shape (n_samples, 2) for classes [0, 1]

**Core API:**

.. code-block:: python

   import boosters as bst

   config = bst.GBDTConfig(
       n_estimators=100,
       objective=bst.Objective.logistic(),
       metric=bst.Metric.auc(),
   )

   model = bst.GBDTModel.train(bst.Dataset(X_train, y_train), config=config)

   # Raw predictions (log-odds)
   raw_preds = model.predict_raw(bst.Dataset(X_test))

   # Probability predictions
   proba = model.predict(bst.Dataset(X_test))

Multiclass Classification
-------------------------

For multiclass classification (3+ classes), use the softmax objective:

**sklearn API:**

.. code-block:: python

   from boosters.sklearn import GBDTClassifier
   from boosters import Objective

   # Multiclass requires explicit objective with n_classes
   clf = GBDTClassifier(
       n_estimators=100,
       objective=Objective.softmax(n_classes=3),
   )
   clf.fit(X_train, y_train)

   # Predict class labels (0, 1, or 2)
   predictions = clf.predict(X_test)

   # Predict probabilities for each class
   probabilities = clf.predict_proba(X_test)
   # Returns array of shape (n_samples, 3)

**Core API:**

.. code-block:: python

   import boosters as bst

   config = bst.GBDTConfig(
       n_estimators=100,
       objective=bst.Objective.softmax(n_classes=3),
       metric=bst.Metric.mlogloss(),  # Multi-class log loss
   )

   model = bst.GBDTModel.train(bst.Dataset(X_train, y_train), config=config)

Evaluation Metrics
------------------

**Binary classification metrics:**

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Metric
     - Description
   * - ``Metric.auc()``
     - Area Under ROC Curve (recommended)
   * - ``Metric.logloss()``
     - Binary cross-entropy
   * - ``Metric.accuracy()``
     - Classification accuracy

**Multiclass metrics:**

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Metric
     - Description
   * - ``Metric.mlogloss()``
     - Multi-class cross-entropy
   * - ``Metric.accuracy()``
     - Classification accuracy

Class Imbalance
---------------

For imbalanced datasets, consider:

1. **Sample weights** — Weight minority class higher
2. **Subsampling** — Use ``subsample < 1.0`` to downsample majority class
3. **Threshold tuning** — Adjust decision threshold post-training

.. code-block:: python

   import numpy as np

   # Compute class weights
   class_weights = len(y_train) / (2 * np.bincount(y_train))
   sample_weights = class_weights[y_train]

   # Train with weights (sklearn API)
   clf.fit(X_train, y_train, sample_weight=sample_weights)

   # Train with weights (Core API)
   dataset = bst.Dataset(X_train, y_train, weights=sample_weights)
   model = bst.GBDTModel.train(dataset, config=config)

See Also
--------

- :doc:`/tutorials/03-classification` — Binary classification tutorial
- :doc:`/tutorials/04-multiclass` — Multiclass classification tutorial

===
FAQ
===

Frequently asked questions about boosters.

General
-------

What is boosters?
^^^^^^^^^^^^^^^^^

boosters is a high-performance gradient boosting library written in Rust with 
Python bindings. It provides efficient training and inference for both 
tree-based (GBDT) and linear (GBLinear) gradient boosting models.

How does boosters compare to XGBoost/LightGBM?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

boosters aims to be:

- **Faster** for inference (cache-efficient tree layout)
- **Simpler** (focused feature set)
- **Modern** (Rust implementation, Python-first API)

See :doc:`/explanations/benchmarks` for detailed comparisons.

Is boosters production-ready?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

boosters is under active development. The core training and inference 
functionality is stable, but the API may change before 1.0.

Installation
------------

How do I install boosters?
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pip install boosters

For development:

.. code-block:: bash

   pip install boosters[dev]

Does boosters support GPU?
^^^^^^^^^^^^^^^^^^^^^^^^^^

Not currently. boosters focuses on CPU efficiency with cache-optimized 
algorithms and SIMD acceleration.

Training
--------

My model is overfitting. What should I do?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Reduce ``max_depth`` (try 4 instead of 6)
2. Increase ``reg_lambda`` (try 5.0 or 10.0)
3. Add subsampling (``subsample=0.8``, ``colsample_bytree=0.8``)
4. Reduce ``n_estimators`` with early stopping

See :doc:`debugging-performance` for more details.

My model is underfitting. What should I do?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Increase ``max_depth`` (try 8 or 10)
2. Increase ``n_estimators``
3. Reduce regularization (``reg_lambda=0``)
4. Use a higher ``learning_rate``

Training is slow. How can I speed it up?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Reduce ``max_depth``
2. Enable subsampling
3. Reduce ``max_bin`` (try 128 instead of 256)
4. Use early stopping to avoid unnecessary iterations

Inference
---------

Predictions are slow. What can I do?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Batch predictions**: Predict multiple samples at once instead of one at a time
2. **Fewer trees**: Use early stopping to find the minimum trees needed
3. **Consider GBLinear**: If accuracy allows, linear models are much faster

Can I use boosters models in a different language?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Not yet. Model serialization is currently Python-only (pickle/joblib). 
Rust and JSON formats are planned.

Data Handling
-------------

Does boosters handle missing values?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Yes, for GBDT. Missing values (NaN) are handled natively without imputation.
See :doc:`missing-values`.

Does boosters handle categorical features?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Not natively. Use scikit-learn's ``OrdinalEncoder`` or ``OneHotEncoder``.
See :doc:`categorical-features`.

What data formats are supported?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- NumPy arrays (recommended)
- Pandas DataFrames
- Any array-like that numpy can convert

API
---

Is boosters compatible with scikit-learn?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Yes. The ``boosters.sklearn`` module provides estimators that follow the 
scikit-learn API, including:

- ``fit(X, y)`` / ``predict(X)``
- ``get_params()`` / ``set_params()``
- ``score(X, y)``
- Pipeline compatibility

How do I get feature importances?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   model.fit(X, y)
   print(model.feature_importances_)

How do I access model internals?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # For GBLinear: linear coefficients
   print(model.coef_)
   print(model.intercept_)

   # For GBDT: tree structure (advanced)
   # API still being finalized

Troubleshooting
---------------

I get a ``RuntimeError`` during training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This usually indicates a data issue:

1. Check for infinite values: ``np.isinf(X).any()``
2. Check array shapes: ``X.shape``, ``y.shape``
3. Ensure y is 1D for regression/binary classification

Memory error during training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Reduce ``max_bin`` (default 256 → 128 or 64)
2. Use subsampling (``subsample < 1.0``)
3. Train on a subset of data

Model predictions are all the same
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This usually means the model didn't learn:

1. Check that y has variance
2. Check that features have variance
3. Try a higher learning rate
4. Increase n_estimators

Still have questions?
---------------------

- Check the :doc:`/tutorials/index` for examples
- Read the :doc:`/api/index` documentation
- Open an issue on `GitHub <https://github.com/egordm/booste-rs/issues>`_

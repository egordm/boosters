==========
Benchmarks
==========

This page explains the benchmarking methodology and summarizes performance 
characteristics of boosters compared to other gradient boosting libraries.

Methodology
-----------

boosters is benchmarked against:

- **XGBoost**: The most widely used gradient boosting library
- **LightGBM**: Known for fast training on large datasets
- **scikit-learn GradientBoosting**: Reference implementation

Benchmark Categories
^^^^^^^^^^^^^^^^^^^^

1. **Training performance**: Time to fit models on standardized datasets
2. **Inference performance**: Predictions per second, latency
3. **Model quality**: Accuracy/loss on held-out test data
4. **Memory usage**: Peak memory during training and inference

Standardized Datasets
^^^^^^^^^^^^^^^^^^^^^

We use consistent dataset sizes for fair comparison:

+----------+------------+-----------------+------------------------+
| Size     | Rows       | Features        | Use Case               |
+==========+============+=================+========================+
| Small    | 5,000      | 50              | Quick iteration        |
+----------+------------+-----------------+------------------------+
| Medium   | 50,000     | 100             | Primary comparison     |
+----------+------------+-----------------+------------------------+
| Large    | 500,000    | 200             | Scaling behavior       |
+----------+------------+-----------------+------------------------+

Training Performance
--------------------

boosters achieves competitive training times through:

- **Histogram-based training**: O(bins × features) split finding
- **Cache-efficient binning**: Quantized features for CPU cache utilization
- **SIMD-accelerated histogram accumulation**: Vectorized gradient sums

Key Observations
^^^^^^^^^^^^^^^^

- Comparable to XGBoost on small-to-medium datasets
- Histogram approach scales linearly with data size
- Memory usage is predictable (binned storage)

Inference Performance
---------------------

boosters excels at inference through:

- **Structure-of-Arrays (SoA) tree layout**: Cache-friendly traversal
- **Branchless prediction**: Minimizes branch mispredictions
- **Batch processing**: Amortizes prediction overhead

Key Observations
^^^^^^^^^^^^^^^^

- Significantly faster single-row prediction than XGBoost
- Batch prediction competitive with fastest implementations
- Minimal Python overhead through efficient bindings

Running Benchmarks
------------------

Training Benchmarks
^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Unified comparison (requires bench-compare feature)
   cargo bench --features bench-compare --bench compare_training

Prediction Benchmarks
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   cargo bench --features bench-compare --bench compare_prediction

Quality Benchmarks
^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Full quality report
   uv run boosters-eval full -o docs/benchmarks/quality-report.md

   # Quick iteration
   uv run boosters-eval quick

Benchmark Reports
-----------------

Detailed benchmark reports with specific numbers are stored in 
``docs/benchmarks/`` with naming format:

::

   YYYY-MM-DD-<commit-short>-<topic>.md

See the :doc:`/design/index` for the latest benchmark reports.

Trade-offs
----------

+-----------------+------------------------------------+-------------------------+
| Aspect          | boosters                           | Trade-off               |
+=================+====================================+=========================+
| Memory          | Higher (binned storage)            | Faster training         |
+-----------------+------------------------------------+-------------------------+
| Features        | Fewer than XGBoost/LightGBM        | Simpler, focused        |
+-----------------+------------------------------------+-------------------------+
| GPU support     | Not yet                            | Focus on CPU efficiency |
+-----------------+------------------------------------+-------------------------+
| Ecosystem       | Growing                            | Modern Rust foundation  |
+-----------------+------------------------------------+-------------------------+

See Also
--------

- `docs/benchmarks/ <https://github.com/egordm/booste-rs/tree/main/docs/benchmarks>`_
  — Detailed benchmark reports
- :doc:`gradient-boosting` — How boosting works
- :doc:`hyperparameters` — Tuning for performance

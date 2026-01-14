=============
boosters-eval
=============

A framework for benchmarking gradient boosting libraries.

``boosters-eval`` makes it easy to compare Boosters against XGBoost and LightGBM 
on various datasets, ensuring fair comparisons with consistent hyperparameters.

Installation
------------

.. code-block:: bash

   # From the repository root
   pip install -e packages/boosters-eval

   # Or using uv
   uv pip install -e packages/boosters-eval

Quick Start
-----------

.. tab-set::

   .. tab-item:: CLI

      .. code-block:: bash

         # Run a quick benchmark
         boosters-eval quick

         # Compare specific libraries
         boosters-eval compare -d california -l boosters -l xgboost

         # Generate a report
         boosters-eval report -s quick -o benchmark.md

   .. tab-item:: Python

      .. code-block:: python

         from boosters_eval import compare, run_suite, QUICK_SUITE

         # Quick comparison
         results = compare(["california"], seeds=[42])
         print(results.to_markdown())

         # Run predefined suite
         results = run_suite(QUICK_SUITE)
         print(results.summary())

CLI Commands
------------

quick
~~~~~

Run a quick benchmark suite (3 seeds, 2 datasets, 50 trees):

.. code-block:: bash

   boosters-eval quick
   boosters-eval quick -o results.md

full
~~~~

Run the full benchmark suite (5 seeds, all datasets, 100 trees):

.. code-block:: bash

   boosters-eval full
   boosters-eval full -o results.md
   boosters-eval full --booster gblinear  # Test linear booster

compare
~~~~~~~

Compare specific libraries on selected datasets:

.. code-block:: bash

   # Compare all libraries on california dataset
   boosters-eval compare -d california

   # Customize comparison
   boosters-eval compare \
       -d california \
       -d breast_cancer \
       -l boosters \
       -l xgboost \
       --trees 100 \
       --seeds 5

baseline
~~~~~~~~

Record and check baselines for CI regression detection:

.. code-block:: bash

   # Record baseline
   boosters-eval baseline record -o baseline.json -s quick

   # Check against baseline
   boosters-eval baseline check baseline.json -s quick --tolerance 0.02

report
~~~~~~

Generate markdown reports with machine fingerprinting:

.. code-block:: bash

   boosters-eval report -s quick -o docs/benchmarks/report.md
   boosters-eval report -s full --title "Release 0.1.0 Benchmark"

list-*
~~~~~~

List available resources:

.. code-block:: bash

   boosters-eval list-datasets
   boosters-eval list-libraries
   boosters-eval list-tasks

Python API
----------

Custom Suites
~~~~~~~~~~~~~

.. code-block:: python

   from boosters_eval import SuiteConfig, run_suite, BoosterType

   suite = SuiteConfig(
       name="custom",
       description="My custom benchmark",
       datasets=["california", "breast_cancer"],
       n_estimators=100,
       seeds=[42, 123, 456],
       libraries=["boosters", "xgboost", "lightgbm"],
       booster_type=BoosterType.GBDT,
   )

   results = run_suite(suite)
   print(results.to_markdown())

Ablation Studies
~~~~~~~~~~~~~~~~

Compare different hyperparameter settings:

.. code-block:: python

   from boosters_eval import QUICK_SUITE, create_ablation_suite, run_suite

   # Compare different tree depths
   depth_variants = {
       "depth_4": {"max_depth": 4},
       "depth_6": {"max_depth": 6},
       "depth_8": {"max_depth": 8},
   }

   depth_suites = create_ablation_suite("depth_study", QUICK_SUITE, depth_variants)

   for suite in depth_suites:
       results = run_suite(suite)
       print(f"\n{suite.name}:")
       print(results.to_markdown())

Baseline Regression Testing
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Detect performance regressions in CI:

.. code-block:: python

   from boosters_eval import (
       record_baseline,
       load_baseline,
       check_baseline,
       run_suite,
       QUICK_SUITE,
   )
   from pathlib import Path

   # Record baseline
   results = run_suite(QUICK_SUITE)
   baseline = record_baseline(results, output_path=Path("baseline.json"))

   # Later: check for regressions
   current_results = run_suite(QUICK_SUITE)
   baseline = load_baseline(Path("baseline.json"))
   report = check_baseline(current_results, baseline, tolerance=0.02)

   if report.has_regressions:
       for reg in report.regressions:
           print(f"⚠️ Regression: {reg['config']} {reg['metric']}")

Available Datasets
------------------

.. list-table::
   :widths: 25 25 25 25
   :header-rows: 1

   * - Dataset
     - Task
     - Size
     - Features
   * - california
     - Regression
     - 20,640
     - 8
   * - breast_cancer
     - Binary Classification
     - 569
     - 30
   * - iris
     - Multiclass Classification
     - 150
     - 4
   * - synthetic_reg_*
     - Synthetic Regression
     - Various
     - Configurable
   * - synthetic_bin_*
     - Synthetic Binary
     - Various
     - Configurable

Supported Libraries
-------------------

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Library
     - Booster Types
     - Notes
   * - boosters
     - gbdt, gblinear
     - Native Rust implementation
   * - xgboost
     - gbdt, gblinear
     - Industry standard
   * - lightgbm
     - gbdt, linear_trees
     - Leaf-wise growth, histogram-based

CI Integration
--------------

Add baseline regression testing to your CI pipeline:

.. code-block:: yaml

   # .github/workflows/benchmark.yml
   name: Benchmark Regression Check

   on:
     pull_request:
       branches: [main]

   jobs:
     benchmark:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4

         - name: Set up Python
           uses: actions/setup-python@v5
           with:
             python-version: "3.11"

         - name: Install dependencies
           run: |
             pip install -e packages/boosters-eval
             pip install xgboost lightgbm

         - name: Check baseline
           run: |
             boosters-eval baseline check \
               tests/baselines/quick.json \
               -s quick \
               --tolerance 0.02

See Also
--------

- :doc:`/benchmarks/2026-01-05-42da16` — Latest benchmark results
- :doc:`/user-guide/hyperparameters` — Understanding hyperparameters
- :doc:`/research/library-comparison` — Comparison of GBDT libraries

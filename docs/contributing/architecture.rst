============
Architecture
============

Overview of the boosters codebase architecture.

Crate Structure
---------------

::

   crates/
   └── boosters/               # Core library
       ├── src/
       │   ├── data/           # Dataset, binning, histograms
       │   ├── model/          # Tree/linear model storage
       │   ├── training/       # Training algorithms
       │   ├── inference/      # Prediction
       │   ├── objective/      # Loss functions
       │   └── explainability/ # SHAP, feature importance

   packages/
   ├── boosters-python/        # PyO3 bindings
   │   ├── src/                # Rust → Python wrappers
   │   └── python/boosters/    # Pure Python code (sklearn)
   └── boosters-eval/          # Benchmarking
       └── src/boosters_eval/  # CLI and evaluation framework

Key Design Decisions
--------------------

See the :doc:`/design/rfcs` section for detailed RFCs on:

- Data representation and binning
- Tree storage (Structure-of-Arrays)
- Histogram-based training
- Parallel processing strategy

Contributing Guidelines
-----------------------

For detailed contribution guidelines, see the :doc:`/contributing/development` section.

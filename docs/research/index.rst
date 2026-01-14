========
Research
========

Deep dives into algorithms, data structures, and techniques used in gradient boosting.

Research documents are **educational** — explaining how gradient boosting algorithms 
work, their optimizations, and trade-offs. They inform implementation decisions but 
are not prescriptive to our library's specific design.

Foundations
-----------

.. toctree::
   :maxdepth: 1

   gradient-boosting

Algorithms
----------

**GBDT (Gradient Boosted Decision Trees)**

.. toctree::
   :maxdepth: 1
   :caption: GBDT Overview

   gbdt/README

.. toctree::
   :maxdepth: 1
   :caption: Training Pipeline

   gbdt/training/quantization
   gbdt/training/histogram-training
   gbdt/training/split-finding
   gbdt/training/tree-growth-strategies
   gbdt/training/sampling-strategies

.. toctree::
   :maxdepth: 1
   :caption: Inference

   gbdt/inference/batch-traversal
   gbdt/inference/multi-output

.. toctree::
   :maxdepth: 1
   :caption: Data Structures

   gbdt/data-structures/histogram-cuts
   gbdt/data-structures/quantized-matrix
   gbdt/data-structures/tree-storage

**GBLinear (Linear Gradient Boosting)**

.. toctree::
   :maxdepth: 1
   :caption: GBLinear Overview

   gblinear/README

.. toctree::
   :maxdepth: 1
   :caption: GBLinear Details

   gblinear/training/coordinate-descent
   gblinear/inference/prediction

Cross-Cutting Topics
--------------------

.. toctree::
   :maxdepth: 1

   categorical-features
   classification-objectives
   explainability
   feature-bundling
   storage-formats

Reference
---------

.. toctree::
   :maxdepth: 1

   library-comparison
   implementation-notes

Research vs RFCs
----------------

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Research
     - RFCs
   * - "How does gradient boosting work?"
     - "How will we build it?"
   * - Algorithm documentation
     - Design decisions
   * - External focus (XGBoost, LightGBM)
     - Internal focus (Boosters)
   * - Educational
     - Prescriptive
   * - Can cite academic papers
     - Should be self-contained

Primary Sources
---------------

These documents synthesize information from:

**XGBoost** — `github.com/dmlc/xgboost <https://github.com/dmlc/xgboost>`_

- Primary reference for histogram-based training
- JSON model format compatibility

**LightGBM** — `github.com/microsoft/LightGBM <https://github.com/microsoft/LightGBM>`_

- Leaf-wise growth strategy
- GOSS sampling
- Native categorical handling

**Academic Papers**

- Chen & Guestrin (2016): *XGBoost: A Scalable Tree Boosting System*
- Ke et al. (2017): *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*

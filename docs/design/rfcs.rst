=======
RFCs
=======

Boosters is developed using a Request for Comments (RFC) process. Each major 
feature or design decision is documented before implementation.

RFCs describe the "why" and "how" of implementation decisions.

.. toctree::
   :maxdepth: 1
   :caption: Core
   
   /rfcs/0001-dataset
   /rfcs/0002-trees
   /rfcs/0016-model-serialization

.. toctree::
   :maxdepth: 1
   :caption: Preprocessing
   
   /rfcs/0003-binning
   /rfcs/0004-efb
   /rfcs/0012-categoricals

.. toctree::
   :maxdepth: 1
   :caption: Training
   
   /rfcs/0005-objectives-metrics
   /rfcs/0006-sampling
   /rfcs/0007-histograms
   /rfcs/0008-gbdt-training

.. toctree::
   :maxdepth: 1
   :caption: Inference & Analysis
   
   /rfcs/0009-gbdt-inference
   /rfcs/0013-explainability

.. toctree::
   :maxdepth: 1
   :caption: Linear Models
   
   /rfcs/0010-gblinear
   /rfcs/0011-linear-leaves

.. toctree::
   :maxdepth: 1
   :caption: Python & Tooling
   
   /rfcs/0014-python-bindings
   /rfcs/0015-evaluation-framework
   /rfcs/0017-documentation


RFC Status Overview
-------------------

.. list-table::
   :widths: 10 50 20
   :header-rows: 1

   * - RFC
     - Title
     - Status
   * - :doc:`0001 </rfcs/0001-dataset>`
     - Dataset
     - ✅ Complete
   * - :doc:`0002 </rfcs/0002-trees>`
     - Trees
     - ✅ Complete
   * - :doc:`0003 </rfcs/0003-binning>`
     - Binning
     - ✅ Complete
   * - :doc:`0004 </rfcs/0004-efb>`
     - Exclusive Feature Bundling
     - ✅ Complete
   * - :doc:`0005 </rfcs/0005-objectives-metrics>`
     - Objectives & Metrics
     - ✅ Complete
   * - :doc:`0006 </rfcs/0006-sampling>`
     - Sampling
     - ✅ Complete
   * - :doc:`0007 </rfcs/0007-histograms>`
     - Histograms
     - ✅ Complete
   * - :doc:`0008 </rfcs/0008-gbdt-training>`
     - GBDT Training
     - ✅ Complete
   * - :doc:`0009 </rfcs/0009-gbdt-inference>`
     - GBDT Inference
     - ✅ Complete
   * - :doc:`0010 </rfcs/0010-gblinear>`
     - GBLinear
     - ✅ Complete
   * - :doc:`0011 </rfcs/0011-linear-leaves>`
     - Linear Leaves
     - ✅ Complete
   * - :doc:`0012 </rfcs/0012-categoricals>`
     - Categorical Features
     - ✅ Complete
   * - :doc:`0013 </rfcs/0013-explainability>`
     - Explainability
     - ✅ Complete
   * - :doc:`0014 </rfcs/0014-python-bindings>`
     - Python Bindings
     - ✅ Complete
   * - :doc:`0015 </rfcs/0015-evaluation-framework>`
     - Evaluation Framework
     - ✅ Complete
   * - :doc:`0016 </rfcs/0016-model-serialization>`
     - Model Serialization
     - ✅ Complete
   * - :doc:`0017 </rfcs/0017-documentation>`
     - Documentation
     - 🔄 In Progress


Creating an RFC
---------------

1. Copy ``docs/rfcs/TEMPLATE.md``
2. Fill in the template sections
3. Submit as a PR for review
4. Once approved, begin implementation

See the `RFC template on GitHub <https://github.com/egordm/booste-rs/blob/main/docs/rfcs/TEMPLATE.md>`_
for the full template.

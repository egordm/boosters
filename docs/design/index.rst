=================
Design Documents
=================

This section contains RFCs (Request for Comments) and design documentation 
that explains the architecture and design decisions behind boosters.

RFCs
----

RFCs document significant design decisions and their rationale.

Core Data Structures
^^^^^^^^^^^^^^^^^^^^

- :doc:`../rfcs/0001-dataset` — Dataset representation
- :doc:`../rfcs/0002-trees` — Tree storage format
- :doc:`../rfcs/0003-binning` — Feature binning

Training
^^^^^^^^

- :doc:`../rfcs/0007-histograms` — Histogram-based training
- :doc:`../rfcs/0008-gbdt-training` — GBDT training algorithm
- :doc:`../rfcs/0010-gblinear` — Linear boosting

Features
^^^^^^^^

- :doc:`../rfcs/0005-objectives-metrics` — Objectives and metrics
- :doc:`../rfcs/0006-sampling` — Row and column sampling
- :doc:`../rfcs/0012-categoricals` — Categorical feature handling
- :doc:`../rfcs/0013-explainability` — Feature importance and SHAP

Integration
^^^^^^^^^^^

- :doc:`../rfcs/0014-python-bindings` — Python API design
- :doc:`../rfcs/0016-model-serialization` — Model serialization formats
- :doc:`../rfcs/0017-documentation` — Documentation infrastructure


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: RFCs

   ../rfcs/0001-dataset
   ../rfcs/0002-trees
   ../rfcs/0003-binning
   ../rfcs/0004-efb
   ../rfcs/0005-objectives-metrics
   ../rfcs/0006-sampling
   ../rfcs/0007-histograms
   ../rfcs/0008-gbdt-training
   ../rfcs/0009-gbdt-inference
   ../rfcs/0010-gblinear
   ../rfcs/0011-linear-leaves
   ../rfcs/0012-categoricals
   ../rfcs/0013-explainability
   ../rfcs/0014-python-bindings
   ../rfcs/0015-evaluation-framework
   ../rfcs/0016-model-serialization
   ../rfcs/0017-documentation

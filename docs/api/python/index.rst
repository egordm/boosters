==========
Python API
==========

This section documents the complete Python API for boosters.

boosters provides two APIs:

- **Core API** — Full control with explicit configuration (``GBDTModel``, ``GBDTConfig``, etc.)
- **sklearn API** — Familiar sklearn-compatible estimators (``GBDTRegressor``, ``GBDTClassifier``, etc.)

See :doc:`/user-guide/choosing-api` for guidance on which to use.

Core API
--------

.. currentmodule:: boosters

The core API provides full control over model training and configuration.

**Models**

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   Dataset
   GBDTModel
   GBLinearModel

**Configuration**

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   GBDTConfig
   GBLinearConfig
   Objective
   Metric

sklearn API
-----------

.. currentmodule:: boosters.sklearn

The sklearn API provides familiar estimators that work with ``Pipeline``, 
``cross_val_score``, ``GridSearchCV``, and other sklearn utilities.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   GBDTRegressor
   GBDTClassifier
   GBLinearRegressor
   GBLinearClassifier

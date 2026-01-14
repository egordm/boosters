========
Rust API
========

The Rust API documentation is generated using rustdoc and is available at:

`Rust API Reference </rustdoc/boosters/>`_

Key Modules
-----------

- ``boosters::data`` — Dataset handling, binning, and preprocessing
- ``boosters::model`` — Model types (GBDTModel, GBLinearModel)
- ``boosters::training`` — Training algorithms and configuration
- ``boosters::inference`` — Prediction and model evaluation
- ``boosters::objective`` — Objective functions and metrics
- ``boosters::explainability`` — Feature importance and SHAP

Building Rustdoc Locally
------------------------

To build the Rust documentation locally:

.. code-block:: bash

   cargo doc --no-deps --package boosters --open

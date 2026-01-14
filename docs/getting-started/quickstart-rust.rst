===============
Rust Quickstart
===============

This guide shows how to use boosters in your Rust project.

Add Dependency
--------------

Add boosters to your ``Cargo.toml``:

.. code-block:: toml

   [dependencies]
   boosters = "0.1"

Basic Training Example
----------------------

.. code-block:: rust

   use boosters::{Dataset, GBDTModel, GBDTConfig, Objective};

   fn main() -> Result<(), Box<dyn std::error::Error>> {
       // Prepare training data (features and labels)
       let features: Vec<f32> = vec![/* your feature data */];
       let labels: Vec<f32> = vec![/* your labels */];
       
       // Create dataset
       let dataset = Dataset::from_dense(
           &features,
           n_samples,
           n_features,
           Some(&labels),
       )?;

       // Configure model
       let config = GBDTConfig::default()
           .with_n_estimators(100)
           .with_max_depth(6)
           .with_learning_rate(0.1)
           .with_objective(Objective::SquaredError);

       // Train
       let model = GBDTModel::train(&config, &dataset)?;

       // Predict
       let predictions = model.predict(&dataset)?;

       Ok(())
   }

API Documentation
-----------------

For complete Rust API documentation, see the `rustdoc </rustdoc/boosters/>`_.

Key Modules
^^^^^^^^^^^

- ``boosters::data`` — Dataset handling and preprocessing
- ``boosters::model`` — Model types (GBDT, GBLinear)
- ``boosters::config`` — Configuration and hyperparameters
- ``boosters::objective`` — Objective functions

Feature Flags
-------------

boosters provides several feature flags:

.. code-block:: toml

   [dependencies]
   boosters = { version = "0.1", features = ["serde"] }

- ``serde`` — Enable serialization support
- ``rayon`` — Enable parallel processing (enabled by default)

Next Steps
----------

- `Rust API Reference </rustdoc/boosters/>`_ — Complete rustdoc documentation
- :doc:`/explanations/gbdt` — How GBDT works

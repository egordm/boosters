//! High-level model wrappers.
//!
//! This module provides user-friendly model types that combine training,
//! prediction, and serialization into a cohesive API.
//!
//! # Overview
//!
//! - [`GBDTModel`]: Tree ensemble model (GBDT/GBRT)
//! - [`GBLinearModel`]: Linear booster model
//! - [`ModelMeta`]: Shared metadata for all model types
//!
//! # Example
//!
//! ```ignore
//! use boosters::model::{GBDTModel, TaskKind};
//! use boosters::model::gbdt::GBDTConfig;
//! use boosters::training::{Objective, Metric};
//!
//! // Train a model
//! let config = GBDTConfig::builder()
//!     .objective(Objective::squared_error())
//!     .metric(Metric::rmse())
//!     .n_trees(50)
//!     .learning_rate(0.1)
//!     .build()
//!     .unwrap();
//! let model = GBDTModel::train(&data, &labels, &[], config)?;
//!
//! // Make predictions
//! let predictions = model.predict_batch(&new_data, n_rows);
//!
//! // Save and load
//! model.save("model.bstr")?;
//! let loaded = GBDTModel::load("model.bstr")?;
//! ```

mod meta;
pub mod gbdt;
pub mod gblinear;

pub use meta::{ModelMeta, TaskKind, FeatureType};
pub use gbdt::GBDTModel;
pub use gblinear::GBLinearModel;

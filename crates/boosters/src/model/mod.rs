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
//! use boosters::training::GBDTParams;
//! use boosters::training::ObjectiveFunction;
//!
//! // Train a model
//! let params = GBDTParams {
//!     n_trees: 50,
//!     ..Default::default()
//! };
//! let model = GBDTModel::train(&data, ObjectiveFunction::SquaredError, params)?;
//!
//! // Make predictions
//! let predictions = model.predict(&new_data);
//!
//! // Save and load
//! model.save("model.bstr")?;
//! let loaded = GBDTModel::load("model.bstr")?;
//! ```

mod meta;
mod gbdt;
mod gblinear;

pub use meta::{ModelMeta, TaskKind, FeatureType};
pub use gbdt::GBDTModel;
pub use gblinear::GBLinearModel;

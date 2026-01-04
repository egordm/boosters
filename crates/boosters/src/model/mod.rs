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
//! # Prediction API
//!
//! Both model types provide two batch prediction methods:
//! - [`predict()`](GBDTModel::predict) - Returns transformed predictions
//!   (probabilities for classification, values for regression)
//! - [`predict_raw()`](GBDTModel::predict_raw) - Returns raw margin scores
//!   (no sigmoid/softmax transformation)
//!
//! These methods accept any [`DataMatrix`](crate::data::DataMatrix) and return
//! a [`ColMatrix<f32>`](crate::data::ColMatrix).
//!
//! # Example
//!
//! ```ignore
//! use boosters::model::GBDTModel;
//! use boosters::model::gbdt::GBDTConfig;
//! use boosters::data::Dataset;
//! use boosters::training::{Objective, Metric};
//!
//! // Train a model
//! let config = GBDTConfig::builder()
//!     .objective(Objective::squared_error())
//!     .metric(Metric::Rmse)
//!     .n_trees(50)
//!     .learning_rate(0.1)
//!     .build()
//!     .unwrap();
//! let model = GBDTModel::train(&dataset, None, config, 0)?;
//!
//! // Make predictions
//! let predictions = model.predict(&dataset, 0);
//! ```

pub mod gbdt;
pub mod gblinear;
mod meta;
mod transform;

pub use gbdt::GBDTModel;
pub use gblinear::GBLinearModel;
pub use meta::{FeatureType, ModelMeta};
pub use transform::OutputTransform;

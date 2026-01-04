//! boosters: A gradient boosting library for Rust.
//!
//! Native Rust implementations for gradient boosted decision trees,
//! with support for loading and saving models in native format, and
//! converting from XGBoost/LightGBM via Python utilities.
//!
//! # Key Types
//!
//! - [`GBDTModel`] / [`GBLinearModel`] - High-level models with train/predict
//! - [`GBDTConfig`] / [`GBLinearConfig`] - Configuration builders
//! - [`Objective`] / [`Metric`] - Training objectives and evaluation metrics
//! - [`Dataset`] - Data handling
//!
//! # Training
//!
//! Use `GBDTConfig::builder()` to configure, then `GBDTModel::train()`.
//! See the [`model`] module for details.
//!
//! # Model Serialization
//!
//! Use [`persist`] module to save/load models in native binary or JSON format.
//! For converting XGBoost/LightGBM models, use the Python utilities in `boosters.convert`.

pub mod data;
pub mod explainability;
pub mod inference;
pub mod model;
#[cfg(feature = "persist")]
pub mod persist;
pub mod repr;
pub mod testing;
pub mod training;
pub mod utils;

// =============================================================================
// Convenience Re-exports
// =============================================================================

// High-level model types
pub use model::{GBDTModel, GBLinearModel, ModelMeta, OutputTransform};

// Configuration types (most users want these)
pub use model::gbdt::GBDTConfig;
pub use model::gblinear::GBLinearConfig;

// Training types (objectives, metrics)
pub use training::{Metric, Objective};

// Data types (for preparing training data)
pub use data::{
    Dataset, DatasetBuilder, DatasetError, DatasetSchema, Feature, FeatureMeta, FeatureType,
    SamplesView, TargetsView, WeightsView,
};

// Shared utilities
pub use utils::{Parallelism, run_with_threads};

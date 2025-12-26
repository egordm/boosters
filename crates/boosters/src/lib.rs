//! boosters: A gradient boosting library for Rust.
//!
//! Native Rust implementations for gradient boosted decision trees,
//! with support for loading models from XGBoost and LightGBM.
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
//! # Loading XGBoost Models
//!
//! Use [`compat::xgboost::XgbModel`] to load JSON models.
//! See the [`compat`] module for details.

// Re-export approx traits for users who want to compare predictions
pub use approx;

pub mod compat;
pub mod data;
pub mod explainability;
pub mod inference;
pub mod model;
pub mod repr;
pub mod testing;
pub mod training;
pub mod utils;

// =============================================================================
// Convenience Re-exports
// =============================================================================

// High-level model types
pub use model::{GBDTModel, GBLinearModel, ModelMeta, TaskKind};

// Configuration types (most users want these)
pub use model::gbdt::GBDTConfig;
pub use model::gblinear::GBLinearConfig;

// Training types (objectives, metrics)
pub use training::{Metric, MetricFn, Objective, ObjectiveFn};

// Data types (for preparing training data)
pub use data::{
    Column, Dataset, DatasetBuilder, DatasetError, DatasetSchema, FeatureMeta, FeatureType,
    FeaturesView, SparseColumn, TargetsView, WeightsView,
};

// Shared utilities
pub use utils::{Parallelism, run_with_threads};

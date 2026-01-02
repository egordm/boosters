//! Raw dataset types for features, targets, and weights.
//!
//! This module provides the user-facing data containers:
//!
//! - [`Dataset`]: Main container for features, targets, and weights
//! - [`DatasetBuilder`]: Fluent builder for dataset construction
//! - [`SamplesView`] / [`TargetsView`] / [`WeightsView`]: Read-only views
//!
//! # Storage Layout
//!
//! Features are stored in **feature-major** layout: `[n_features, n_samples]`.
//! This is optimal for training (histogram building, coordinate descent).
//!
//! # Sample-Major Access
//!
//! For prediction and SHAP, use [`Dataset::buffer_samples`] to fill a
//! sample-major buffer. Callers manage their own buffers and use
//! [`Parallelism::maybe_par_for_each_init`] for parallel processing
//! with per-thread buffer reuse.
//!
//! # Usage Note
//!
//! For training, use [`super::binned::BinnedDataset::from_dataset`] to create
//! a binned dataset. The binned dataset handles feature quantization,
//! bundling, and optimization automatically.

#![allow(clippy::all)] // Legacy code - avoid churn from clippy updates
#![allow(dead_code)] // Some fields used conditionally
#![allow(unused_imports)] // Re-exports used by parent module

pub mod dataset;
pub mod feature;
pub mod schema;
pub mod views;

// Re-export public types
pub use dataset::{Dataset, DatasetBuilder};
pub use feature::Feature;
pub use schema::{DatasetSchema, FeatureMeta, FeatureType};
pub use views::{SamplesView, TargetsView, WeightsIter, WeightsView};

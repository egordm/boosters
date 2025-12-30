//! Raw dataset types for features, targets, and weights.
//!
//! This module provides the user-facing data containers:
//!
//! - [`Dataset`]: Main container for features, targets, and weights
//! - [`DatasetBuilder`]: Fluent builder for dataset construction
//! - [`FeaturesView`] / [`TargetsView`] / [`WeightsView`]: Read-only views
//! - [`SampleBlocks`]: Efficient block-based iteration for prediction
//!
//! # Storage Layout
//!
//! Features are stored in **feature-major** layout: `[n_features, n_samples]`.
//! This is optimal for training (histogram building, coordinate descent).
//!
//! # Usage Note
//!
//! For training, use [`super::binned::BinnedDataset::from_dataset`] to create
//! a binned dataset. The binned dataset handles feature quantization,
//! bundling, and optimization automatically.

#![allow(clippy::all)] // Legacy code - avoid churn from clippy updates
#![allow(dead_code)] // Some fields used conditionally
#![allow(unused_imports)] // Re-exports used by parent module

pub mod accessor;
pub mod dataset;
pub mod feature;
pub mod sample_blocks;
pub mod schema;
pub mod views;

// Re-export public types
pub use accessor::{DataAccessor, SampleAccessor};
pub use dataset::{Dataset, DatasetBuilder};
pub use feature::Feature;
pub use sample_blocks::{SampleBlocks, SampleBlocksIter};
pub use schema::{DatasetSchema, FeatureMeta, FeatureType};
pub use views::{FeaturesView, SamplesView, TargetsView, WeightsIter, WeightsView};

// Legacy compatibility aliases (will be removed in Epic 6)
#[allow(deprecated)]
pub use feature::Column;

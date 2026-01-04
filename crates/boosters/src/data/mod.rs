//! Data structures for feature matrices and datasets.
//!
//! This module provides the data layer for boosters, including:
//!
//! # User-Facing Types
//!
//! - [`Dataset`]: Main container for features, targets, and weights
//! - [`DatasetBuilder`]: Fluent builder for complex dataset construction
//! - [`SamplesView`] / [`TargetsView`] / [`WeightsView`]: Read-only views
//!
//! # Training-Specific Types
//!
//! - [`binned::BinnedDataset`]: Quantized feature data for histogram-based GBDT
//! - [`BinningConfig`]: Configuration for feature quantization
//!
//! # Sample-Major Access
//!
//! Use [`Dataset::buffer_samples`] to fill a sample-major buffer for prediction.
//! Callers manage their own buffers and use [`crate::Parallelism::maybe_par_for_each_init`]
//! for parallel processing with per-thread buffer reuse.
//!
//! # Storage Layout
//!
//! Features are stored in **feature-major** layout: `[n_features, n_samples]`.
//! This is optimal for training (histogram building, coordinate descent).
//!
//! # Missing Values
//!
//! Missing values are represented as `f32::NAN`.

pub mod binned;
mod error;
mod ndarray;

// Raw data types (Dataset, views, accessors, schema, sample_blocks)
pub(crate) mod raw;

// =============================================================================
// Core Dataset Types (user-facing)
// =============================================================================

pub use error::DatasetError;
pub use raw::dataset::{Dataset, DatasetBuilder};
pub use raw::feature::Feature;
pub use raw::schema::{DatasetSchema, FeatureMeta, FeatureType};
pub use raw::views::{SamplesView, TargetsView, WeightsIter, WeightsView};

// =============================================================================
// ndarray Utilities
// =============================================================================

pub use ndarray::{axis, init_predictions, init_predictions_into, transpose_to_c_order};

// =============================================================================
// Binned Data Types (re-exports for convenience)
// =============================================================================

pub use binned::{
    BinMapper, BinnedDataset, BinnedFeatureColumnSource, BinnedFeatureInfo, BinnedFeatureViews,
    BinningConfig, BuildError, FeatureGroup, FeatureMetadata, FeatureView, MissingType,
};

// Internal types for tests/benchmarks
#[doc(hidden)]
pub use binned::GroupSpec;

//! Data structures for feature matrices and datasets.
//!
//! This module provides the data layer for boosters, including:
//!
//! # User-Facing Types
//!
//! - [`Dataset`]: Main container for features, targets, and weights
//! - [`DatasetBuilder`]: Fluent builder for complex dataset construction
//! - [`FeaturesView`] / [`TargetsView`] / [`WeightsView`]: Read-only views
//!
//! # Training-Specific Types
//!
//! - [`binned::BinnedDataset`]: Quantized feature data for histogram-based GBDT
//! - [`BinningConfig`]: Configuration for feature quantization
//!
//! # Internal Types
//!
//! - [`SampleAccessor`]: Access features for a single sample (row)
//! - [`DataAccessor`]: Access samples from a dataset (matrix)
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
mod accessor;
mod column;
mod dataset;
mod error;
mod ndarray;
mod schema;
mod views;

#[cfg(feature = "io-parquet")]
pub mod io;

// =============================================================================
// Core Dataset Types (user-facing)
// =============================================================================

pub use column::{Column, SparseColumn};
pub use dataset::{Dataset, DatasetBuilder};
pub use error::DatasetError;
pub use schema::{DatasetSchema, FeatureMeta, FeatureType};
pub use views::{FeaturesView, SamplesView, TargetsView, WeightsIter, WeightsView};

// =============================================================================
// Accessor Traits (internal)
// =============================================================================

pub use accessor::{DataAccessor, SampleAccessor};

// =============================================================================
// ndarray Utilities
// =============================================================================

pub use ndarray::{
    axis, init_predictions, init_predictions_into, transpose_to_c_order,
};

// =============================================================================
// Binned Data Types (re-exports for convenience)
// =============================================================================

pub use binned::{
    BinMapper, BinStorage, BinType, BinnedDataset, BinnedDatasetBuilder, BinnedFeatureMeta,
    BinningConfig, BinningStrategy, BuildError, FeatureGroup, FeatureView,
    GroupLayout, MissingType, BinnedSample,
};

// Internal types for tests/benchmarks
#[doc(hidden)]
pub use binned::{GroupSpec, GroupStrategy};


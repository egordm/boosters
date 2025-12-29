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
mod error;
mod ndarray;

// Deprecated module containing old implementation during RFC-0018 migration
// All old data types (Dataset, Column, accessor, schema, views) live here
#[allow(deprecated)]
pub(crate) mod deprecated;

#[cfg(feature = "io-parquet")]
pub mod io;

// =============================================================================
// Core Dataset Types (user-facing) - re-exported from deprecated
// =============================================================================

#[allow(deprecated)]
pub use deprecated::column::{Column, SparseColumn};
#[allow(deprecated)]
pub use deprecated::dataset::{Dataset, DatasetBuilder};
pub use error::DatasetError;
#[allow(deprecated)]
pub use deprecated::schema::{DatasetSchema, FeatureMeta, FeatureType};
#[allow(deprecated)]
pub use deprecated::views::{FeaturesView, SamplesView, TargetsView, WeightsIter, WeightsView};

// =============================================================================
// Accessor Traits (internal) - re-exported from deprecated
// =============================================================================

#[allow(deprecated)]
pub use deprecated::accessor::{DataAccessor, SampleAccessor};

// =============================================================================
// ndarray Utilities
// =============================================================================

pub use ndarray::{axis, init_predictions, init_predictions_into, transpose_to_c_order};

// =============================================================================
// Binned Data Types (re-exports for convenience)
// =============================================================================

pub use binned::{
    BinMapper, BinStorage, BinType, BinnedDataset, BinnedDatasetBuilder, BinnedFeatureInfo,
    BinnedSampleView, BinningConfig, BinningStrategy, BuildError, FeatureGroup, FeatureMetadata,
    FeatureView, MissingType,
};

// Internal types for tests/benchmarks
#[doc(hidden)]
pub use binned::{GroupSpec, GroupStrategy};

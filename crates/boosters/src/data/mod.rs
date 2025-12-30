//! Data structures for feature matrices and datasets.
//!
//! This module provides the data layer for boosters, including:
//!
//! # User-Facing Types
//!
//! - [`Dataset`]: Main container for features, targets, and weights
//! - [`DatasetBuilder`]: Fluent builder for complex dataset construction
//! - [`FeaturesView`] / [`TargetsView`] / [`WeightsView`]: Read-only views
//! - [`SampleBlocks`]: Efficient block-based iteration for prediction
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

// Raw data types (Dataset, views, accessors, schema, sample_blocks)
pub(crate) mod raw;

// =============================================================================
// Core Dataset Types (user-facing)
// =============================================================================

pub use raw::feature::Feature;
pub use raw::dataset::{Dataset, DatasetBuilder};
pub use error::DatasetError;
pub use raw::schema::{DatasetSchema, FeatureMeta, FeatureType};
pub use raw::views::{FeaturesView, SamplesView, TargetsView, WeightsIter, WeightsView};
pub use raw::sample_blocks::{SampleBlocks, SampleBlocksIter};

// Legacy compatibility aliases (will be removed in Epic 6)
#[allow(deprecated)]
pub use raw::feature::Column;

// =============================================================================
// Accessor Traits (internal)
// =============================================================================

pub use raw::accessor::{DataAccessor, SampleAccessor};

// =============================================================================
// ndarray Utilities
// =============================================================================

pub use ndarray::{axis, init_predictions, init_predictions_into, transpose_to_c_order};

// =============================================================================
// Binned Data Types (re-exports for convenience)
// =============================================================================

pub use binned::{
    BinMapper, BinnedDataset, BinnedDatasetBuilder, BinnedFeatureInfo,
    BinnedSampleView, BinningConfig, BuildError, EffectiveViews, FeatureGroup,
    FeatureMetadata, FeatureView, MissingType,
};

// Internal types for tests/benchmarks
#[doc(hidden)]
pub use binned::GroupSpec;

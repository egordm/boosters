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

// Core data types (Dataset, views, accessors, schema)
pub(crate) mod types;

// =============================================================================
// Core Dataset Types (user-facing)
// =============================================================================

pub use types::column::{Column, SparseColumn};
pub use types::dataset::{Dataset, DatasetBuilder};
pub use error::DatasetError;
pub use types::schema::{DatasetSchema, FeatureMeta, FeatureType};
pub use types::views::{FeaturesView, SamplesView, TargetsView, WeightsIter, WeightsView};

// =============================================================================
// Accessor Traits (internal)
// =============================================================================

pub use types::accessor::{DataAccessor, SampleAccessor};

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

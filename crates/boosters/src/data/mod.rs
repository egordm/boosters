//! Data input abstractions for feature matrices.
//!
//! This module provides data types for accessing feature data during
//! tree traversal and training.
//!
//! # Overview
//!
//! The core abstraction is [`FeatureAccessor`], which provides a uniform interface
//! for accessing feature values regardless of the underlying storage format.
//!
//! # Storage Types
//!
//! - [`SamplesView`]: Sample-major view `[n_samples, n_features]` - samples on rows
//! - [`FeaturesView`]: Feature-major view `[n_features, n_samples]` - features on rows
//! - [`binned::BinnedDataset`]: Quantized feature data for GBDT training
//!
//! # ndarray Integration
//!
//! This module works with ndarray arrays directly. The wrapper types ([`SamplesView`],
//! [`FeaturesView`]) provide semantic clarity about which axis represents what.
//!
//! # Missing Values
//!
//! Missing values are represented as `f32::NAN`. This is the modern standard
//! used by XGBoost and other libraries.
//!
//! See RFC-0004 for design rationale, RFC-0021 for ndarray migration.

pub mod binned;
mod dataset;
mod traits;
mod ndarray;

#[cfg(feature = "io-parquet")]
pub mod io;

pub use dataset::{Dataset, DatasetError, FeatureColumn};
pub use traits::FeatureAccessor;

pub use ndarray::{
    axis, init_predictions, init_predictions_vec, transpose_to_c_order,
    FeaturesView, SamplesView,
};

// Re-export binned types for convenience
pub use binned::{
    BinMapper, BinStorage, BinType, BinnedDataset, BinnedDatasetBuilder, BinningConfig,
    BinningStrategy, BuildError, FeatureGroup, FeatureMeta, FeatureType, FeatureView,
    GroupLayout, GroupSpec, GroupStrategy, MissingType, RowView as BinnedRowView,
};

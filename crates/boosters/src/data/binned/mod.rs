//! BinnedDataset - Feature group-based quantized data for GBDT training.
//!
//! This module contains the RFC-0018 implementation for binned datasets
//! with raw feature storage for linear leaf fitting.

// RFC-0018 implementation modules
mod bin_data;
mod bin_mapper;
pub(crate) mod builder;
pub(crate) mod bundling;
pub(crate) mod dataset;
pub(crate) mod feature_analysis;
pub(crate) mod group;
mod sample_blocks;
mod storage;
pub(crate) mod view;

// =============================================================================
// Public API (RFC-0018 types)
// =============================================================================

// Core types
pub use bin_data::BinData;
pub use bin_mapper::{BinMapper, FeatureType, MissingType};
pub use dataset::{BinnedDataset, BinnedFeatureInfo, BinnedSampleView, EffectiveViews, FeatureLocation};
pub use group::FeatureGroup;
pub use storage::{
    BundleStorage, CategoricalStorage, FeatureStorage, NumericStorage, SparseCategoricalStorage,
    SparseNumericStorage,
};
pub use view::FeatureView;

// Builder types
pub use builder::{BuiltGroups, DatasetBuilder, DatasetError};
pub use feature_analysis::{BinningConfig, FeatureAnalysis, FeatureMetadata, GroupSpec};

// Bundling types (RFC-0018 native)
pub use bundling::{BundlePlan as NewBundlePlan, BundlingConfig as NewBundlingConfig};

// For backward compatibility during migration, re-export some builder types with old names
pub use builder::DatasetBuilder as BinnedDatasetBuilder;
pub use builder::DatasetError as BuildError;

// =============================================================================
// Deprecated re-exports (bundling and legacy storage)
// These will be removed once all consumers migrate away from bundling
// =============================================================================

// Bundling types - still needed for deprecated BinnedDataset consumers
// Note: New BinnedDataset does NOT support bundling yet
#[allow(deprecated)]
pub use super::deprecated::binned::{
    BinStorage, BinType, BundlePlan, BundlingConfig, BundlingFeatures, BundlingStats,
    FeatureBundle, GroupStrategy,
};

// Legacy builder types for backward compat
#[allow(deprecated)]
pub use super::deprecated::binned::BinningStrategy;


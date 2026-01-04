//! BinnedDataset - Feature group-based quantized data for GBDT training.
//!
//! This module contains internal binned (quantized) data structures used for
//! histogram-based training.

// RFC-0018 implementation modules
mod bin_data;
mod bin_mapper;
pub(crate) mod bundling;
pub(crate) mod dataset;
pub(crate) mod feature_analysis;
pub(crate) mod group;
mod storage;
pub(crate) mod view;

// =============================================================================
// Public API (RFC-0018 types)
// =============================================================================

// Core types
pub use bin_data::BinData;
pub use bin_mapper::{BinMapper, FeatureType, MissingType};
pub use dataset::{
    BinnedDataset, BinnedFeatureColumnSource, BinnedFeatureInfo, BinnedFeatureViews,
    FeatureLocation,
};
pub use feature_analysis::{BinningConfig, FeatureAnalysis, FeatureMetadata, GroupSpec};
pub use group::FeatureGroup;
pub use storage::{
    BundleStorage, CategoricalStorage, FeatureStorage, NumericStorage, SparseCategoricalStorage,
    SparseNumericStorage,
};
pub use view::FeatureView;

// Bundling types (RFC-0018 native)
pub use bundling::{BundlePlan, BundlingConfig};

// Error type re-export
pub use dataset::DatasetError;
pub use dataset::DatasetError as BuildError;

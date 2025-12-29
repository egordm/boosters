//! BinnedDataset - Feature group-based quantized data for GBDT training.
//!
//! This module contains the new implementation following RFC-0018.
//! Deprecated types are re-exported for backward compatibility during migration.

// New RFC-0018 implementation
mod bin_data;
mod bin_mapper;
mod builder;
mod dataset;
mod feature_analysis;
pub(crate) mod group;
mod storage;
pub(crate) mod view;

// Public exports for new types
// Note: FeatureGroup and FeatureView are NOT exported publicly yet to avoid
// conflict with deprecated types. They will be exported when we switch over in Epic 7.
// Note: BinMapper and FeatureAnalysis are also NOT exported yet - deprecated versions
// are still used via re-export. New types will be used by the new builder.
pub use bin_data::BinData;
pub use storage::{
    CategoricalStorage, FeatureStorage, NumericStorage, SparseCategoricalStorage,
    SparseNumericStorage,
};

// Re-export everything from deprecated for backward compatibility
#[allow(deprecated)]
pub use super::deprecated::binned::*;


//! BinnedDataset - Feature group-based quantized data for GBDT training.
//!
//! This module contains the new implementation following RFC-0018.
//! Deprecated types are re-exported for backward compatibility during migration.

// New RFC-0018 implementation
mod bin_data;
mod bin_mapper;
pub(crate) mod group;
mod storage;
pub(crate) mod view;

// Public exports for new types
// Note: FeatureGroup and FeatureView are NOT exported publicly yet to avoid
// conflict with deprecated types. They will be exported when we switch over in Epic 7.
// Note: BinMapper is also NOT exported yet - deprecated BinMapper is still used via re-export.
pub use bin_data::BinData;
pub use storage::{
    CategoricalStorage, FeatureStorage, NumericStorage, SparseCategoricalStorage,
    SparseNumericStorage,
};

// Re-export everything from deprecated for backward compatibility
#[allow(deprecated)]
pub use super::deprecated::binned::*;

//! BinnedDataset - Feature group-based quantized data for GBDT training.
//!
//! This module contains the new implementation following RFC-0018.
//! Deprecated types are re-exported for backward compatibility during migration.

// New RFC-0018 implementation
mod bin_data;
pub(crate) mod group;
mod storage;

// Public exports for new types
// Note: FeatureGroup is NOT exported publicly yet to avoid conflict with deprecated.
// It will be exported when we switch over in Epic 7.
pub use bin_data::BinData;
pub use storage::{
    CategoricalStorage, FeatureStorage, NumericStorage, SparseCategoricalStorage,
    SparseNumericStorage,
};

// Re-export everything from deprecated for backward compatibility
#[allow(deprecated)]
pub use super::deprecated::binned::*;

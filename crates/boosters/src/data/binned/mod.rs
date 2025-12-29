//! BinnedDataset - Feature group-based quantized data for GBDT training.
//!
//! This module contains the new implementation following RFC-0018.
//! Deprecated types are re-exported for backward compatibility during migration.

// New RFC-0018 implementation
mod bin_data;
mod storage;

// Public exports for new types
pub use bin_data::BinData;
pub use storage::{
    CategoricalStorage, NumericStorage, SparseCategoricalStorage, SparseNumericStorage,
};

// Re-export everything from deprecated for backward compatibility
#[allow(deprecated)]
pub use super::deprecated::binned::*;

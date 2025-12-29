//! BinnedDataset - Feature group-based quantized data for GBDT training.
//!
//! This module re-exports from the deprecated implementation during migration.
//! New implementation will be added here following RFC-0018.

// Re-export everything from deprecated for backward compatibility
#[allow(deprecated)]
pub use super::deprecated::binned::*;

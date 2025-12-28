//! v2 Storage Types - Clean Implementation for RFC-0018
//!
//! This module contains the new typed storage hierarchy for BinnedDataset:
//!
//! - `BinData`: Replaces `BinType` - typed bin storage (U8/U16)
//! - `NumericStorage`: Dense storage with raw values for numeric features
//! - `CategoricalStorage`: Dense storage for categorical features (no raw values)
//! - `SparseNumericStorage`: Sparse storage with raw values
//! - `SparseCategoricalStorage`: Sparse storage for categorical features
//! - `BundleStorage`: EFB bundle storage with decode logic
//! - `FeatureStorage`: Unified enum wrapping all storage types
//!
//! # Design Principles
//!
//! 1. **Type Safety**: Storage types encode whether raw values are available
//! 2. **Homogeneous Groups**: Feature groups contain only one storage type
//! 3. **Column-Major Only**: All dense storage is column-major (no stride)
//! 4. **Minimal API**: Only essential methods, no deprecated cruft
//!
//! # Migration Path
//!
//! Old types in parent module are deprecated. New code should use these types.
//! After migration is complete, old types will be removed.
//!
//! ```ignore
//! // Old (deprecated)
//! use boosters::data::binned::{BinType, BinStorage};
//!
//! // New
//! use boosters::data::binned::v2::{BinData, FeatureStorage};
//! ```

mod bin_data;
mod storage;

pub use bin_data::BinData;
pub use storage::{
    CategoricalStorage, FeatureStorage, NumericStorage, SparseCategoricalStorage,
    SparseNumericStorage,
};

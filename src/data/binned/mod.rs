//! BinnedDataset - Feature group-based quantized data for GBDT training.
//!
//! This module provides a flexible dataset structure where features are organized
//! into groups with different storage layouts (row-major or column-major) and
//! bin types (u8, u16, u32).
//!
//! # Architecture
//!
//! ```text
//! BinnedDataset
//! ├── FeatureGroup 0 (dense numeric, row-major, u8)
//! │   ├── Feature 0
//! │   ├── Feature 1
//! │   └── Feature 2
//! ├── FeatureGroup 1 (sparse, column-major, u8)
//! │   └── Feature 3
//! └── FeatureGroup 2 (wide features, column-major, u16)
//!     └── Feature 4
//! ```
//!
//! # Design Principles
//!
//! - **No labels/weights**: Pass separately to training functions
//! - **Direct storage access**: Match on `BinStorage` enum for typed access
//! - **Per-group layout**: Row-major for dense (row-parallel), column-major for sparse
//! - **Exhaustive matching**: Storage enum forces handling all types

mod bin_mapper;
mod storage;
mod group;
mod dataset;
mod builder;

pub use bin_mapper::{BinMapper, FeatureType, MissingType};
pub use storage::{FeatureView, BinStorage, BinType, GroupLayout};
pub use group::{FeatureGroup, FeatureMeta};
pub use dataset::{BinnedDataset, RowView};
pub use builder::{BinnedDatasetBuilder, GroupStrategy, GroupSpec, BuildError, BinningConfig};

//! BinnedDataset - Feature group-based quantized data for GBDT training.
//!
//! This module provides a flexible dataset structure where features are organized
//! into groups for efficient histogram building during GBDT training.
//!
//! # Usage
//!
//! ```ignore
//! use boosters::data::{BinnedDatasetBuilder, BinningConfig};
//!
//! let binned = BinnedDatasetBuilder::new(config)
//!     .add_features(dataset.features(), Parallelism::Parallel)
//!     .build()?;
//! ```
//!
//! # Architecture
//!
//! Features are automatically organized into groups with optimal storage:
//! - Dense numeric features: column-major u8 for fast histogram building
//! - Wide features (>256 bins): column-major u16
//! - Sparse features: column-major with optional bundling

mod bin_mapper;
mod storage;
mod group;
mod dataset;
mod builder;
mod feature_analysis;
mod bundling;

pub use bin_mapper::{BinMapper, FeatureType, MissingType};
pub use storage::{FeatureView, BinStorage, BinType, GroupLayout};
pub use group::{FeatureGroup, BinnedFeatureMeta};
pub use dataset::{BinnedDataset, BundlingStats, BinnedSample};
pub use builder::{BinnedDatasetBuilder, BuildError, BinningConfig, BinningStrategy};
pub use feature_analysis::{FeatureInfo, analyze_features, analyze_features_sequential};
pub use bundling::BundlingFeatures;
pub use bundling::{BundlingConfig, FeatureBundle, FeatureLocation, BundlePlan, create_bundle_plan};

// Internal types exposed for tests and benchmarks
#[doc(hidden)]
pub use builder::{GroupStrategy, GroupSpec};

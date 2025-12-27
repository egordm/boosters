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
mod builder;
mod bundling;
mod dataset;
mod feature_analysis;
mod group;
mod storage;

pub use bin_mapper::{BinMapper, FeatureType, MissingType};
pub use builder::{BinnedDatasetBuilder, BinningConfig, BinningStrategy, BuildError};
pub use bundling::BundlingFeatures;
pub use bundling::{
    BundleDecoder, BundleHistogramCache, BundlePlan, BundlingConfig, FeatureBundle,
    FeatureLocation, SubFeatureInfo, create_bundle_plan,
};
pub use dataset::{BinnedDataset, BinnedSample, BundlingStats};
pub use feature_analysis::{FeatureInfo, analyze_features, analyze_features_sequential};
pub use group::{BinnedFeatureInfo, FeatureGroup};
pub use storage::{BinStorage, BinType, FeatureView, GroupLayout};

// Internal types exposed for tests and benchmarks
#[doc(hidden)]
pub use builder::{GroupSpec, GroupStrategy};

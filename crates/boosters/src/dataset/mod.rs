//! Unified Dataset type for gradient boosting models.
//!
//! This module provides [`Dataset`], the standard input container for all boosters
//! model types (GBDT, GBLinear). It replaces the old `data::Dataset` with a cleaner
//! design based on RFC-0019.
//!
//! # Key Types
//!
//! - [`Dataset`]: Main container holding features, targets, and weights
//! - [`DatasetBuilder`]: Fluent builder for complex dataset construction
//! - [`Column`]: Dense or sparse feature column
//! - [`DatasetSchema`]: Feature metadata (names, types)
//! - [`FeaturesView`] / [`TargetsView`]: Read-only views for algorithms
//!
//! # Storage Layout
//!
//! Features are stored in **feature-major** layout: `[n_features, n_samples]`.
//! Each feature's values across all samples are contiguous in memory.
//! This is optimal for training (histogram building, coordinate descent).
//!
//! For prediction (which needs sample-major access), the predictor handles
//! block buffering internally.
//!
//! # Example
//!
//! ```
//! use boosters::dataset::{Dataset, DatasetBuilder, FeatureType};
//! use ndarray::array;
//!
//! // Simple construction from dense matrix (row-major: [n_samples, n_features])
//! let features = array![[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]; // 3 samples, 2 features
//! let targets = array![[0.0], [1.0], [0.0]]; // 3 samples, 1 output
//! let ds = Dataset::new(features.view(), targets.view());
//!
//! assert_eq!(ds.n_samples(), 3);
//! assert_eq!(ds.n_features(), 2);
//! ```

mod column;
mod dataset;
mod error;
mod schema;
mod views;

pub use column::{Column, SparseColumn};
pub use dataset::{Dataset, DatasetBuilder};
pub use error::DatasetError;
pub use schema::{DatasetSchema, FeatureMeta, FeatureType};
pub use views::{FeaturesView, TargetsView};

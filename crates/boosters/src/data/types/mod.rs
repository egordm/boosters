//! Core data types for features, targets, and weights.
//!
//! This module provides the fundamental data containers used throughout boosters:
//!
//! - [`Dataset`]: Main container for features, targets, and weights
//! - [`FeaturesView`] / [`TargetsView`] / [`WeightsView`]: Read-only views
//! - [`SampleAccessor`] / [`DataAccessor`]: Traits for data access
//! - [`FeatureType`]: Categorical vs numeric feature types
//!
//! # Usage Note
//!
//! For training, use [`super::binned::BinnedDatasetBuilder`] to create a
//! [`super::binned::BinnedDataset`] from a `Dataset`. The builder handles
//! feature binning, bundling, and optimization automatically.

#![allow(clippy::all)] // Legacy code - avoid churn from clippy updates
#![allow(dead_code)] // Some fields used conditionally
#![allow(unused_imports)] // Re-exports used by parent module

pub mod accessor;
pub mod column;
pub mod dataset;
pub mod schema;
pub mod views;

pub use accessor::{DataAccessor, SampleAccessor};
pub use column::{Column, SparseColumn};
pub use dataset::{Dataset, DatasetBuilder};
pub use schema::{DatasetSchema, FeatureMeta, FeatureType};
pub use views::{FeaturesView, SamplesView, TargetsView, WeightsIter, WeightsView};

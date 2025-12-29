//! Deprecated data structures moved during RFC-0018 migration.
//!
//! This module contains the old binned dataset implementation that is being
//! replaced by the new implementation in `data::binned`. These types are
//! re-exported at their original paths for backward compatibility during
//! the migration period.
//!
//! **Do not add new code to this module.** All new development should happen
//! in the new `data::binned` module.
//!
//! This entire module will be deleted once migration is complete.

#![deprecated(note = "Use new binned implementation from data::binned")]
#![allow(clippy::all)] // Don't fix clippy warnings in deprecated code
#![allow(deprecated)] // Allow using deprecated items within deprecated module
#![allow(unused_imports)] // Re-exports may appear unused but are intentional
#![allow(dead_code)] // Fields may be unused during migration period

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

//! Sampling strategies for training regularization.
//!
//! This module provides row and column sampling for gradient boosting.
//!
//! # Row Sampling
//!
//! Row sampling selects which samples to use for each boosting round.
//! See [`row`] module for details.
//!
//! Available strategies:
//! - [`RowSampling::None`]: Use all rows (default)
//! - [`RowSampling::Random`]: Random fraction without replacement
//! - [`RowSampling::Goss`]: Gradient-based one-side sampling
//!
//! # Column Sampling
//!
//! Column sampling selects which features to consider during tree building.
//! See [`column`] module for details.
//!
//! Sampling cascades at three levels:
//! - `colsample_bytree`: Per tree
//! - `colsample_bylevel`: Per depth level
//! - `colsample_bynode`: Per node

mod column;
mod row;

pub use column::ColumnSampler;
pub use row::{GossSampler, NoSampler, RandomSampler, RowSample, RowSampler, RowSampling};

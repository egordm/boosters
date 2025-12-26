//! Dataset and data conversion types for Python bindings.
//!
//! This module provides the `Dataset` wrapper for NumPy arrays and pandas DataFrames,
//! along with `EvalSet` for named evaluation datasets.
//!
//! # Design Notes
//!
//! - Dataset is marked as `subclass` so Python can extend it with convenience methods
//! - Categorical features can be auto-detected from pandas or specified explicitly
//! - NaN in features is allowed (treated as missing values)
//! - Inf in features or NaN/Inf in labels raise errors

mod dataset;
mod evalset;

pub use dataset::PyDataset;
pub use evalset::PyEvalSet;

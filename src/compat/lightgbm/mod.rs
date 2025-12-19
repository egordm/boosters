//! LightGBM model format support.
//!
//! This module provides parsing of LightGBM's text model format and conversion
//! to native booste-rs types.
//!
//! # Format Overview
//!
//! LightGBM text models have three main sections:
//! 1. **Header**: Model metadata (num_class, num_trees, objective, feature info)
//! 2. **Trees**: Each tree's structure (splits, leaf values, decision types)
//! 3. **Footer**: Feature importances, parameters (optional, skipped during parsing)
//!
//! # Key Differences from XGBoost
//!
//! - Split condition uses `<=` (left if value ≤ threshold) vs XGBoost's `<`
//! - Leaf indices are encoded as negative values in child arrays
//! - Decision type is a bitfield encoding categorical flag, default direction, and missing type
//! - Categorical features use bitset encoding
//!
//! # Example
//!
//! ```ignore
//! use boosters::compat::lightgbm::LgbModel;
//!
//! let model = LgbModel::from_file("model.txt")?;
//! let forest = model.to_forest()?;
//! let predictions = forest.predict_row(&features);
//! ```

mod convert;
mod text;

pub use convert::ConversionError;
pub use text::*;

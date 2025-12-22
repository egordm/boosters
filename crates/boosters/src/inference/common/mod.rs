//! Shared inference types and utilities.
//!
//! This module contains types shared across GBDT and GBLinear inference:
//! - [`PredictionKind`]: What do prediction values represent
//! - [`Predictions`]: Semantic wrapper around predictions

mod predictions;

pub use predictions::{PredictionKind, Predictions};

// Note: Transform functions (sigmoid, softmax) are provided by objectives.
// Use `Objective::transform_predictions()` for applying transforms.

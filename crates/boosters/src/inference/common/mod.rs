//! Shared inference types and utilities.
//!
//! This module contains types shared across GBDT and GBLinear inference:
//! - [`PredictionOutput`]: Column-major prediction storage

mod output;
mod predictions;

pub use output::PredictionOutput;
pub use predictions::{PredictionKind, Predictions};

// Note: Transform functions (sigmoid, softmax) are provided by objectives.
// Use `Objective::transform_predictions()` for applying transforms.

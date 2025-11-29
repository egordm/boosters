//! Prediction pipeline for tree ensemble inference.
//!
//! This module provides the prediction infrastructure for traversing trees
//! and accumulating leaf values.
//!
//! # Quick Start
//!
//! ```ignore
//! use booste_rs::predict::{SimplePredictor, UnrolledPredictor6};
//! use booste_rs::data::DenseMatrix;
//!
//! // Simple predictor (no pre-computation)
//! let predictor = SimplePredictor::new(&forest);
//!
//! // Unrolled predictor (faster for large batches)
//! let fast_predictor = UnrolledPredictor6::new(&forest);
//!
//! // Predict
//! let output = predictor.predict(&features);
//! ```
//!
//! # Choosing a Predictor
//!
//! | Scenario | Recommended |
//! |----------|-------------|
//! | Single row | `SimplePredictor` |
//! | Small batches (<100 rows) | `SimplePredictor` |
//! | Large batches (100+ rows) | `UnrolledPredictor6` |
//! | Very deep trees (>6 levels) | `UnrolledPredictor8` |
//!
//! # Block Size
//!
//! All predictors use block-based processing for cache efficiency. Default is 64 rows.
//! Customize with [`Predictor::with_block_size`].
//!
//! # Unroll Depth
//!
//! [`UnrolledTraversal`] uses compile-time depth for optimization:
//!
//! - `UnrolledPredictor4`: 4 levels (15 nodes) — shallow trees
//! - `UnrolledPredictor6`: 6 levels (63 nodes) — default, matches XGBoost
//! - `UnrolledPredictor8`: 8 levels (255 nodes) — deep trees
//!
//! # Output Format
//!
//! Predictions are returned as [`PredictionOutput`], a flat row-major buffer
//! with shape `(num_rows, num_groups)`. For regression, `num_groups = 1`.
//! For multiclass with K classes, `num_groups = K`.

mod output;
mod predictor;
mod traversal;

// Re-export predictor types
pub use predictor::{
    Predictor, SimplePredictor, UnrolledPredictor4, UnrolledPredictor6, UnrolledPredictor8,
    DEFAULT_BLOCK_SIZE,
};

// Re-export traversal types
pub use traversal::{
    StandardTraversal, TreeTraversal, UnrolledTraversal, UnrolledTraversal4, UnrolledTraversal6,
    UnrolledTraversal8,
};

// Re-export output type
pub use output::PredictionOutput;


//! Prediction pipeline for tree ensemble inference.
//!
//! This module provides the prediction infrastructure for traversing trees
//! and accumulating leaf values.
//!
//! # Quick Start
//!
//! ```ignore
//! use booste_rs::predict::{Predictor, StandardTraversal, UnrolledTraversal6};
//! use booste_rs::data::DenseMatrix;
//!
//! // Simple predictor (no pre-computation, good for single rows)
//! let predictor = Predictor::<StandardTraversal>::new(&forest);
//!
//! // Unrolled predictor (faster for large batches, 2-3x speedup)
//! let fast_predictor = Predictor::<UnrolledTraversal6>::new(&forest);
//!
//! // Sequential prediction
//! let output = predictor.predict(&features);
//!
//! // Parallel prediction (for large batches on multi-core)
//! let output = fast_predictor.par_predict(&features);
//! ```
//!
//! # Choosing a Traversal Strategy
//!
//! | Scenario | Traversal | Method |
//! |----------|-----------|--------|
//! | Single row | `StandardTraversal` | `predict()` |
//! | Small batches (<100 rows) | `StandardTraversal` | `predict()` |
//! | Large batches (100+ rows) | `UnrolledTraversal6` | `predict()` |
//! | Large batches + multi-core | `UnrolledTraversal6` | `par_predict()` |
//! | Very deep trees (>6 levels) | `UnrolledTraversal8` | `predict()` |
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
//! - `UnrolledTraversal4`: 4 levels (15 nodes) — shallow trees
//! - `UnrolledTraversal6`: 6 levels (63 nodes) — default, matches XGBoost
//! - `UnrolledTraversal8`: 8 levels (255 nodes) — deep trees
//!
//! Type aliases `UnrolledPredictor4/6/8` are provided for convenience.
//!
//! # Output Format
//!
//! Predictions are returned as [`PredictionOutput`], a flat row-major buffer
//! with shape `(num_rows, num_groups)`. For regression, `num_groups = 1`.
//! For multiclass with K classes, `num_groups = K`.

mod output;
mod predictor;
mod traversal;

#[cfg(feature = "simd")]
mod simd;

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

// Re-export SIMD types when feature is enabled
#[cfg(feature = "simd")]
pub use simd::{SimdTraversal, SimdTraversal4, SimdTraversal6, SimdTraversal8, SIMD_WIDTH};

// Re-export output type
pub use output::PredictionOutput;

// Type aliases for SIMD predictors (experimental, see simd.rs for status)
#[cfg(feature = "simd")]
/// SIMD-accelerated predictor with depth 4 (experimental).
pub type SimdPredictor4<'f> = Predictor<'f, SimdTraversal4>;
#[cfg(feature = "simd")]
/// SIMD-accelerated predictor with depth 6 (experimental).
pub type SimdPredictor6<'f> = Predictor<'f, SimdTraversal6>;
#[cfg(feature = "simd")]
/// SIMD-accelerated predictor with depth 8 (experimental).
pub type SimdPredictor8<'f> = Predictor<'f, SimdTraversal8>;


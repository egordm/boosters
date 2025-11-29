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
//! // SIMD predictor (fastest for large batches, requires `simd` feature)
//! #[cfg(feature = "simd")]
//! let simd_predictor = booste_rs::predict::SimdPredictor6::new(&forest);
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
//! | Large batches + AVX2 | `SimdPredictor6` (requires `simd` feature) |
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

// Type aliases for SIMD predictors
#[cfg(feature = "simd")]
/// SIMD-accelerated predictor with depth 4.
pub type SimdPredictor4<'f> = Predictor<'f, SimdTraversal4>;
#[cfg(feature = "simd")]
/// SIMD-accelerated predictor with depth 6 (default).
pub type SimdPredictor6<'f> = Predictor<'f, SimdTraversal6>;
#[cfg(feature = "simd")]
/// SIMD-accelerated predictor with depth 8.
pub type SimdPredictor8<'f> = Predictor<'f, SimdTraversal8>;


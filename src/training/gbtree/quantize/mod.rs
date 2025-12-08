//! Quantization and binning for histogram-based training.
//!
//! This module provides the infrastructure to discretize continuous feature values
//! into bins for histogram-based gradient boosting (RFC-0011).
//!
//! # Overview
//!
//! Histogram-based training requires converting continuous features into a small
//! number of discrete bins (typically 256). This enables:
//!
//! - O(n_bins) split search instead of O(n_samples)
//! - Efficient histogram aggregation
//! - Better cache locality (u8 bin indices fit more data in cache)
//! - Histogram subtraction optimization
//!
//! # Key Types
//!
//! - [`BinCuts`]: Bin boundaries for all features (thresholds)
//! - [`QuantizedMatrix`]: Quantized feature matrix storing bin indices
//! - [`BinIndex`]: Trait for bin index types (u8, u16, u32)
//! - [`Quantizer`]: Transforms raw features into quantized form
//!
//! # Example
//!
//! ```ignore
//! use booste_rs::training::quantize::{ExactQuantileCuts, Quantizer, CutFinder};
//! use booste_rs::data::ColMatrix;
//!
//! // Create feature matrix
//! let data: ColMatrix<f32> = /* ... */;
//!
//! // Find bin boundaries using exact quantiles
//! let cut_finder = ExactQuantileCuts::default();
//! let cuts = cut_finder.find_cuts(&data, 256);
//!
//! // Quantize the data
//! let quantizer = Quantizer::new(cuts);
//! let quantized = quantizer.quantize(&data);
//!
//! // Access bin indices
//! let bin = quantized.get(row, feature);
//! let column = quantized.feature_column(feature);
//! ```
//!
//! # Missing Values
//!
//! Missing values (NaN) are mapped to bin 0 by convention. This allows the
//! split finder to handle missing values by checking if `bin == 0`.
//!
//! See RFC-0011 for design rationale.

mod cuts;
mod matrix;
mod quantizer;

pub use cuts::{BinCuts, BinIndex, CategoricalInfo};
pub use matrix::QuantizedMatrix;
pub use quantizer::{CutFinder, ExactQuantileCuts, Quantizer};

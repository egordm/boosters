//! Histogram-based gradient aggregation for tree building.
//!
//! This module provides the histogram infrastructure for histogram-based
//! gradient boosting (RFC-0011).
//!
//! # Overview
//!
//! Histogram-based training discretizes features into bins and accumulates
//! gradients per bin. This enables:
//!
//! - O(n_bins) split search instead of O(n_samples)
//! - Efficient histogram subtraction optimization
//! - Better cache locality
//!
//! # Key Types
//!
//! - [`FeatureHistogram`]: Per-feature gradient/hessian histogram
//! - [`NodeHistogram`]: Collection of feature histograms for a tree node
//! - [`HistogramBuilder`]: Builds histograms from quantized data
//!
//! # Histogram Subtraction
//!
//! Both [`FeatureHistogram`] and [`NodeHistogram`] implement `Sub<&Self>`
//! for efficient sibling derivation: `parent - smaller_child = larger_child`.
//!
//! # Example
//!
//! ```ignore
//! use booste_rs::training::histogram::{HistogramBuilder, NodeHistogram};
//!
//! let mut hist = NodeHistogram::new(&cuts);
//! HistogramBuilder.build(&mut hist, &quantized, &grads, &hess, &rows);
//!
//! // Access per-feature histograms
//! let feat_hist = hist.feature(0);
//! let (grad, hess, count) = feat_hist.bin_stats(bin_idx);
//!
//! // Compute sibling via subtraction
//! let sibling = &parent - &child;
//! ```
//!
//! See RFC-0011 for design rationale.

mod builder;
mod feature;
mod node;

pub use builder::HistogramBuilder;
pub use feature::FeatureHistogram;
pub use node::NodeHistogram;

//! Histogram-based gradient aggregation for tree building.
//!
//! This module provides the histogram infrastructure for histogram-based
//! gradient boosting (RFC-0011, RFC-0025).
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
//! ## Per-Node Histograms
//! - [`FeatureHistogram`]: Per-feature gradient/hessian histogram (legacy, owned)
//! - [`NodeHistogram`]: Collection of feature histograms for a tree node (legacy)
//! - [`FeatureSlice`], [`FeatureSliceMut`]: Views into flat histogram storage
//!
//! ## Histogram Layout
//! - [`HistogramLayout`]: Maps features to bin ranges in flat histograms
//!
//! ## Sequential Building
//! - [`HistogramBuilder`]: Builds histograms from quantized data
//!
//! ## Parallel Building (RFC-0025)
//! - [`ContiguousHistogramPool`]: LRU-cached contiguous histogram storage
//! - [`RowParallelScratch`]: Per-thread scratch buffers for row-parallel building
//! - [`NodeId`], [`SlotId`]: Type-safe identifiers for pool management
//! - [`PoolMetrics`]: Statistics for monitoring pool behavior
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
//! See RFC-0011 and RFC-0025 for design rationale.

mod builder;
mod feature;
mod node;
pub mod parallel_builder;
pub mod pool;
pub mod scratch;
mod slice;
pub mod types;

pub use builder::HistogramBuilder;
pub use feature::FeatureHistogram;
pub use node::NodeHistogram;
pub use parallel_builder::{ParallelHistogramBuilder, ParallelHistogramConfig};
pub use pool::{ContiguousHistogramPool, HistogramSlot, HistogramSlotMut};
pub use scratch::{subtract_histograms, RowParallelScratch, ScratchSlotMut};
pub use slice::{FeatureSlice, FeatureSliceMut, HistogramBins};
pub use types::{recommended_pool_capacity, HistogramLayout, NodeId, PoolMetrics};

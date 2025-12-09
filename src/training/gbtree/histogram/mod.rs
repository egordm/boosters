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
//! - [`FeatureHistogram`]: Per-feature gradient/hessian histogram (owned)
//! - [`NodeHistogram`]: Collection of feature histograms for a tree node
//! - [`FeatureSlice`], [`FeatureSliceMut`]: Borrowed views into flat storage
//!
//! ## Histogram Layout
//! - [`HistogramLayout`]: Maps features to bin ranges in flat histograms
//!
//! ## Building (RFC-0025)
//! - [`HistogramBuilder`]: Unified builder with multiple strategies:
//!   - `build_sequential()` - single-threaded
//!   - `build_feature_parallel()` - parallelizes across features
//!   - `build_row_parallel()` - parallelizes across rows
//! - [`HistogramConfig`]: Configuration for builder strategies
//!
//! ## Pool & Scratch
//! - [`ContiguousHistogramPool`]: LRU-cached contiguous histogram storage
//! - [`RowParallelScratch`]: Per-thread scratch buffers for row-parallel
//! - [`NodeId`], [`SlotId`]: Type-safe identifiers for pool management
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
//! let builder = HistogramBuilder::default();
//! let mut hist = NodeHistogram::new(&cuts);
//! builder.build_sequential(&mut hist, &quantized, &grads, &hess, &rows);
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
pub mod pool;
pub mod scratch;
mod slice;
pub mod types;

pub use builder::{HistogramBuilder, HistogramConfig};
pub use feature::FeatureHistogram;
pub use node::NodeHistogram;
pub use pool::{ContiguousHistogramPool, HistogramSlot, HistogramSlotMut};
pub use scratch::{subtract_histograms, RowParallelScratch, ScratchSlotMut};
pub use slice::{FeatureSlice, FeatureSliceMut, HistogramBins};
pub use types::{recommended_pool_capacity, HistogramLayout, NodeId, PoolMetrics};

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
//! ## Pool-Based Histograms
//! - [`ContiguousHistogramPool`]: LRU-cached contiguous histogram storage
//! - [`HistogramSlot`], [`HistogramSlotMut`]: Borrowed views into pool storage
//! - [`FeatureSlice`]: Borrowed view into a single feature's bins
//!
//! ## Layout
//! - [`HistogramLayout`]: Maps features to bin ranges in flat histograms
//!
//! ## Building (RFC-0025)
//! - [`HistogramBuilder`]: Unified builder with multiple strategies
//! - [`HistogramConfig`]: Configuration for builder strategies
//!
//! # Example
//!
//! ```ignore
//! use booste_rs::training::gbtree::histogram::{
//!     HistogramBuilder, HistogramConfig, ContiguousHistogramPool, HistogramLayout,
//! };
//!
//! // Create builder with row-parallel support
//! let builder = HistogramBuilder::new(&cuts, HistogramConfig::default());
//! let layout = HistogramLayout::new(&cuts);
//! let mut pool = ContiguousHistogramPool::new(16, layout.total_bins());
//!
//! // Build into pool slot
//! let mut slot = pool.get_or_allocate(node_id);
//! builder.build(&mut slot, &layout, strategy, &quantized, &grads, &hess, &rows);
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

// Core building API
pub use builder::{HistogramBuilder, HistogramConfig};

// Pool-based storage
#[allow(unused_imports)]
pub use pool::{ContiguousHistogramPool, HistogramSlot, HistogramSlotMut};

// Feature access
pub use slice::{FeatureSlice, HistogramBins};

// Layout
#[allow(unused_imports)]
pub use types::{HistogramLayout, NodeId};

// Row-parallel internals (exposed for advanced use)
#[allow(unused_imports)]
pub use scratch::RowParallelScratch;

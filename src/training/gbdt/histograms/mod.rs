//! Histogram data structures for gradient boosting tree training.
//!
//! This module provides:
//! - [`build_histograms`] for building histograms with automatic parallel strategy
//! - [`build_histograms_ordered`] for building with pre-gathered ordered gradients
//! - [`HistogramPool`] for LRU-cached histogram storage
//!
//! # Module Organization
//!
//! - [`ops`] - Histogram building and operations
//! - [`pool`] - LRU-cached histogram storage pool
//!
//! # Design Philosophy
//!
//! This module uses simple `(f64, f64)` tuples for histogram bins rather than
//! complex trait hierarchies. Benchmarks showed that:
//!
//! - LLVM auto-vectorizes scalar loops effectively
//! - Manual SIMD (pulp) added overhead on ARM, minimal benefit on x86
//! - Quantization to int8/16 added unpacking overhead that outweighed bandwidth savings
//! - Prefetching was 2x slower than hardware prefetching on both platforms
//! - Row-parallel strategy was 2.8x slower due to merge overhead
//!
//! The subtraction trick (sibling = parent - child) provides 10-44x speedup and
//! is the main optimization worth keeping.
//!
//! **Ordered gradients** (pre-gathering gradients into partition order) provides
//! significant cache efficiency gains by converting random gradient access into
//! sequential reads, following LightGBM's approach.

pub mod ops;
pub mod pool;

// Re-export main types
pub use ops::{
    build_histograms, build_histograms_ordered, ParallelStrategy, HistogramBin,
    subtract_histogram, merge_histogram, clear_histogram, sum_histogram,
};
pub use pool::{
    AcquireResult, FeatureMeta, HistogramPool, 
    HistogramSlot, HistogramSlotMut, SlotId,
};

// Re-export FeatureView from data module for convenience
pub use crate::data::binned::FeatureView;

//! Histogram data structures for gradient boosting tree training.
//!
//! This module provides:
//! - [`build_histograms_ordered`] - **Preferred** for training with pre-gathered gradients
//! - [`build_histograms`] - Legacy/testing; use ordered version in production
//! - [`HistogramPool`] for LRU-cached histogram storage
//!
//! # Recommended Usage
//!
//! Always use [`build_histograms_ordered`] in production. The "ordered gradients"
//! technique pre-gathers gradients into partition order, converting random memory
//! access into sequential reads. This provides significant cache efficiency gains
//! (following LightGBM's approach).
//!
//! The non-ordered [`build_histograms`] is kept for testing and edge cases where
//! gradients cannot be pre-gathered.
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

pub mod ops;
pub mod pool;

// Re-export main types
pub use ops::{
    build_histograms, build_histograms_ordered, build_histograms_ordered_sequential, ParallelStrategy, HistogramBin,
    subtract_histogram, merge_histogram, clear_histogram, sum_histogram,
};
pub use pool::{
    AcquireResult, FeatureMeta, HistogramPool, 
    HistogramSlot, HistogramSlotMut, SlotId,
};

// Re-export FeatureView from data module for convenience
pub use crate::data::binned::FeatureView;

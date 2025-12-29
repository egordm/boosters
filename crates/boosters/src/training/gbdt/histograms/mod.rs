//! Histogram data structures for gradient boosting tree training.
//!
//! This module provides:
//! - [`HistogramBuilder`] - Main interface for building histograms
//! - [`HistogramPool`] - LRU-cached histogram storage
//!
//! # Recommended Usage
//!
//! Use [`HistogramBuilder`] for all histogram construction. It handles parallel
//! strategy selection and kernel dispatch internally.
//!
//! # Module Organization
//!
//! - [`ops`] - Histogram building kernels and operations
//! - [`pool`] - LRU-cached histogram storage pool
//! - [`slices`] - Safe iteration over disjoint feature histogram regions
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
pub mod slices;

// Re-export main types
pub use ops::{
    HistogramBin, HistogramBuilder, clear_histogram, merge_histogram, subtract_histogram,
    sum_histogram,
};
pub use pool::{
    AcquireResult, HistogramLayout, HistogramPool, HistogramSlot, HistogramSlotMut, SlotId,
};
pub use slices::HistogramFeatureIter;

// Re-export new FeatureView from data module
// Note: Uses the new RFC-0018 FeatureView with tuple variants (U8, U16) and
// sample_indices field (instead of row_indices) for sparse variants.
pub use crate::data::binned::view::FeatureView;

// Deprecated FeatureView for transition - will be removed in Epic 7 switchover
#[allow(deprecated)]
pub use crate::data::deprecated::binned::FeatureView as DeprecatedFeatureView;

/// Convert deprecated FeatureView to new FeatureView.
///
/// This is a zero-cost conversion since the underlying data is the same,
/// just the enum variant syntax differs.
///
/// # Transition Note
///
/// This function exists only during the Epic 6-7 migration period.
/// Once grower.rs uses the new BinnedDataset, this can be removed.
#[allow(deprecated)]
pub fn convert_feature_view<'a>(deprecated: &DeprecatedFeatureView<'a>) -> FeatureView<'a> {
    match deprecated {
        DeprecatedFeatureView::U8 { bins } => FeatureView::U8(bins),
        DeprecatedFeatureView::U16 { bins } => FeatureView::U16(bins),
        DeprecatedFeatureView::SparseU8 {
            row_indices,
            bin_values,
        } => FeatureView::SparseU8 {
            sample_indices: row_indices,
            bin_values,
        },
        DeprecatedFeatureView::SparseU16 {
            row_indices,
            bin_values,
        } => FeatureView::SparseU16 {
            sample_indices: row_indices,
            bin_values,
        },
    }
}

/// Convert a slice of deprecated FeatureViews to new FeatureViews.
///
/// Allocates a new Vec since we can't do in-place conversion between different types.
///
/// # Transition Note
///
/// This function exists only during the Epic 6-7 migration period.
#[allow(deprecated)]
pub fn convert_feature_views<'a>(deprecated: &[DeprecatedFeatureView<'a>]) -> Vec<FeatureView<'a>> {
    deprecated.iter().map(convert_feature_view).collect()
}

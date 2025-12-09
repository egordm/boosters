//! Core types for histogram pool and parallel building.
//!
//! This module defines the fundamental types used by the histogram pooling
//! and row-parallel building systems (RFC-0025).
//!
//! # Key Types
//!
//! - [`NodeId`]: Identifies a tree node for histogram caching
//! - [`SlotId`]: Internal slot identifier within the pool
//! - [`PoolMetrics`]: Statistics for pool usage monitoring
//! - [`HistogramLayout`]: Describes feature-to-bin mapping for flat histograms

use std::fmt;

use crate::training::gbtree::quantize::BinCuts;

/// Identifies a tree node for histogram caching.
///
/// Node IDs are assigned during tree building and used to index
/// histograms in the pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u32);

impl NodeId {
    /// Create a new node ID.
    #[inline]
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    /// Get the underlying ID value.
    #[inline]
    pub fn value(self) -> u32 {
        self.0
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Node({})", self.0)
    }
}

impl From<u32> for NodeId {
    #[inline]
    fn from(id: u32) -> Self {
        Self(id)
    }
}

/// Internal slot identifier within the histogram pool.
///
/// Slots are indices into the contiguous backing store.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SlotId(pub(crate) u32);

impl SlotId {
    #[inline]
    pub(crate) fn index(self) -> usize {
        self.0 as usize
    }
}

/// Statistics for histogram pool usage.
///
/// Useful for tuning pool capacity and understanding access patterns.
#[derive(Debug, Clone, Default)]
pub struct PoolMetrics {
    /// Number of times a requested histogram was already in pool.
    pub hits: u64,
    /// Number of times a new slot had to be allocated.
    pub misses: u64,
    /// Number of times a slot was evicted via LRU.
    pub evictions: u64,
    /// Maximum number of slots in use simultaneously.
    pub peak_usage: usize,
}

impl PoolMetrics {
    /// Create new metrics with all counters at zero.
    pub fn new() -> Self {
        Self::default()
    }

    /// Calculate hit rate as a percentage.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            (self.hits as f64 / total as f64) * 100.0
        }
    }

    /// Reset all counters to zero.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

impl fmt::Display for PoolMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PoolMetrics {{ hits: {}, misses: {}, evictions: {}, peak: {}, hit_rate: {:.1}% }}",
            self.hits,
            self.misses,
            self.evictions,
            self.peak_usage,
            self.hit_rate()
        )
    }
}

/// Calculate recommended pool capacity based on tree parameters.
///
/// # Strategy
///
/// - Base: 2 × max_leaves (need parent + both children during split)
/// - Buffer: +25% for histogram subtraction working space
/// - Minimum: 8 (even for very shallow trees)
/// - Cap: available memory / histogram_size
///
/// # Arguments
///
/// * `max_leaves` - Maximum number of leaves in the tree
/// * `max_depth` - Optional maximum tree depth (overrides max_leaves heuristic)
/// * `histogram_size_bytes` - Size of one histogram in bytes
/// * `available_memory_bytes` - Maximum memory budget for histograms
///
/// # Returns
///
/// Recommended number of histogram slots for the pool.
#[allow(dead_code)]
pub fn recommended_pool_capacity(
    max_leaves: usize,
    max_depth: Option<usize>,
    histogram_size_bytes: usize,
    available_memory_bytes: usize,
) -> usize {
    let base = if let Some(depth) = max_depth {
        // Depth-wise: need nodes at current + next level
        // At depth d, we can have up to 2^d nodes, plus 2^(d+1) children
        2usize.saturating_pow(depth as u32 + 1)
    } else {
        // Leaf-wise: need all candidate leaves + working space
        max_leaves.saturating_mul(2)
    };

    // Add 25% buffer for working space
    let with_buffer = (base as f64 * 1.25) as usize;

    // Cap by available memory
    let memory_cap = if histogram_size_bytes > 0 {
        available_memory_bytes / histogram_size_bytes
    } else {
        usize::MAX
    };

    // Enforce minimum of 8, then cap by memory
    with_buffer.max(8).min(memory_cap)
}

// ============================================================================
// HistogramLayout
// ============================================================================

/// Describes the layout of features within a flat histogram.
///
/// A flat histogram stores all features' bins contiguously:
/// ```text
/// [feat0_bins | feat1_bins | feat2_bins | ...]
/// ```
///
/// `HistogramLayout` tracks where each feature's bins start and end,
/// enabling efficient slicing into per-feature views.
///
/// # Example
///
/// ```ignore
/// let layout = HistogramLayout::from_cuts(&cuts);
///
/// // Get feature 2's bin range
/// let (start, end) = layout.feature_range(2);
/// let feat2_grads = &all_grads[start..end];
/// ```
#[derive(Debug, Clone)]
pub struct HistogramLayout {
    /// Offset of each feature's first bin.
    /// `feature_offsets[f]` = starting bin index for feature f.
    /// Length = num_features + 1 (last entry = total_bins for easy slicing).
    feature_offsets: Box<[usize]>,

    /// Number of features.
    num_features: u32,

    /// Total bins across all features.
    total_bins: usize,
}

impl HistogramLayout {
    /// Create a layout from bin cuts.
    ///
    /// Computes feature offsets from the cuts' per-feature bin counts.
    pub fn from_cuts(cuts: &BinCuts) -> Self {
        let num_features = cuts.num_features();
        let mut feature_offsets = Vec::with_capacity(num_features as usize + 1);

        let mut offset = 0usize;
        for f in 0..num_features {
            feature_offsets.push(offset);
            offset += cuts.num_bins(f);
        }
        feature_offsets.push(offset); // Sentinel for easy slicing

        Self {
            feature_offsets: feature_offsets.into_boxed_slice(),
            num_features,
            total_bins: offset,
        }
    }

    /// Create a layout with uniform bins per feature.
    ///
    /// Useful for testing or when all features have the same bin count.
    pub fn uniform(num_features: u32, bins_per_feature: usize) -> Self {
        let mut feature_offsets = Vec::with_capacity(num_features as usize + 1);

        for f in 0..=num_features {
            feature_offsets.push(f as usize * bins_per_feature);
        }

        Self {
            feature_offsets: feature_offsets.into_boxed_slice(),
            num_features,
            total_bins: num_features as usize * bins_per_feature,
        }
    }

    /// Number of features.
    #[inline]
    pub fn num_features(&self) -> u32 {
        self.num_features
    }

    /// Total bins across all features.
    #[inline]
    pub fn total_bins(&self) -> usize {
        self.total_bins
    }

    /// Get the bin range for a feature.
    ///
    /// Returns `(start, end)` where `start..end` is the range of bins.
    #[inline]
    pub fn feature_range(&self, feature: u32) -> (usize, usize) {
        debug_assert!(
            (feature as usize) < self.feature_offsets.len() - 1,
            "feature {} out of range",
            feature
        );
        unsafe {
            let start = *self.feature_offsets.get_unchecked(feature as usize);
            let end = *self.feature_offsets.get_unchecked(feature as usize + 1);
            (start, end)
        }
    }

    /// Number of bins for a feature.
    #[inline]
    pub fn num_bins(&self, feature: u32) -> usize {
        let (start, end) = self.feature_range(feature);
        end - start
    }

    /// Get the starting offset for a feature.
    #[inline]
    pub fn feature_offset(&self, feature: u32) -> usize {
        debug_assert!((feature as usize) < self.feature_offsets.len());
        unsafe { *self.feature_offsets.get_unchecked(feature as usize) }
    }

    /// Slice the feature offsets array.
    #[inline]
    pub fn offsets(&self) -> &[usize] {
        &self.feature_offsets
    }

    /// Extract a feature slice from raw histogram arrays.
    ///
    /// # Arguments
    ///
    /// * `feature` - Feature index
    /// * `sum_grad` - Full gradient array for the histogram
    /// * `sum_hess` - Full hessian array for the histogram
    /// * `count` - Full count array for the histogram
    ///
    /// # Returns
    ///
    /// [`FeatureSlice`] for the specified feature.
    #[inline]
    pub fn feature_slice<'a>(
        &self,
        feature: u32,
        sum_grad: &'a [f32],
        sum_hess: &'a [f32],
        count: &'a [u32],
    ) -> super::slice::FeatureSlice<'a> {
        let (start, end) = self.feature_range(feature);
        super::slice::FeatureSlice::new(
            &sum_grad[start..end],
            &sum_hess[start..end],
            &count[start..end],
        )
    }

    /// Extract a mutable feature slice from raw histogram arrays.
    ///
    /// # Arguments
    ///
    /// * `feature` - Feature index
    /// * `sum_grad` - Full gradient array for the histogram
    /// * `sum_hess` - Full hessian array for the histogram
    /// * `count` - Full count array for the histogram
    ///
    /// # Returns
    ///
    /// [`FeatureSliceMut`] for the specified feature.
    #[inline]
    pub fn feature_slice_mut<'a>(
        &self,
        feature: u32,
        sum_grad: &'a mut [f32],
        sum_hess: &'a mut [f32],
        count: &'a mut [u32],
    ) -> super::slice::FeatureSliceMut<'a> {
        let (start, end) = self.feature_range(feature);
        super::slice::FeatureSliceMut::new(
            &mut sum_grad[start..end],
            &mut sum_hess[start..end],
            &mut count[start..end],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_id() {
        let id = NodeId::new(42);
        assert_eq!(id.value(), 42);
        assert_eq!(format!("{}", id), "Node(42)");

        let id2: NodeId = 42.into();
        assert_eq!(id, id2);
    }

    #[test]
    fn test_pool_metrics() {
        let mut metrics = PoolMetrics::new();
        assert_eq!(metrics.hits, 0);
        assert_eq!(metrics.hit_rate(), 0.0);

        metrics.hits = 80;
        metrics.misses = 20;
        assert!((metrics.hit_rate() - 80.0).abs() < 0.01);

        metrics.evictions = 5;
        metrics.peak_usage = 32;
        let display = format!("{}", metrics);
        assert!(display.contains("80.0%"));
    }

    #[test]
    fn test_recommended_pool_capacity() {
        // Depth-wise: depth 4 needs up to 2^5 = 32 nodes
        let cap = recommended_pool_capacity(31, Some(4), 1000, 1_000_000);
        assert!(cap >= 32); // At least 2^5
        assert!(cap <= 50); // Not too much more with buffer

        // Leaf-wise: 31 leaves × 2 = 62, + 25% = ~78
        let cap = recommended_pool_capacity(31, None, 1000, 1_000_000);
        assert!(cap >= 62);
        assert!(cap <= 100);

        // Memory cap kicks in
        let cap = recommended_pool_capacity(100, None, 10_000, 50_000);
        assert_eq!(cap, 5); // 50_000 / 10_000 = 5, but minimum is 8
        // Actually minimum is 8, so:
        let cap = recommended_pool_capacity(100, None, 10_000, 100_000);
        assert!(cap <= 10); // Memory cap should limit

        // Minimum 8
        let cap = recommended_pool_capacity(2, Some(1), 1000, 1_000_000);
        assert!(cap >= 8);
    }

    #[test]
    fn test_histogram_layout_uniform() {
        let layout = HistogramLayout::uniform(3, 10);

        assert_eq!(layout.num_features(), 3);
        assert_eq!(layout.total_bins(), 30);

        // Feature ranges
        assert_eq!(layout.feature_range(0), (0, 10));
        assert_eq!(layout.feature_range(1), (10, 20));
        assert_eq!(layout.feature_range(2), (20, 30));

        // Bin counts
        assert_eq!(layout.num_bins(0), 10);
        assert_eq!(layout.num_bins(1), 10);
        assert_eq!(layout.num_bins(2), 10);

        // Offsets
        assert_eq!(layout.feature_offset(0), 0);
        assert_eq!(layout.feature_offset(1), 10);
        assert_eq!(layout.feature_offset(2), 20);
    }

    #[test]
    fn test_histogram_layout_offsets() {
        let layout = HistogramLayout::uniform(4, 8);

        let offsets = layout.offsets();
        assert_eq!(offsets, &[0, 8, 16, 24, 32]);
    }

    #[test]
    fn test_layout_feature_slice() {
        let layout = HistogramLayout::uniform(2, 4);

        // Create some test data
        let sum_grad = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let sum_hess = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let count = vec![1u32, 2, 3, 4, 5, 6, 7, 8];

        // Get slice for feature 0
        let slice0 = layout.feature_slice(0, &sum_grad, &sum_hess, &count);
        assert_eq!(slice0.num_bins(), 4);
        assert_eq!(slice0.bin_stats(0), (1.0, 0.1, 1));
        assert_eq!(slice0.bin_stats(3), (4.0, 0.4, 4));

        // Get slice for feature 1
        let slice1 = layout.feature_slice(1, &sum_grad, &sum_hess, &count);
        assert_eq!(slice1.num_bins(), 4);
        assert_eq!(slice1.bin_stats(0), (5.0, 0.5, 5));
        assert_eq!(slice1.bin_stats(3), (8.0, 0.8, 8));
    }

    #[test]
    fn test_layout_feature_slice_mut() {
        let layout = HistogramLayout::uniform(2, 4);

        let mut sum_grad = vec![0.0f32; 8];
        let mut sum_hess = vec![0.0f32; 8];
        let mut count = vec![0u32; 8];

        // Modify feature 1's slice
        {
            let mut slice = layout.feature_slice_mut(1, &mut sum_grad, &mut sum_hess, &mut count);
            slice.add(0, 1.0, 0.5);
            slice.add(2, 3.0, 1.5);
        }

        // Verify modification
        assert_eq!(sum_grad, vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 3.0, 0.0]);
        assert_eq!(sum_hess, vec![0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.5, 0.0]);
        assert_eq!(count, vec![0, 0, 0, 0, 1, 0, 1, 0]);
    }
}
//! Feature slice views into flat histogram storage.
//!
//! This module provides [`FeatureSlice`] and [`FeatureSliceMut`] types that
//! give per-feature views into a flat histogram (like [`ContiguousHistogramPool`]).
//!
//! These types provide the same interface as [`FeatureHistogram`] but work with
//! borrowed slices rather than owned storage. This enables:
//!
//! - Using the pool for all strategies (sequential, feature-parallel, row-parallel)
//! - Efficient split finding without copying
//! - Unified histogram building regardless of storage backend
//!
//! # Trait Abstraction
//!
//! The [`HistogramBins`] trait provides a common read-only interface for accessing
//! histogram bin statistics. Both [`FeatureHistogram`] and [`FeatureSlice`] implement
//! this trait, allowing split finders to be generic over histogram storage.
//!
//! # Example
//!
//! ```ignore
//! use booste_rs::training::gbtree::histogram::{HistogramBins, FeatureSlice};
//!
//! fn find_split<H: HistogramBins>(hist: &H) {
//!     for bin in 0..hist.num_bins() as usize {
//!         let (g, h, c) = hist.bin_stats(bin);
//!         // ... process bins
//!     }
//! }
//!
//! // Works with both FeatureHistogram and FeatureSlice
//! find_split(&feature_histogram);
//! find_split(&feature_slice);
//! ```
//!
//! [`ContiguousHistogramPool`]: super::pool::ContiguousHistogramPool
//! [`FeatureHistogram`]: super::feature::FeatureHistogram

use super::pool::{HistogramSlot, HistogramSlotMut};

use super::types::HistogramLayout;

// ============================================================================
// HistogramBins trait
// ============================================================================

/// Trait for read-only access to histogram bin statistics.
///
/// This trait provides a common interface for split finding that works with
/// both owned histograms ([`FeatureHistogram`]) and borrowed views ([`FeatureSlice`]).
///
/// # Required Methods
///
/// Only `num_bins()` and `bin_stats()` are required, as these are sufficient
/// for split finding operations.
///
/// # Example
///
/// ```ignore
/// fn find_best_bin<H: HistogramBins>(hist: &H) -> usize {
///     let mut best_gain = 0.0f32;
///     let mut best_bin = 0;
///     for bin in 0..hist.num_bins() as usize {
///         let (g, h, _) = hist.bin_stats(bin);
///         let gain = compute_gain(g, h);
///         if gain > best_gain {
///             best_gain = gain;
///             best_bin = bin;
///         }
///     }
///     best_bin
/// }
/// ```
///
/// [`FeatureHistogram`]: super::feature::FeatureHistogram
/// [`FeatureSlice`]: FeatureSlice
pub trait HistogramBins {
    /// Number of bins in this histogram.
    fn num_bins(&self) -> u16;

    /// Get gradient, hessian, and count for a specific bin.
    ///
    /// Returns `(sum_grad, sum_hess, count)`.
    fn bin_stats(&self, bin: usize) -> (f32, f32, u32);
}

// ============================================================================
// FeatureSlice (immutable)
// ============================================================================

/// Immutable view into one feature's bins within a flat histogram.
///
/// Provides the same read interface as [`FeatureHistogram`] but borrows
/// from a pool slot or scratch buffer.
///
/// [`FeatureHistogram`]: super::feature::FeatureHistogram
#[derive(Debug, Clone, Copy)]
pub struct FeatureSlice<'a> {
    /// Gradient sums for this feature's bins.
    sum_grad: &'a [f32],
    /// Hessian sums for this feature's bins.
    sum_hess: &'a [f32],
    /// Sample counts for this feature's bins.
    count: &'a [u32],
}

impl<'a> FeatureSlice<'a> {
    /// Create a feature slice from raw slices.
    ///
    /// # Panics
    ///
    /// Panics if the slices have different lengths.
    pub fn new(sum_grad: &'a [f32], sum_hess: &'a [f32], count: &'a [u32]) -> Self {
        debug_assert_eq!(sum_grad.len(), sum_hess.len());
        debug_assert_eq!(sum_grad.len(), count.len());
        Self {
            sum_grad,
            sum_hess,
            count,
        }
    }

    /// Create a feature slice from a histogram slot.
    pub fn from_slot(layout: &HistogramLayout, slot: &HistogramSlot<'a>, feature: u32) -> Self {
        let (start, end) = layout.feature_range(feature);
        Self {
            sum_grad: &slot.sum_grad[start..end],
            sum_hess: &slot.sum_hess[start..end],
            count: &slot.count[start..end],
        }
    }

    /// Number of bins in this feature.
    #[inline]
    pub fn num_bins(&self) -> u16 {
        self.sum_grad.len() as u16
    }

    /// Get statistics for a bin.
    ///
    /// Returns (sum_grad, sum_hess, count).
    #[inline]
    pub fn bin_stats(&self, bin: usize) -> (f32, f32, u32) {
        debug_assert!(bin < self.sum_grad.len());
        unsafe {
            (
                *self.sum_grad.get_unchecked(bin),
                *self.sum_hess.get_unchecked(bin),
                *self.count.get_unchecked(bin),
            )
        }
    }

    /// Get sum of gradients for a bin.
    #[inline]
    pub fn grad(&self, bin: usize) -> f32 {
        debug_assert!(bin < self.sum_grad.len());
        unsafe { *self.sum_grad.get_unchecked(bin) }
    }

    /// Get sum of hessians for a bin.
    #[inline]
    pub fn hess(&self, bin: usize) -> f32 {
        debug_assert!(bin < self.sum_hess.len());
        unsafe { *self.sum_hess.get_unchecked(bin) }
    }

    /// Get count for a bin.
    #[inline]
    pub fn count(&self, bin: usize) -> u32 {
        debug_assert!(bin < self.count.len());
        unsafe { *self.count.get_unchecked(bin) }
    }

    /// Get slice of gradient sums (all bins).
    #[inline]
    pub fn grads(&self) -> &[f32] {
        self.sum_grad
    }

    /// Get slice of hessian sums (all bins).
    #[inline]
    pub fn hesses(&self) -> &[f32] {
        self.sum_hess
    }

    /// Get slice of counts (all bins).
    #[inline]
    pub fn counts(&self) -> &[u32] {
        self.count
    }

    /// Compute total gradient sum across all bins.
    pub fn total_grad(&self) -> f32 {
        self.sum_grad.iter().sum()
    }

    /// Compute total hessian sum across all bins.
    pub fn total_hess(&self) -> f32 {
        self.sum_hess.iter().sum()
    }

    /// Compute total count across all bins.
    pub fn total_count(&self) -> u32 {
        self.count.iter().sum()
    }
}

impl HistogramBins for FeatureSlice<'_> {
    #[inline]
    fn num_bins(&self) -> u16 {
        self.sum_grad.len() as u16
    }

    #[inline]
    fn bin_stats(&self, bin: usize) -> (f32, f32, u32) {
        (self.sum_grad[bin], self.sum_hess[bin], self.count[bin])
    }
}

// ============================================================================
// FeatureSliceMut (mutable)
// ============================================================================

/// Mutable view into one feature's bins within a flat histogram.
///
/// Provides the same interface as [`FeatureHistogram`] but borrows mutably
/// from a pool slot or scratch buffer.
///
/// [`FeatureHistogram`]: super::feature::FeatureHistogram
#[derive(Debug)]
pub struct FeatureSliceMut<'a> {
    /// Gradient sums for this feature's bins.
    sum_grad: &'a mut [f32],
    /// Hessian sums for this feature's bins.
    sum_hess: &'a mut [f32],
    /// Sample counts for this feature's bins.
    count: &'a mut [u32],
}

impl<'a> FeatureSliceMut<'a> {
    /// Create a feature slice from raw mutable slices.
    ///
    /// # Panics
    ///
    /// Panics if the slices have different lengths.
    pub fn new(sum_grad: &'a mut [f32], sum_hess: &'a mut [f32], count: &'a mut [u32]) -> Self {
        debug_assert_eq!(sum_grad.len(), sum_hess.len());
        debug_assert_eq!(sum_grad.len(), count.len());
        Self {
            sum_grad,
            sum_hess,
            count,
        }
    }

    /// Create a feature slice from a histogram slot.
    ///
    /// Note: This takes the full slot mutably, so you can only have one
    /// feature slice at a time. For parallel feature building, use raw
    /// pointer access with appropriate synchronization.
    pub fn from_slot(
        layout: &HistogramLayout,
        slot: &'a mut HistogramSlotMut<'_>,
        feature: u32,
    ) -> Self {
        let (start, end) = layout.feature_range(feature);
        Self {
            sum_grad: &mut slot.sum_grad[start..end],
            sum_hess: &mut slot.sum_hess[start..end],
            count: &mut slot.count[start..end],
        }
    }

    /// Number of bins in this feature.
    #[inline]
    pub fn num_bins(&self) -> u16 {
        self.sum_grad.len() as u16
    }

    /// Add a sample to a bin.
    ///
    /// Accumulates gradient, hessian, and increments count.
    #[inline]
    pub fn add(&mut self, bin: usize, grad: f32, hess: f32) {
        debug_assert!(bin < self.sum_grad.len());
        unsafe {
            *self.sum_grad.get_unchecked_mut(bin) += grad;
            *self.sum_hess.get_unchecked_mut(bin) += hess;
            *self.count.get_unchecked_mut(bin) += 1;
        }
    }

    /// Get statistics for a bin.
    #[inline]
    pub fn bin_stats(&self, bin: usize) -> (f32, f32, u32) {
        debug_assert!(bin < self.sum_grad.len());
        unsafe {
            (
                *self.sum_grad.get_unchecked(bin),
                *self.sum_hess.get_unchecked(bin),
                *self.count.get_unchecked(bin),
            )
        }
    }

    /// Get sum of gradients for a bin.
    #[inline]
    pub fn grad(&self, bin: usize) -> f32 {
        debug_assert!(bin < self.sum_grad.len());
        unsafe { *self.sum_grad.get_unchecked(bin) }
    }

    /// Get sum of hessians for a bin.
    #[inline]
    pub fn hess(&self, bin: usize) -> f32 {
        debug_assert!(bin < self.sum_hess.len());
        unsafe { *self.sum_hess.get_unchecked(bin) }
    }

    /// Get count for a bin.
    #[inline]
    pub fn count(&self, bin: usize) -> u32 {
        debug_assert!(bin < self.count.len());
        unsafe { *self.count.get_unchecked(bin) }
    }

    /// Reset all bins to zero.
    pub fn reset(&mut self) {
        self.sum_grad.fill(0.0);
        self.sum_hess.fill(0.0);
        self.count.fill(0);
    }

    /// Get slice of gradient sums (all bins).
    #[inline]
    pub fn grads(&self) -> &[f32] {
        self.sum_grad
    }

    /// Get slice of hessian sums (all bins).
    #[inline]
    pub fn hesses(&self) -> &[f32] {
        self.sum_hess
    }

    /// Get slice of counts (all bins).
    #[inline]
    pub fn counts(&self) -> &[u32] {
        self.count
    }

    /// Compute total gradient sum across all bins.
    pub fn total_grad(&self) -> f32 {
        self.sum_grad.iter().sum()
    }

    /// Compute total hessian sum across all bins.
    pub fn total_hess(&self) -> f32 {
        self.sum_hess.iter().sum()
    }

    /// Compute total count across all bins.
    pub fn total_count(&self) -> u32 {
        self.count.iter().sum()
    }

    /// Convert to immutable slice.
    pub fn as_immut(&self) -> FeatureSlice<'_> {
        FeatureSlice {
            sum_grad: self.sum_grad,
            sum_hess: self.sum_hess,
            count: self.count,
        }
    }
}

impl HistogramBins for FeatureSliceMut<'_> {
    #[inline]
    fn num_bins(&self) -> u16 {
        self.sum_grad.len() as u16
    }

    #[inline]
    fn bin_stats(&self, bin: usize) -> (f32, f32, u32) {
        (self.sum_grad[bin], self.sum_hess[bin], self.count[bin])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_slice_basic() {
        let grads = vec![1.0, 2.0, 3.0, 4.0];
        let hess = vec![0.1, 0.2, 0.3, 0.4];
        let counts = vec![10, 20, 30, 40];

        let slice = FeatureSlice::new(&grads, &hess, &counts);

        assert_eq!(slice.num_bins(), 4);
        assert_eq!(slice.bin_stats(0), (1.0, 0.1, 10));
        assert_eq!(slice.bin_stats(2), (3.0, 0.3, 30));
        assert_eq!(slice.grad(1), 2.0);
        assert_eq!(slice.hess(3), 0.4);
        assert_eq!(slice.count(2), 30);
    }

    #[test]
    fn test_feature_slice_totals() {
        let grads = vec![1.0, 2.0, 3.0];
        let hess = vec![0.1, 0.2, 0.3];
        let counts = vec![10, 20, 30];

        let slice = FeatureSlice::new(&grads, &hess, &counts);

        assert!((slice.total_grad() - 6.0).abs() < 1e-6);
        assert!((slice.total_hess() - 0.6).abs() < 1e-6);
        assert_eq!(slice.total_count(), 60);
    }

    #[test]
    fn test_feature_slice_mut_add() {
        let mut grads = vec![0.0; 4];
        let mut hess = vec![0.0; 4];
        let mut counts = vec![0u32; 4];

        {
            let mut slice = FeatureSliceMut::new(&mut grads, &mut hess, &mut counts);

            slice.add(0, 1.0, 0.1);
            slice.add(0, 2.0, 0.2);
            slice.add(2, 5.0, 0.5);

            assert_eq!(slice.bin_stats(0), (3.0, 0.3, 2));
            assert_eq!(slice.bin_stats(2), (5.0, 0.5, 1));
            assert_eq!(slice.bin_stats(1), (0.0, 0.0, 0));
        }

        // Verify mutations persisted
        assert_eq!(grads, vec![3.0, 0.0, 5.0, 0.0]);
        assert_eq!(counts, vec![2, 0, 1, 0]);
    }

    #[test]
    fn test_feature_slice_mut_reset() {
        let mut grads = vec![1.0, 2.0, 3.0];
        let mut hess = vec![0.1, 0.2, 0.3];
        let mut counts = vec![10, 20, 30];

        {
            let mut slice = FeatureSliceMut::new(&mut grads, &mut hess, &mut counts);
            slice.reset();
        }

        assert_eq!(grads, vec![0.0, 0.0, 0.0]);
        assert_eq!(hess, vec![0.0, 0.0, 0.0]);
        assert_eq!(counts, vec![0, 0, 0]);
    }

    #[test]
    fn test_feature_slice_mut_as_immut() {
        let mut grads = vec![1.0, 2.0];
        let mut hess = vec![0.1, 0.2];
        let mut counts = vec![10, 20];

        let slice_mut = FeatureSliceMut::new(&mut grads, &mut hess, &mut counts);
        let slice = slice_mut.as_immut();

        assert_eq!(slice.num_bins(), 2);
        assert_eq!(slice.bin_stats(0), (1.0, 0.1, 10));
    }

    /// Verify HistogramBins trait can be used generically.
    #[test]
    fn test_histogram_bins_trait() {
        fn compute_total_grad<H: HistogramBins>(hist: &H) -> f32 {
            let mut total = 0.0;
            for bin in 0..hist.num_bins() as usize {
                let (g, _, _) = hist.bin_stats(bin);
                total += g;
            }
            total
        }

        // Test with FeatureSlice
        let grads = vec![1.0, 2.0, 3.0];
        let hess = vec![0.1, 0.2, 0.3];
        let counts = vec![10, 20, 30];
        let slice = FeatureSlice::new(&grads, &hess, &counts);

        let total = compute_total_grad(&slice);
        assert!((total - 6.0).abs() < 1e-6);
    }

    // ========================================================================
    // Integration tests: Pool -> Layout -> Slice
    // ========================================================================

    /// Integration test demonstrating the full path:
    /// 1. Build histogram using row-parallel into pool
    /// 2. Access histogram via layout + slices
    /// 3. Verify data is correct
    #[test]
    fn test_integration_pool_layout_slice() {
        use crate::data::ColMatrix;
        use crate::training::gbtree::histogram::{
            ContiguousHistogramPool, HistogramBuilder, HistogramConfig, HistogramLayout, NodeId,
        };
        use crate::training::gbtree::quantize::{ExactQuantileCuts, Quantizer};

        // Create test data with a clear pattern
        // Feature 0: values 0..20 (will have varying gradients)
        // Feature 1: constant (no good split)
        let n_rows = 20;
        let n_features = 2;
        let mut data = Vec::with_capacity(n_rows * n_features);
        for row in 0..n_rows {
            data.push(row as f32); // Feature 0: 0..19
        }
        for _ in 0..n_rows {
            data.push(5.0); // Feature 1: constant
        }
        let matrix = ColMatrix::from_vec(data, n_rows, n_features);

        // Quantize
        let quantizer = Quantizer::from_data(&matrix, &ExactQuantileCuts::default(), 16);
        let quantized = quantizer.quantize_u8(&matrix);
        let cuts = quantized.cuts().clone();

        // Create gradients: first half positive, second half negative
        // This creates a clear split point at row 10
        let grads: Vec<f32> = (0..n_rows)
            .map(|i| if i < 10 { 1.0 } else { -1.0 })
            .collect();
        let hess: Vec<f32> = vec![1.0; n_rows];
        let rows: Vec<u32> = (0..n_rows as u32).collect();

        // Create histogram layout from cuts
        let layout = HistogramLayout::from_cuts(&cuts);
        assert_eq!(layout.num_features(), 2);

        // Create pool and builder with row-parallel support
        let mut pool = ContiguousHistogramPool::new(2, layout.total_bins());
        let config = HistogramConfig {
            num_threads: 2,
            ..Default::default()
        };
        let mut builder = HistogramBuilder::with_config(config, &cuts);

        // Build histogram using row-parallel into pool
        let node_id = NodeId(0);
        builder.build_row_parallel(&mut pool, node_id, &quantized, &grads, &hess, &rows);

        // Get the histogram slot
        let slot = pool.get(node_id).expect("histogram should exist");

        // Use layout to get FeatureSlice for each feature
        let feat0_slice = layout.feature_slice(0, slot.sum_grad, slot.sum_hess, slot.count);
        let feat1_slice = layout.feature_slice(1, slot.sum_grad, slot.sum_hess, slot.count);

        // Verify slice totals for feature 0
        // Total grad: 10 * 1.0 + 10 * (-1.0) = 0.0
        assert!(
            (feat0_slice.total_grad() - 0.0).abs() < 1e-5,
            "feat0 total_grad mismatch: {}",
            feat0_slice.total_grad()
        );
        assert!(
            (feat0_slice.total_hess() - 20.0).abs() < 1e-5,
            "feat0 total_hess mismatch: {}",
            feat0_slice.total_hess()
        );
        assert_eq!(feat0_slice.total_count(), 20);

        // Feature 1 should have same totals (same data)
        assert!(
            (feat1_slice.total_grad() - 0.0).abs() < 1e-5,
            "feat1 total_grad mismatch"
        );
        assert!((feat1_slice.total_hess() - 20.0).abs() < 1e-5);
        assert_eq!(feat1_slice.total_count(), 20);

        // Verify HistogramBins trait works with the slices
        fn count_bins<H: HistogramBins>(h: &H) -> u32 {
            (0..h.num_bins() as usize)
                .map(|b| h.bin_stats(b).2)
                .sum()
        }

        assert_eq!(count_bins(&feat0_slice), 20);
        assert_eq!(count_bins(&feat1_slice), 20);
    }

    /// Test that building the same node twice resets properly
    #[test]
    fn test_integration_pool_rebuild() {
        use crate::data::ColMatrix;
        use crate::training::gbtree::histogram::{
            ContiguousHistogramPool, HistogramBuilder, HistogramConfig, HistogramLayout, NodeId,
        };
        use crate::training::gbtree::quantize::{ExactQuantileCuts, Quantizer};

        let n_rows = 10;
        let matrix = ColMatrix::from_vec((0..n_rows).map(|i| i as f32).collect(), n_rows, 1);
        let quantizer = Quantizer::from_data(&matrix, &ExactQuantileCuts::default(), 8);
        let quantized = quantizer.quantize_u8(&matrix);
        let cuts = quantized.cuts().clone();

        let layout = HistogramLayout::from_cuts(&cuts);
        let mut pool = ContiguousHistogramPool::new(1, layout.total_bins());
        let config = HistogramConfig::default();
        let mut builder = HistogramBuilder::with_config(config, &cuts);

        let node_id = NodeId(0);

        // Build with positive grads
        let grads1: Vec<f32> = vec![1.0; n_rows];
        let hess1: Vec<f32> = vec![1.0; n_rows];
        let rows: Vec<u32> = (0..n_rows as u32).collect();
        builder.build_row_parallel(&mut pool, node_id, &quantized, &grads1, &hess1, &rows);

        let slot = pool.get(node_id).unwrap();
        let slice = layout.feature_slice(0, slot.sum_grad, slot.sum_hess, slot.count);
        assert!((slice.total_grad() - 10.0).abs() < 1e-5);

        // Rebuild with negative grads
        let grads2: Vec<f32> = vec![-2.0; n_rows];
        builder.build_row_parallel(&mut pool, node_id, &quantized, &grads2, &hess1, &rows);

        let slot = pool.get(node_id).unwrap();
        let slice = layout.feature_slice(0, slot.sum_grad, slot.sum_hess, slot.count);
        assert!(
            (slice.total_grad() - (-20.0)).abs() < 1e-5,
            "Rebuild should reset: got {}",
            slice.total_grad()
        );
    }
}

//! Gradient histogram building for split finding.
//!
//! This module implements RFC-0012: gradient histogram construction and storage.
//!
//! # Overview
//!
//! For each tree node, we build histograms that aggregate gradients by bin:
//! - For each feature, accumulate (grad, hess, count) into bins
//! - Use aggregated histograms to find the best split
//!
//! # Key Types
//!
//! - [`FeatureHistogram`]: Per-feature gradient aggregates (SoA layout)
//! - [`NodeHistogram`]: Collection of feature histograms for a node
//! - [`HistogramBuilder`]: Builds histograms from quantized data
//! - [`HistogramSubtractor`]: Computes sibling via parent - child
//!
//! # Optimization: Histogram Subtraction
//!
//! When splitting a node, we build the histogram for the smaller child and
//! derive the larger child via subtraction: `parent - smaller = larger`.
//! This nearly halves the build cost since subtraction is O(bins) while
//! building is O(rows × features).
//!
//! # Example
//!
//! ```ignore
//! use booste_rs::training::histogram::{NodeHistogram, HistogramBuilder};
//!
//! // Build histogram for a node
//! let mut hist = NodeHistogram::new(&cuts);
//! HistogramBuilder.build(&mut hist, &quantized, &gradients, &row_indices);
//!
//! // Use histogram subtraction for the sibling
//! let mut sibling = NodeHistogram::new(&cuts);
//! HistogramSubtractor::compute_sibling(&parent_hist, &hist, &mut sibling);
//! ```
//!
//! See RFC-0012 for design rationale.

use rayon::prelude::*;

use crate::training::GradientBuffer;

use super::quantize::{BinCuts, BinIndex, QuantizedMatrix};

// ============================================================================
// FeatureHistogram
// ============================================================================

/// Gradient histogram for a single feature.
///
/// Each bin stores sum of gradients, sum of hessians, and sample count.
///
/// # Memory Layout
///
/// For 256 bins:
/// - `sum_grad`: 1024 bytes (256 × f32)
/// - `sum_hess`: 1024 bytes (256 × f32)
/// - `count`:    1024 bytes (256 × u32)
/// - Total: ~3 KB per feature
#[derive(Debug, Clone)]
pub struct FeatureHistogram {
    /// Sum of gradients per bin: [num_bins]
    sum_grad: Box<[f32]>,
    /// Sum of hessians per bin: [num_bins]
    sum_hess: Box<[f32]>,
    /// Count of samples per bin: [num_bins]
    count: Box<[u32]>,
    /// Number of bins (including bin 0 for missing)
    num_bins: u16,
}

impl FeatureHistogram {
    /// Create a new histogram with the specified number of bins.
    ///
    /// All bins are initialized to zero.
    pub fn new(num_bins: u16) -> Self {
        let n = num_bins as usize;
        Self {
            sum_grad: vec![0.0; n].into_boxed_slice(),
            sum_hess: vec![0.0; n].into_boxed_slice(),
            count: vec![0; n].into_boxed_slice(),
            num_bins,
        }
    }

    /// Number of bins in this histogram.
    #[inline]
    pub fn num_bins(&self) -> u16 {
        self.num_bins
    }

    /// Add a sample to a bin.
    ///
    /// Accumulates gradient, hessian, and increments count.
    #[inline]
    pub fn add(&mut self, bin: usize, grad: f32, hess: f32) {
        debug_assert!(bin < self.num_bins as usize, "bin {} >= num_bins {}", bin, self.num_bins);
        // SAFETY: bounds checked by debug_assert
        unsafe {
            *self.sum_grad.get_unchecked_mut(bin) += grad;
            *self.sum_hess.get_unchecked_mut(bin) += hess;
            *self.count.get_unchecked_mut(bin) += 1;
        }
    }

    /// Get statistics for a bin.
    ///
    /// Returns (sum_grad, sum_hess, count).
    #[inline]
    pub fn bin_stats(&self, bin: usize) -> (f32, f32, u32) {
        debug_assert!(bin < self.num_bins as usize);
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
        debug_assert!(bin < self.num_bins as usize);
        unsafe { *self.sum_grad.get_unchecked(bin) }
    }

    /// Get sum of hessians for a bin.
    #[inline]
    pub fn hess(&self, bin: usize) -> f32 {
        debug_assert!(bin < self.num_bins as usize);
        unsafe { *self.sum_hess.get_unchecked(bin) }
    }

    /// Get count for a bin.
    #[inline]
    pub fn count(&self, bin: usize) -> u32 {
        debug_assert!(bin < self.num_bins as usize);
        unsafe { *self.count.get_unchecked(bin) }
    }

    /// Reset all bins to zero.
    ///
    /// Faster than allocating a new histogram.
    pub fn reset(&mut self) {
        self.sum_grad.fill(0.0);
        self.sum_hess.fill(0.0);
        self.count.fill(0);
    }

    /// Compute sibling histogram via subtraction.
    ///
    /// After calling this, `self` contains `parent - self` (the sibling).
    /// This is the histogram subtraction trick: build smaller child,
    /// derive larger child via `parent - smaller = larger`.
    pub fn subtract_from(&mut self, parent: &Self) {
        debug_assert_eq!(self.num_bins, parent.num_bins);
        
        for i in 0..self.sum_grad.len() {
            self.sum_grad[i] = parent.sum_grad[i] - self.sum_grad[i];
            self.sum_hess[i] = parent.sum_hess[i] - self.sum_hess[i];
        }
        for i in 0..self.count.len() {
            self.count[i] = parent.count[i] - self.count[i];
        }
    }

    /// Create a new histogram by subtracting another from self.
    ///
    /// Returns `self - other`.
    pub fn subtract(&self, other: &Self) -> Self {
        debug_assert_eq!(self.num_bins, other.num_bins);
        
        let n = self.num_bins as usize;
        
        let mut sum_grad = vec![0.0f32; n].into_boxed_slice();
        let mut sum_hess = vec![0.0f32; n].into_boxed_slice();
        let mut count = vec![0u32; n].into_boxed_slice();
        
        for i in 0..n {
            sum_grad[i] = self.sum_grad[i] - other.sum_grad[i];
            sum_hess[i] = self.sum_hess[i] - other.sum_hess[i];
            count[i] = self.count[i] - other.count[i];
        }
        
        Self {
            sum_grad,
            sum_hess,
            count,
            num_bins: self.num_bins,
        }
    }

    /// Copy from another histogram.
    ///
    /// Overwrites current contents with source.
    pub fn copy_from(&mut self, src: &Self) {
        debug_assert_eq!(self.num_bins, src.num_bins);
        self.sum_grad.copy_from_slice(&src.sum_grad);
        self.sum_hess.copy_from_slice(&src.sum_hess);
        self.count.copy_from_slice(&src.count);
    }

    /// Get slice of gradient sums (all bins).
    #[inline]
    pub fn grads(&self) -> &[f32] {
        &self.sum_grad
    }

    /// Get slice of hessian sums (all bins).
    #[inline]
    pub fn hesses(&self) -> &[f32] {
        &self.sum_hess
    }

    /// Get slice of counts (per bin).
    #[inline]
    pub fn counts(&self) -> &[u32] {
        &self.count
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

// ============================================================================
// NodeHistogram
// ============================================================================

/// Histograms for all features at a single tree node.
///
/// Contains one [`FeatureHistogram`] per feature, plus cached totals.
///
/// # Memory Usage
///
/// For 100 features × 256 bins:
/// - 100 × 3 KB = 300 KB per node
///
/// # Example
///
/// ```ignore
/// let mut hist = NodeHistogram::new(&cuts);
/// HistogramBuilder.build(&mut hist, &quantized, &gradients, &rows);
/// ```
#[derive(Debug, Clone)]
pub struct NodeHistogram {
    /// Per-feature histograms
    features: Box<[FeatureHistogram]>,
    /// Cached total gradient (sum across all features, all bins)
    /// Updated after build()
    total_grad: f32,
    /// Cached total hessian
    total_hess: f32,
    /// Cached total count
    total_count: u32,
}

impl NodeHistogram {
    /// Create histograms for all features based on bin cuts.
    ///
    /// Each feature gets a histogram with the appropriate number of bins.
    pub fn new(cuts: &BinCuts) -> Self {
        let features: Box<[FeatureHistogram]> = (0..cuts.num_features())
            .map(|f| {
                let num_bins = cuts.num_bins(f as u32);
                FeatureHistogram::new(num_bins as u16)
            })
            .collect();

        Self {
            features,
            total_grad: 0.0,
            total_hess: 0.0,
            total_count: 0,
        }
    }

    /// Create from explicit bin counts per feature.
    ///
    /// Useful for testing or when cuts aren't available.
    pub fn from_bin_counts(bin_counts: &[u16]) -> Self {
        let features: Box<[FeatureHistogram]> = bin_counts
            .iter()
            .map(|&n| FeatureHistogram::new(n))
            .collect();

        Self {
            features,
            total_grad: 0.0,
            total_hess: 0.0,
            total_count: 0,
        }
    }

    /// Number of features.
    #[inline]
    pub fn num_features(&self) -> usize {
        self.features.len()
    }

    /// Get histogram for a specific feature.
    #[inline]
    pub fn feature(&self, feat: usize) -> &FeatureHistogram {
        &self.features[feat]
    }

    /// Get mutable histogram for a specific feature.
    #[inline]
    pub fn feature_mut(&mut self, feat: usize) -> &mut FeatureHistogram {
        &mut self.features[feat]
    }

    /// Get all feature histograms as a slice.
    #[inline]
    pub fn features(&self) -> &[FeatureHistogram] {
        &self.features
    }

    /// Get all feature histograms as a mutable slice.
    #[inline]
    pub fn features_mut(&mut self) -> &mut [FeatureHistogram] {
        &mut self.features
    }

    /// Reset all histograms to zero.
    pub fn reset(&mut self) {
        for hist in self.features.iter_mut() {
            hist.reset();
        }
        self.total_grad = 0.0;
        self.total_hess = 0.0;
        self.total_count = 0;
    }

    /// Update cached totals from histogram contents.
    ///
    /// Should be called after building histograms.
    /// Totals are computed from feature 0 (all features should have same totals).
    pub fn update_totals(&mut self) {
        if self.features.is_empty() {
            return;
        }

        let first = &self.features[0];
        self.total_grad = first.total_grad();
        self.total_hess = first.total_hess();
        self.total_count = first.total_count();
    }

    /// Set totals directly.
    pub fn set_totals(&mut self, grad: f32, hess: f32, count: u32) {
        self.total_grad = grad;
        self.total_hess = hess;
        self.total_count = count;
    }

    /// Get cached total gradient.
    #[inline]
    pub fn total_grad(&self) -> f32 {
        self.total_grad
    }

    /// Get cached total hessian.
    #[inline]
    pub fn total_hess(&self) -> f32 {
        self.total_hess
    }

    /// Get cached total count.
    #[inline]
    pub fn total_count(&self) -> u32 {
        self.total_count
    }

    /// Compute sibling histograms via subtraction.
    ///
    /// After calling, `self` contains `parent - self`.
    pub fn subtract_from(&mut self, parent: &Self) {
        debug_assert_eq!(self.features.len(), parent.features.len());
        
        for (child, par) in self.features.iter_mut().zip(parent.features.iter()) {
            child.subtract_from(par);
        }
        self.total_grad = parent.total_grad - self.total_grad;
        self.total_hess = parent.total_hess - self.total_hess;
        self.total_count = parent.total_count - self.total_count;
    }

    /// Create a new histogram by subtracting another from self.
    ///
    /// Returns `self - other`. Useful for histogram subtraction optimization.
    pub fn subtract(&self, other: &Self) -> Self {
        debug_assert_eq!(self.features.len(), other.features.len());
        
        let features: Box<[FeatureHistogram]> = self
            .features
            .iter()
            .zip(other.features.iter())
            .map(|(a, b)| a.subtract(b))
            .collect();
        
        Self {
            features,
            total_grad: self.total_grad - other.total_grad,
            total_hess: self.total_hess - other.total_hess,
            total_count: self.total_count - other.total_count,
        }
    }

    /// Copy from another histogram.
    pub fn copy_from(&mut self, src: &Self) {
        debug_assert_eq!(self.features.len(), src.features.len());
        
        for (dst, src) in self.features.iter_mut().zip(src.features.iter()) {
            dst.copy_from(src);
        }
        self.total_grad = src.total_grad;
        self.total_hess = src.total_hess;
        self.total_count = src.total_count;
    }
}

// ============================================================================
// HistogramBuilder
// ============================================================================

/// Builds histograms from quantized features and gradients.
///
/// The builder iterates over rows belonging to a node and accumulates
/// gradients into per-feature histograms based on bin assignments.
///
/// # Algorithm
///
/// 1. Reset histogram to zero
/// 2. For each row in the node:
///    - Look up gradient and hessian
///    - For each feature, add (grad, hess) to the corresponding bin
/// 3. Update cached totals
///
/// # Parallelization
///
/// Two parallelization strategies are available:
/// - [`build`](Self::build): Single-threaded, good baseline
/// - [`build_parallel`](Self::build_parallel): Per-feature parallelism
///
/// Per-feature parallelism works well because each feature histogram is
/// independent — no synchronization needed.
#[derive(Debug, Default, Clone, Copy)]
pub struct HistogramBuilder;

impl HistogramBuilder {
    /// Build histogram for a node from its row indices (single-threaded).
    ///
    /// # Arguments
    ///
    /// * `hist` - Histogram to fill (will be reset first)
    /// * `index` - Quantized feature matrix
    /// * `grads` - Gradient buffer (grad/hess for each row)
    /// * `rows` - Row indices belonging to this node
    ///
    /// # Panics
    ///
    /// Panics if row indices are out of bounds.
    pub fn build<B: BinIndex>(
        &self,
        hist: &mut NodeHistogram,
        index: &QuantizedMatrix<B>,
        grads: &GradientBuffer,
        rows: &[u32],
    ) {
        hist.reset();
        let num_features = hist.num_features();

        for &row in rows {
            let row_idx = row as usize;
            let (grad, hess) = grads.get(row_idx, 0);

            for feat in 0..num_features {
                let bin = index.get(row, feat as u32).to_usize();
                hist.features[feat].add(bin, grad, hess);
            }
        }

        hist.update_totals();
    }

    /// Build histogram with per-feature parallelism.
    ///
    /// Each feature histogram is built independently in parallel using Rayon.
    /// This is the preferred method for datasets with many features.
    ///
    /// # Arguments
    ///
    /// Same as [`build`](Self::build).
    pub fn build_parallel<B: BinIndex>(
        &self,
        hist: &mut NodeHistogram,
        index: &QuantizedMatrix<B>,
        grads: &GradientBuffer,
        rows: &[u32],
    ) {
        hist.features
            .par_iter_mut()
            .enumerate()
            .for_each(|(feat, feat_hist)| {
                feat_hist.reset();

                for (row, bin) in rows.iter().zip(index.iter_rows_for_feature(feat as u32, rows)) {
                    let row_idx = *row as usize;
                    let (grad, hess) = grads.get(row_idx, 0);
                    feat_hist.add(bin.to_usize(), grad, hess);
                }
            });

        hist.update_totals();
    }

    /// Build histogram using column iteration (cache-friendly).
    ///
    /// Processes one feature at a time, which is cache-friendly for
    /// column-major quantized matrices.
    pub fn build_column_wise<B: BinIndex>(
        &self,
        hist: &mut NodeHistogram,
        index: &QuantizedMatrix<B>,
        grads: &GradientBuffer,
        rows: &[u32],
    ) {
        hist.reset();
        let num_features = hist.num_features();

        for feat in 0..num_features {
            let feat_hist = &mut hist.features[feat];

            for (&row, bin) in rows.iter().zip(index.iter_rows_for_feature(feat as u32, rows)) {
                let (grad, hess) = grads.get(row as usize, 0);
                feat_hist.add(bin.to_usize(), grad, hess);
            }
        }

        hist.update_totals();
    }
}

// ============================================================================
// HistogramSubtractor
// ============================================================================

/// Which child to build directly (the other derived via subtraction).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChildSide {
    /// Build left child, derive right via subtraction
    Left,
    /// Build right child, derive left via subtraction
    Right,
}

/// Utilities for histogram subtraction optimization.
///
/// When splitting a node, we build the histogram for the smaller child
/// and derive the larger child via `parent - smaller = larger`.
/// This nearly halves histogram building cost since:
/// - Building: O(rows × features)
/// - Subtraction: O(bins × features)
pub struct HistogramSubtractor;

impl HistogramSubtractor {
    /// Determine which child to build directly.
    ///
    /// Prefers building the smaller child (fewer rows to iterate).
    pub fn select_build_child(left_count: u32, right_count: u32) -> ChildSide {
        if left_count <= right_count {
            ChildSide::Left
        } else {
            ChildSide::Right
        }
    }

    /// Compute sibling histogram via subtraction.
    ///
    /// Given parent histogram and one child's histogram, computes the
    /// other child's histogram: `sibling = parent - child`.
    ///
    /// # Arguments
    ///
    /// * `parent` - Parent node's histogram
    /// * `child` - Built child's histogram
    /// * `sibling` - Output: will contain parent - child
    pub fn compute_sibling(
        parent: &NodeHistogram,
        child: &NodeHistogram,
        sibling: &mut NodeHistogram,
    ) {
        // Copy child to sibling, then subtract from parent
        sibling.copy_from(child);
        sibling.subtract_from(parent);
    }

    /// In-place sibling computation.
    ///
    /// Transforms `child` into the sibling histogram.
    /// Original child data is lost.
    pub fn compute_sibling_inplace(parent: &NodeHistogram, child: &mut NodeHistogram) {
        child.subtract_from(parent);
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_histogram_basic() {
        let mut hist = FeatureHistogram::new(4);
        
        assert_eq!(hist.num_bins(), 4);
        assert_eq!(hist.bin_stats(0), (0.0, 0.0, 0));
        
        // Add some samples
        hist.add(0, 1.0, 0.5);
        hist.add(0, 2.0, 1.0);
        hist.add(1, 3.0, 1.5);
        
        assert_eq!(hist.bin_stats(0), (3.0, 1.5, 2));
        assert_eq!(hist.bin_stats(1), (3.0, 1.5, 1));
        assert_eq!(hist.bin_stats(2), (0.0, 0.0, 0));
        
        // Check individual accessors
        assert_eq!(hist.grad(0), 3.0);
        assert_eq!(hist.hess(0), 1.5);
        assert_eq!(hist.count(0), 2);
    }

    #[test]
    fn test_feature_histogram_totals() {
        let mut hist = FeatureHistogram::new(4);
        
        hist.add(0, 1.0, 0.5);
        hist.add(1, 2.0, 1.0);
        hist.add(2, 3.0, 1.5);
        hist.add(3, 4.0, 2.0);
        
        assert_eq!(hist.total_grad(), 10.0);
        assert_eq!(hist.total_hess(), 5.0);
        assert_eq!(hist.total_count(), 4);
    }

    #[test]
    fn test_feature_histogram_reset() {
        let mut hist = FeatureHistogram::new(4);
        
        hist.add(0, 1.0, 0.5);
        hist.add(1, 2.0, 1.0);
        
        hist.reset();
        
        assert_eq!(hist.bin_stats(0), (0.0, 0.0, 0));
        assert_eq!(hist.bin_stats(1), (0.0, 0.0, 0));
        assert_eq!(hist.total_count(), 0);
    }

    #[test]
    fn test_feature_histogram_subtraction() {
        // Parent histogram
        let mut parent = FeatureHistogram::new(4);
        parent.add(0, 10.0, 5.0);  // bin 0: 2 samples
        parent.add(0, 10.0, 5.0);
        parent.add(1, 6.0, 3.0);   // bin 1: 1 sample
        parent.add(2, 4.0, 2.0);   // bin 2: 1 sample
        
        // Child (left) histogram - subset of parent
        let mut child = FeatureHistogram::new(4);
        child.add(0, 10.0, 5.0);   // bin 0: 1 sample
        child.add(1, 6.0, 3.0);    // bin 1: 1 sample
        
        // Compute sibling via subtraction
        child.subtract_from(&parent);
        
        // Sibling should have: bin 0: 1 sample, bin 2: 1 sample
        assert_eq!(child.bin_stats(0), (10.0, 5.0, 1));  // 20-10=10
        assert_eq!(child.bin_stats(1), (0.0, 0.0, 0));   // 6-6=0
        assert_eq!(child.bin_stats(2), (4.0, 2.0, 1));   // 4-0=4
    }

    #[test]
    fn test_node_histogram_creation() {
        // Create cuts with different bin counts per feature
        let cut_values = vec![0.5, 1.5, 2.5, 10.0];  // Feature 0: 3 cuts, Feature 1: 1 cut
        let cut_ptrs = vec![0, 3, 4];
        let cuts = BinCuts::new(cut_values, cut_ptrs);
        
        let hist = NodeHistogram::new(&cuts);
        
        assert_eq!(hist.num_features(), 2);
        // 3 cuts creates 4 regions + 1 missing bin = 5 bins
        assert_eq!(hist.feature(0).num_bins(), 5);
        // 1 cut creates 2 regions + 1 missing bin = 3 bins
        assert_eq!(hist.feature(1).num_bins(), 3);
    }

    #[test]
    fn test_node_histogram_subtraction() {
        let mut parent = NodeHistogram::from_bin_counts(&[4, 4]);
        let mut child = NodeHistogram::from_bin_counts(&[4, 4]);
        
        // Parent: feature 0, bin 0 = (10, 5, 2); feature 1, bin 1 = (8, 4, 2)
        parent.feature_mut(0).add(0, 5.0, 2.5);
        parent.feature_mut(0).add(0, 5.0, 2.5);
        parent.feature_mut(1).add(1, 4.0, 2.0);
        parent.feature_mut(1).add(1, 4.0, 2.0);
        parent.set_totals(18.0, 9.0, 4);
        
        // Child: feature 0, bin 0 = (5, 2.5, 1); feature 1, bin 1 = (4, 2, 1)
        child.feature_mut(0).add(0, 5.0, 2.5);
        child.feature_mut(1).add(1, 4.0, 2.0);
        child.set_totals(9.0, 4.5, 2);
        
        // Subtract
        child.subtract_from(&parent);
        
        // Check sibling
        assert_eq!(child.feature(0).bin_stats(0), (5.0, 2.5, 1));
        assert_eq!(child.feature(1).bin_stats(1), (4.0, 2.0, 1));
        assert_eq!(child.total_grad(), 9.0);
        assert_eq!(child.total_hess(), 4.5);
        assert_eq!(child.total_count(), 2);
    }

    #[test]
    fn test_select_build_child() {
        assert_eq!(HistogramSubtractor::select_build_child(10, 20), ChildSide::Left);
        assert_eq!(HistogramSubtractor::select_build_child(20, 10), ChildSide::Right);
        assert_eq!(HistogramSubtractor::select_build_child(10, 10), ChildSide::Left);
    }

    mod integration {
        use super::*;
        use crate::data::ColMatrix;
        use crate::training::gbtree::quantize::{ExactQuantileCuts, Quantizer};

        fn make_test_data() -> (QuantizedMatrix<u8>, GradientBuffer) {
            // Create simple data: 10 rows, 2 features
            let data: Vec<f32> = vec![
                // Feature 0: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
                // Feature 1: all same value
                5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
            ];
            let matrix = ColMatrix::from_vec(data, 10, 2);
            
            // Quantize
            let quantizer = Quantizer::from_data(&matrix, &ExactQuantileCuts::default(), 256);
            let quantized: QuantizedMatrix<u8> = quantizer.quantize(&matrix);
            
            // Create gradients: grad = row_id, hess = 1.0
            let mut grads = GradientBuffer::new(10, 1);
            for i in 0..10 {
                grads.set(i, 0, i as f32, 1.0);
            }
            
            (quantized, grads)
        }

        #[test]
        fn test_histogram_builder_basic() {
            let (quantized, grads) = make_test_data();
            let rows: Vec<u32> = (0..10).collect();
            
            let mut hist = NodeHistogram::new(quantized.cuts());
            HistogramBuilder.build(&mut hist, &quantized, &grads, &rows);
            
            // Total grad should be 0+1+2+...+9 = 45
            // Total hess should be 10 (each row contributes 1.0)
            assert_eq!(hist.total_count(), 10);
            assert!((hist.total_grad() - 45.0).abs() < 1e-5);
            assert!((hist.total_hess() - 10.0).abs() < 1e-5);
        }

        #[test]
        fn test_histogram_builder_subset() {
            let (quantized, grads) = make_test_data();
            
            // Build histogram for subset of rows
            let rows: Vec<u32> = vec![0, 2, 4, 6, 8];  // Even rows
            
            let mut hist = NodeHistogram::new(quantized.cuts());
            HistogramBuilder.build(&mut hist, &quantized, &grads, &rows);
            
            // Grad sum: 0+2+4+6+8 = 20, hess: 5
            assert_eq!(hist.total_count(), 5);
            assert!((hist.total_grad() - 20.0).abs() < 1e-5);
            assert!((hist.total_hess() - 5.0).abs() < 1e-5);
        }

        #[test]
        fn test_histogram_builder_parallel_matches_sequential() {
            let (quantized, grads) = make_test_data();
            let rows: Vec<u32> = (0..10).collect();
            
            let mut hist_seq = NodeHistogram::new(quantized.cuts());
            HistogramBuilder.build(&mut hist_seq, &quantized, &grads, &rows);
            
            let mut hist_par = NodeHistogram::new(quantized.cuts());
            HistogramBuilder.build_parallel(&mut hist_par, &quantized, &grads, &rows);
            
            // Compare totals
            assert_eq!(hist_seq.total_count(), hist_par.total_count());
            assert!((hist_seq.total_grad() - hist_par.total_grad()).abs() < 1e-5);
            assert!((hist_seq.total_hess() - hist_par.total_hess()).abs() < 1e-5);
            
            // Compare per-feature histograms
            for feat in 0..hist_seq.num_features() {
                let seq = hist_seq.feature(feat);
                let par = hist_par.feature(feat);
                for bin in 0..seq.num_bins() as usize {
                    let (sg, sh, sc) = seq.bin_stats(bin);
                    let (pg, ph, pc) = par.bin_stats(bin);
                    assert!((sg - pg).abs() < 1e-5, "Feature {} bin {} grad mismatch", feat, bin);
                    assert!((sh - ph).abs() < 1e-5, "Feature {} bin {} hess mismatch", feat, bin);
                    assert_eq!(sc, pc, "Feature {} bin {} count mismatch", feat, bin);
                }
            }
        }

        #[test]
        fn test_histogram_subtraction_correctness() {
            let (quantized, grads) = make_test_data();
            
            // Build parent histogram (all rows)
            let all_rows: Vec<u32> = (0..10).collect();
            let mut parent = NodeHistogram::new(quantized.cuts());
            HistogramBuilder.build(&mut parent, &quantized, &grads, &all_rows);
            
            // Build left child histogram (first 6 rows)
            let left_rows: Vec<u32> = vec![0, 1, 2, 3, 4, 5];
            let mut left = NodeHistogram::new(quantized.cuts());
            HistogramBuilder.build(&mut left, &quantized, &grads, &left_rows);
            
            // Build right child directly for comparison
            let right_rows: Vec<u32> = vec![6, 7, 8, 9];
            let mut right_direct = NodeHistogram::new(quantized.cuts());
            HistogramBuilder.build(&mut right_direct, &quantized, &grads, &right_rows);
            
            // Compute right via subtraction
            let mut right_subtracted = NodeHistogram::new(quantized.cuts());
            HistogramSubtractor::compute_sibling(&parent, &left, &mut right_subtracted);
            
            // Compare
            assert_eq!(right_direct.total_count(), right_subtracted.total_count());
            assert!((right_direct.total_grad() - right_subtracted.total_grad()).abs() < 1e-5);
            assert!((right_direct.total_hess() - right_subtracted.total_hess()).abs() < 1e-5);
            
            // Verify actual values: right should have rows 6,7,8,9
            // Grad sum: 6+7+8+9 = 30
            assert!((right_subtracted.total_grad() - 30.0).abs() < 1e-5);
            assert_eq!(right_subtracted.total_count(), 4);
        }

        #[test]
        fn test_column_wise_matches_row_wise() {
            let (quantized, grads) = make_test_data();
            let rows: Vec<u32> = (0..10).collect();
            
            let mut hist_row = NodeHistogram::new(quantized.cuts());
            HistogramBuilder.build(&mut hist_row, &quantized, &grads, &rows);
            
            let mut hist_col = NodeHistogram::new(quantized.cuts());
            HistogramBuilder.build_column_wise(&mut hist_col, &quantized, &grads, &rows);
            
            // Compare totals
            assert_eq!(hist_row.total_count(), hist_col.total_count());
            assert!((hist_row.total_grad() - hist_col.total_grad()).abs() < 1e-5);
            assert!((hist_row.total_hess() - hist_col.total_hess()).abs() < 1e-5);
        }
    }
}

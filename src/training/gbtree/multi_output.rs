//! Multi-output tree training structures.
//!
//! Supports training trees with K-dimensional leaf values for multi-class
//! classification or multi-target regression using the "multi_output_tree" strategy.
//!
//! ## Design
//!
//! Multi-output histograms store K grad/hess sums per bin:
//! ```text
//! Layout: [bin0_out0, bin0_out1, ..., bin0_outK, bin1_out0, ...]
//! Index: bin * n_outputs + output
//! ```
//!
//! Split gain is the sum of per-output gains:
//! ```text
//! total_gain = Σ_k gain(left_k, right_k, λ) - γ
//! ```
//!
//! ## Usage
//!
//! ```ignore
//! use booste_rs::training::gbtree::{MultiOutputHistogram, MultiOutputNodeHistogram};
//!
//! let n_outputs = 3;  // e.g., 3 classes
//! let hist = MultiOutputFeatureHistogram::new(256, n_outputs);
//! ```

use crate::training::GradientBuffer;
use super::quantize::{BinCuts, BinIndex, QuantizedMatrix};

// =============================================================================
// MultiOutputFeatureHistogram
// =============================================================================

/// Histogram for a single feature with multi-output gradient/hessian sums.
///
/// Each bin stores K gradient and K hessian values (one per output).
/// Layout: `[bin0_out0, bin0_out1, ..., bin0_outK, bin1_out0, ...]`
#[derive(Debug, Clone)]
pub struct MultiOutputFeatureHistogram {
    /// Sum of gradients per (bin, output): [num_bins * n_outputs]
    sum_grad: Box<[f32]>,
    /// Sum of hessians per (bin, output): [num_bins * n_outputs]
    sum_hess: Box<[f32]>,
    /// Count of samples per bin: [num_bins]
    count: Box<[u32]>,
    /// Number of bins (including bin 0 for missing)
    num_bins: u16,
    /// Number of outputs (K)
    n_outputs: u16,
}

impl MultiOutputFeatureHistogram {
    /// Create a new multi-output histogram.
    ///
    /// # Arguments
    /// - `num_bins`: Number of bins (including missing bin)
    /// - `n_outputs`: Number of outputs per sample (K)
    pub fn new(num_bins: u16, n_outputs: u16) -> Self {
        let total = num_bins as usize * n_outputs as usize;
        Self {
            sum_grad: vec![0.0; total].into_boxed_slice(),
            sum_hess: vec![0.0; total].into_boxed_slice(),
            count: vec![0; num_bins as usize].into_boxed_slice(),
            num_bins,
            n_outputs,
        }
    }

    /// Number of bins in this histogram.
    #[inline]
    pub fn num_bins(&self) -> u16 {
        self.num_bins
    }

    /// Number of outputs (K).
    #[inline]
    pub fn n_outputs(&self) -> u16 {
        self.n_outputs
    }

    /// Add a sample to a bin with K gradient/hessian values.
    ///
    /// # Arguments
    /// - `bin`: Bin index
    /// - `grads`: Slice of K gradient values
    /// - `hess`: Slice of K hessian values
    #[inline]
    pub fn add(&mut self, bin: usize, grads: &[f32], hess: &[f32]) {
        debug_assert!(bin < self.num_bins as usize);
        debug_assert_eq!(grads.len(), self.n_outputs as usize);
        debug_assert_eq!(hess.len(), self.n_outputs as usize);

        let base = bin * self.n_outputs as usize;
        for k in 0..self.n_outputs as usize {
            // SAFETY: bounds checked by debug_assert
            unsafe {
                *self.sum_grad.get_unchecked_mut(base + k) += grads[k];
                *self.sum_hess.get_unchecked_mut(base + k) += hess[k];
            }
        }
        self.count[bin] += 1;
    }

    /// Get gradient sum for a specific (bin, output).
    #[inline]
    pub fn grad(&self, bin: usize, output: usize) -> f32 {
        debug_assert!(bin < self.num_bins as usize);
        debug_assert!(output < self.n_outputs as usize);
        self.sum_grad[bin * self.n_outputs as usize + output]
    }

    /// Get hessian sum for a specific (bin, output).
    #[inline]
    pub fn hess(&self, bin: usize, output: usize) -> f32 {
        debug_assert!(bin < self.num_bins as usize);
        debug_assert!(output < self.n_outputs as usize);
        self.sum_hess[bin * self.n_outputs as usize + output]
    }

    /// Get count for a bin.
    #[inline]
    pub fn count(&self, bin: usize) -> u32 {
        debug_assert!(bin < self.num_bins as usize);
        self.count[bin]
    }

    /// Get all K gradients for a bin.
    #[inline]
    pub fn bin_grads(&self, bin: usize) -> &[f32] {
        let base = bin * self.n_outputs as usize;
        &self.sum_grad[base..base + self.n_outputs as usize]
    }

    /// Get all K hessians for a bin.
    #[inline]
    pub fn bin_hess(&self, bin: usize) -> &[f32] {
        let base = bin * self.n_outputs as usize;
        &self.sum_hess[base..base + self.n_outputs as usize]
    }

    /// Reset all bins to zero.
    pub fn reset(&mut self) {
        self.sum_grad.fill(0.0);
        self.sum_hess.fill(0.0);
        self.count.fill(0);
    }

    /// Compute sibling histogram via subtraction.
    ///
    /// After calling, `self` contains `parent - self` (the sibling).
    pub fn subtract_from(&mut self, parent: &Self) {
        debug_assert_eq!(self.num_bins, parent.num_bins);
        debug_assert_eq!(self.n_outputs, parent.n_outputs);

        for i in 0..self.sum_grad.len() {
            self.sum_grad[i] = parent.sum_grad[i] - self.sum_grad[i];
            self.sum_hess[i] = parent.sum_hess[i] - self.sum_hess[i];
        }
        for i in 0..self.count.len() {
            self.count[i] = parent.count[i] - self.count[i];
        }
    }
}

// =============================================================================
// MultiOutputNodeHistogram
// =============================================================================

/// Collection of multi-output histograms for all features in a node.
#[derive(Debug, Clone)]
pub struct MultiOutputNodeHistogram {
    /// Histogram per feature
    pub features: Vec<MultiOutputFeatureHistogram>,
    /// Total gradient sums across all samples in node [K]
    total_grad: Vec<f32>,
    /// Total hessian sums across all samples in node [K]
    total_hess: Vec<f32>,
    /// Total sample count
    total_count: u32,
    /// Number of outputs
    n_outputs: u16,
}

impl MultiOutputNodeHistogram {
    /// Create histograms for all features.
    ///
    /// # Arguments
    /// - `num_bins_per_feature`: Number of bins for each feature
    /// - `n_outputs`: Number of outputs (K)
    pub fn new(num_bins_per_feature: &[u16], n_outputs: u16) -> Self {
        let features = num_bins_per_feature
            .iter()
            .map(|&num_bins| MultiOutputFeatureHistogram::new(num_bins, n_outputs))
            .collect();

        Self {
            features,
            total_grad: vec![0.0; n_outputs as usize],
            total_hess: vec![0.0; n_outputs as usize],
            total_count: 0,
            n_outputs,
        }
    }

    /// Number of features.
    pub fn num_features(&self) -> usize {
        self.features.len()
    }

    /// Number of outputs (K).
    pub fn n_outputs(&self) -> u16 {
        self.n_outputs
    }

    /// Total gradient sums for the node [K values].
    pub fn total_grad(&self) -> &[f32] {
        &self.total_grad
    }

    /// Total hessian sums for the node [K values].
    pub fn total_hess(&self) -> &[f32] {
        &self.total_hess
    }

    /// Total sample count in the node.
    pub fn total_count(&self) -> u32 {
        self.total_count
    }

    /// Reset all histograms.
    pub fn reset(&mut self) {
        for hist in &mut self.features {
            hist.reset();
        }
        self.total_grad.fill(0.0);
        self.total_hess.fill(0.0);
        self.total_count = 0;
    }

    /// Update totals by summing across all bins of feature 0.
    pub fn update_totals(&mut self) {
        if self.features.is_empty() {
            return;
        }

        // Sum from feature 0 (any feature would give same totals)
        let feat = &self.features[0];
        self.total_grad.fill(0.0);
        self.total_hess.fill(0.0);
        self.total_count = 0;

        for bin in 0..feat.num_bins() as usize {
            for k in 0..self.n_outputs as usize {
                self.total_grad[k] += feat.grad(bin, k);
                self.total_hess[k] += feat.hess(bin, k);
            }
            self.total_count += feat.count(bin);
        }
    }

    /// Compute sibling histogram via subtraction.
    pub fn subtract_from(&mut self, parent: &Self) {
        for (child, parent_feat) in self.features.iter_mut().zip(parent.features.iter()) {
            child.subtract_from(parent_feat);
        }
        for k in 0..self.n_outputs as usize {
            self.total_grad[k] = parent.total_grad[k] - self.total_grad[k];
            self.total_hess[k] = parent.total_hess[k] - self.total_hess[k];
        }
        self.total_count = parent.total_count - self.total_count;
    }
}

// =============================================================================
// MultiOutputHistogramBuilder
// =============================================================================

/// Builder for multi-output histograms.
pub struct MultiOutputHistogramBuilder {
    /// Number of bins per feature
    num_bins: Vec<u16>,
    /// Number of outputs (K)
    n_outputs: u16,
}

impl MultiOutputHistogramBuilder {
    /// Create a builder from bin cuts.
    pub fn new(cuts: &BinCuts, n_outputs: u16) -> Self {
        let num_bins: Vec<u16> = (0..cuts.num_features())
            .map(|f| cuts.num_bins(f) as u16)
            .collect();

        Self { num_bins, n_outputs }
    }

    /// Create a new node histogram.
    pub fn create_node_histogram(&self) -> MultiOutputNodeHistogram {
        MultiOutputNodeHistogram::new(&self.num_bins, self.n_outputs)
    }

    /// Build histogram for a node.
    pub fn build<B: BinIndex>(
        &self,
        hist: &mut MultiOutputNodeHistogram,
        index: &QuantizedMatrix<B>,
        grads: &GradientBuffer,
        rows: &[u32],
    ) {
        hist.reset();

        let num_features = hist.num_features();
        let _n_outputs = grads.n_outputs();

        // For each row in the node
        for &row in rows {
            let row_idx = row as usize;
            let row_grads = grads.sample_grads(row_idx);
            let row_hess = grads.sample_hess(row_idx);

            // For each feature, add to the appropriate bin
            for feat in 0..num_features {
                let bin = index.get(row, feat as u32).to_usize();
                hist.features[feat].add(bin, row_grads, row_hess);
            }
        }

        hist.update_totals();
    }
}

// =============================================================================
// Multi-output gain computation
// =============================================================================

/// Compute aggregated gain across all K outputs.
///
/// For multi-output trees, the gain is the sum of per-output gains:
/// ```text
/// total_gain = Σ_k [gain_k(left, right)] - γ
/// ```
///
/// Single γ penalty since we're making one structural decision.
pub fn multi_output_split_gain(
    left_grad: &[f32],
    left_hess: &[f32],
    right_grad: &[f32],
    right_hess: &[f32],
    lambda: f32,
    min_split_gain: f32,
) -> f32 {
    debug_assert_eq!(left_grad.len(), left_hess.len());
    debug_assert_eq!(left_grad.len(), right_grad.len());
    debug_assert_eq!(left_grad.len(), right_hess.len());

    let mut total_gain = 0.0f32;

    for k in 0..left_grad.len() {
        // Per-output gain: (G_L^2 / (H_L + λ)) + (G_R^2 / (H_R + λ)) - (G^2 / (H + λ))
        let gl = left_grad[k];
        let hl = left_hess[k];
        let gr = right_grad[k];
        let hr = right_hess[k];

        let g = gl + gr;
        let h = hl + hr;

        let left_score = (gl * gl) / (hl + lambda);
        let right_score = (gr * gr) / (hr + lambda);
        let parent_score = (g * g) / (h + lambda);

        total_gain += left_score + right_score - parent_score;
    }

    // Apply single γ penalty
    total_gain * 0.5 - min_split_gain
}

/// Compute vector leaf weights from aggregated statistics.
///
/// For each output k: weight_k = -sum_grad_k / (sum_hess_k + λ)
pub fn multi_output_leaf_weight(grad_sums: &[f32], hess_sums: &[f32], lambda: f32) -> Vec<f32> {
    debug_assert_eq!(grad_sums.len(), hess_sums.len());

    grad_sums
        .iter()
        .zip(hess_sums.iter())
        .map(|(&g, &h)| -g / (h + lambda))
        .collect()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_output_feature_histogram() {
        let mut hist = MultiOutputFeatureHistogram::new(4, 3);

        // Add sample to bin 1
        hist.add(1, &[1.0, 2.0, 3.0], &[0.5, 0.5, 0.5]);
        hist.add(1, &[0.5, 1.0, 1.5], &[0.5, 0.5, 0.5]);

        // Check bin 1 stats
        assert_eq!(hist.count(1), 2);
        assert!((hist.grad(1, 0) - 1.5).abs() < 1e-6);
        assert!((hist.grad(1, 1) - 3.0).abs() < 1e-6);
        assert!((hist.grad(1, 2) - 4.5).abs() < 1e-6);
        assert!((hist.hess(1, 0) - 1.0).abs() < 1e-6);

        // Check bin 0 (empty)
        assert_eq!(hist.count(0), 0);
        assert_eq!(hist.grad(0, 0), 0.0);
    }

    #[test]
    fn test_multi_output_node_histogram() {
        let num_bins = vec![4u16, 3, 5];
        let mut hist = MultiOutputNodeHistogram::new(&num_bins, 2);

        assert_eq!(hist.num_features(), 3);
        assert_eq!(hist.n_outputs(), 2);

        // Add to feature 0, bin 1
        hist.features[0].add(1, &[1.0, 2.0], &[0.5, 0.5]);

        hist.update_totals();

        assert!((hist.total_grad()[0] - 1.0).abs() < 1e-6);
        assert!((hist.total_grad()[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_multi_output_split_gain() {
        let left_grad = vec![-5.0, -3.0];
        let left_hess = vec![2.0, 2.0];
        let right_grad = vec![5.0, 3.0];
        let right_hess = vec![2.0, 2.0];

        let gain = multi_output_split_gain(
            &left_grad,
            &left_hess,
            &right_grad,
            &right_hess,
            1.0,  // lambda
            0.0,  // gamma
        );

        // Expected: 0.5 * sum of per-output gains
        // Output 0: (25/3 + 25/3 - 0/5) = 50/3 ≈ 16.67
        // Output 1: (9/3 + 9/3 - 0/5) = 18/3 = 6
        // Total: 0.5 * (16.67 + 6) ≈ 11.33
        assert!(gain > 10.0);
    }

    #[test]
    fn test_multi_output_leaf_weight() {
        let grad_sums = vec![-6.0, -4.0, 2.0];
        let hess_sums = vec![3.0, 2.0, 2.0];

        let weights = multi_output_leaf_weight(&grad_sums, &hess_sums, 1.0);

        // weight[k] = -grad[k] / (hess[k] + λ)
        assert!((weights[0] - 1.5).abs() < 1e-6);  // -(-6) / 4 = 1.5
        assert!((weights[1] - 4.0 / 3.0).abs() < 1e-6);  // -(-4) / 3 ≈ 1.33
        assert!((weights[2] - (-2.0 / 3.0)).abs() < 1e-6);  // -(2) / 3 ≈ -0.67
    }

    #[test]
    fn test_histogram_subtraction() {
        let mut parent = MultiOutputFeatureHistogram::new(4, 2);
        let mut child = MultiOutputFeatureHistogram::new(4, 2);

        // Parent has samples in bins 1 and 2
        parent.add(1, &[2.0, 3.0], &[1.0, 1.0]);
        parent.add(2, &[4.0, 5.0], &[2.0, 2.0]);

        // Child has sample in bin 1 only
        child.add(1, &[2.0, 3.0], &[1.0, 1.0]);

        // Subtract: sibling = parent - child
        child.subtract_from(&parent);

        // Sibling should have bin 2 only
        assert_eq!(child.count(1), 0);
        assert_eq!(child.count(2), 1);
        assert!((child.grad(2, 0) - 4.0).abs() < 1e-6);
        assert!((child.grad(2, 1) - 5.0).abs() < 1e-6);
    }
}

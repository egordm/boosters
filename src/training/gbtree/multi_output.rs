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

use std::collections::HashMap;

use crate::training::GradientBuffer;
use super::grower::TreeParams;
use super::partition::RowPartitioner;
use super::quantize::{BinCuts, BinIndex, QuantizedMatrix};
use super::split::GainParams;

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
// MultiOutputSplitInfo
// =============================================================================

/// Split information for multi-output trees.
///
/// Unlike scalar `SplitInfo`, this stores K weights per child.
#[derive(Clone, Debug)]
pub struct MultiOutputSplitInfo {
    /// Feature index to split on
    pub feature: u32,
    /// Bin index for the split (values in bins <= split_bin go left)
    pub split_bin: u32,
    /// Split threshold (go left if value <= threshold)
    pub threshold: f32,
    /// Aggregated gain from this split (sum across all outputs)
    pub gain: f32,
    /// Sum of gradients in left child [K]
    pub grad_left: Vec<f32>,
    /// Sum of hessians in left child [K]
    pub hess_left: Vec<f32>,
    /// Sum of gradients in right child [K]
    pub grad_right: Vec<f32>,
    /// Sum of hessians in right child [K]
    pub hess_right: Vec<f32>,
    /// Optimal weights for left leaf [K]
    pub weight_left: Vec<f32>,
    /// Optimal weights for right leaf [K]
    pub weight_right: Vec<f32>,
    /// Default direction for missing values (true = left)
    pub default_left: bool,
}

impl MultiOutputSplitInfo {
    /// Create an invalid split (no split found).
    pub fn invalid(n_outputs: usize) -> Self {
        Self {
            feature: 0,
            split_bin: 0,
            threshold: 0.0,
            gain: f32::NEG_INFINITY,
            grad_left: vec![0.0; n_outputs],
            hess_left: vec![0.0; n_outputs],
            grad_right: vec![0.0; n_outputs],
            hess_right: vec![0.0; n_outputs],
            weight_left: vec![0.0; n_outputs],
            weight_right: vec![0.0; n_outputs],
            default_left: true,
        }
    }

    /// Check if this split is valid (has positive gain).
    pub fn is_valid(&self) -> bool {
        self.gain > 0.0
    }

    /// Number of outputs (K).
    pub fn n_outputs(&self) -> usize {
        self.weight_left.len()
    }
}

// =============================================================================
// MultiOutputSplitFinder
// =============================================================================

/// Split finder for multi-output trees.
///
/// Finds best split by computing aggregated gain across all K outputs.
pub struct MultiOutputSplitFinder {
    /// Optional subset of features to consider
    pub feature_subset: Option<Vec<u32>>,
}

impl Default for MultiOutputSplitFinder {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiOutputSplitFinder {
    /// Create a new split finder.
    pub fn new() -> Self {
        Self { feature_subset: None }
    }

    /// Find the best split from a multi-output histogram.
    pub fn find_best_split(
        &self,
        hist: &MultiOutputNodeHistogram,
        cuts: &BinCuts,
        params: &GainParams,
    ) -> MultiOutputSplitInfo {
        let n_outputs = hist.n_outputs() as usize;
        let mut best = MultiOutputSplitInfo::invalid(n_outputs);

        let features: Vec<u32> = match &self.feature_subset {
            Some(subset) => subset.clone(),
            None => (0..hist.num_features() as u32).collect(),
        };

        for &feat in &features {
            let feat_hist = &hist.features[feat as usize];
            let feature_cuts = cuts.feature_cuts(feat);
            let num_bins = feat_hist.num_bins() as usize;

            // Accumulators for left side (scanning from bin 1)
            let mut left_grad = vec![0.0f32; n_outputs];
            let mut left_hess = vec![0.0f32; n_outputs];
            let mut left_count = 0u32;

            // Total (excluding missing bin 0)
            let total_grad: Vec<f32> = (0..n_outputs)
                .map(|k| (1..num_bins).map(|b| feat_hist.grad(b, k)).sum())
                .collect();
            let total_hess: Vec<f32> = (0..n_outputs)
                .map(|k| (1..num_bins).map(|b| feat_hist.hess(b, k)).sum())
                .collect();
            let total_count: u32 = (1..num_bins).map(|b| feat_hist.count(b)).sum();

            // Missing values (bin 0)
            let missing_grad: Vec<f32> = (0..n_outputs).map(|k| feat_hist.grad(0, k)).collect();
            let missing_hess: Vec<f32> = (0..n_outputs).map(|k| feat_hist.hess(0, k)).collect();
            let missing_count = feat_hist.count(0);

            // Scan through bins to find best split
            for bin in 1..num_bins {
                // Accumulate left stats
                for k in 0..n_outputs {
                    left_grad[k] += feat_hist.grad(bin, k);
                    left_hess[k] += feat_hist.hess(bin, k);
                }
                left_count += feat_hist.count(bin);

                // Right stats = total - left
                let right_grad: Vec<f32> = (0..n_outputs)
                    .map(|k| total_grad[k] - left_grad[k])
                    .collect();
                let right_hess: Vec<f32> = (0..n_outputs)
                    .map(|k| total_hess[k] - left_hess[k])
                    .collect();
                let right_count = total_count - left_count;

                // Skip if either child is too small
                let total_hess_left: f32 = left_hess.iter().sum();
                let total_hess_right: f32 = right_hess.iter().sum();
                if total_hess_left < params.min_child_weight
                    || total_hess_right < params.min_child_weight
                {
                    continue;
                }
                if left_count == 0 || right_count == 0 {
                    continue;
                }

                // Try missing values going left
                let mut cand_left_grad = left_grad.clone();
                let mut cand_left_hess = left_hess.clone();
                let mut cand_right_grad = right_grad.clone();
                let mut cand_right_hess = right_hess.clone();

                for k in 0..n_outputs {
                    cand_left_grad[k] += missing_grad[k];
                    cand_left_hess[k] += missing_hess[k];
                }

                let gain_left = multi_output_split_gain(
                    &cand_left_grad,
                    &cand_left_hess,
                    &cand_right_grad,
                    &cand_right_hess,
                    params.lambda,
                    params.min_split_gain,
                );

                // Try missing values going right
                cand_left_grad = left_grad.clone();
                cand_left_hess = left_hess.clone();
                cand_right_grad = right_grad.clone();
                cand_right_hess = right_hess.clone();

                for k in 0..n_outputs {
                    cand_right_grad[k] += missing_grad[k];
                    cand_right_hess[k] += missing_hess[k];
                }

                let gain_right = multi_output_split_gain(
                    &cand_left_grad,
                    &cand_left_hess,
                    &cand_right_grad,
                    &cand_right_hess,
                    params.lambda,
                    params.min_split_gain,
                );

                // Choose best direction for missing
                let (gain, default_left, final_left_grad, final_left_hess, final_right_grad, final_right_hess) =
                    if missing_count == 0 || gain_left >= gain_right {
                        // Missing left or no missing
                        let mut lg = left_grad.clone();
                        let mut lh = left_hess.clone();
                        for k in 0..n_outputs {
                            lg[k] += missing_grad[k];
                            lh[k] += missing_hess[k];
                        }
                        (gain_left, true, lg, lh, right_grad.clone(), right_hess.clone())
                    } else {
                        // Missing right
                        let mut rg = right_grad.clone();
                        let mut rh = right_hess.clone();
                        for k in 0..n_outputs {
                            rg[k] += missing_grad[k];
                            rh[k] += missing_hess[k];
                        }
                        (gain_right, false, left_grad.clone(), left_hess.clone(), rg, rh)
                    };

                if gain > best.gain {
                    let threshold = if bin < feature_cuts.len() {
                        feature_cuts[bin]
                    } else {
                        f32::INFINITY
                    };

                    best = MultiOutputSplitInfo {
                        feature: feat,
                        split_bin: bin as u32,
                        threshold,
                        gain,
                        grad_left: final_left_grad.clone(),
                        hess_left: final_left_hess.clone(),
                        grad_right: final_right_grad.clone(),
                        hess_right: final_right_hess.clone(),
                        weight_left: multi_output_leaf_weight(&final_left_grad, &final_left_hess, params.lambda),
                        weight_right: multi_output_leaf_weight(&final_right_grad, &final_right_hess, params.lambda),
                        default_left,
                    };
                }
            }
        }

        best
    }
}

// =============================================================================
// MultiOutputBuildingTree
// =============================================================================

/// A node in a multi-output tree being built.
#[derive(Clone, Debug)]
pub struct MultiOutputBuildingNode {
    /// Is this a leaf node?
    pub is_leaf: bool,
    /// Vector leaf weights [K] (valid if is_leaf)
    pub weights: Vec<f32>,
    /// Split info (valid if !is_leaf)
    pub split: Option<MultiOutputSplitInfo>,
    /// Left child node ID (0 means no child)
    pub left: u32,
    /// Right child node ID (0 means no child)
    pub right: u32,
    /// Partition ID in RowPartitioner
    pub partition_id: u32,
}

impl MultiOutputBuildingNode {
    /// Create a new leaf node.
    pub fn leaf(weights: Vec<f32>, partition_id: u32) -> Self {
        Self {
            is_leaf: true,
            weights,
            split: None,
            left: 0,
            right: 0,
            partition_id,
        }
    }
}

/// A multi-output tree being built.
#[derive(Clone, Debug)]
pub struct MultiOutputBuildingTree {
    /// All nodes in the tree
    nodes: Vec<MultiOutputBuildingNode>,
    /// Number of outputs (K)
    #[allow(dead_code)]
    n_outputs: usize,
}

impl MultiOutputBuildingTree {
    /// Create a new tree with a root node.
    pub fn new(n_outputs: usize) -> Self {
        let root = MultiOutputBuildingNode::leaf(vec![0.0; n_outputs], 0);
        Self {
            nodes: vec![root],
            n_outputs,
        }
    }

    /// Get a node by ID.
    pub fn node(&self, id: u32) -> &MultiOutputBuildingNode {
        &self.nodes[id as usize]
    }

    /// Get a mutable node by ID.
    pub fn node_mut(&mut self, id: u32) -> &mut MultiOutputBuildingNode {
        &mut self.nodes[id as usize]
    }

    /// Number of nodes in the tree.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Expand a leaf node into an internal node with two children.
    ///
    /// Returns (left_id, right_id).
    pub fn expand(
        &mut self,
        node_id: u32,
        split: MultiOutputSplitInfo,
        left_partition: u32,
        right_partition: u32,
    ) -> (u32, u32) {
        let left_id = self.nodes.len() as u32;
        let right_id = left_id + 1;

        // Create child nodes
        let left_node = MultiOutputBuildingNode::leaf(split.weight_left.clone(), left_partition);
        let right_node = MultiOutputBuildingNode::leaf(split.weight_right.clone(), right_partition);

        self.nodes.push(left_node);
        self.nodes.push(right_node);

        // Update parent
        let parent = &mut self.nodes[node_id as usize];
        parent.is_leaf = false;
        parent.split = Some(split);
        parent.left = left_id;
        parent.right = right_id;

        (left_id, right_id)
    }

    /// Apply learning rate to all leaf weights.
    pub fn apply_learning_rate(&mut self, eta: f32) {
        for node in &mut self.nodes {
            if node.is_leaf {
                for w in &mut node.weights {
                    *w *= eta;
                }
            }
        }
    }

    /// Iterate over leaf node IDs.
    pub fn leaves(&self) -> impl Iterator<Item = u32> + '_ {
        self.nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.is_leaf)
            .map(|(i, _)| i as u32)
    }
}

// =============================================================================
// MultiOutputTreeGrower
// =============================================================================

/// Growth strategy for multi-output trees.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MultiOutputGrowthStrategy {
    /// Depth-wise: expand all nodes at each level (XGBoost style)
    DepthWise,
    /// Leaf-wise: expand best-gain leaf first (LightGBM style)
    LeafWise,
}

impl Default for MultiOutputGrowthStrategy {
    fn default() -> Self {
        Self::DepthWise
    }
}

/// Tree grower for multi-output trees.
///
/// Grows a single tree with K-dimensional leaf values.
/// Supports both depth-wise (XGBoost) and leaf-wise (LightGBM) growth strategies.
pub struct MultiOutputTreeGrower<'a> {
    /// Histogram builder
    hist_builder: MultiOutputHistogramBuilder,
    /// Split finder
    split_finder: MultiOutputSplitFinder,
    /// Bin cuts
    cuts: &'a BinCuts,
    /// Training parameters
    params: TreeParams,
    /// Number of outputs
    n_outputs: usize,
    /// Growth strategy
    strategy: MultiOutputGrowthStrategy,
}

impl<'a> MultiOutputTreeGrower<'a> {
    /// Create a new multi-output tree grower with depth-wise strategy.
    pub fn new(cuts: &'a BinCuts, params: TreeParams, n_outputs: usize) -> Self {
        Self::with_strategy(cuts, params, n_outputs, MultiOutputGrowthStrategy::DepthWise)
    }

    /// Create a new multi-output tree grower with specified strategy.
    pub fn with_strategy(
        cuts: &'a BinCuts,
        params: TreeParams,
        n_outputs: usize,
        strategy: MultiOutputGrowthStrategy,
    ) -> Self {
        let hist_builder = MultiOutputHistogramBuilder::new(cuts, n_outputs as u16);
        Self {
            hist_builder,
            split_finder: MultiOutputSplitFinder::new(),
            cuts,
            params,
            n_outputs,
            strategy,
        }
    }

    /// Build a multi-output tree.
    pub fn build_tree<B: BinIndex>(
        &mut self,
        quantized: &QuantizedMatrix<B>,
        grads: &GradientBuffer,
        partitioner: &mut RowPartitioner,
    ) -> MultiOutputBuildingTree {
        match self.strategy {
            MultiOutputGrowthStrategy::DepthWise => {
                self.build_tree_depth_wise(quantized, grads, partitioner)
            }
            MultiOutputGrowthStrategy::LeafWise => {
                self.build_tree_leaf_wise(quantized, grads, partitioner)
            }
        }
    }

    /// Build tree using depth-wise expansion (XGBoost style).
    fn build_tree_depth_wise<B: BinIndex>(
        &mut self,
        quantized: &QuantizedMatrix<B>,
        grads: &GradientBuffer,
        partitioner: &mut RowPartitioner,
    ) -> MultiOutputBuildingTree {
        let mut tree = MultiOutputBuildingTree::new(self.n_outputs);

        // Histogram storage per node
        let mut histograms: HashMap<u32, MultiOutputNodeHistogram> = HashMap::new();

        // Build root histogram
        let root_rows = partitioner.node_rows(0);
        let mut root_hist = self.hist_builder.create_node_histogram();
        self.hist_builder.build(&mut root_hist, quantized, grads, root_rows);
        
        // Compute root weights from totals
        let root_weights = multi_output_leaf_weight(
            root_hist.total_grad(),
            root_hist.total_hess(),
            self.params.gain.lambda,
        );
        tree.node_mut(0).weights = root_weights;

        // Find initial split for root
        let root_split = self.split_finder.find_best_split(&root_hist, self.cuts, &self.params.gain);
        histograms.insert(0, root_hist);

        // Candidates for expansion: (node_id, split_info, depth)
        let mut candidates: HashMap<u32, (MultiOutputSplitInfo, u32)> = HashMap::new();
        if root_split.is_valid() {
            candidates.insert(0, (root_split, 0));
        }

        let max_depth = if self.params.max_depth == 0 { u32::MAX } else { self.params.max_depth };

        // Main growth loop - depth-wise expansion
        while !candidates.is_empty() {
            // Select nodes to expand at current depth that pass constraints
            let nodes_to_expand: Vec<u32> = candidates
                .iter()
                .filter(|(_, (split, depth))| {
                    split.is_valid() && *depth < max_depth
                })
                .map(|(&id, _)| id)
                .collect();

            if nodes_to_expand.is_empty() {
                break;
            }

            for node_id in nodes_to_expand {
                let (split, depth) = candidates.remove(&node_id).unwrap();
                self.expand_node(
                    &mut tree,
                    &mut histograms,
                    &mut candidates,
                    node_id,
                    split,
                    depth,
                    quantized,
                    grads,
                    partitioner,
                );
            }
        }

        // Apply learning rate
        tree.apply_learning_rate(self.params.learning_rate);

        tree
    }

    /// Build tree using leaf-wise expansion (LightGBM style).
    ///
    /// Always expands the leaf with highest gain until max_leaves is reached.
    fn build_tree_leaf_wise<B: BinIndex>(
        &mut self,
        quantized: &QuantizedMatrix<B>,
        grads: &GradientBuffer,
        partitioner: &mut RowPartitioner,
    ) -> MultiOutputBuildingTree {
        use std::collections::BinaryHeap;
        use std::cmp::Ordering;

        // Priority queue entry: (gain, node_id, depth)
        #[derive(Debug, Clone)]
        struct LeafCandidate {
            gain: f32,
            node_id: u32,
            depth: u32,
        }

        impl PartialEq for LeafCandidate {
            fn eq(&self, other: &Self) -> bool {
                self.gain == other.gain && self.node_id == other.node_id
            }
        }
        impl Eq for LeafCandidate {}

        impl PartialOrd for LeafCandidate {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for LeafCandidate {
            fn cmp(&self, other: &Self) -> Ordering {
                // Max-heap by gain
                self.gain.partial_cmp(&other.gain)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| self.node_id.cmp(&other.node_id))
            }
        }

        let mut tree = MultiOutputBuildingTree::new(self.n_outputs);

        // Histogram storage per node
        let mut histograms: HashMap<u32, MultiOutputNodeHistogram> = HashMap::new();
        // Split storage per node
        let mut splits: HashMap<u32, MultiOutputSplitInfo> = HashMap::new();

        // Build root histogram
        let root_rows = partitioner.node_rows(0);
        let mut root_hist = self.hist_builder.create_node_histogram();
        self.hist_builder.build(&mut root_hist, quantized, grads, root_rows);
        
        // Compute root weights
        let root_weights = multi_output_leaf_weight(
            root_hist.total_grad(),
            root_hist.total_hess(),
            self.params.gain.lambda,
        );
        tree.node_mut(0).weights = root_weights;

        // Find initial split for root
        let root_split = self.split_finder.find_best_split(&root_hist, self.cuts, &self.params.gain);
        histograms.insert(0, root_hist);

        // Priority queue of candidates
        let mut pq: BinaryHeap<LeafCandidate> = BinaryHeap::new();
        if root_split.is_valid() {
            pq.push(LeafCandidate {
                gain: root_split.gain,
                node_id: 0,
                depth: 0,
            });
            splits.insert(0, root_split);
        }

        let max_depth = if self.params.max_depth == 0 { u32::MAX } else { self.params.max_depth };
        let max_leaves = self.params.max_leaves.max(2); // At least 2 leaves
        let mut num_leaves = 1u32;

        // Main growth loop - leaf-wise: always expand best-gain candidate
        while let Some(candidate) = pq.pop() {
            // Check stopping conditions
            if num_leaves >= max_leaves {
                break;
            }
            if candidate.depth >= max_depth {
                continue; // Skip this candidate but try others
            }

            let node_id = candidate.node_id;
            let split = match splits.remove(&node_id) {
                Some(s) => s,
                None => continue,
            };

            // Expand this node
            let partition_id = tree.node(node_id).partition_id;
            let scalar_split = self.to_scalar_split(&split);
            let (left_partition, right_partition) =
                partitioner.apply_split(partition_id, &scalar_split, quantized);

            let (left_id, right_id) = tree.expand(node_id, split, left_partition, right_partition);
            num_leaves += 1; // Net +1 (remove 1 leaf, add 2)

            // Build child histograms
            let parent_hist = histograms.get(&node_id).unwrap();
            let left_size = partitioner.node_size(left_partition);
            let right_size = partitioner.node_size(right_partition);

            let (left_hist, right_hist) = if left_size <= right_size {
                let mut left = self.hist_builder.create_node_histogram();
                let left_rows = partitioner.node_rows(left_partition);
                self.hist_builder.build(&mut left, quantized, grads, left_rows);
                let right = self.subtract_histogram(parent_hist, &left);
                (left, right)
            } else {
                let mut right = self.hist_builder.create_node_histogram();
                let right_rows = partitioner.node_rows(right_partition);
                self.hist_builder.build(&mut right, quantized, grads, right_rows);
                let left = self.subtract_histogram(parent_hist, &right);
                (left, right)
            };

            // Find splits for children and add to priority queue
            let left_split = self.split_finder.find_best_split(&left_hist, self.cuts, &self.params.gain);
            let right_split = self.split_finder.find_best_split(&right_hist, self.cuts, &self.params.gain);

            histograms.insert(left_id, left_hist);
            histograms.insert(right_id, right_hist);

            if left_split.is_valid() {
                pq.push(LeafCandidate {
                    gain: left_split.gain,
                    node_id: left_id,
                    depth: candidate.depth + 1,
                });
                splits.insert(left_id, left_split);
            }
            if right_split.is_valid() {
                pq.push(LeafCandidate {
                    gain: right_split.gain,
                    node_id: right_id,
                    depth: candidate.depth + 1,
                });
                splits.insert(right_id, right_split);
            }
        }

        // Apply learning rate
        tree.apply_learning_rate(self.params.learning_rate);

        tree
    }

    /// Expand a single node (helper for depth-wise).
    #[allow(clippy::too_many_arguments)]
    fn expand_node<B: BinIndex>(
        &mut self,
        tree: &mut MultiOutputBuildingTree,
        histograms: &mut HashMap<u32, MultiOutputNodeHistogram>,
        candidates: &mut HashMap<u32, (MultiOutputSplitInfo, u32)>,
        node_id: u32,
        split: MultiOutputSplitInfo,
        depth: u32,
        quantized: &QuantizedMatrix<B>,
        grads: &GradientBuffer,
        partitioner: &mut RowPartitioner,
    ) {
        let partition_id = tree.node(node_id).partition_id;

        // Apply split to partitioner
        let scalar_split = self.to_scalar_split(&split);
        let (left_partition, right_partition) =
            partitioner.apply_split(partition_id, &scalar_split, quantized);

        // Expand tree
        let (left_id, right_id) = tree.expand(node_id, split, left_partition, right_partition);

        // Build child histograms using histogram subtraction
        let parent_hist = histograms.get(&node_id).unwrap();
        
        let left_size = partitioner.node_size(left_partition);
        let right_size = partitioner.node_size(right_partition);
        
        let (left_hist, right_hist) = if left_size <= right_size {
            let mut left = self.hist_builder.create_node_histogram();
            let left_rows = partitioner.node_rows(left_partition);
            self.hist_builder.build(&mut left, quantized, grads, left_rows);
            
            let right = self.subtract_histogram(parent_hist, &left);
            (left, right)
        } else {
            let mut right = self.hist_builder.create_node_histogram();
            let right_rows = partitioner.node_rows(right_partition);
            self.hist_builder.build(&mut right, quantized, grads, right_rows);
            
            let left = self.subtract_histogram(parent_hist, &right);
            (left, right)
        };

        // Find splits for children
        let left_split = self.split_finder.find_best_split(&left_hist, self.cuts, &self.params.gain);
        let right_split = self.split_finder.find_best_split(&right_hist, self.cuts, &self.params.gain);

        histograms.insert(left_id, left_hist);
        histograms.insert(right_id, right_hist);

        if left_split.is_valid() {
            candidates.insert(left_id, (left_split, depth + 1));
        }
        if right_split.is_valid() {
            candidates.insert(right_id, (right_split, depth + 1));
        }
    }

    /// Subtract child histogram from parent to get sibling histogram.
    fn subtract_histogram(
        &self,
        parent: &MultiOutputNodeHistogram,
        child: &MultiOutputNodeHistogram,
    ) -> MultiOutputNodeHistogram {
        let mut result = self.hist_builder.create_node_histogram();
        
        for (f, (ph, ch)) in parent.features.iter().zip(child.features.iter()).enumerate() {
            for bin in 0..ph.num_bins() as usize {
                for k in 0..self.n_outputs {
                    let pg = ph.grad(bin, k);
                    let ph_val = ph.hess(bin, k);
                    let cg = ch.grad(bin, k);
                    let ch_val = ch.hess(bin, k);
                    result.features[f].sum_grad[bin * self.n_outputs + k] = pg - cg;
                    result.features[f].sum_hess[bin * self.n_outputs + k] = ph_val - ch_val;
                }
                result.features[f].count[bin] = ph.count(bin) - ch.count(bin);
            }
        }
        result.update_totals();
        result
    }

    /// Convert multi-output split to scalar split for partitioner.
    fn to_scalar_split(&self, split: &MultiOutputSplitInfo) -> super::split::SplitInfo {
        super::split::SplitInfo {
            feature: split.feature,
            split_bin: split.split_bin,
            threshold: split.threshold,
            gain: split.gain,
            grad_left: split.grad_left.iter().sum(),
            hess_left: split.hess_left.iter().sum(),
            grad_right: split.grad_right.iter().sum(),
            hess_right: split.hess_right.iter().sum(),
            weight_left: split.weight_left.iter().sum::<f32>() / split.weight_left.len() as f32,
            weight_right: split.weight_right.iter().sum::<f32>() / split.weight_right.len() as f32,
            default_left: split.default_left,
            is_categorical: false,
            categories_left: Vec::new(),
        }
    }
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

    #[test]
    fn test_multi_output_tree_grower() {
        use crate::training::gbtree::{BinCuts, TreeParams, RowPartitioner, QuantizedMatrix};
        use crate::training::GradientBuffer;
        use std::sync::Arc;
        
        // 2-output problem (binary classification)
        let n_outputs = 2;
        let n_samples = 100usize;

        // Generate simple data: x < 0.5 -> class 0, x >= 0.5 -> class 1
        let mut features = vec![0.0f32; n_samples];
        let mut targets: Vec<usize> = vec![0; n_samples];

        for i in 0..n_samples {
            features[i] = i as f32 / n_samples as f32;
            targets[i] = if features[i] < 0.5 { 0 } else { 1 };
        }

        // Create simple cuts: 10 equal bins for one feature
        let cut_values: Vec<f32> = (1..=10).map(|i| i as f32 * 0.1).collect();
        let cut_ptrs = vec![0u32, cut_values.len() as u32];
        let cuts = BinCuts::new(cut_values, cut_ptrs);
        let cuts_arc = Arc::new(cuts);
        
        // Manual quantization: value to bin
        let mut index = vec![0u8; n_samples];
        for i in 0..n_samples {
            let v = features[i];
            // Bin 0 = missing, bins 1-11 = values
            let bin = if v.is_nan() {
                0u8
            } else {
                let b = ((v * 10.0) as usize).min(10);
                (b + 1) as u8  // Shift by 1 for missing bin
            };
            index[i] = bin;
        }
        let quantized = QuantizedMatrix::<u8>::new(index, n_samples as u32, 1, cuts_arc.clone());

        // Initial predictions (uniform)
        let preds = vec![0.5f32; n_samples * n_outputs];

        // Compute gradients for cross-entropy loss
        let mut grads = GradientBuffer::new(n_samples, n_outputs);
        for i in 0..n_samples {
            for k in 0..n_outputs {
                let p = preds[i * n_outputs + k];
                let y = if targets[i] == k { 1.0 } else { 0.0 };
                let g = p - y;
                let h = (p * (1.0 - p)).max(0.0001);
                grads.set(i, k, g, h);
            }
        }

        // Build tree using defaults
        let params = TreeParams::default();

        let mut partitioner = RowPartitioner::new(n_samples as u32);
        let mut grower = MultiOutputTreeGrower::new(&cuts_arc, params, n_outputs);
        let tree = grower.build_tree(&quantized, &grads, &mut partitioner);

        // Should have more than just root
        assert!(tree.num_nodes() >= 3, "Tree should have at least root + 2 children, got {}", tree.num_nodes());

        // Check leaf weights have correct dimension
        for leaf_id in tree.leaves() {
            let node = tree.node(leaf_id);
            assert_eq!(node.weights.len(), n_outputs);
        }
    }

    #[test]
    fn test_multi_output_tree_grower_leaf_wise() {
        use crate::training::gbtree::{BinCuts, TreeParams, RowPartitioner, QuantizedMatrix};
        use crate::training::GradientBuffer;
        use std::sync::Arc;
        
        // 2-output problem (binary classification)
        let n_outputs = 2;
        let n_samples = 100usize;

        // Generate simple data: x < 0.5 -> class 0, x >= 0.5 -> class 1
        let mut features = vec![0.0f32; n_samples];
        let mut targets: Vec<usize> = vec![0; n_samples];

        for i in 0..n_samples {
            features[i] = i as f32 / n_samples as f32;
            targets[i] = if features[i] < 0.5 { 0 } else { 1 };
        }

        // Create simple cuts: 10 equal bins for one feature
        let cut_values: Vec<f32> = (1..=10).map(|i| i as f32 * 0.1).collect();
        let cut_ptrs = vec![0u32, cut_values.len() as u32];
        let cuts = BinCuts::new(cut_values, cut_ptrs);
        let cuts_arc = Arc::new(cuts);
        
        // Manual quantization
        let mut index = vec![0u8; n_samples];
        for i in 0..n_samples {
            let v = features[i];
            let bin = if v.is_nan() {
                0u8
            } else {
                let b = ((v * 10.0) as usize).min(10);
                (b + 1) as u8
            };
            index[i] = bin;
        }
        let quantized = QuantizedMatrix::<u8>::new(index, n_samples as u32, 1, cuts_arc.clone());

        // Initial predictions and gradients
        let preds = vec![0.5f32; n_samples * n_outputs];
        let mut grads = GradientBuffer::new(n_samples, n_outputs);
        for i in 0..n_samples {
            for k in 0..n_outputs {
                let p = preds[i * n_outputs + k];
                let y = if targets[i] == k { 1.0 } else { 0.0 };
                grads.set(i, k, p - y, (p * (1.0 - p)).max(0.0001));
            }
        }

        // Build tree with leaf-wise strategy and max_leaves limit
        let mut params = TreeParams::default();
        params.max_leaves = 8; // Limit to 8 leaves
        params.max_depth = 10; // High depth to let leaf-wise decide

        let mut partitioner = RowPartitioner::new(n_samples as u32);
        let mut grower = MultiOutputTreeGrower::with_strategy(
            &cuts_arc, params, n_outputs, MultiOutputGrowthStrategy::LeafWise
        );
        let tree = grower.build_tree(&quantized, &grads, &mut partitioner);

        // Should have expanded (at least root + 2 children)
        assert!(tree.num_nodes() >= 3, "Tree should have at least 3 nodes, got {}", tree.num_nodes());

        // Count leaves - should be <= max_leaves
        let num_leaves: usize = tree.leaves().count();
        assert!(num_leaves <= 8, "Should have at most 8 leaves, got {}", num_leaves);

        // Check leaf weights have correct dimension
        for leaf_id in tree.leaves() {
            let node = tree.node(leaf_id);
            assert_eq!(node.weights.len(), n_outputs);
        }
    }
}

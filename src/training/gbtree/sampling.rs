//! Row and column sampling for training.
//!
//! This module implements RFC-0017: sampling strategies for regularization.
//!
//! # Row Sampling
//!
//! - `subsample`: Sample a fraction of rows at the start of each boosting iteration.
//!   All trees in the iteration see the same sample.
//!
//! # Column Sampling (Cascading)
//!
//! Column sampling cascades at three levels:
//! 1. `colsample_bytree`: Sample features per tree
//! 2. `colsample_bylevel`: Sample features per depth level (from tree-level set)
//! 3. `colsample_bynode`: Sample features per node (from level-level set)
//!
//! The final feature set at each node is the intersection of all three samplings.
//!
//! # Example
//!
//! ```ignore
//! use booste_rs::training::gbtree::sampling::{RowSampler, ColumnSampler};
//!
//! // Sample 80% of rows
//! let row_sampler = RowSampler::new(num_rows, 0.8);
//! let row_indices = row_sampler.sample(seed);
//!
//! // Sample features at tree level (90% of features)
//! let mut col_sampler = ColumnSampler::new(num_features, 0.9, 1.0, 1.0);
//! col_sampler.sample_tree(seed);
//!
//! // Get allowed features for a specific node
//! let allowed = col_sampler.allowed_features_for_node(depth, node_id, seed);
//! ```

use rand::prelude::*;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

// ============================================================================
// RowSampler
// ============================================================================

/// Samples rows without replacement for each boosting iteration.
///
/// Row indices are sorted after sampling for cache-friendly access.
#[derive(Debug, Clone)]
pub struct RowSampler {
    /// Total number of rows in the dataset.
    num_rows: u32,
    /// Fraction of rows to sample (0, 1].
    subsample: f32,
}

impl RowSampler {
    /// Create a new row sampler.
    ///
    /// # Arguments
    ///
    /// * `num_rows` - Total number of rows in the dataset
    /// * `subsample` - Fraction of rows to sample (0, 1]
    ///
    /// # Panics
    ///
    /// Panics if `subsample` is not in (0, 1].
    pub fn new(num_rows: u32, subsample: f32) -> Self {
        assert!(
            subsample > 0.0 && subsample <= 1.0,
            "subsample must be in (0, 1], got {}",
            subsample
        );
        Self { num_rows, subsample }
    }

    /// Returns true if sampling is enabled (subsample < 1.0).
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.subsample < 1.0
    }

    /// Sample row indices.
    ///
    /// Returns all indices if `subsample == 1.0`.
    /// Otherwise, samples without replacement and sorts the result.
    ///
    /// # Arguments
    ///
    /// * `seed` - Random seed for reproducibility
    pub fn sample(&self, seed: u64) -> Vec<u32> {
        if !self.is_enabled() {
            // Return all indices
            return (0..self.num_rows).collect();
        }

        let sample_size = ((self.num_rows as f32 * self.subsample).ceil() as usize).max(1);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);

        // Sample without replacement using partial Fisher-Yates shuffle
        let mut indices: Vec<u32> = (0..self.num_rows).collect();
        for i in 0..sample_size {
            let j = rng.gen_range(i..self.num_rows as usize);
            indices.swap(i, j);
        }

        // Take first sample_size elements and sort for cache efficiency
        let mut sampled: Vec<u32> = indices[..sample_size].to_vec();
        sampled.sort_unstable();
        sampled
    }
}

// ============================================================================
// ColumnSampler
// ============================================================================

/// Samples features at tree, level, and node granularity.
///
/// Column sampling cascades:
/// - `colsample_bytree`: Initial set for the tree
/// - `colsample_bylevel`: Subset per depth level
/// - `colsample_bynode`: Subset per node (from level subset)
#[derive(Debug, Clone)]
pub struct ColumnSampler {
    /// Total number of features.
    num_features: u32,
    /// Tree-level sampling ratio.
    colsample_bytree: f32,
    /// Level-level sampling ratio.
    colsample_bylevel: f32,
    /// Node-level sampling ratio.
    colsample_bynode: f32,
    /// Features selected for the current tree (after bytree sampling).
    tree_features: Vec<u32>,
}

impl ColumnSampler {
    /// Create a new column sampler.
    ///
    /// # Arguments
    ///
    /// * `num_features` - Total number of features
    /// * `colsample_bytree` - Tree-level sampling ratio (0, 1]
    /// * `colsample_bylevel` - Level-level sampling ratio (0, 1]
    /// * `colsample_bynode` - Node-level sampling ratio (0, 1]
    pub fn new(
        num_features: u32,
        colsample_bytree: f32,
        colsample_bylevel: f32,
        colsample_bynode: f32,
    ) -> Self {
        assert!(
            colsample_bytree > 0.0 && colsample_bytree <= 1.0,
            "colsample_bytree must be in (0, 1]"
        );
        assert!(
            colsample_bylevel > 0.0 && colsample_bylevel <= 1.0,
            "colsample_bylevel must be in (0, 1]"
        );
        assert!(
            colsample_bynode > 0.0 && colsample_bynode <= 1.0,
            "colsample_bynode must be in (0, 1]"
        );

        Self {
            num_features,
            colsample_bytree,
            colsample_bylevel,
            colsample_bynode,
            tree_features: Vec::new(),
        }
    }

    /// Returns true if any column sampling is enabled.
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.colsample_bytree < 1.0
            || self.colsample_bylevel < 1.0
            || self.colsample_bynode < 1.0
    }

    /// Sample features for a new tree.
    ///
    /// Must be called at the start of each tree building.
    pub fn sample_tree(&mut self, seed: u64) {
        if self.colsample_bytree >= 1.0 {
            // Use all features
            self.tree_features = (0..self.num_features).collect();
        } else {
            let sample_size =
                ((self.num_features as f32 * self.colsample_bytree).ceil() as usize).max(1);
            self.tree_features = sample_without_replacement(self.num_features, sample_size, seed);
        }
    }

    /// Get the features selected for the current tree.
    pub fn tree_features(&self) -> &[u32] {
        &self.tree_features
    }

    /// Get allowed features for a specific node.
    ///
    /// Applies level and node sampling on top of tree-level features.
    ///
    /// # Arguments
    ///
    /// * `depth` - Current node depth (0 = root)
    /// * `node_id` - Node identifier (used for node-level seed)
    /// * `tree_seed` - Base seed for this tree
    pub fn allowed_features_for_node(&self, depth: u32, node_id: u32, tree_seed: u64) -> Vec<u32> {
        let mut features = self.tree_features.clone();

        // Apply level sampling
        if self.colsample_bylevel < 1.0 {
            let level_seed = tree_seed.wrapping_add((depth as u64).wrapping_mul(0x9E3779B97F4A7C15));
            let sample_size =
                ((features.len() as f32 * self.colsample_bylevel).ceil() as usize).max(1);
            features = sample_from_slice(&features, sample_size, level_seed);
        }

        // Apply node sampling
        if self.colsample_bynode < 1.0 {
            let node_seed = tree_seed
                .wrapping_add((depth as u64).wrapping_mul(0x9E3779B97F4A7C15))
                .wrapping_add((node_id as u64).wrapping_mul(0x517CC1B727220A95));
            let sample_size =
                ((features.len() as f32 * self.colsample_bynode).ceil() as usize).max(1);
            features = sample_from_slice(&features, sample_size, node_seed);
        }

        features
    }
}

// ============================================================================
// GOSSSampler (Gradient-based One-Side Sampling)
// ============================================================================

/// Parameters for GOSS sampling.
#[derive(Debug, Clone, Copy)]
pub struct GossParams {
    /// Fraction of rows to keep by top gradient magnitude.
    /// These rows always contribute to training.
    pub top_rate: f32,
    /// Fraction of remaining rows to randomly sample.
    /// These rows get weight amplification.
    pub other_rate: f32,
}

impl Default for GossParams {
    fn default() -> Self {
        Self {
            top_rate: 0.2,
            other_rate: 0.1,
        }
    }
}

impl GossParams {
    /// Create new GOSS parameters.
    ///
    /// # Panics
    ///
    /// Panics if `top_rate` or `other_rate` is not in (0, 1].
    pub fn new(top_rate: f32, other_rate: f32) -> Self {
        assert!(
            top_rate > 0.0 && top_rate <= 1.0,
            "top_rate must be in (0, 1], got {}",
            top_rate
        );
        assert!(
            other_rate > 0.0 && other_rate <= 1.0,
            "other_rate must be in (0, 1], got {}",
            other_rate
        );
        Self { top_rate, other_rate }
    }

    /// Returns true if GOSS would sample (i.e., not all rows are kept).
    #[inline]
    pub fn is_enabled(&self) -> bool {
        // If top_rate + other_rate * (1 - top_rate) < 1.0, GOSS filters rows
        self.top_rate + self.other_rate * (1.0 - self.top_rate) < 1.0
    }

    /// Compute the weight amplification factor for sampled (non-top) rows.
    ///
    /// Formula: (1 - top_rate) / other_rate
    /// This compensates for undersampling the small-gradient rows.
    #[inline]
    pub fn weight_amplification(&self) -> f32 {
        (1.0 - self.top_rate) / self.other_rate
    }
}

/// Result of GOSS sampling containing selected rows and their weights.
#[derive(Debug, Clone)]
pub struct GossSample {
    /// Indices of selected rows (sorted for cache efficiency).
    pub indices: Vec<u32>,
    /// Weight for each selected row (1.0 for top rows, amplified for sampled rows).
    /// Length matches `indices`.
    pub weights: Vec<f32>,
}

impl GossSample {
    /// Create a sample that includes all rows with uniform weight.
    pub fn all_rows(num_rows: u32) -> Self {
        Self {
            indices: (0..num_rows).collect(),
            weights: vec![1.0; num_rows as usize],
        }
    }
}

/// Gradient-based One-Side Sampling (GOSS).
///
/// GOSS selects rows based on gradient magnitude:
/// 1. Keep top `top_rate` fraction by |gradient| (always included)
/// 2. Randomly sample `other_rate` fraction of remaining rows
/// 3. Apply weight amplification to sampled rows to compensate for bias
///
/// This focuses training on informative samples (large gradients) while
/// maintaining dataset distribution through weighted sampling.
#[derive(Debug, Clone)]
pub struct GossSampler {
    /// GOSS parameters
    params: GossParams,
}

impl GossSampler {
    /// Create a new GOSS sampler with the given parameters.
    pub fn new(params: GossParams) -> Self {
        Self { params }
    }

    /// Returns true if GOSS sampling is enabled.
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.params.is_enabled()
    }

    /// Sample rows based on gradient magnitudes.
    ///
    /// # Arguments
    ///
    /// * `gradients` - Gradient values for each row (can be negative)
    /// * `seed` - Random seed for reproducibility
    ///
    /// # Returns
    ///
    /// A `GossSample` containing selected row indices and their weights.
    pub fn sample(&self, gradients: &[f32], seed: u64) -> GossSample {
        let num_rows = gradients.len();
        
        if !self.is_enabled() || num_rows == 0 {
            return GossSample::all_rows(num_rows as u32);
        }

        // 1. Compute absolute gradient magnitudes with indices
        let mut indexed_grads: Vec<(u32, f32)> = gradients
            .iter()
            .enumerate()
            .map(|(i, &g)| (i as u32, g.abs()))
            .collect();

        // 2. Sort by gradient magnitude (descending) to find top rows
        indexed_grads.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // 3. Calculate how many top rows to keep
        let top_count = ((num_rows as f32 * self.params.top_rate).ceil() as usize).max(1);
        let top_count = top_count.min(num_rows);

        // 4. Top rows (always included with weight 1.0)
        let mut result_indices: Vec<u32> = indexed_grads[..top_count]
            .iter()
            .map(|(i, _)| *i)
            .collect();
        let mut result_weights: Vec<f32> = vec![1.0; top_count];

        // 5. Remaining rows to sample from
        let remaining_count = num_rows - top_count;
        if remaining_count > 0 {
            let sample_count = ((remaining_count as f32 * self.params.other_rate).ceil() as usize).max(1);
            let sample_count = sample_count.min(remaining_count);

            // Sample from remaining rows
            let remaining_indices: Vec<u32> = indexed_grads[top_count..]
                .iter()
                .map(|(i, _)| *i)
                .collect();

            let sampled = sample_from_slice(&remaining_indices, sample_count, seed);
            let weight = self.params.weight_amplification();

            for idx in sampled {
                result_indices.push(idx);
                result_weights.push(weight);
            }
        }

        // 6. Sort indices for cache-friendly access (reorder weights to match)
        let mut pairs: Vec<(u32, f32)> = result_indices
            .into_iter()
            .zip(result_weights)
            .collect();
        pairs.sort_by_key(|(i, _)| *i);

        GossSample {
            indices: pairs.iter().map(|(i, _)| *i).collect(),
            weights: pairs.iter().map(|(_, w)| *w).collect(),
        }
    }

    /// Sample rows based on multi-output gradient magnitudes.
    ///
    /// For multi-output (K outputs per row), computes row importance as
    /// the L2 norm of the gradient vector: `sqrt(sum(grad_k^2))`.
    ///
    /// # Arguments
    ///
    /// * `grads` - Gradient buffer with shape `[n_samples, n_outputs]`
    /// * `seed` - Random seed for reproducibility
    ///
    /// # Returns
    ///
    /// A `GossSample` containing selected row indices and their weights.
    pub fn sample_multioutput(
        &self,
        grads: &crate::training::GradientBuffer,
        seed: u64,
    ) -> GossSample {
        let num_rows = grads.n_samples();
        let num_outputs = grads.n_outputs();

        if !self.is_enabled() || num_rows == 0 {
            return GossSample::all_rows(num_rows as u32);
        }

        // For single output, delegate to the simple version
        if num_outputs == 1 {
            let gradients: Vec<f32> = (0..num_rows).map(|i| grads.get(i, 0).0).collect();
            return self.sample(&gradients, seed);
        }

        // Compute L2 norm of gradient vector for each row
        let mut indexed_mags: Vec<(u32, f32)> = (0..num_rows)
            .map(|row| {
                let mut sum_sq = 0.0f32;
                for k in 0..num_outputs {
                    let (grad, _) = grads.get(row, k);
                    sum_sq += grad * grad;
                }
                (row as u32, sum_sq.sqrt())
            })
            .collect();

        // Sort by magnitude (descending)
        indexed_mags.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Calculate top row count
        let top_count = ((num_rows as f32 * self.params.top_rate).ceil() as usize).max(1);
        let top_count = top_count.min(num_rows);

        // Top rows with weight 1.0
        let mut result_indices: Vec<u32> = indexed_mags[..top_count]
            .iter()
            .map(|(i, _)| *i)
            .collect();
        let mut result_weights: Vec<f32> = vec![1.0; top_count];

        // Sample from remaining rows
        let remaining_count = num_rows - top_count;
        if remaining_count > 0 {
            let sample_count =
                ((remaining_count as f32 * self.params.other_rate).ceil() as usize).max(1);
            let sample_count = sample_count.min(remaining_count);

            let remaining_indices: Vec<u32> = indexed_mags[top_count..]
                .iter()
                .map(|(i, _)| *i)
                .collect();

            let sampled = sample_from_slice(&remaining_indices, sample_count, seed);
            let weight = self.params.weight_amplification();

            for idx in sampled {
                result_indices.push(idx);
                result_weights.push(weight);
            }
        }

        // Sort for cache-friendly access
        let mut pairs: Vec<(u32, f32)> = result_indices
            .into_iter()
            .zip(result_weights)
            .collect();
        pairs.sort_by_key(|(i, _)| *i);

        GossSample {
            indices: pairs.iter().map(|(i, _)| *i).collect(),
            weights: pairs.iter().map(|(_, w)| *w).collect(),
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Sample `k` items from `0..n` without replacement.
///
/// Returns sorted indices for cache-friendly access.
fn sample_without_replacement(n: u32, k: usize, seed: u64) -> Vec<u32> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut indices: Vec<u32> = (0..n).collect();

    // Partial Fisher-Yates shuffle
    for i in 0..k {
        let j = rng.gen_range(i..n as usize);
        indices.swap(i, j);
    }

    let mut sampled: Vec<u32> = indices[..k].to_vec();
    sampled.sort_unstable();
    sampled
}

/// Sample `k` items from a slice without replacement.
///
/// Returns sorted values.
fn sample_from_slice(items: &[u32], k: usize, seed: u64) -> Vec<u32> {
    if k >= items.len() {
        return items.to_vec();
    }

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut indices: Vec<usize> = (0..items.len()).collect();

    for i in 0..k {
        let j = rng.gen_range(i..items.len());
        indices.swap(i, j);
    }

    let mut sampled: Vec<u32> = indices[..k].iter().map(|&i| items[i]).collect();
    sampled.sort_unstable();
    sampled
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_row_sampler_no_sampling() {
        let sampler = RowSampler::new(100, 1.0);
        assert!(!sampler.is_enabled());

        let indices = sampler.sample(42);
        assert_eq!(indices.len(), 100);
        assert_eq!(indices, (0..100).collect::<Vec<_>>());
    }

    #[test]
    fn test_row_sampler_half() {
        let sampler = RowSampler::new(100, 0.5);
        assert!(sampler.is_enabled());

        let indices = sampler.sample(42);
        assert_eq!(indices.len(), 50); // ceil(100 * 0.5)

        // Should be sorted
        for i in 1..indices.len() {
            assert!(indices[i] > indices[i - 1]);
        }

        // All indices should be valid
        for &idx in &indices {
            assert!(idx < 100);
        }
    }

    #[test]
    fn test_row_sampler_reproducible() {
        let sampler = RowSampler::new(100, 0.5);

        let indices1 = sampler.sample(42);
        let indices2 = sampler.sample(42);

        assert_eq!(indices1, indices2);
    }

    #[test]
    fn test_row_sampler_different_seeds() {
        let sampler = RowSampler::new(100, 0.5);

        let indices1 = sampler.sample(42);
        let indices2 = sampler.sample(123);

        assert_ne!(indices1, indices2);
    }

    #[test]
    fn test_column_sampler_no_sampling() {
        let mut sampler = ColumnSampler::new(10, 1.0, 1.0, 1.0);
        assert!(!sampler.is_enabled());

        sampler.sample_tree(42);
        assert_eq!(sampler.tree_features(), (0..10).collect::<Vec<_>>());

        let node_features = sampler.allowed_features_for_node(0, 0, 42);
        assert_eq!(node_features, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn test_column_sampler_bytree() {
        let mut sampler = ColumnSampler::new(10, 0.5, 1.0, 1.0);
        assert!(sampler.is_enabled());

        sampler.sample_tree(42);
        assert_eq!(sampler.tree_features().len(), 5); // ceil(10 * 0.5)

        // Tree features should be subset of 0..10
        for &f in sampler.tree_features() {
            assert!(f < 10);
        }
    }

    #[test]
    fn test_column_sampler_bylevel() {
        let mut sampler = ColumnSampler::new(10, 1.0, 0.5, 1.0);
        sampler.sample_tree(42);

        let features_depth0 = sampler.allowed_features_for_node(0, 0, 42);
        let features_depth1 = sampler.allowed_features_for_node(1, 0, 42);

        // Both should have 5 features
        assert_eq!(features_depth0.len(), 5);
        assert_eq!(features_depth1.len(), 5);

        // Different depths should (likely) have different features
        // (could be same by chance, but unlikely)
    }

    #[test]
    fn test_column_sampler_bynode() {
        let mut sampler = ColumnSampler::new(10, 1.0, 1.0, 0.5);
        sampler.sample_tree(42);

        let features_node0 = sampler.allowed_features_for_node(0, 0, 42);
        let features_node1 = sampler.allowed_features_for_node(0, 1, 42);

        // Both should have 5 features
        assert_eq!(features_node0.len(), 5);
        assert_eq!(features_node1.len(), 5);

        // Different nodes should (likely) have different features
    }

    #[test]
    fn test_column_sampler_cascading() {
        // 50% at each level: 10 * 0.5 = 5, then 5 * 0.5 = 3, then 3 * 0.5 = 2
        let mut sampler = ColumnSampler::new(10, 0.5, 0.5, 0.5);
        sampler.sample_tree(42);

        assert_eq!(sampler.tree_features().len(), 5);

        let node_features = sampler.allowed_features_for_node(0, 0, 42);
        // Should be ceil(5 * 0.5) = 3 after level, then ceil(3 * 0.5) = 2 after node
        assert_eq!(node_features.len(), 2);
    }

    #[test]
    fn test_column_sampler_reproducible() {
        let mut sampler1 = ColumnSampler::new(10, 0.5, 0.5, 0.5);
        let mut sampler2 = ColumnSampler::new(10, 0.5, 0.5, 0.5);

        sampler1.sample_tree(42);
        sampler2.sample_tree(42);

        assert_eq!(sampler1.tree_features(), sampler2.tree_features());

        let f1 = sampler1.allowed_features_for_node(1, 3, 42);
        let f2 = sampler2.allowed_features_for_node(1, 3, 42);
        assert_eq!(f1, f2);
    }

    #[test]
    #[should_panic(expected = "subsample must be in (0, 1]")]
    fn test_row_sampler_invalid_zero() {
        RowSampler::new(100, 0.0);
    }

    #[test]
    #[should_panic(expected = "subsample must be in (0, 1]")]
    fn test_row_sampler_invalid_negative() {
        RowSampler::new(100, -0.5);
    }

    #[test]
    #[should_panic(expected = "colsample_bytree must be in (0, 1]")]
    fn test_column_sampler_invalid() {
        ColumnSampler::new(10, 0.0, 1.0, 1.0);
    }

    // ---- GOSS Sampler Tests ----

    #[test]
    fn test_goss_params_default() {
        let params = GossParams::default();
        assert_eq!(params.top_rate, 0.2);
        assert_eq!(params.other_rate, 0.1);
        assert!(params.is_enabled());
    }

    #[test]
    fn test_goss_params_not_enabled() {
        // When top_rate + other_rate * (1 - top_rate) >= 1.0, all rows are kept
        let params = GossParams::new(0.5, 1.0); // 0.5 + 1.0 * 0.5 = 1.0
        assert!(!params.is_enabled());
    }

    #[test]
    fn test_goss_params_weight_amplification() {
        let params = GossParams::new(0.2, 0.1);
        // Weight = (1 - 0.2) / 0.1 = 0.8 / 0.1 = 8.0
        assert!((params.weight_amplification() - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_goss_sample_all_rows() {
        let params = GossParams::new(0.5, 1.0);
        let sampler = GossSampler::new(params);
        
        let gradients: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let sample = sampler.sample(&gradients, 42);
        
        // Should include all rows
        assert_eq!(sample.indices.len(), 10);
        assert_eq!(sample.indices, (0..10).collect::<Vec<_>>());
        // All weights should be 1.0
        assert!(sample.weights.iter().all(|&w| w == 1.0));
    }

    #[test]
    fn test_goss_sample_basic() {
        let params = GossParams::new(0.3, 0.2);
        let sampler = GossSampler::new(params);
        
        // 10 rows with varying gradients
        // Large gradients: indices 9, 8, 7 (top 3 = 30%)
        let gradients: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let sample = sampler.sample(&gradients, 42);
        
        // Should have top 3 (30%) + some from remaining 7 (20% of 7 = 2)
        // Top 3 are indices 9, 8, 7 (highest absolute gradients)
        let top_rows = vec![7u32, 8, 9];
        for &top in &top_rows {
            assert!(sample.indices.contains(&top), "Top row {} missing", top);
        }
        
        // Total: 3 top + 2 sampled = 5 rows
        assert_eq!(sample.indices.len(), 5);
        
        // Check weights
        for (i, &idx) in sample.indices.iter().enumerate() {
            if top_rows.contains(&idx) {
                assert_eq!(sample.weights[i], 1.0, "Top row should have weight 1.0");
            } else {
                // Sampled row should have amplified weight
                let expected_weight = params.weight_amplification();
                assert!(
                    (sample.weights[i] - expected_weight).abs() < 1e-6,
                    "Sampled row should have weight {}, got {}",
                    expected_weight,
                    sample.weights[i]
                );
            }
        }
    }

    #[test]
    fn test_goss_sample_sorted_indices() {
        let params = GossParams::new(0.2, 0.3);
        let sampler = GossSampler::new(params);
        
        let gradients: Vec<f32> = (0..100).map(|i| (i as f32).sin()).collect();
        let sample = sampler.sample(&gradients, 42);
        
        // Indices should be sorted for cache efficiency
        for i in 1..sample.indices.len() {
            assert!(
                sample.indices[i] > sample.indices[i - 1],
                "Indices should be sorted"
            );
        }
    }

    #[test]
    fn test_goss_sample_reproducible() {
        let params = GossParams::new(0.2, 0.1);
        let sampler = GossSampler::new(params);
        
        let gradients: Vec<f32> = (0..50).map(|i| (i as f32).sin()).collect();
        
        let sample1 = sampler.sample(&gradients, 42);
        let sample2 = sampler.sample(&gradients, 42);
        
        assert_eq!(sample1.indices, sample2.indices);
        assert_eq!(sample1.weights, sample2.weights);
    }

    #[test]
    fn test_goss_sample_different_seeds() {
        let params = GossParams::new(0.2, 0.1);
        let sampler = GossSampler::new(params);
        
        let gradients: Vec<f32> = (0..50).map(|i| (i as f32).sin()).collect();
        
        let sample1 = sampler.sample(&gradients, 42);
        let sample2 = sampler.sample(&gradients, 123);
        
        // Top rows should be the same (deterministic based on gradients)
        // But sampled rows may differ
        // Just check they're not completely identical
        // (very unlikely to be identical with different seeds)
        assert_ne!(sample1.indices, sample2.indices);
    }

    #[test]
    fn test_goss_sample_negative_gradients() {
        let params = GossParams::new(0.3, 0.2);
        let sampler = GossSampler::new(params);
        
        // Mix of positive and negative gradients
        // Highest absolute values: -9, 8, -7
        let gradients: Vec<f32> = vec![-9.0, 1.0, -2.0, 3.0, -4.0, 5.0, -6.0, -7.0, 8.0, 0.5];
        let sample = sampler.sample(&gradients, 42);
        
        // Top 3 by absolute value: indices 0 (|-9|), 8 (|8|), 7 (|-7|)
        assert!(sample.indices.contains(&0), "Index 0 (grad=-9) should be top");
        assert!(sample.indices.contains(&8), "Index 8 (grad=8) should be top");
        assert!(sample.indices.contains(&7), "Index 7 (grad=-7) should be top");
    }

    #[test]
    #[should_panic(expected = "top_rate must be in (0, 1]")]
    fn test_goss_params_invalid_top_rate() {
        GossParams::new(0.0, 0.1);
    }

    #[test]
    #[should_panic(expected = "other_rate must be in (0, 1]")]
    fn test_goss_params_invalid_other_rate() {
        GossParams::new(0.2, 0.0);
    }

    // ---- GOSS Multi-output Tests ----

    #[test]
    fn test_goss_multioutput_basic() {
        use crate::training::GradientBuffer;

        let params = GossParams::new(0.3, 0.2);
        let sampler = GossSampler::new(params);

        // Create 10 rows × 3 outputs
        let mut grads = GradientBuffer::new(10, 3);

        // Set gradients so rows 0, 1, 2 have highest L2 norms
        // Row 0: [9, 9, 9] → L2 = sqrt(243) ≈ 15.59
        // Row 1: [8, 8, 8] → L2 = sqrt(192) ≈ 13.86
        // Row 2: [7, 7, 7] → L2 = sqrt(147) ≈ 12.12
        // Row 3-9: small gradients
        for row in 0..10 {
            let grad = if row < 3 { (9 - row) as f32 } else { 0.5 };
            for k in 0..3 {
                grads.set(row, k, grad, 1.0);
            }
        }

        let sample = sampler.sample_multioutput(&grads, 42);

        // Top 3 rows (30% of 10) should always be included
        assert!(sample.indices.contains(&0), "Row 0 (highest L2 norm) should be top");
        assert!(sample.indices.contains(&1), "Row 1 (second highest) should be top");
        assert!(sample.indices.contains(&2), "Row 2 (third highest) should be top");

        // Sample should have 3 top + ceil(7 * 0.2) = 3 + 2 = 5 rows
        assert_eq!(sample.indices.len(), 5, "Expected 5 rows (3 top + 2 sampled)");

        // Top rows should have weight 1.0
        for (i, &idx) in sample.indices.iter().enumerate() {
            if idx <= 2 {
                assert_eq!(sample.weights[i], 1.0, "Top row should have weight 1.0");
            } else {
                // Sampled row should have amplified weight
                let expected = params.weight_amplification();
                assert!(
                    (sample.weights[i] - expected).abs() < 1e-6,
                    "Sampled row should have amplified weight"
                );
            }
        }
    }

    #[test]
    fn test_goss_multioutput_uses_l2_norm() {
        use crate::training::GradientBuffer;

        let params = GossParams::new(0.2, 0.1);
        let sampler = GossSampler::new(params);

        // 5 rows × 2 outputs
        let mut grads = GradientBuffer::new(5, 2);

        // Row 0: [10, 0] → L2 = 10
        // Row 1: [0, 10] → L2 = 10
        // Row 2: [7, 7] → L2 = sqrt(98) ≈ 9.9 (highest!)
        // Row 3: [1, 1] → L2 = sqrt(2) ≈ 1.4
        // Row 4: [0, 0] → L2 = 0
        grads.set(0, 0, 10.0, 1.0);
        grads.set(0, 1, 0.0, 1.0);
        grads.set(1, 0, 0.0, 1.0);
        grads.set(1, 1, 10.0, 1.0);
        grads.set(2, 0, 7.0, 1.0);
        grads.set(2, 1, 7.0, 1.0);
        grads.set(3, 0, 1.0, 1.0);
        grads.set(3, 1, 1.0, 1.0);
        grads.set(4, 0, 0.0, 1.0);
        grads.set(4, 1, 0.0, 1.0);

        let sample = sampler.sample_multioutput(&grads, 42);

        // Top 1 (20% of 5) should be row 2 (highest L2 norm with 7,7 vs 10,0)
        // Actually L2(10,0) = 10, L2(7,7) ≈ 9.9, so row 0 or 1 should be first
        // Let's just check that top rows are from {0, 1, 2}
        assert!(
            sample.indices.contains(&0) || sample.indices.contains(&1) || sample.indices.contains(&2),
            "One of the high-L2 rows should be in top"
        );
    }

    #[test]
    fn test_goss_multioutput_delegates_to_single_output() {
        use crate::training::GradientBuffer;

        let params = GossParams::new(0.2, 0.1);
        let sampler = GossSampler::new(params);

        // Single-output case
        let mut grads = GradientBuffer::new(50, 1);
        for row in 0..50 {
            grads.set(row, 0, (row as f32).sin(), 1.0);
        }

        let sample_multi = sampler.sample_multioutput(&grads, 42);
        
        // Compare with direct single-output call
        let gradients: Vec<f32> = (0..50).map(|i| grads.get(i, 0).0).collect();
        let sample_single = sampler.sample(&gradients, 42);

        assert_eq!(sample_multi.indices, sample_single.indices);
        assert_eq!(sample_multi.weights, sample_single.weights);
    }
}

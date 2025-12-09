//! Column (feature) sampling for training.
//!
//! Column sampling selects which features to consider at different granularities
//! during tree building. This provides regularization and can speed up training.
//!
//! # Cascading Sampling
//!
//! Column sampling cascades at three levels:
//! 1. `colsample_bytree`: Sample features once per tree
//! 2. `colsample_bylevel`: Sample from tree features at each depth level
//! 3. `colsample_bynode`: Sample from level features at each node
//!
//! The final feature set at each node is the intersection of all three samplings.
//!
//! # Example
//!
//! ```ignore
//! use booste_rs::training::gbtree::ColumnSampler;
//!
//! // Sample 80% of features at tree level, 90% at level, 100% at node
//! let mut sampler = ColumnSampler::new(100, 0.8, 0.9, 1.0);
//!
//! // Start a new tree
//! sampler.sample_for_tree(seed);
//!
//! // Get features for a specific node
//! let features = sampler.sample_for_node(depth, node_id, seed);
//! ```

use rand::prelude::*;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

// ============================================================================
// ColumnSampler
// ============================================================================

/// Samples features at tree, level, and node granularity.
///
/// Column sampling cascades:
/// - `colsample_bytree`: Initial set for the tree
/// - `colsample_bylevel`: Subset per depth level
/// - `colsample_bynode`: Subset per node (from level subset)
///
/// Unlike row sampling, column sampling is stateful - it tracks the features
/// selected for the current tree and applies cascading at each level.
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
    ///
    /// # Panics
    ///
    /// Panics if any ratio is not in (0, 1].
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
    pub fn sample_for_tree(&mut self, seed: u64) {
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
    pub fn sample_for_node(&self, depth: u32, node_id: u32, tree_seed: u64) -> Vec<u32> {
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
    fn test_column_sampler_no_sampling() {
        let mut sampler = ColumnSampler::new(10, 1.0, 1.0, 1.0);
        assert!(!sampler.is_enabled());

        sampler.sample_for_tree(42);
        assert_eq!(sampler.tree_features(), (0..10).collect::<Vec<_>>());

        let node_features = sampler.sample_for_node(0, 0, 42);
        assert_eq!(node_features, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn test_column_sampler_bytree() {
        let mut sampler = ColumnSampler::new(10, 0.5, 1.0, 1.0);
        assert!(sampler.is_enabled());

        sampler.sample_for_tree(42);
        assert_eq!(sampler.tree_features().len(), 5);

        for &f in sampler.tree_features() {
            assert!(f < 10);
        }
    }

    #[test]
    fn test_column_sampler_bylevel() {
        let mut sampler = ColumnSampler::new(10, 1.0, 0.5, 1.0);
        sampler.sample_for_tree(42);

        let features_depth0 = sampler.sample_for_node(0, 0, 42);
        let features_depth1 = sampler.sample_for_node(1, 0, 42);

        assert_eq!(features_depth0.len(), 5);
        assert_eq!(features_depth1.len(), 5);
    }

    #[test]
    fn test_column_sampler_bynode() {
        let mut sampler = ColumnSampler::new(10, 1.0, 1.0, 0.5);
        sampler.sample_for_tree(42);

        let features_node0 = sampler.sample_for_node(0, 0, 42);
        let features_node1 = sampler.sample_for_node(0, 1, 42);

        assert_eq!(features_node0.len(), 5);
        assert_eq!(features_node1.len(), 5);
    }

    #[test]
    fn test_column_sampler_cascading() {
        let mut sampler = ColumnSampler::new(10, 0.5, 0.5, 0.5);
        sampler.sample_for_tree(42);

        assert_eq!(sampler.tree_features().len(), 5);

        let node_features = sampler.sample_for_node(0, 0, 42);
        // 5 * 0.5 = 2.5 → ceil = 3 at level, then 3 * 0.5 = 1.5 → ceil = 2 at node
        assert_eq!(node_features.len(), 2);
    }

    #[test]
    fn test_column_sampler_reproducible() {
        let mut sampler1 = ColumnSampler::new(10, 0.5, 0.5, 0.5);
        let mut sampler2 = ColumnSampler::new(10, 0.5, 0.5, 0.5);

        sampler1.sample_for_tree(42);
        sampler2.sample_for_tree(42);

        assert_eq!(sampler1.tree_features(), sampler2.tree_features());

        let f1 = sampler1.sample_for_node(1, 3, 42);
        let f2 = sampler2.sample_for_node(1, 3, 42);
        assert_eq!(f1, f2);
    }

    #[test]
    fn test_column_sampler_different_seeds() {
        let mut sampler1 = ColumnSampler::new(10, 0.5, 1.0, 1.0);
        let mut sampler2 = ColumnSampler::new(10, 0.5, 1.0, 1.0);

        sampler1.sample_for_tree(42);
        sampler2.sample_for_tree(123);

        // Different seeds should (very likely) produce different features
        // Could be same by chance but extremely unlikely with 10 features
        assert_ne!(sampler1.tree_features(), sampler2.tree_features());
    }

    #[test]
    #[should_panic(expected = "colsample_bytree must be in (0, 1]")]
    fn test_column_sampler_invalid_bytree() {
        ColumnSampler::new(10, 0.0, 1.0, 1.0);
    }

    #[test]
    #[should_panic(expected = "colsample_bylevel must be in (0, 1]")]
    fn test_column_sampler_invalid_bylevel() {
        ColumnSampler::new(10, 1.0, 0.0, 1.0);
    }

    #[test]
    #[should_panic(expected = "colsample_bynode must be in (0, 1]")]
    fn test_column_sampler_invalid_bynode() {
        ColumnSampler::new(10, 1.0, 1.0, 0.0);
    }
}

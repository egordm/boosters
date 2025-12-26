//! Column (feature) sampling for gradient boosted trees.
//!
//! Provides three-level column sampling following XGBoost's design:
//! - `colsample_bytree`: Sample features once per tree
//! - `colsample_bylevel`: Further sample per depth level
//! - `colsample_bynode`: Further sample per node
//!
//! The sampling is multiplicative: effective rate = bytree × bylevel × bynode.

use rand::prelude::*;
use rand::rngs::SmallRng;

/// Configuration for column (feature) sampling.
///
/// Three levels of sampling that apply multiplicatively:
/// - `bytree`: Applied once when tree starts
/// - `bylevel`: Applied at each new depth level
/// - `bynode`: Applied for each node during split finding
#[derive(Clone, Debug, Default)]
pub enum ColSamplingParams {
    /// No column sampling (use all features).
    #[default]
    None,
    /// Sample columns with specified rates.
    Sample {
        /// Fraction of features to sample per tree (1.0 = all).
        colsample_bytree: f32,
        /// Fraction of tree features to sample per level (1.0 = all).
        colsample_bylevel: f32,
        /// Fraction of level features to sample per node (1.0 = all).
        colsample_bynode: f32,
    },
}

impl ColSamplingParams {
    /// Create a config with only `colsample_bytree`.
    pub fn bytree(rate: f32) -> Self {
        Self::Sample {
            colsample_bytree: rate,
            colsample_bylevel: 1.0,
            colsample_bynode: 1.0,
        }
    }

    /// Create a config with only `colsample_bylevel`.
    pub fn bylevel(rate: f32) -> Self {
        Self::Sample {
            colsample_bytree: 1.0,
            colsample_bylevel: rate,
            colsample_bynode: 1.0,
        }
    }

    /// Create a config with only `colsample_bynode`.
    pub fn bynode(rate: f32) -> Self {
        Self::Sample {
            colsample_bytree: 1.0,
            colsample_bylevel: 1.0,
            colsample_bynode: rate,
        }
    }

    /// Create a config with all three rates.
    pub fn new(bytree: f32, bylevel: f32, bynode: f32) -> Self {
        // If all rates are 1.0, use None for efficiency
        if (bytree - 1.0).abs() < 1e-6
            && (bylevel - 1.0).abs() < 1e-6
            && (bynode - 1.0).abs() < 1e-6
        {
            Self::None
        } else {
            Self::Sample {
                colsample_bytree: bytree,
                colsample_bylevel: bylevel,
                colsample_bynode: bynode,
            }
        }
    }

    /// Check if any sampling is enabled.
    pub fn is_enabled(&self) -> bool {
        !matches!(self, Self::None)
    }

    /// Get the effective sample rate (bytree × bylevel × bynode).
    pub fn effective_rate(&self) -> f32 {
        match self {
            Self::None => 1.0,
            Self::Sample {
                colsample_bytree,
                colsample_bylevel,
                colsample_bynode,
            } => colsample_bytree * colsample_bylevel * colsample_bynode,
        }
    }
}

/// Column sampler implementing three-level feature sampling.
///
/// # Usage Pattern
///
/// ```text
/// // At start of each tree
/// sampler.sample_tree();
///
/// // At each depth level
/// sampler.sample_level(depth);
///
/// // For each node's split finding
/// let features = sampler.sample_node();
/// splitter.find_split(histogram, features, ...);
/// ```
pub struct ColSampler {
    /// Configuration (rates for each level).
    config: ColSamplingParams,
    /// Total number of features.
    n_features: u32,
    /// Random number generator.
    rng: SmallRng,
    /// Features selected for current tree (after bytree sampling).
    tree_features: Vec<u32>,
    /// Features selected for current level (after bylevel sampling).
    level_features: Vec<u32>,
    /// Features selected for current node (after bynode sampling).
    node_features: Vec<u32>,
    /// Current depth (to detect level changes).
    current_depth: u16,
}

impl ColSampler {
    /// Create a new column sampler.
    ///
    /// # Arguments
    /// * `config` - Sampling configuration
    /// * `n_features` - Total number of features in dataset
    /// * `seed` - Random seed (0 = use entropy)
    pub fn new(config: ColSamplingParams, n_features: u32, seed: u64) -> Self {
        let rng = if seed == 0 {
            SmallRng::from_entropy()
        } else {
            SmallRng::seed_from_u64(seed)
        };

        let all_features: Vec<u32> = (0..n_features).collect();

        Self {
            config,
            n_features,
            rng,
            tree_features: all_features.clone(),
            level_features: all_features.clone(),
            node_features: all_features,
            current_depth: 0,
        }
    }

    /// Get the configuration.
    pub fn config(&self) -> &ColSamplingParams {
        &self.config
    }

    /// Get the total number of features.
    pub fn n_features(&self) -> u32 {
        self.n_features
    }

    /// Sample features for a new tree.
    ///
    /// This applies `colsample_bytree` to select a subset of features
    /// that will be used throughout the entire tree.
    ///
    /// Call this at the start of each tree in the ensemble.
    pub fn sample_tree(&mut self) {
        self.current_depth = 0;

        match &self.config {
            ColSamplingParams::None => {
                // Use all features
                self.tree_features.clear();
                self.tree_features.extend(0..self.n_features);
            }
            ColSamplingParams::Sample {
                colsample_bytree, ..
            } => {
                self.tree_features.clear();
                self.tree_features.extend(0..self.n_features);

                if *colsample_bytree < 1.0 {
                    let k = ((self.n_features as f32) * colsample_bytree).ceil() as usize;
                    let k = k.max(1).min(self.n_features as usize);
                    Self::sample_k_inplace(&mut self.rng, &mut self.tree_features, k);
                }
            }
        }

        // Initially, level features = tree features
        self.level_features.clone_from(&self.tree_features);
    }

    /// Sample features for a new depth level.
    ///
    /// This applies `colsample_bylevel` to further filter the tree features.
    /// Only call when depth actually changes.
    ///
    /// # Arguments
    /// * `depth` - The current tree depth (0 = root)
    pub fn sample_level(&mut self, depth: u16) {
        // Only resample if depth changed
        if depth == self.current_depth && depth != 0 {
            return;
        }
        self.current_depth = depth;

        match &self.config {
            ColSamplingParams::None => {
                // Level features = tree features (no additional filtering)
                self.level_features.clone_from(&self.tree_features);
            }
            ColSamplingParams::Sample {
                colsample_bylevel, ..
            } => {
                if *colsample_bylevel < 1.0 {
                    let k = ((self.tree_features.len() as f32) * colsample_bylevel).ceil() as usize;
                    let k = k.max(1).min(self.tree_features.len());
                    self.level_features.clone_from(&self.tree_features);
                    Self::sample_k_inplace(&mut self.rng, &mut self.level_features, k);
                } else {
                    self.level_features.clone_from(&self.tree_features);
                }
            }
        }
    }

    /// Sample features for a node's split finding.
    ///
    /// This applies `colsample_bynode` to further filter the level features.
    /// Returns the final set of features to use for split finding.
    ///
    /// # Returns
    /// Slice of feature indices to consider for split finding.
    /// Always returns a valid slice (all features when no sampling is active).
    pub fn sample_node(&mut self) -> &[u32] {
        match &self.config {
            ColSamplingParams::None => {
                // No sampling - return all features
                &self.node_features
            }
            ColSamplingParams::Sample {
                colsample_bynode, ..
            } => {
                if *colsample_bynode < 1.0 {
                    let k = ((self.level_features.len() as f32) * colsample_bynode).ceil() as usize;
                    let k = k.max(1).min(self.level_features.len());
                    self.node_features.clone_from(&self.level_features);
                    Self::sample_k_inplace(&mut self.rng, &mut self.node_features, k);
                } else {
                    // No bynode sampling, but level/tree sampling may be active
                    self.node_features.clone_from(&self.level_features);
                }
                &self.node_features
            }
        }
    }

    /// Sample k elements in place using partial Fisher-Yates, then truncate and sort.
    fn sample_k_inplace(rng: &mut SmallRng, data: &mut Vec<u32>, k: usize) {
        if k >= data.len() {
            return;
        }

        // Partial Fisher-Yates: only shuffle first k elements
        let n = data.len();
        for i in 0..k {
            let j = rng.r#gen::<usize>() % (n - i) + i;
            data.swap(i, j);
        }

        // Truncate to k elements
        data.truncate(k);

        // Sort for cache-friendly access during split finding
        data.sort_unstable();
    }

    /// Get tree-level features (for debugging/testing).
    #[cfg(test)]
    pub fn tree_features(&self) -> &[u32] {
        &self.tree_features
    }

    /// Get level features (for debugging/testing).
    #[cfg(test)]
    pub fn level_features(&self) -> &[u32] {
        &self.level_features
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_sampling() {
        let config = ColSamplingParams::None;
        let mut sampler = ColSampler::new(config, 10, 42);

        sampler.sample_tree();
        assert_eq!(sampler.tree_features.len(), 10);

        sampler.sample_level(0);
        assert_eq!(sampler.level_features.len(), 10);

        let node_features = sampler.sample_node();
        assert_eq!(node_features.len(), 10); // No sampling = all features returned
    }

    #[test]
    fn test_bytree_sampling() {
        let config = ColSamplingParams::bytree(0.5);
        let mut sampler = ColSampler::new(config, 10, 42);

        sampler.sample_tree();
        // 50% of 10 = 5 features
        assert_eq!(sampler.tree_features.len(), 5);

        // Level should inherit tree features
        sampler.sample_level(0);
        assert_eq!(sampler.level_features.len(), 5);

        // Node should return the filtered features
        let node_features = sampler.sample_node();
        assert_eq!(node_features.len(), 5);
    }

    #[test]
    fn test_bylevel_sampling() {
        let config = ColSamplingParams::bylevel(0.5);
        let mut sampler = ColSampler::new(config, 10, 42);

        sampler.sample_tree();
        assert_eq!(sampler.tree_features.len(), 10); // bytree = 1.0

        sampler.sample_level(0);
        // 50% of 10 = 5 features at this level
        assert_eq!(sampler.level_features.len(), 5);

        // Different level should resample
        let _level0_features = sampler.level_features.clone();
        sampler.sample_level(1);
        // May get different features (depends on RNG)
        assert_eq!(sampler.level_features.len(), 5);

        // Same level shouldn't resample
        sampler.sample_level(1);
        assert_eq!(sampler.level_features.len(), 5);

        // Going back to level 0 should resample (different depth)
        sampler.sample_level(0);
        assert_eq!(sampler.level_features.len(), 5);
    }

    #[test]
    fn test_bynode_sampling() {
        let config = ColSamplingParams::bynode(0.5);
        let mut sampler = ColSampler::new(config, 10, 42);

        sampler.sample_tree();
        sampler.sample_level(0);

        let node1 = sampler.sample_node().to_vec();
        let node2 = sampler.sample_node().to_vec();

        // Both should have 5 features
        assert_eq!(node1.len(), 5);
        assert_eq!(node2.len(), 5);

        // Different nodes may get different features (depends on RNG)
        // Just verify they're valid subsets
        for &f in &node1 {
            assert!(f < 10);
        }
        for &f in &node2 {
            assert!(f < 10);
        }
    }

    #[test]
    fn test_combined_sampling() {
        let config = ColSamplingParams::new(0.5, 0.5, 0.5);
        let mut sampler = ColSampler::new(config, 100, 42);

        sampler.sample_tree();
        // 50% of 100 = 50
        assert_eq!(sampler.tree_features.len(), 50);

        sampler.sample_level(0);
        // 50% of 50 = 25
        assert_eq!(sampler.level_features.len(), 25);

        let node_features = sampler.sample_node();
        // 50% of 25 = 13 (ceil)
        assert_eq!(node_features.len(), 13);
    }

    #[test]
    fn test_effective_rate() {
        let config = ColSamplingParams::new(0.5, 0.5, 0.5);
        assert!((config.effective_rate() - 0.125).abs() < 1e-6);

        let config = ColSamplingParams::bytree(0.8);
        assert!((config.effective_rate() - 0.8).abs() < 1e-6);

        let config = ColSamplingParams::None;
        assert!((config.effective_rate() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_minimum_one_feature() {
        let config = ColSamplingParams::bytree(0.01); // Very small rate
        let mut sampler = ColSampler::new(config, 10, 42);

        sampler.sample_tree();
        // Should always have at least 1 feature
        assert!(!sampler.tree_features.is_empty());
    }

    #[test]
    fn test_sorted_features() {
        let config = ColSamplingParams::bytree(0.5);
        let mut sampler = ColSampler::new(config, 100, 42);

        sampler.sample_tree();

        // Features should be sorted for cache-friendly access
        let features = &sampler.tree_features;
        for i in 1..features.len() {
            assert!(features[i] > features[i - 1]);
        }
    }

    #[test]
    fn test_reproducibility() {
        let config = ColSamplingParams::new(0.5, 0.5, 0.5);

        let mut sampler1 = ColSampler::new(config.clone(), 100, 42);
        let mut sampler2 = ColSampler::new(config, 100, 42);

        sampler1.sample_tree();
        sampler2.sample_tree();

        assert_eq!(sampler1.tree_features, sampler2.tree_features);

        sampler1.sample_level(0);
        sampler2.sample_level(0);

        assert_eq!(sampler1.level_features, sampler2.level_features);
    }

    #[test]
    fn test_new_tree_resets() {
        let config = ColSamplingParams::bytree(0.5);
        let mut sampler = ColSampler::new(config, 100, 42);

        sampler.sample_tree();
        let tree1_features = sampler.tree_features.clone();

        // New tree should potentially get different features
        sampler.sample_tree();
        let tree2_features = sampler.tree_features.clone();

        // Both should have 50 features
        assert_eq!(tree1_features.len(), 50);
        assert_eq!(tree2_features.len(), 50);

        // Features may differ (probabilistic, but very likely with 50% sampling)
        // This is just for coverage - not a deterministic test
    }

    #[test]
    fn test_config_none_optimization() {
        // All rates = 1.0 should collapse to None
        let config = ColSamplingParams::new(1.0, 1.0, 1.0);
        assert!(!config.is_enabled());
    }
}

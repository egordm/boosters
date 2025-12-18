//! Feature bundling for exclusive feature grouping.
//!
//! This module implements the Exclusive Feature Bundling (EFB) algorithm from
//! LightGBM. It groups mutually exclusive sparse features into bundles to reduce
//! the effective feature count and improve histogram building performance.
//!
//! # Algorithm Overview
//!
//! 1. **Feature Analysis**: Identify sparse features (density < threshold)
//! 2. **Conflict Detection**: Build conflict graph using bitsets
//! 3. **Bundle Assignment**: Greedy assignment to minimize bundles
//!
//! See RFC-0017 for detailed design rationale.

use super::FeatureInfo;
use crate::data::DataMatrix;
use fixedbitset::FixedBitSet;
use rand::prelude::*;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use std::collections::HashMap;

/// Configuration for feature bundling.
///
/// # Example
///
/// ```
/// use booste_rs::data::binned::BundlingConfig;
///
/// // Use defaults for most datasets
/// let config = BundlingConfig::auto();
///
/// // Aggressive bundling for very sparse data
/// let aggressive = BundlingConfig::aggressive();
///
/// // Disable bundling entirely
/// let disabled = BundlingConfig::disabled();
/// ```
#[derive(Clone, Debug)]
pub struct BundlingConfig {
    /// Enable exclusive feature bundling. Default: true.
    pub enable_bundling: bool,

    /// Maximum allowed conflict rate (fraction of samples where
    /// multiple features in a bundle are non-zero). Default: 0.0001.
    pub max_conflict_rate: f32,

    /// Minimum sparsity (fraction of zeros) for a feature to be
    /// considered for bundling. Default: 0.9.
    pub min_sparsity: f32,

    /// Maximum features per bundle. Default: 256 (to fit in u8 bin).
    pub max_bundle_size: usize,

    /// Maximum sparse features to consider for bundling.
    /// If exceeded, bundling is skipped. Default: 1000.
    pub max_sparse_features: usize,

    /// Maximum rows to sample for conflict detection. Default: 10000.
    pub max_sample_rows: usize,

    /// Optional pre-defined bundle hints. Skip conflict detection for these groups.
    /// Example: `vec![vec![0,1,2], vec![3,4,5]]` bundles features 0-2 and 3-5.
    pub bundle_hints: Option<Vec<Vec<usize>>>,

    /// Random seed for row sampling. Default: 42.
    pub seed: u64,
}

impl Default for BundlingConfig {
    fn default() -> Self {
        Self {
            enable_bundling: true,
            max_conflict_rate: 0.0001,
            min_sparsity: 0.9,
            max_bundle_size: 256,
            max_sparse_features: 1000,
            max_sample_rows: 10000,
            bundle_hints: None,
            seed: 42,
        }
    }
}

impl BundlingConfig {
    /// Create default configuration optimized for most datasets.
    /// Use this as the starting point for typical sparse/one-hot data.
    ///
    /// **When in doubt, use `auto()` and check `BundlingStats::is_effective()`
    /// after building to see if bundling helped your dataset.**
    pub fn auto() -> Self {
        Self::default()
    }

    /// Disable bundling entirely.
    /// Use when: debugging, A/B testing, or datasets with no sparse features.
    pub fn disabled() -> Self {
        Self {
            enable_bundling: false,
            ..Default::default()
        }
    }

    /// Aggressive bundling for very sparse datasets.
    /// Use when: very high sparsity (>95% zeros), many one-hot encoded categoricals,
    /// or when you're willing to trade tiny accuracy for significant speedup.
    pub fn aggressive() -> Self {
        Self {
            enable_bundling: true,
            max_conflict_rate: 0.001, // Allow 0.1% conflicts
            min_sparsity: 0.8,        // Include moderately sparse
            max_bundle_size: 256,
            max_sparse_features: 2000, // Allow more features
            max_sample_rows: 10000,
            bundle_hints: None,
            seed: 42,
        }
    }

    /// Conservative bundling with zero tolerance for conflicts.
    /// Use when: accuracy is paramount, debugging bundling issues.
    pub fn strict() -> Self {
        Self {
            enable_bundling: true,
            max_conflict_rate: 0.0,
            min_sparsity: 0.95, // Only very sparse
            max_bundle_size: 256,
            max_sparse_features: 1000,
            max_sample_rows: 10000,
            bundle_hints: None,
            seed: 42,
        }
    }

    /// Compute maximum allowed conflicts for a given number of rows.
    pub fn max_conflicts(&self, n_rows: usize) -> usize {
        (self.max_conflict_rate * n_rows as f32).ceil() as usize
    }
}

/// A bundle of mutually exclusive features.
#[derive(Clone, Debug)]
pub struct FeatureBundle {
    /// Indices of features in this bundle (in original feature space).
    pub feature_indices: Vec<usize>,
}

impl FeatureBundle {
    /// Create a new bundle with a single feature.
    pub fn new(feature_idx: usize) -> Self {
        Self {
            feature_indices: vec![feature_idx],
        }
    }

    /// Add a feature to this bundle.
    pub fn add(&mut self, feature_idx: usize) {
        self.feature_indices.push(feature_idx);
    }

    /// Number of features in this bundle.
    pub fn len(&self) -> usize {
        self.feature_indices.len()
    }

    /// Check if this bundle is empty.
    pub fn is_empty(&self) -> bool {
        self.feature_indices.is_empty()
    }
}

/// Result of the bundling analysis.
#[derive(Clone, Debug)]
pub struct BundlePlan {
    /// The computed bundles.
    pub bundles: Vec<FeatureBundle>,

    /// Total conflicts in the plan.
    pub total_conflicts: usize,

    /// Number of rows sampled for conflict detection.
    pub rows_sampled: usize,

    /// Original number of sparse features considered.
    pub sparse_feature_count: usize,

    /// Whether bundling was skipped (too many sparse features).
    pub skipped: bool,

    /// Skip reason if skipped.
    pub skip_reason: Option<String>,
}

impl BundlePlan {
    /// Check if bundling was effective (reduced feature count).
    pub fn is_effective(&self) -> bool {
        !self.skipped && self.bundles.len() < self.sparse_feature_count
    }

    /// Reduction ratio: bundles / original_sparse_features.
    /// Lower is better (more compression).
    pub fn reduction_ratio(&self) -> f32 {
        if self.sparse_feature_count == 0 {
            1.0
        } else {
            self.bundles.len() as f32 / self.sparse_feature_count as f32
        }
    }
}

/// Conflict graph for sparse features.
///
/// Uses bitsets to track which rows have non-zero values for each feature,
/// then counts intersections to determine conflicts.
struct ConflictGraph {
    /// Number of features in the graph.
    n_features: usize,

    /// Conflict counts between feature pairs.
    /// Key: (min_idx, max_idx) to avoid duplicate pairs.
    conflicts: HashMap<(usize, usize), usize>,
}

impl ConflictGraph {
    /// Build a conflict graph from sampled rows.
    ///
    /// # Arguments
    /// * `matrix` - The data matrix
    /// * `sparse_features` - Indices of sparse features to analyze
    /// * `sampled_rows` - Row indices to check for conflicts
    fn build<M: DataMatrix<Element = f32> + Sync>(
        matrix: &M,
        sparse_features: &[usize],
        sampled_rows: &[usize],
    ) -> Self {
        let n_features = sparse_features.len();
        let n_sampled = sampled_rows.len();

        if n_features == 0 {
            return Self {
                n_features: 0,
                conflicts: HashMap::new(),
            };
        }

        // Build bitsets for each feature in parallel
        let bitsets: Vec<FixedBitSet> = sparse_features
            .par_iter()
            .map(|&feat_idx| {
                let mut bits = FixedBitSet::with_capacity(n_sampled);
                for (row_pos, &row_idx) in sampled_rows.iter().enumerate() {
                    if matrix.get(row_idx, feat_idx).is_some_and(|val| val != 0.0) {
                        bits.insert(row_pos);
                    }
                }
                bits
            })
            .collect();

        // Count pairwise conflicts in parallel
        let pairs: Vec<(usize, usize)> = (0..n_features)
            .flat_map(|i| ((i + 1)..n_features).map(move |j| (i, j)))
            .collect();

        let conflict_counts: Vec<((usize, usize), usize)> = pairs
            .par_iter()
            .filter_map(|&(i, j)| {
                // Count intersection using bitset AND + popcount
                let intersection = bitsets[i].intersection(&bitsets[j]).count();
                if intersection > 0 {
                    Some(((i, j), intersection))
                } else {
                    None
                }
            })
            .collect();

        let conflicts: HashMap<(usize, usize), usize> = conflict_counts.into_iter().collect();

        Self {
            n_features,
            conflicts,
        }
    }

    /// Get the conflict count between two features (by their index in sparse_features).
    fn get_conflict(&self, i: usize, j: usize) -> usize {
        let key = if i < j { (i, j) } else { (j, i) };
        *self.conflicts.get(&key).unwrap_or(&0)
    }

    /// Total number of conflicting pairs.
    fn conflict_pair_count(&self) -> usize {
        self.conflicts.len()
    }

    /// Total number of possible pairs.
    fn total_pair_count(&self) -> usize {
        if self.n_features < 2 {
            0
        } else {
            self.n_features * (self.n_features - 1) / 2
        }
    }

    /// Conflict rate: conflicting_pairs / total_pairs.
    fn conflict_rate(&self) -> f32 {
        let total = self.total_pair_count();
        if total == 0 {
            0.0
        } else {
            self.conflict_pair_count() as f32 / total as f32
        }
    }
}

/// Sample rows for conflict detection using stratified sampling.
///
/// Strategy: 80% random + 10% first rows + 10% last rows.
/// This catches temporal patterns that pure random sampling might miss.
fn sample_rows(n_rows: usize, max_samples: usize, seed: u64) -> Vec<usize> {
    if n_rows <= max_samples {
        return (0..n_rows).collect();
    }

    let n_first = max_samples / 10; // 10% first
    let n_last = max_samples / 10; // 10% last
    let n_random = max_samples - n_first - n_last; // 80% random

    let mut result = Vec::with_capacity(max_samples);

    // First rows
    result.extend(0..n_first.min(n_rows));

    // Last rows
    let last_start = n_rows.saturating_sub(n_last);
    result.extend(last_start..n_rows);

    // Random middle rows (avoiding duplicates with first/last)
    let middle_start = n_first;
    let middle_end = last_start;

    if middle_end > middle_start {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let middle_range: Vec<usize> = (middle_start..middle_end).collect();

        // Reservoir sampling for random selection
        let mut selected: Vec<usize> = middle_range
            .iter()
            .take(n_random.min(middle_range.len()))
            .copied()
            .collect();

        for (i, &idx) in middle_range.iter().enumerate().skip(selected.len()) {
            let j = rng.gen_range(0..=i);
            if j < selected.len() {
                selected[j] = idx;
            }
        }

        result.extend(selected);
    }

    // Sort for cache-friendly access
    result.sort_unstable();
    result.dedup();
    result
}

/// Assign features to bundles using greedy algorithm.
///
/// Features are sorted by density (denser first = more restrictive),
/// then assigned to the best-fit bundle or a new bundle.
fn assign_bundles(
    feature_infos: &[FeatureInfo],
    conflict_graph: &ConflictGraph,
    config: &BundlingConfig,
    n_sampled_rows: usize,
) -> (Vec<FeatureBundle>, usize) {
    if feature_infos.is_empty() {
        return (Vec::new(), 0);
    }

    let max_conflicts = config.max_conflicts(n_sampled_rows);

    // Sort features by density (denser first)
    let mut sorted_indices: Vec<usize> = (0..feature_infos.len()).collect();
    sorted_indices.sort_by(|&a, &b| {
        feature_infos[b]
            .density
            .partial_cmp(&feature_infos[a].density)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut bundles: Vec<FeatureBundle> = Vec::new();
    let mut bundle_conflicts: Vec<usize> = Vec::new();
    let mut total_conflicts = 0usize;

    for &feat_sparse_idx in &sorted_indices {
        let feat_info = &feature_infos[feat_sparse_idx];
        let original_idx = feat_info.original_idx;

        // Find the best bundle for this feature
        let mut best_bundle: Option<usize> = None;
        let mut best_conflict_increase = usize::MAX;

        for (bundle_idx, bundle) in bundles.iter().enumerate() {
            // Check bundle size limit
            if bundle.len() >= config.max_bundle_size {
                continue;
            }

            // Calculate conflict increase if we add this feature
            // Bundles store original indices. We need sparse indices for conflict lookup.
            let mut conflict_sum = 0usize;
            for existing_orig in &bundle.feature_indices {
                // Find the sparse index for this original index
                if let Some(existing_sparse_idx) = feature_infos
                    .iter()
                    .position(|f| f.original_idx == *existing_orig)
                {
                    conflict_sum += conflict_graph.get_conflict(feat_sparse_idx, existing_sparse_idx);
                }
            }
            let new_conflicts = conflict_sum;

            // Check if adding this feature would exceed conflict limit
            if bundle_conflicts[bundle_idx] + new_conflicts <= max_conflicts
                && new_conflicts < best_conflict_increase
            {
                best_bundle = Some(bundle_idx);
                best_conflict_increase = new_conflicts;
            }
        }

        if let Some(bundle_idx) = best_bundle {
            bundles[bundle_idx].add(original_idx);
            bundle_conflicts[bundle_idx] += best_conflict_increase;
            total_conflicts += best_conflict_increase;
        } else {
            // Create a new bundle
            bundles.push(FeatureBundle::new(original_idx));
            bundle_conflicts.push(0);
        }
    }

    (bundles, total_conflicts)
}

/// Create a bundle plan from feature analysis results.
///
/// This is the main entry point for feature bundling.
///
/// # Arguments
/// * `matrix` - The data matrix
/// * `feature_infos` - Results from `analyze_features()`
/// * `config` - Bundling configuration
///
/// # Returns
/// A `BundlePlan` describing how features should be bundled.
pub fn create_bundle_plan<M: DataMatrix<Element = f32> + Sync>(
    matrix: &M,
    feature_infos: &[FeatureInfo],
    config: &BundlingConfig,
) -> BundlePlan {
    // Check if bundling is disabled
    if !config.enable_bundling {
        return BundlePlan {
            bundles: Vec::new(),
            total_conflicts: 0,
            rows_sampled: 0,
            sparse_feature_count: 0,
            skipped: true,
            skip_reason: Some("Bundling disabled".to_string()),
        };
    }

    // Filter to sparse features only
    let sparse_threshold = 1.0 - config.min_sparsity;
    let sparse_features: Vec<&FeatureInfo> = feature_infos
        .iter()
        .filter(|f| f.density <= sparse_threshold && !f.is_trivial)
        .collect();

    let sparse_count = sparse_features.len();

    // Check if too few sparse features
    if sparse_count < 2 {
        return BundlePlan {
            bundles: sparse_features
                .iter()
                .map(|f| FeatureBundle::new(f.original_idx))
                .collect(),
            total_conflicts: 0,
            rows_sampled: 0,
            sparse_feature_count: sparse_count,
            skipped: true,
            skip_reason: Some("Too few sparse features for bundling".to_string()),
        };
    }

    // Check if too many sparse features
    if sparse_count > config.max_sparse_features {
        // Log warning would go here in production
        // For now, we just mark as skipped
        return BundlePlan {
            bundles: sparse_features
                .iter()
                .map(|f| FeatureBundle::new(f.original_idx))
                .collect(),
            total_conflicts: 0,
            rows_sampled: 0,
            sparse_feature_count: sparse_count,
            skipped: true,
            skip_reason: Some(format!(
                "Too many sparse features ({} > {})",
                sparse_count, config.max_sparse_features
            )),
        };
    }

    // Sample rows for conflict detection
    let n_rows = matrix.num_rows();
    let sampled_rows = sample_rows(n_rows, config.max_sample_rows, config.seed);
    let n_sampled = sampled_rows.len();

    // Get sparse feature indices in original space
    let sparse_original_indices: Vec<usize> = sparse_features.iter().map(|f| f.original_idx).collect();

    // Build conflict graph
    let conflict_graph = ConflictGraph::build(matrix, &sparse_original_indices, &sampled_rows);

    // Check for high conflict rate (early termination)
    if conflict_graph.conflict_rate() > 0.5 {
        // More than 50% of pairs conflict - bundling won't help much
        return BundlePlan {
            bundles: sparse_features
                .iter()
                .map(|f| FeatureBundle::new(f.original_idx))
                .collect(),
            total_conflicts: 0,
            rows_sampled: n_sampled,
            sparse_feature_count: sparse_count,
            skipped: true,
            skip_reason: Some(format!(
                "High conflict rate ({:.1}%)",
                conflict_graph.conflict_rate() * 100.0
            )),
        };
    }

    // Create a local copy of sparse feature infos for assignment
    let sparse_feature_infos: Vec<FeatureInfo> = sparse_features.iter().map(|f| (*f).clone()).collect();

    // Assign features to bundles
    let (bundles, total_conflicts) =
        assign_bundles(&sparse_feature_infos, &conflict_graph, config, n_sampled);

    BundlePlan {
        bundles,
        total_conflicts,
        rows_sampled: n_sampled,
        sparse_feature_count: sparse_count,
        skipped: false,
        skip_reason: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::ColMatrix;

    #[test]
    fn test_bundling_config_defaults() {
        let config = BundlingConfig::default();
        assert!(config.enable_bundling);
        assert_eq!(config.max_conflict_rate, 0.0001);
        assert_eq!(config.min_sparsity, 0.9);
        assert_eq!(config.max_bundle_size, 256);
        assert_eq!(config.max_sparse_features, 1000);
        assert_eq!(config.max_sample_rows, 10000);
    }

    #[test]
    fn test_bundling_config_auto() {
        let config = BundlingConfig::auto();
        assert!(config.enable_bundling);
    }

    #[test]
    fn test_bundling_config_disabled() {
        let config = BundlingConfig::disabled();
        assert!(!config.enable_bundling);
    }

    #[test]
    fn test_bundling_config_aggressive() {
        let config = BundlingConfig::aggressive();
        assert!(config.enable_bundling);
        assert_eq!(config.max_conflict_rate, 0.001);
        assert_eq!(config.min_sparsity, 0.8);
    }

    #[test]
    fn test_bundling_config_strict() {
        let config = BundlingConfig::strict();
        assert!(config.enable_bundling);
        assert_eq!(config.max_conflict_rate, 0.0);
        assert_eq!(config.min_sparsity, 0.95);
    }

    #[test]
    fn test_max_conflicts_calculation() {
        let config = BundlingConfig::default();
        // 0.0001 * 10000 = 1
        assert_eq!(config.max_conflicts(10000), 1);
        // 0.0001 * 100000 = 10
        assert_eq!(config.max_conflicts(100000), 10);
    }

    #[test]
    fn test_feature_bundle_operations() {
        let mut bundle = FeatureBundle::new(0);
        assert_eq!(bundle.len(), 1);
        assert!(!bundle.is_empty());
        assert_eq!(bundle.feature_indices, vec![0]);

        bundle.add(5);
        bundle.add(10);
        assert_eq!(bundle.len(), 3);
        assert_eq!(bundle.feature_indices, vec![0, 5, 10]);
    }

    #[test]
    fn test_sample_rows_small_dataset() {
        // When n_rows <= max_samples, return all rows
        let rows = sample_rows(100, 1000, 42);
        assert_eq!(rows.len(), 100);
        assert_eq!(rows, (0..100).collect::<Vec<_>>());
    }

    #[test]
    fn test_sample_rows_large_dataset() {
        let rows = sample_rows(100000, 1000, 42);
        assert_eq!(rows.len(), 1000);

        // Should be sorted
        let mut sorted = rows.clone();
        sorted.sort();
        assert_eq!(rows, sorted);

        // Should include first 100 rows (10%)
        for i in 0..100 {
            assert!(rows.contains(&i), "Missing first row {}", i);
        }

        // Should include last 100 rows (10%)
        for i in 99900..100000 {
            assert!(rows.contains(&i), "Missing last row {}", i);
        }
    }

    #[test]
    fn test_sample_rows_deterministic() {
        let rows1 = sample_rows(100000, 1000, 42);
        let rows2 = sample_rows(100000, 1000, 42);
        assert_eq!(rows1, rows2);

        // Different seed gives different result
        let rows3 = sample_rows(100000, 1000, 123);
        assert_ne!(rows1, rows3);
    }

    #[test]
    fn test_conflict_graph_no_conflicts() {
        // Two mutually exclusive features (one-hot style)
        // Feature 0: [1, 0, 0, 0]
        // Feature 1: [0, 1, 0, 0]
        // ColMatrix is column-major, so we flatten by column
        // Col 0: [1, 0, 0, 0], Col 1: [0, 1, 0, 0]
        let matrix = ColMatrix::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], 4, 2);

        let sparse_features = vec![0, 1];
        let sampled_rows: Vec<usize> = (0..4).collect();

        let graph = ConflictGraph::build(&matrix, &sparse_features, &sampled_rows);

        assert_eq!(graph.get_conflict(0, 1), 0);
        assert_eq!(graph.conflict_pair_count(), 0);
    }

    #[test]
    fn test_conflict_graph_with_conflicts() {
        // Two features that conflict on row 0
        // Feature 0: [1, 0, 0, 0]
        // Feature 1: [1, 1, 0, 0]
        // ColMatrix is column-major
        let matrix = ColMatrix::from_vec(vec![1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0], 4, 2);

        let sparse_features = vec![0, 1];
        let sampled_rows: Vec<usize> = (0..4).collect();

        let graph = ConflictGraph::build(&matrix, &sparse_features, &sampled_rows);

        assert_eq!(graph.get_conflict(0, 1), 1);
        assert_eq!(graph.conflict_pair_count(), 1);
    }

    #[test]
    fn test_bundle_plan_disabled() {
        // 2 rows, 2 cols. Col 0: [1, 0], Col 1: [0, 1]
        let matrix = ColMatrix::from_vec(vec![1.0, 0.0, 0.0, 1.0], 2, 2);
        let feature_infos = vec![
            FeatureInfo {
                original_idx: 0,
                density: 0.5,
                is_binary: true,
                is_trivial: false,
            },
            FeatureInfo {
                original_idx: 1,
                density: 0.5,
                is_binary: true,
                is_trivial: false,
            },
        ];

        let config = BundlingConfig::disabled();
        let plan = create_bundle_plan(&matrix, &feature_infos, &config);

        assert!(plan.skipped);
        assert_eq!(plan.skip_reason, Some("Bundling disabled".to_string()));
    }

    #[test]
    fn test_bundle_plan_too_few_sparse() {
        // Only 1 sparse feature - not enough to bundle
        // 2 rows, 2 cols. Col 0: [1, 0], Col 1: [0, 0]
        let matrix = ColMatrix::from_vec(vec![1.0, 0.0, 0.0, 0.0], 2, 2);
        let feature_infos = vec![
            FeatureInfo {
                original_idx: 0,
                density: 0.5, // Not sparse
                is_binary: true,
                is_trivial: false,
            },
            FeatureInfo {
                original_idx: 1,
                density: 0.0, // Sparse (all zeros)
                is_binary: false,
                is_trivial: true,
            },
        ];

        let config = BundlingConfig::auto();
        let plan = create_bundle_plan(&matrix, &feature_infos, &config);

        assert!(plan.skipped);
        assert!(plan
            .skip_reason
            .as_ref()
            .unwrap()
            .contains("Too few sparse features"));
    }

    #[test]
    fn test_bundle_plan_exclusive_features() {
        // 4 mutually exclusive features (one-hot encoding)
        // 10 rows, 4 cols
        // Each feature has 1 non-zero value in different rows
        // Col 0: [1,0,0,0,0,0,0,0,0,0]
        // Col 1: [0,1,0,0,0,0,0,0,0,0]
        // Col 2: [0,0,1,0,0,0,0,0,0,0]
        // Col 3: [0,0,0,1,0,0,0,0,0,0]
        let mut data = vec![0.0f32; 40]; // 10 rows * 4 cols
        data[0] = 1.0;  // col 0, row 0
        data[11] = 1.0; // col 1, row 1
        data[22] = 1.0; // col 2, row 2
        data[33] = 1.0; // col 3, row 3
        let matrix = ColMatrix::from_vec(data, 10, 4);

        // density = 0.1 for each feature (1 non-zero out of 10)
        let feature_infos: Vec<FeatureInfo> = (0..4)
            .map(|i| FeatureInfo {
                original_idx: i,
                density: 0.1,
                is_binary: true,
                is_trivial: false,
            })
            .collect();

        let config = BundlingConfig::auto();
        let plan = create_bundle_plan(&matrix, &feature_infos, &config);

        assert!(!plan.skipped, "Plan should not be skipped");
        assert_eq!(plan.sparse_feature_count, 4);
        assert_eq!(plan.total_conflicts, 0);

        // All 4 features should be bundled into 1 bundle (no conflicts)
        assert_eq!(plan.bundles.len(), 1);
        assert_eq!(plan.bundles[0].len(), 4);

        assert!(plan.is_effective());
        assert!(plan.reduction_ratio() < 0.5); // 1/4 = 0.25
    }

    #[test]
    fn test_bundle_plan_conflicting_features() {
        // 3 features where 2 conflict but 1 doesn't
        // This tests that conflicting features end up in separate bundles
        // 10 rows, 3 cols
        // Col 0: [1,1,0,0,0,0,0,0,0,0] - conflicts with col 1
        // Col 1: [1,1,0,0,0,0,0,0,0,0] - conflicts with col 0
        // Col 2: [0,0,1,0,0,0,0,0,0,0] - doesn't conflict with anyone
        let mut data = vec![0.0f32; 30];
        data[0] = 1.0;  // col 0, row 0
        data[1] = 1.0;  // col 0, row 1
        data[10] = 1.0; // col 1, row 0
        data[11] = 1.0; // col 1, row 1
        data[22] = 1.0; // col 2, row 2
        let matrix = ColMatrix::from_vec(data, 10, 3);

        // strict() uses min_sparsity = 0.95, so sparse_threshold = 0.05
        let feature_infos = vec![
            FeatureInfo {
                original_idx: 0,
                density: 0.01,
                is_binary: true,
                is_trivial: false,
            },
            FeatureInfo {
                original_idx: 1,
                density: 0.01,
                is_binary: true,
                is_trivial: false,
            },
            FeatureInfo {
                original_idx: 2,
                density: 0.01,
                is_binary: true,
                is_trivial: false,
            },
        ];

        let config = BundlingConfig::strict(); // 0 conflicts allowed
        let plan = create_bundle_plan(&matrix, &feature_infos, &config);

        assert!(!plan.skipped, "skip_reason: {:?}", plan.skip_reason);
        // Feature 2 can bundle with either 0 or 1, but 0 and 1 must be separate
        // So we expect at least 2 bundles
        assert!(plan.bundles.len() >= 2, "Should have at least 2 bundles, got {}", plan.bundles.len());
    }

    #[test]
    fn test_bundle_plan_high_conflict_skipped() {
        // 2 features that conflict on every non-zero row
        // This should trigger early termination due to high conflict rate (>50%)
        let mut data = vec![0.0f32; 20];
        data[0] = 1.0;  // col 0, row 0
        data[1] = 1.0;  // col 0, row 1
        data[10] = 1.0; // col 1, row 0
        data[11] = 1.0; // col 1, row 1
        let matrix = ColMatrix::from_vec(data, 10, 2);

        let feature_infos = vec![
            FeatureInfo {
                original_idx: 0,
                density: 0.01,
                is_binary: true,
                is_trivial: false,
            },
            FeatureInfo {
                original_idx: 1,
                density: 0.01,
                is_binary: true,
                is_trivial: false,
            },
        ];

        let config = BundlingConfig::strict();
        let plan = create_bundle_plan(&matrix, &feature_infos, &config);

        // With only 2 features, the only pair conflicts = 100% conflict rate
        // Should be skipped due to high conflict rate
        assert!(plan.skipped);
        assert!(plan.skip_reason.as_ref().unwrap().contains("High conflict rate"));
    }

    #[test]
    fn test_bundle_plan_effectiveness() {
        let plan = BundlePlan {
            bundles: vec![FeatureBundle::new(0)],
            total_conflicts: 0,
            rows_sampled: 100,
            sparse_feature_count: 4,
            skipped: false,
            skip_reason: None,
        };

        assert!(plan.is_effective()); // 1 bundle from 4 features
        assert_eq!(plan.reduction_ratio(), 0.25);
    }

    #[test]
    fn test_bundle_plan_not_effective() {
        let plan = BundlePlan {
            bundles: vec![
                FeatureBundle::new(0),
                FeatureBundle::new(1),
                FeatureBundle::new(2),
                FeatureBundle::new(3),
            ],
            total_conflicts: 0,
            rows_sampled: 100,
            sparse_feature_count: 4,
            skipped: false,
            skip_reason: None,
        };

        assert!(!plan.is_effective()); // 4 bundles from 4 features
        assert_eq!(plan.reduction_ratio(), 1.0);
    }

    #[test]
    fn test_conflict_rate_calculation() {
        let mut conflicts = HashMap::new();
        conflicts.insert((0, 1), 5);
        conflicts.insert((0, 2), 3);

        let graph = ConflictGraph {
            n_features: 4,
            conflicts,
        };

        // 4 features = 6 possible pairs, 2 have conflicts
        assert_eq!(graph.total_pair_count(), 6);
        assert_eq!(graph.conflict_pair_count(), 2);
        assert!((graph.conflict_rate() - 2.0 / 6.0).abs() < 0.001);
    }
}

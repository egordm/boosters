//! Feature bundling for Exclusive Feature Bundling (EFB).
//!
//! This module implements the core bundling algorithms from LightGBM,
//! adapted for the RFC-0018 storage system.
//!
//! # Algorithm Overview
//!
//! 1. **Identify candidates**: Sparse categorical features
//! 2. **Build conflict graph**: Pairwise conflicts via bitset intersection
//! 3. **Greedy assignment**: Assign features to bundles minimizing conflicts
//! 4. **Encode bundles**: Create `BundleStorage` with offset encoding

use fixedbitset::FixedBitSet;
use rand::SeedableRng;
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use std::collections::HashMap;

use super::feature_analysis::{FeatureAnalysis, GroupSpec, GroupType, GroupingResult};
use crate::data::{Dataset, Feature};

// =============================================================================
// BundlingConfig
// =============================================================================

/// Configuration for feature bundling.
#[derive(Clone, Debug)]
pub struct BundlingConfig {
    /// Maximum allowed conflict rate (fraction of samples where
    /// multiple features in a bundle are non-zero). Default: 0.0001.
    pub max_conflict_rate: f32,

    /// Minimum sparsity (fraction of zeros) for a feature to be
    /// bundled. Default: 0.9.
    pub min_sparsity: f32,

    /// Maximum features per bundle. Default: 256 (to fit encoded bins).
    pub max_bundle_size: usize,

    /// Maximum sparse features to consider for bundling.
    /// If exceeded, bundling is skipped. Default: 1000.
    pub max_sparse_features: usize,

    /// Maximum rows to sample for conflict detection. Default: 10000.
    pub max_sample_rows: usize,

    /// Random seed for row sampling. Default: 42.
    pub seed: u64,
}

impl Default for BundlingConfig {
    fn default() -> Self {
        Self {
            max_conflict_rate: 0.0001,
            min_sparsity: 0.9,
            max_bundle_size: 256,
            max_sparse_features: 1000,
            max_sample_rows: 10000,
            seed: 42,
        }
    }
}

impl BundlingConfig {
    /// Compute maximum allowed conflicts for a given number of rows.
    pub fn max_conflicts(&self, n_rows: usize) -> usize {
        (self.max_conflict_rate * n_rows as f32).ceil() as usize
    }
}

// =============================================================================
// Conflict Detection
// =============================================================================

/// Conflict graph for sparse features.
///
/// Uses bitsets to efficiently count pairwise conflicts.
struct ConflictGraph {
    /// Number of features in the graph.
    n_features: usize,

    /// Conflict counts between feature pairs.
    /// Key: (min_idx, max_idx) for deduplication.
    conflicts: HashMap<(usize, usize), usize>,
}

impl ConflictGraph {
    /// Build a conflict graph from sampled rows.
    fn build(dataset: &Dataset, sparse_features: &[usize], sampled_rows: &[usize]) -> Self {
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
                match dataset.feature(feat_idx) {
                    Feature::Dense(values) => {
                        if let Some(slice) = values.as_slice() {
                            for (row_pos, &row_idx) in sampled_rows.iter().enumerate() {
                                let v = slice[row_idx];
                                if v != 0.0 && !v.is_nan() {
                                    bits.insert(row_pos);
                                }
                            }
                        } else {
                            for (row_pos, &row_idx) in sampled_rows.iter().enumerate() {
                                let v = values[row_idx];
                                if v != 0.0 && !v.is_nan() {
                                    bits.insert(row_pos);
                                }
                            }
                        }
                    }
                    Feature::Sparse {
                        indices, values, ..
                    } => {
                        // Two-pointer scan: both `sampled_rows` and `indices` are sorted.
                        let mut p = 0usize;
                        for (row_pos, &row_idx) in sampled_rows.iter().enumerate() {
                            let row_u32 = row_idx as u32;
                            while p < indices.len() && indices[p] < row_u32 {
                                p += 1;
                            }
                            if p < indices.len() && indices[p] == row_u32 {
                                let v = values[p];
                                if v != 0.0 && !v.is_nan() {
                                    bits.insert(row_pos);
                                }
                            }
                        }
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

    /// Get the conflict count between two features.
    fn get_conflict(&self, i: usize, j: usize) -> usize {
        let key = if i < j { (i, j) } else { (j, i) };
        *self.conflicts.get(&key).unwrap_or(&0)
    }

    /// Total conflicting pairs.
    fn conflict_pair_count(&self) -> usize {
        self.conflicts.len()
    }

    /// Total possible pairs.
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
fn sample_rows(n_rows: usize, max_samples: usize, seed: u64) -> Vec<usize> {
    if n_rows <= max_samples {
        return (0..n_rows).collect();
    }

    let n_first = max_samples / 10;
    let n_last = max_samples / 10;
    let n_random = max_samples - n_first - n_last;

    let mut result = Vec::with_capacity(max_samples);

    // First rows
    result.extend(0..n_first.min(n_rows));

    // Last rows
    let last_start = n_rows.saturating_sub(n_last);
    result.extend(last_start..n_rows);

    // Random middle rows
    let middle_start = n_first;
    let middle_end = last_start;

    if middle_end > middle_start {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let middle_range: Vec<usize> = (middle_start..middle_end).collect();

        // Reservoir sampling
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

    result.sort_unstable();
    result.dedup();
    result
}

// =============================================================================
// Bundle Assignment
// =============================================================================

/// A bundle assignment: feature indices that will be bundled together.
#[derive(Clone, Debug)]
struct BundleAssignment {
    /// Original feature indices.
    feature_indices: Vec<usize>,
}

impl BundleAssignment {
    fn new(feature_idx: usize) -> Self {
        Self {
            feature_indices: vec![feature_idx],
        }
    }

    fn add(&mut self, feature_idx: usize) {
        self.feature_indices.push(feature_idx);
    }

    fn len(&self) -> usize {
        self.feature_indices.len()
    }
}

/// Assign features to bundles using greedy algorithm.
fn assign_bundles(
    analyses: &[FeatureAnalysis],
    sparse_indices: &[usize], // Indices into analyses
    conflict_graph: &ConflictGraph,
    config: &BundlingConfig,
    n_sampled_rows: usize,
) -> Vec<BundleAssignment> {
    if sparse_indices.is_empty() {
        return Vec::new();
    }

    let max_conflicts = config.max_conflicts(n_sampled_rows);

    // Sort features by density (denser first = more restrictive)
    let mut sorted_sparse: Vec<usize> = sparse_indices.to_vec();
    sorted_sparse.sort_by(|&a, &b| {
        let density_a = analyses[a].density();
        let density_b = analyses[b].density();
        density_b
            .partial_cmp(&density_a)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut bundles: Vec<BundleAssignment> = Vec::new();
    let mut bundle_conflicts: Vec<usize> = Vec::new();

    // Map from sparse_indices position to sorted position
    let sparse_to_sorted: HashMap<usize, usize> = sparse_indices
        .iter()
        .enumerate()
        .map(|(i, &idx)| (idx, i))
        .collect();

    for &feat_analysis_idx in &sorted_sparse {
        let original_idx = analyses[feat_analysis_idx].feature_idx;
        let sparse_idx = sparse_to_sorted[&feat_analysis_idx];

        // Find the best bundle for this feature
        let mut best_bundle: Option<usize> = None;
        let mut best_conflict_increase = usize::MAX;

        for (bundle_idx, bundle) in bundles.iter().enumerate() {
            // Check bundle size limit
            if bundle.len() >= config.max_bundle_size {
                continue;
            }

            // Calculate conflict increase
            let mut conflict_sum = 0usize;
            for &existing_orig in &bundle.feature_indices {
                // Find sparse index for existing feature
                if let Some(existing_analysis_idx) =
                    analyses.iter().position(|a| a.feature_idx == existing_orig)
                    && let Some(&existing_sparse_idx) = sparse_to_sorted.get(&existing_analysis_idx)
                {
                    conflict_sum += conflict_graph.get_conflict(sparse_idx, existing_sparse_idx);
                }
            }

            if bundle_conflicts[bundle_idx] + conflict_sum <= max_conflicts
                && conflict_sum < best_conflict_increase
            {
                best_bundle = Some(bundle_idx);
                best_conflict_increase = conflict_sum;
            }
        }

        if let Some(bundle_idx) = best_bundle {
            bundles[bundle_idx].add(original_idx);
            bundle_conflicts[bundle_idx] += best_conflict_increase;
        } else {
            // Create a new bundle
            bundles.push(BundleAssignment::new(original_idx));
            bundle_conflicts.push(0);
        }
    }

    bundles
}

// =============================================================================
// Public API
// =============================================================================

/// Result of bundle planning.
#[derive(Clone, Debug)]
pub struct BundlePlan {
    /// Bundles to create (each contains 2+ features).
    pub bundles: Vec<Vec<usize>>,
    /// Features that remain standalone (not bundled).
    pub standalone_sparse: Vec<usize>,
    /// Whether bundling was skipped.
    pub skipped: bool,
    /// Reason for skipping (if skipped).
    pub skip_reason: Option<String>,
}

/// Create a bundle plan from sparse categorical features.
///
/// # Arguments
/// * `analyses` - Feature analysis results
/// * `data` - Raw feature data in column-major format
/// * `config` - Bundling configuration
///
/// # Returns
/// A `BundlePlan` describing which features should be bundled.
pub fn create_bundle_plan(
    analyses: &[FeatureAnalysis],
    dataset: &Dataset,
    config: &BundlingConfig,
) -> BundlePlan {
    // Identify sparse categorical features
    let sparse_threshold = 1.0 - config.min_sparsity;
    let sparse_categorical: Vec<usize> = analyses
        .iter()
        .enumerate()
        .filter(|(_, a)| !a.is_trivial && !a.is_numeric && a.density() <= sparse_threshold)
        .map(|(i, _)| i)
        .collect();

    // Check if too few sparse features
    if sparse_categorical.len() < 2 {
        return BundlePlan {
            bundles: Vec::new(),
            standalone_sparse: sparse_categorical
                .iter()
                .map(|&i| analyses[i].feature_idx)
                .collect(),
            skipped: true,
            skip_reason: Some("Too few sparse features for bundling".to_string()),
        };
    }

    // Check if too many sparse features
    if sparse_categorical.len() > config.max_sparse_features {
        return BundlePlan {
            bundles: Vec::new(),
            standalone_sparse: sparse_categorical
                .iter()
                .map(|&i| analyses[i].feature_idx)
                .collect(),
            skipped: true,
            skip_reason: Some(format!(
                "Too many sparse features ({} > {})",
                sparse_categorical.len(),
                config.max_sparse_features
            )),
        };
    }

    // Get original feature indices for sparse features
    let sparse_original: Vec<usize> = sparse_categorical
        .iter()
        .map(|&i| analyses[i].feature_idx)
        .collect();

    // Sample rows for conflict detection
    let n_samples = dataset.n_samples();
    let sampled_rows = sample_rows(n_samples, config.max_sample_rows, config.seed);
    let n_sampled = sampled_rows.len();

    // Build conflict graph
    let conflict_graph = ConflictGraph::build(dataset, &sparse_original, &sampled_rows);

    // Check for high conflict rate
    if conflict_graph.conflict_rate() > 0.5 {
        return BundlePlan {
            bundles: Vec::new(),
            standalone_sparse: sparse_original,
            skipped: true,
            skip_reason: Some(format!(
                "High conflict rate ({:.1}%)",
                conflict_graph.conflict_rate() * 100.0
            )),
        };
    }

    // Assign features to bundles
    let bundle_assignments = assign_bundles(
        analyses,
        &sparse_categorical,
        &conflict_graph,
        config,
        n_sampled,
    );

    // Separate bundles (2+ features) from singletons
    let mut bundles: Vec<Vec<usize>> = Vec::new();
    let mut standalone: Vec<usize> = Vec::new();

    for assignment in bundle_assignments {
        if assignment.len() >= 2 {
            bundles.push(assignment.feature_indices);
        } else {
            standalone.extend(assignment.feature_indices);
        }
    }

    BundlePlan {
        bundles,
        standalone_sparse: standalone,
        skipped: false,
        skip_reason: None,
    }
}

/// Apply bundling to a grouping result.
///
/// This replaces individual `SparseCategorical` groups with `Bundle` groups
/// where appropriate, based on the bundle plan.
pub fn apply_bundling(
    mut grouping: GroupingResult,
    plan: &BundlePlan,
    _analyses: &[FeatureAnalysis],
) -> GroupingResult {
    if plan.skipped || plan.bundles.is_empty() {
        return grouping;
    }

    // Collect features that are being bundled
    let bundled_features: std::collections::HashSet<usize> =
        plan.bundles.iter().flatten().copied().collect();

    // Remove SparseCategorical groups for bundled features
    grouping.groups.retain(|g| {
        if g.group_type == GroupType::SparseCategorical && g.feature_indices.len() == 1 {
            !bundled_features.contains(&g.feature_indices[0])
        } else {
            true
        }
    });

    // Add Bundle groups
    for bundle_features in &plan.bundles {
        // Calculate if we need U16 (based on total bins across all features)
        // Bundle always uses U16 for encoded bins
        let needs_u16 = true;

        grouping.groups.push(GroupSpec::new(
            bundle_features.clone(),
            GroupType::Bundle,
            needs_u16,
        ));
    }

    grouping
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, array};

    fn make_analysis(idx: usize, is_numeric: bool, density: f32) -> FeatureAnalysis {
        FeatureAnalysis {
            feature_idx: idx,
            is_numeric,
            is_sparse: density < 0.5,
            needs_u16: false,
            is_trivial: false,
            is_binary: false,
            n_unique: 10,
            n_nonzero: (density * 100.0) as usize,
            n_valid: 100,
            n_nan: 0,
            n_samples: 100,
            min_val: 0.0,
            max_val: 10.0,
        }
    }

    #[test]
    fn test_bundling_config_defaults() {
        let config = BundlingConfig::default();
        assert_eq!(config.max_conflict_rate, 0.0001);
        assert_eq!(config.min_sparsity, 0.9);
        assert_eq!(config.max_bundle_size, 256);
    }

    #[test]
    fn test_max_conflicts_calculation() {
        let config = BundlingConfig::default();
        assert_eq!(config.max_conflicts(10000), 1); // 0.0001 * 10000 = 1
        assert_eq!(config.max_conflicts(100000), 10); // 0.0001 * 100000 = 10
    }

    #[test]
    fn test_sample_rows_small() {
        let rows = sample_rows(50, 100, 42);
        assert_eq!(rows.len(), 50);
        assert_eq!(rows, (0..50).collect::<Vec<_>>());
    }

    #[test]
    fn test_sample_rows_large() {
        let rows = sample_rows(100000, 1000, 42);
        assert!(rows.len() <= 1000);
        assert!(rows.len() >= 900); // Should be close to 1000
        // Should be sorted
        for i in 1..rows.len() {
            assert!(rows[i] > rows[i - 1]);
        }
    }

    #[test]
    fn test_create_bundle_plan_too_few_sparse() {
        let analyses = vec![
            make_analysis(0, false, 0.05), // Only one sparse categorical
        ];
        let features = Array2::from_shape_vec((1, 100), vec![0.0; 100]).unwrap();
        let dataset = Dataset::from_array(features.view(), None, None);
        let config = BundlingConfig::default();

        let plan = create_bundle_plan(&analyses, &dataset, &config);

        assert!(plan.skipped);
        assert_eq!(plan.bundles.len(), 0);
    }

    #[test]
    fn test_create_bundle_plan_no_conflicts() {
        // Two sparse categorical features with no conflicts (mutually exclusive)
        let analyses = vec![make_analysis(0, false, 0.05), make_analysis(1, false, 0.05)];

        // Feature 0 is non-zero only for rows 0-4
        // Feature 1 is non-zero only for rows 5-9
        let mut data0 = vec![0.0; 100];
        let mut data1 = vec![0.0; 100];
        for v in data0.iter_mut().take(5) {
            *v = 1.0;
        }
        for v in data1.iter_mut().take(10).skip(5) {
            *v = 1.0;
        }

        let flat: Vec<f32> = data0.into_iter().chain(data1).collect();
        let features = Array2::from_shape_vec((2, 100), flat).unwrap();
        let dataset = Dataset::from_array(features.view(), None, None);
        let config = BundlingConfig::default();

        let plan = create_bundle_plan(&analyses, &dataset, &config);

        assert!(!plan.skipped);
        assert_eq!(plan.bundles.len(), 1);
        assert_eq!(plan.bundles[0].len(), 2);
    }

    #[test]
    fn test_conflict_graph_no_conflicts() {
        // Two features with no overlap
        let dataset = Dataset::from_array(
            array![[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]].view(),
            None,
            None,
        );
        let sparse_features = vec![0, 1];
        let sampled_rows: Vec<usize> = (0..4).collect();

        let graph = ConflictGraph::build(&dataset, &sparse_features, &sampled_rows);

        assert_eq!(graph.get_conflict(0, 1), 0);
        assert_eq!(graph.conflict_pair_count(), 0);
    }

    #[test]
    fn test_conflict_graph_with_conflicts() {
        // Two features with overlap at row 0
        let dataset = Dataset::from_array(
            array![[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]].view(),
            None,
            None,
        );
        let sparse_features = vec![0, 1];
        let sampled_rows: Vec<usize> = (0..4).collect();

        let graph = ConflictGraph::build(&dataset, &sparse_features, &sampled_rows);

        assert_eq!(graph.get_conflict(0, 1), 1);
        assert_eq!(graph.conflict_pair_count(), 1);
    }
}

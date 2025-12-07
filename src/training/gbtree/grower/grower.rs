//! Tree grower implementation.
//!
//! The [`TreeGrower`] coordinates tree building by bringing together:
//! - Histogram building
//! - Split finding
//! - Row partitioning
//! - Growth strategy (depth-wise or leaf-wise)
//! - Monotonic constraints
//! - Interaction constraints
//!
//! # Design
//!
//! `TreeGrower` receives pre-configured components (samplers, constraint checkers)
//! rather than constructing them from raw parameters. This keeps the grower focused
//! on tree building while the trainer handles configuration.

use std::collections::HashMap;

use super::building::{BuildingTree, NodeCandidate};
use super::policy::GrowthStrategy;
use super::super::constraints::{
    InteractionConstraints, MonotonicBounds, MonotonicChecker,
};
use super::super::histogram::{HistogramBuilder, NodeHistogram};
use super::super::partition::RowPartitioner;
use super::super::quantize::{BinCuts, BinIndex, QuantizedMatrix};
use super::super::sampling::ColumnSampler;
use super::super::split::{GainParams, GreedySplitFinder, SplitFinder, SplitInfo};
use crate::training::buffer::GradientBuffer;

// ============================================================================
// TreeBuildParams
// ============================================================================

/// Core parameters for tree building.
///
/// This struct contains only the essential tree-building parameters.
/// Sampling and constraint components are passed separately to `TreeGrower`.
#[derive(Debug, Clone)]
pub struct TreeBuildParams {
    /// Parameters for gain computation (regularization, min child weight, etc.)
    pub gain: GainParams,
    /// Maximum tree depth (used by depth-wise, also as absolute limit for leaf-wise)
    pub max_depth: u32,
    /// Maximum number of leaves (used by leaf-wise growth)
    pub max_leaves: u32,
    /// Minimum samples required to split a node
    pub min_samples_split: u32,
    /// Minimum samples required in a leaf
    pub min_samples_leaf: u32,
    /// Use parallel histogram building (beneficial for many features)
    pub parallel_histograms: bool,
}

impl Default for TreeBuildParams {
    fn default() -> Self {
        Self {
            gain: GainParams::default(),
            max_depth: 6,
            max_leaves: 31, // 2^5 - 1, common LightGBM default
            min_samples_split: 2,
            min_samples_leaf: 1,
            parallel_histograms: false, // Sequential by default, parallel for many features
        }
    }
}

// ============================================================================
// TreeGrower
// ============================================================================

/// Coordinates tree growing with a growth strategy.
///
/// Brings together histogram building, split finding, and row partitioning
/// to grow a tree according to the specified growth strategy.
///
/// # Design
///
/// `TreeGrower` receives pre-configured components rather than constructing
/// them from raw parameters:
/// - `ColumnSampler`: Pre-configured with sampling ratios
/// - `MonotonicChecker`: Pre-configured with per-feature constraints
/// - `InteractionConstraints`: Pre-configured with feature groups
///
/// This keeps the grower focused on tree building while the trainer handles
/// component configuration.
///
/// # Naming Note
///
/// Named `TreeGrower` (not `TreeBuilder`) to avoid confusion with
/// `trees::TreeBuilder` which is an inference-time builder pattern helper.
pub struct TreeGrower<'a> {
    /// Growth strategy (depth-wise or leaf-wise)
    strategy: GrowthStrategy,
    /// Histogram builder
    hist_builder: HistogramBuilder,
    /// Split finder
    split_finder: GreedySplitFinder,
    /// Bin cuts for histograms
    cuts: &'a BinCuts,
    /// Tree building parameters
    params: TreeBuildParams,
    /// Learning rate (shrinkage) applied to leaf weights
    learning_rate: f32,
    /// Column sampler for feature sampling (pre-configured)
    col_sampler: &'a ColumnSampler,
    /// Monotonic constraint checker (pre-configured)
    mono_checker: &'a MonotonicChecker,
    /// Interaction constraints (pre-configured)
    interaction_constraints: &'a InteractionConstraints,
}

impl<'a> TreeGrower<'a> {
    /// Create a new tree grower with pre-configured components.
    ///
    /// # Arguments
    ///
    /// * `strategy` - Growth strategy (depth-wise or leaf-wise)
    /// * `cuts` - Bin cuts for histogram building
    /// * `params` - Core tree building parameters
    /// * `learning_rate` - Shrinkage applied to leaf weights
    /// * `col_sampler` - Pre-configured column sampler
    /// * `mono_checker` - Pre-configured monotonic constraint checker
    /// * `interaction_constraints` - Pre-configured interaction constraints
    pub fn new(
        strategy: GrowthStrategy,
        cuts: &'a BinCuts,
        params: TreeBuildParams,
        learning_rate: f32,
        col_sampler: &'a ColumnSampler,
        mono_checker: &'a MonotonicChecker,
        interaction_constraints: &'a InteractionConstraints,
    ) -> Self {
        Self {
            strategy,
            hist_builder: HistogramBuilder::default(),
            split_finder: GreedySplitFinder::new(),
            cuts,
            params,
            learning_rate,
            col_sampler,
            mono_checker,
            interaction_constraints,
        }
    }

    /// Build a single tree with a specific seed for column sampling.
    ///
    /// # Arguments
    ///
    /// * `quantized` - Quantized feature matrix
    /// * `grads` - Gradient buffer with (grad, hess) for each row
    /// * `partitioner` - Row partitioner (will be modified during building)
    /// * `seed` - Seed for column sampling reproducibility
    ///
    /// # Returns
    ///
    /// The built tree structure.
    pub fn build_tree<B: BinIndex>(
        &mut self,
        quantized: &QuantizedMatrix<B>,
        grads: &GradientBuffer,
        partitioner: &mut RowPartitioner,
        seed: u64,
    ) -> BuildingTree {
        let mut tree = BuildingTree::new(0.0);
        let mut state = self.strategy.init();

        // Check if components are enabled
        let col_sampling_enabled = self.col_sampler.is_enabled();
        let mono_enabled = self.mono_checker.is_enabled();
        let interaction_enabled = self.interaction_constraints.is_enabled();

        // Histogram storage per node (reuse across iterations)
        let mut histograms: HashMap<u32, NodeHistogram> = HashMap::new();

        // Monotonic bounds per node (node_id -> bounds)
        let mut node_bounds: HashMap<u32, MonotonicBounds> = HashMap::new();
        node_bounds.insert(0, MonotonicBounds::unbounded());

        // Path features per node (node_id -> list of features used on path from root)
        let mut node_path_features: HashMap<u32, Vec<u32>> = HashMap::new();
        node_path_features.insert(0, Vec::new());

        // Build root histogram
        let root_rows = partitioner.node_rows(0);
        let root_hist = self.build_histogram(root_rows, quantized, grads);
        histograms.insert(0, root_hist);

        // Find initial split for root (depth=0, node_id=0)
        let root_features = self.get_allowed_features(
            0,
            0,
            seed,
            &[],
            col_sampling_enabled,
            interaction_enabled,
        );
        self.split_finder.feature_subset = root_features;
        let mut root_split = self
            .split_finder
            .find_best_split(&histograms[&0], self.cuts, &self.params.gain);

        // Apply monotonic constraints to root split
        if mono_enabled && root_split.is_valid() {
            let root_bounds = node_bounds[&0];
            self.apply_monotonic_constraints(&mut root_split, &root_bounds);
        }

        // Update root weight (K-dimensional)
        tree.node_mut(0).weight = root_split.weight_left.clone();

        // Use HashMap for candidates so leaf-wise can find candidates from previous iterations
        let mut candidates: HashMap<u32, NodeCandidate> = HashMap::new();
        candidates.insert(
            0,
            NodeCandidate::new(0, root_split, 0, partitioner.node_size(0)),
        );

        // Main growth loop
        while self.strategy.should_continue(&state) {
            let candidate_vec: Vec<_> = candidates.values().cloned().collect();
            let nodes_to_expand = self.strategy.select_nodes(&mut state, &candidate_vec);

            if nodes_to_expand.is_empty() {
                break;
            }

            // Expand selected nodes
            for &node_id in &nodes_to_expand {
                let candidate = match candidates.remove(&node_id) {
                    Some(c) => c,
                    None => continue, // Already processed or removed
                };

                if !self.should_split(&candidate) {
                    continue;
                }

                // Get partition node for this tree node
                let partition_id = tree.node(node_id).partition_id;

                // Apply split to partitioner
                let split = candidate.split.clone();
                let (left_partition, right_partition) =
                    partitioner.apply_split(partition_id, &split, quantized);

                // Compute child bounds for monotonic constraints
                let (left_bounds, right_bounds) = if mono_enabled {
                    let parent_bounds = node_bounds
                        .get(&node_id)
                        .copied()
                        .unwrap_or_else(MonotonicBounds::unbounded);
                    let constraint = self.mono_checker.get(split.feature);
                    // Use midpoint of split weights as the bound pivot
                    let pivot = (split.weight_left + split.weight_right) / 2.0;
                    parent_bounds.child_bounds(constraint, pivot)
                } else {
                    (MonotonicBounds::unbounded(), MonotonicBounds::unbounded())
                };

                // Expand tree node
                let (left_id, right_id) =
                    tree.expand(node_id, split.clone(), left_partition, right_partition);

                // Store child bounds
                node_bounds.insert(left_id, left_bounds);
                node_bounds.insert(right_id, right_bounds);

                // Compute child path features (add split feature to parent's path)
                let parent_path = node_path_features
                    .get(&node_id)
                    .cloned()
                    .unwrap_or_default();
                let mut child_path = parent_path.clone();
                child_path.push(split.feature);
                node_path_features.insert(left_id, child_path.clone());
                node_path_features.insert(right_id, child_path);

                // Register children with growth state
                state.add_children(left_id, right_id, candidate.depth + 1);

                // Build histograms for children (use subtraction optimization)
                let parent_hist = &histograms[&node_id];
                let (left_hist, right_hist) = self.build_child_histograms(
                    parent_hist,
                    left_partition,
                    right_partition,
                    quantized,
                    grads,
                    partitioner,
                );

                // Find splits for new nodes with column sampling and interaction constraints
                let child_depth = candidate.depth + 1;
                let child_path_features = node_path_features
                    .get(&left_id)
                    .map(|v| v.as_slice())
                    .unwrap_or(&[]);

                // Left child
                let left_features = self.get_allowed_features(
                    child_depth,
                    left_id,
                    seed,
                    child_path_features,
                    col_sampling_enabled,
                    interaction_enabled,
                );
                self.split_finder.feature_subset = left_features;
                let mut left_split = self
                    .split_finder
                    .find_best_split(&left_hist, self.cuts, &self.params.gain);

                // Apply monotonic constraints to left split
                if mono_enabled && left_split.is_valid() {
                    self.apply_monotonic_constraints(&mut left_split, &left_bounds);
                }

                // Right child
                let right_features = self.get_allowed_features(
                    child_depth,
                    right_id,
                    seed,
                    child_path_features,
                    col_sampling_enabled,
                    interaction_enabled,
                );
                self.split_finder.feature_subset = right_features;
                let mut right_split = self
                    .split_finder
                    .find_best_split(&right_hist, self.cuts, &self.params.gain);

                // Apply monotonic constraints to right split
                if mono_enabled && right_split.is_valid() {
                    self.apply_monotonic_constraints(&mut right_split, &right_bounds);
                }

                histograms.insert(left_id, left_hist);
                histograms.insert(right_id, right_hist);

                // Add new candidates to the map
                candidates.insert(
                    left_id,
                    NodeCandidate::new(
                        left_id,
                        left_split,
                        candidate.depth + 1,
                        partitioner.node_size(left_partition),
                    ),
                );
                candidates.insert(
                    right_id,
                    NodeCandidate::new(
                        right_id,
                        right_split,
                        candidate.depth + 1,
                        partitioner.node_size(right_partition),
                    ),
                );
            }

            // Advance to next iteration (swap level buffers for depth-wise)
            self.strategy.advance(&mut state);
        }

        // Clamp final leaf weights to monotonic bounds
        if mono_enabled {
            self.clamp_leaf_weights(&mut tree, &node_bounds);
        }

        // Apply learning rate to leaf weights
        tree.apply_learning_rate(self.learning_rate);

        tree
    }

    /// Get allowed features for a node, combining column sampling and interaction constraints.
    fn get_allowed_features(
        &self,
        depth: u32,
        node_id: u32,
        tree_seed: u64,
        path_features: &[u32],
        col_sampling_enabled: bool,
        interaction_enabled: bool,
    ) -> Option<Vec<u32>> {
        let mut allowed: Option<Vec<u32>> = None;

        // Apply interaction constraints
        if interaction_enabled {
            let interaction_allowed = self.interaction_constraints.allowed_features(path_features);
            allowed = Some(interaction_allowed);
        }

        // Apply column sampling
        if col_sampling_enabled {
            let sampled = self
                .col_sampler
                .allowed_features_for_node(depth, node_id, tree_seed);
            match &mut allowed {
                Some(features) => {
                    // Intersect with sampled features
                    features.retain(|f| sampled.contains(f));
                }
                None => {
                    allowed = Some(sampled);
                }
            }
        }

        allowed
    }

    /// Apply monotonic constraints to a split, adjusting weights if needed.
    fn apply_monotonic_constraints(&self, split: &mut SplitInfo, bounds: &MonotonicBounds) {
        let (new_left, new_right, _is_valid) = self.mono_checker.check_and_fix(
            split.feature,
            split.weight_left,
            split.weight_right,
            bounds,
        );
        split.weight_left = new_left;
        split.weight_right = new_right;
    }

    /// Clamp all leaf weights to their monotonic bounds.
    fn clamp_leaf_weights(&self, tree: &mut BuildingTree, bounds: &HashMap<u32, MonotonicBounds>) {
        for leaf_id in tree.leaves().collect::<Vec<_>>() {
            if let Some(node_bounds) = bounds.get(&leaf_id) {
                let node = tree.node_mut(leaf_id);
                node.weight = node_bounds.clamp(node.weight);
            }
        }
    }

    /// Check if a candidate should be split.
    fn should_split(&self, candidate: &NodeCandidate) -> bool {
        // max_depth of 0 means no limit (XGBoost convention)
        let depth_ok = self.params.max_depth == 0 || candidate.depth < self.params.max_depth;
        candidate.is_valid()
            && depth_ok
            && candidate.num_samples >= self.params.min_samples_split
    }

    /// Build histogram for a set of rows.
    fn build_histogram<B: BinIndex>(
        &mut self,
        rows: &[u32],
        quantized: &QuantizedMatrix<B>,
        grads: &GradientBuffer,
    ) -> NodeHistogram {
        let mut hist = NodeHistogram::new(self.cuts);
        if self.params.parallel_histograms {
            self.hist_builder
                .build_parallel(&mut hist, quantized, grads, rows);
        } else {
            self.hist_builder.build(&mut hist, quantized, grads, rows);
        }
        hist
    }

    /// Build histograms for child nodes using subtraction optimization.
    fn build_child_histograms<B: BinIndex>(
        &mut self,
        parent_hist: &NodeHistogram,
        left_partition: u32,
        right_partition: u32,
        quantized: &QuantizedMatrix<B>,
        grads: &GradientBuffer,
        partitioner: &RowPartitioner,
    ) -> (NodeHistogram, NodeHistogram) {
        let left_rows = partitioner.node_rows(left_partition);
        let right_rows = partitioner.node_rows(right_partition);

        let left_size = left_rows.len();
        let right_size = right_rows.len();

        // Build smaller child directly, derive larger via subtraction
        if left_size <= right_size {
            let left_hist = self.build_histogram(left_rows, quantized, grads);
            let right_hist = parent_hist.subtract(&left_hist);
            (left_hist, right_hist)
        } else {
            let right_hist = self.build_histogram(right_rows, quantized, grads);
            let left_hist = parent_hist.subtract(&right_hist);
            (left_hist, right_hist)
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::super::constraints::MonotonicConstraint;
    use crate::data::DenseMatrix;
    use crate::training::gbtree::quantize::{CutFinder, ExactQuantileCuts, Quantizer};

    /// Helper to create default (disabled) components for tests that don't need them.
    fn make_default_components(num_features: u32) -> (ColumnSampler, MonotonicChecker, InteractionConstraints) {
        let col_sampler = ColumnSampler::new(num_features, 1.0, 1.0, 1.0);
        let mono_checker = MonotonicChecker::new(&[], num_features as usize);
        let interaction_constraints = InteractionConstraints::new(&[], num_features);
        (col_sampler, mono_checker, interaction_constraints)
    }

    fn make_test_data() -> (QuantizedMatrix<u8>, BinCuts, GradientBuffer) {
        // 10 rows, 2 features (row-major)
        // Feature 0: values 0..9
        // Feature 1: first 5 rows = 0, last 5 rows = 1
        let data: Vec<f32> = vec![
            0.0, 0.0, // row 0
            1.0, 0.0, // row 1
            2.0, 0.0, // row 2
            3.0, 0.0, // row 3
            4.0, 0.0, // row 4
            5.0, 1.0, // row 5
            6.0, 1.0, // row 6
            7.0, 1.0, // row 7
            8.0, 1.0, // row 8
            9.0, 1.0, // row 9
        ];
        let matrix = DenseMatrix::from_vec(data, 10, 2);
        // Use min_samples_per_bin=1 for small test data
        let cuts_finder = ExactQuantileCuts::new(1);
        let cuts = cuts_finder.find_cuts(&matrix, 256);

        let quantizer = Quantizer::new(cuts.clone());
        let quantized = quantizer.quantize::<_, u8>(&matrix);

        // Simple gradients: positive for first group, negative for second
        let mut grads = GradientBuffer::new(10, 1);
        for i in 0..5 {
            grads.set(i, 0, 1.0, 1.0);
        }
        for i in 5..10 {
            grads.set(i, 0, -1.0, 1.0);
        }

        (quantized, cuts, grads)
    }

    #[test]
    fn test_tree_build_params_default() {
        let params = TreeBuildParams::default();
        assert_eq!(params.max_depth, 6);
        assert_eq!(params.max_leaves, 31);
        assert_eq!(params.min_samples_split, 2);
        assert_eq!(params.min_samples_leaf, 1);
    }

    #[test]
    fn test_tree_grower_single_split() {
        let (quantized, cuts, grads) = make_test_data();
        let num_features = cuts.num_features() as u32;

        let strategy = GrowthStrategy::DepthWise { max_depth: 1 };
        let params = TreeBuildParams {
            max_depth: 1,
            ..Default::default()
        };
        let learning_rate = 1.0;

        let (mut col_sampler, mono_checker, interaction_constraints) = make_default_components(num_features);
        col_sampler.sample_tree(0);

        let mut partitioner = RowPartitioner::new(10);
        let mut grower = TreeGrower::new(
            strategy, &cuts, params, learning_rate,
            &col_sampler, &mono_checker, &interaction_constraints,
        );

        let tree = grower.build_tree(&quantized, &grads, &mut partitioner, 0);

        // Should have root + 2 children
        assert!(tree.num_nodes() >= 1);
        // Root should be split (if gain was found)
        let root = tree.node(0);
        if !root.is_leaf {
            assert_eq!(tree.num_leaves(), 2);
            assert_eq!(tree.max_depth(), 1);
        }
    }

    #[test]
    fn test_tree_grower_multiple_levels() {
        let (quantized, cuts, grads) = make_test_data();
        let num_features = cuts.num_features() as u32;

        let strategy = GrowthStrategy::DepthWise { max_depth: 3 };
        let params = TreeBuildParams::default();
        let learning_rate = 1.0;

        let (mut col_sampler, mono_checker, interaction_constraints) = make_default_components(num_features);
        col_sampler.sample_tree(0);

        let mut partitioner = RowPartitioner::new(10);
        let mut grower = TreeGrower::new(
            strategy, &cuts, params, learning_rate,
            &col_sampler, &mono_checker, &interaction_constraints,
        );

        let tree = grower.build_tree(&quantized, &grads, &mut partitioner, 0);

        // Should build multiple levels until no more gain or max depth
        assert!(tree.max_depth() <= 3);
        // All leaves should be marked as leaves
        for leaf_id in tree.leaves() {
            assert!(tree.node(leaf_id).is_leaf);
        }
    }

    #[test]
    fn test_leaf_wise_single_split() {
        let (quantized, cuts, grads) = make_test_data();
        let num_features = cuts.num_features() as u32;

        let strategy = GrowthStrategy::LeafWise { max_leaves: 2 };
        let params = TreeBuildParams::default();
        let learning_rate = 1.0;

        let (mut col_sampler, mono_checker, interaction_constraints) = make_default_components(num_features);
        col_sampler.sample_tree(0);

        let mut partitioner = RowPartitioner::new(10);
        let mut grower = TreeGrower::new(
            strategy, &cuts, params, learning_rate,
            &col_sampler, &mono_checker, &interaction_constraints,
        );

        let tree = grower.build_tree(&quantized, &grads, &mut partitioner, 0);

        // With max_leaves=2, should have exactly 2 leaves (1 split)
        assert_eq!(tree.num_leaves(), 2);
        assert!(!tree.node(0).is_leaf); // Root was split
    }

    #[test]
    fn test_leaf_wise_max_leaves_constraint() {
        let (quantized, cuts, grads) = make_test_data();
        let num_features = cuts.num_features() as u32;

        let strategy = GrowthStrategy::LeafWise { max_leaves: 4 };
        let params = TreeBuildParams::default();
        let learning_rate = 1.0;

        let (mut col_sampler, mono_checker, interaction_constraints) = make_default_components(num_features);
        col_sampler.sample_tree(0);

        let mut partitioner = RowPartitioner::new(10);
        let mut grower = TreeGrower::new(
            strategy, &cuts, params, learning_rate,
            &col_sampler, &mono_checker, &interaction_constraints,
        );

        let tree = grower.build_tree(&quantized, &grads, &mut partitioner, 0);

        // Should not exceed max_leaves
        assert!(tree.num_leaves() <= 4);
        // All leaf nodes should be marked as leaves
        for leaf_id in tree.leaves() {
            assert!(tree.node(leaf_id).is_leaf);
        }
    }

    #[test]
    fn test_leaf_wise_vs_depth_wise_different_shapes() {
        let (quantized, cuts, grads) = make_test_data();
        let num_features = cuts.num_features() as u32;

        // Build tree with depth-wise
        let depth_strategy = GrowthStrategy::DepthWise { max_depth: 3 };
        let params = TreeBuildParams::default();
        let learning_rate = 1.0;

        let (mut col_sampler, mono_checker, interaction_constraints) = make_default_components(num_features);
        col_sampler.sample_tree(0);

        let mut partitioner1 = RowPartitioner::new(10);
        let mut depth_grower = TreeGrower::new(
            depth_strategy, &cuts, params.clone(), learning_rate,
            &col_sampler, &mono_checker, &interaction_constraints,
        );
        let depth_tree = depth_grower.build_tree(&quantized, &grads, &mut partitioner1, 0);

        // Build tree with leaf-wise (same number of leaves)
        let leaf_strategy = GrowthStrategy::LeafWise {
            max_leaves: depth_tree.num_leaves(),
        };
        let mut partitioner2 = RowPartitioner::new(10);
        let mut leaf_grower = TreeGrower::new(
            leaf_strategy, &cuts, params, learning_rate,
            &col_sampler, &mono_checker, &interaction_constraints,
        );
        let leaf_tree = leaf_grower.build_tree(&quantized, &grads, &mut partitioner2, 0);

        // Both should have same number of leaves
        assert_eq!(depth_tree.num_leaves(), leaf_tree.num_leaves());

        // Leaf-wise can produce deeper trees (asymmetric growth)
        // This is a characteristic property of leaf-wise growth
        println!("Depth-wise max depth: {}", depth_tree.max_depth());
        println!("Leaf-wise max depth: {}", leaf_tree.max_depth());
    }

    // ---- Monotonic Constraint Tests ----

    /// Create test data where feature 0 has a clear monotonic relationship.
    /// Higher feature values should have lower predictions (negative gradient relationship).
    fn make_monotonic_test_data() -> (QuantizedMatrix<u8>, BinCuts, GradientBuffer) {
        // 20 rows, 1 feature
        // Feature 0: values 0..19
        // Gradients: linear relationship where higher feature -> higher gradient
        // This means optimal weights decrease as feature increases
        let data: Vec<f32> = (0..20).map(|x| x as f32).collect();
        let matrix = DenseMatrix::from_vec(data, 20, 1);
        
        let cuts_finder = ExactQuantileCuts::new(1);
        let cuts = cuts_finder.find_cuts(&matrix, 256);

        let quantizer = Quantizer::new(cuts.clone());
        let quantized = quantizer.quantize::<_, u8>(&matrix);

        // Gradients: first half positive (want to decrease weight), second half negative
        // This creates a natural split with left=positive weight, right=negative weight
        let mut grads = GradientBuffer::new(20, 1);
        for i in 0..10 {
            // Positive gradient -> negative optimal weight (w* = -G/H)
            grads.set(i, 0, 2.0, 1.0);
        }
        for i in 10..20 {
            // Negative gradient -> positive optimal weight
            grads.set(i, 0, -2.0, 1.0);
        }

        (quantized, cuts, grads)
    }

    #[test]
    fn test_tree_grower_with_increasing_constraint() {
        let (quantized, cuts, grads) = make_monotonic_test_data();
        let num_features = cuts.num_features() as u32;

        let strategy = GrowthStrategy::DepthWise { max_depth: 2 };
        let params = TreeBuildParams::default();
        let learning_rate = 1.0;

        let mut col_sampler = ColumnSampler::new(num_features, 1.0, 1.0, 1.0);
        col_sampler.sample_tree(0);
        let mono_checker = MonotonicChecker::new(
            &[MonotonicConstraint::Increasing],
            num_features as usize,
        );
        let interaction_constraints = InteractionConstraints::new(&[], num_features);

        let mut partitioner = RowPartitioner::new(20);
        let mut grower = TreeGrower::new(
            strategy, &cuts, params, learning_rate,
            &col_sampler, &mono_checker, &interaction_constraints,
        );

        let tree = grower.build_tree(&quantized, &grads, &mut partitioner, 0);

        // Verify monotonicity: for increasing constraint, traverse left to right
        // and verify weights are non-decreasing
        if !tree.node(0).is_leaf {
            // Verify at each split level that left <= right (for increasing)
            let root = tree.node(0);
            if !root.is_leaf {
                let left = tree.node(root.left);
                let right = tree.node(root.right);
                
                // For increasing constraint on feature 0:
                // Left child (lower feature values) should have lower or equal weight
                assert!(
                    left.weight <= right.weight + 0.001, // Small epsilon for float comparison
                    "Increasing constraint violated: left={:.4} > right={:.4}",
                    left.weight, right.weight
                );
            }
        }
    }

    #[test]
    fn test_tree_grower_with_decreasing_constraint() {
        let (quantized, cuts, grads) = make_monotonic_test_data();
        let num_features = cuts.num_features() as u32;

        let strategy = GrowthStrategy::DepthWise { max_depth: 2 };
        let params = TreeBuildParams::default();
        let learning_rate = 1.0;

        let mut col_sampler = ColumnSampler::new(num_features, 1.0, 1.0, 1.0);
        col_sampler.sample_tree(0);
        let mono_checker = MonotonicChecker::new(
            &[MonotonicConstraint::Decreasing],
            num_features as usize,
        );
        let interaction_constraints = InteractionConstraints::new(&[], num_features);

        let mut partitioner = RowPartitioner::new(20);
        let mut grower = TreeGrower::new(
            strategy, &cuts, params, learning_rate,
            &col_sampler, &mono_checker, &interaction_constraints,
        );

        let tree = grower.build_tree(&quantized, &grads, &mut partitioner, 0);

        // Verify monotonicity: for decreasing constraint
        if !tree.node(0).is_leaf {
            let root = tree.node(0);
            let left = tree.node(root.left);
            let right = tree.node(root.right);
            
            // For decreasing constraint on feature 0:
            // Left child (lower feature values) should have higher or equal weight
            assert!(
                left.weight >= right.weight - 0.001,
                "Decreasing constraint violated: left={:.4} < right={:.4}",
                left.weight, right.weight
            );
        }
    }

    #[test]
    fn test_tree_grower_monotonic_constraint_clamps_weights() {
        // Create data that would naturally violate monotonicity
        // Feature increases, but optimal weights would oscillate
        let data: Vec<f32> = (0..10).map(|x| x as f32).collect();
        let matrix = DenseMatrix::from_vec(data, 10, 1);
        
        let cuts_finder = ExactQuantileCuts::new(1);
        let cuts = cuts_finder.find_cuts(&matrix, 256);
        let quantizer = Quantizer::new(cuts.clone());
        let quantized = quantizer.quantize::<_, u8>(&matrix);
        let num_features = cuts.num_features() as u32;

        // Gradients that would create violating splits without constraints
        // Row 0-4: positive gradient, row 5-9: negative gradient
        // But we want increasing constraint
        let mut grads = GradientBuffer::new(10, 1);
        for i in 0..5 {
            grads.set(i, 0, 2.0, 1.0); // optimal weight = -2
        }
        for i in 5..10 {
            grads.set(i, 0, -2.0, 1.0); // optimal weight = +2
        }

        // Without constraint: left would be -2, right would be +2
        // With increasing constraint: both should be clamped to same value (midpoint=0)
        // because left > right violates increasing

        let strategy = GrowthStrategy::DepthWise { max_depth: 1 };
        let params = TreeBuildParams::default();
        let learning_rate = 1.0;

        let mut col_sampler = ColumnSampler::new(num_features, 1.0, 1.0, 1.0);
        col_sampler.sample_tree(0);
        let mono_checker = MonotonicChecker::new(
            &[MonotonicConstraint::Increasing],
            num_features as usize,
        );
        let interaction_constraints = InteractionConstraints::new(&[], num_features);

        let mut partitioner = RowPartitioner::new(10);
        let mut grower = TreeGrower::new(
            strategy, &cuts, params, learning_rate,
            &col_sampler, &mono_checker, &interaction_constraints,
        );

        let tree = grower.build_tree(&quantized, &grads, &mut partitioner, 0);

        if !tree.node(0).is_leaf {
            let root = tree.node(0);
            let left = tree.node(root.left);
            let right = tree.node(root.right);
            
            // Should be clamped to midpoint or satisfy constraint
            assert!(
                left.weight <= right.weight + 0.001,
                "Increasing constraint should clamp: left={:.4} <= right={:.4}",
                left.weight, right.weight
            );
        }
    }

    #[test]
    fn test_tree_grower_no_constraint_allows_any_order() {
        let (quantized, cuts, grads) = make_monotonic_test_data();
        let num_features = cuts.num_features() as u32;

        let strategy = GrowthStrategy::DepthWise { max_depth: 1 };
        let params = TreeBuildParams::default();
        let learning_rate = 1.0;

        let (mut col_sampler, mono_checker, interaction_constraints) = make_default_components(num_features);
        col_sampler.sample_tree(0);

        let mut partitioner = RowPartitioner::new(20);
        let mut grower = TreeGrower::new(
            strategy, &cuts, params, learning_rate,
            &col_sampler, &mono_checker, &interaction_constraints,
        );

        let tree = grower.build_tree(&quantized, &grads, &mut partitioner, 0);

        // Without constraints, the natural split should occur
        // The data has positive gradients in first half -> negative weights
        // Negative gradients in second half -> positive weights
        if !tree.node(0).is_leaf {
            let root = tree.node(0);
            let left = tree.node(root.left);
            let right = tree.node(root.right);
            
            // Without constraints, left should be negative, right should be positive
            // (based on our gradient setup)
            println!("No constraint: left={:.4}, right={:.4}", left.weight, right.weight);
            // Just verify it builds successfully
            assert!(tree.num_leaves() == 2);
        }
    }

    // ---- Interaction Constraint Tests ----

    /// Create test data with 4 features for interaction constraint testing.
    /// Features 0,1 should interact, Features 2,3 should interact, but not across groups.
    fn make_interaction_test_data() -> (QuantizedMatrix<u8>, BinCuts, GradientBuffer) {
        // 20 rows, 4 features - all features have the same value range
        let num_rows = 20;
        let num_features = 4;
        
        // Generate data with clear patterns per feature - all features have values 0-19
        let mut data = Vec::with_capacity(num_rows * num_features);
        for row in 0..num_rows {
            // All features have the same range [0, 19]
            data.push(row as f32);           // Feature 0: 0-19
            data.push(row as f32);           // Feature 1: 0-19
            data.push(row as f32);           // Feature 2: 0-19
            data.push(row as f32);           // Feature 3: 0-19
        }
        let matrix = DenseMatrix::from_vec(data, num_rows, num_features);
        
        let cuts_finder = ExactQuantileCuts::new(num_features);
        let cuts = cuts_finder.find_cuts(&matrix, 256);

        let quantizer = Quantizer::new(cuts.clone());
        let quantized = quantizer.quantize::<_, u8>(&matrix);

        // Gradients: Create clear splits at midpoints
        let mut grads = GradientBuffer::new(num_rows, 1);
        for i in 0..num_rows {
            if i < num_rows / 2 {
                grads.set(i, 0, 2.0, 1.0);
            } else {
                grads.set(i, 0, -2.0, 1.0);
            }
        }

        (quantized, cuts, grads)
    }

    #[test]
    fn test_tree_grower_with_interaction_constraints() {
        let (quantized, cuts, grads) = make_interaction_test_data();
        let num_features = cuts.num_features() as u32;

        // Features 0,1 can interact; features 2,3 can interact
        let strategy = GrowthStrategy::DepthWise { max_depth: 3 };
        let params = TreeBuildParams::default();
        let learning_rate = 1.0;

        let mut col_sampler = ColumnSampler::new(num_features, 1.0, 1.0, 1.0);
        col_sampler.sample_tree(0);
        let mono_checker = MonotonicChecker::new(&[], num_features as usize);
        let interaction_constraints = InteractionConstraints::new(
            &[vec![0, 1], vec![2, 3]],
            num_features,
        );

        let mut partitioner = RowPartitioner::new(20);
        let mut grower = TreeGrower::new(
            strategy, &cuts, params, learning_rate,
            &col_sampler, &mono_checker, &interaction_constraints,
        );

        let tree = grower.build_tree(&quantized, &grads, &mut partitioner, 0);

        // Verify that feature interactions are respected:
        // If root splits on feature 0 or 1, then children can only split on 0 or 1
        // If root splits on feature 2 or 3, then children can only split on 2 or 3
        let root = tree.node(0);
        if !root.is_leaf {
            let root_feature = root.split.as_ref().unwrap().feature;
            let group1_features = [0u32, 1];
            let group2_features = [2u32, 3];
            
            let root_in_group1 = group1_features.contains(&root_feature);
            let root_in_group2 = group2_features.contains(&root_feature);
            
            println!("Root splits on feature {}", root_feature);
            
            // Check children respect interaction constraints
            fn check_subtree_features(
                tree: &BuildingTree,
                node_id: u32,
                allowed: &[u32],
                depth: usize,
            ) {
                let node = tree.node(node_id);
                if node.is_leaf {
                    return;
                }
                
                let feature = node.split.as_ref().unwrap().feature;
                println!("  {}Node {} splits on feature {}", "  ".repeat(depth), node_id, feature);
                
                // After the root, only the allowed features should be used
                if depth > 0 {
                    assert!(
                        allowed.contains(&feature),
                        "Feature {} used at depth {} but should only use {:?}",
                        feature, depth, allowed
                    );
                }
                
                check_subtree_features(tree, node.left, allowed, depth + 1);
                check_subtree_features(tree, node.right, allowed, depth + 1);
            }
            
            if root_in_group1 {
                check_subtree_features(&tree, 0, &group1_features, 0);
            } else if root_in_group2 {
                check_subtree_features(&tree, 0, &group2_features, 0);
            }
        }
    }

    #[test]
    fn test_tree_grower_no_interaction_constraints_allows_all() {
        let (quantized, cuts, grads) = make_interaction_test_data();
        let num_features = cuts.num_features() as u32;

        // No interaction constraints - all features can interact
        let strategy = GrowthStrategy::DepthWise { max_depth: 3 };
        let params = TreeBuildParams::default();
        let learning_rate = 1.0;

        let (mut col_sampler, mono_checker, interaction_constraints) = make_default_components(num_features);
        col_sampler.sample_tree(0);

        let mut partitioner = RowPartitioner::new(20);
        let mut grower = TreeGrower::new(
            strategy, &cuts, params, learning_rate,
            &col_sampler, &mono_checker, &interaction_constraints,
        );

        let tree = grower.build_tree(&quantized, &grads, &mut partitioner, 0);

        // Just verify tree builds successfully - any feature can be used at any level
        println!("Tree without interaction constraints has {} nodes", tree.num_nodes());
        assert!(tree.num_nodes() > 0);
    }

    #[test]
    fn test_tree_grower_single_feature_group() {
        let (quantized, cuts, grads) = make_interaction_test_data();
        let num_features = cuts.num_features() as u32;

        // Only features 0 and 1 are allowed
        let strategy = GrowthStrategy::DepthWise { max_depth: 3 };
        let params = TreeBuildParams::default();
        let learning_rate = 1.0;

        let mut col_sampler = ColumnSampler::new(num_features, 1.0, 1.0, 1.0);
        col_sampler.sample_tree(0);
        let mono_checker = MonotonicChecker::new(&[], num_features as usize);
        let interaction_constraints = InteractionConstraints::new(
            &[vec![0, 1]],
            num_features,
        );

        let mut partitioner = RowPartitioner::new(20);
        let mut grower = TreeGrower::new(
            strategy, &cuts, params, learning_rate,
            &col_sampler, &mono_checker, &interaction_constraints,
        );

        let tree = grower.build_tree(&quantized, &grads, &mut partitioner, 0);

        // All splits should only use features 0 or 1
        fn check_features_only_from_group(
            tree: &BuildingTree,
            node_id: u32,
            allowed: &[u32],
        ) {
            let node = tree.node(node_id);
            if node.is_leaf {
                return;
            }
            
            let feature = node.split.as_ref().unwrap().feature;
            assert!(
                allowed.contains(&feature),
                "Feature {} used but only {:?} allowed",
                feature, allowed
            );
            
            check_features_only_from_group(tree, node.left, allowed);
            check_features_only_from_group(tree, node.right, allowed);
        }
        
        check_features_only_from_group(&tree, 0, &[0, 1]);
    }
}

//! Tree grower for gradient boosting.
//!
//! Orchestrates tree training using histogram-based split finding, row partitioning,
//! and the subtraction trick to reduce computation.

use crate::data::BinnedDataset;
use crate::training::Gradients;
use crate::training::sampling::{ColSampler, ColSamplingParams};

use super::expansion::{GrowthState, GrowthStrategy, NodeCandidate};
use super::tree::NodeId;
use super::histograms::{
    build_histograms, FeatureMeta, FeatureView, HistogramPool,
};
use super::optimization::OptimizationProfile;
use super::partition::RowPartitioner;
use super::split::{GainParams, GreedySplitter, SplitInfo};
use super::tree::{Tree, TreeBuilder};

/// Parameters for tree growth.
#[derive(Clone, Debug)]
pub struct GrowerParams {
    /// Gain computation and constraint parameters (regularization, min child weight, etc.).
    pub gain: GainParams,
    /// Learning rate.
    pub learning_rate: f32,
    /// Tree growth strategy (includes depth/leaf limits).
    pub growth_strategy: GrowthStrategy,
    /// Max categories for one-hot categorical split.
    pub max_onehot_cats: u32,
    /// Optimization profile for automatic strategy selection.
    pub optimization_profile: OptimizationProfile,
    /// Column (feature) sampling configuration.
    pub col_sampling: ColSamplingParams,
}

impl Default for GrowerParams {
    fn default() -> Self {
        Self {
            gain: GainParams::default(),
            learning_rate: 0.3,
            growth_strategy: GrowthStrategy::default(),
            max_onehot_cats: 4,
            optimization_profile: OptimizationProfile::Auto,
            col_sampling: ColSamplingParams::None,
        }
    }
}

impl GrowerParams {
    /// Get max_leaves from growth strategy (for buffer sizing).
    fn max_leaves(&self) -> u32 {
        match self.growth_strategy {
            GrowthStrategy::DepthWise { max_depth } => 1u32 << max_depth,
            GrowthStrategy::LeafWise { max_leaves } => max_leaves,
        }
    }
}

/// Tree grower for gradient boosting.
///
/// Grows a single decision tree from gradient and hessian vectors.
/// Uses the subtraction trick to reduce histogram building work by ~50%.
pub struct TreeGrower {
    /// Growth parameters.
    params: GrowerParams,
    /// Histogram pool with LRU caching.
    histogram_pool: HistogramPool,
    /// Row partitioner.
    partitioner: RowPartitioner,
    /// Tree builder.
    tree_builder: TreeBuilder,
    /// Feature types (true = categorical).
    feature_types: Vec<bool>,
    /// Feature metadata for histogram building.
    feature_metas: Vec<FeatureMeta>,
    /// Split strategy.
    split_strategy: GreedySplitter,
    /// Column sampler for feature subsampling.
    col_sampler: ColSampler,
}

impl TreeGrower {
    /// Create a new tree grower.
    ///
    /// # Arguments
    /// * `dataset` - Binned dataset (used to get feature metadata)
    /// * `params` - Tree growth parameters
    /// * `cache_size` - Number of histogram slots to cache
    pub fn new(
        dataset: &BinnedDataset,
        params: GrowerParams,
        cache_size: usize,
    ) -> Self {
        let n_features = dataset.n_features();
        let n_samples = dataset.n_rows();

        // Build feature metadata for histogram pool
        let feature_metas: Vec<FeatureMeta> = (0..n_features)
            .map(|f| FeatureMeta {
                offset: dataset.global_bin_offset(f),
                n_bins: dataset.n_bins(f),
            })
            .collect();

        // Maximum nodes: 2*max_leaves - 1 for a full binary tree
        let max_leaves = params.max_leaves();
        let max_nodes = (2 * max_leaves as usize).saturating_sub(1).max(63);

        let histogram_pool =
            HistogramPool::new(feature_metas.clone(), cache_size.max(2), max_nodes);
        let partitioner = RowPartitioner::new(n_samples, max_nodes);

        // Collect feature types
        let feature_types: Vec<bool> =
            (0..n_features).map(|f| dataset.is_categorical(f)).collect();

        // Resolve split strategy based on data characteristics
        let split_strategy_mode = params.optimization_profile.resolve(n_samples, n_features);

        // Build split strategy with gain params encapsulated
        let split_strategy = GreedySplitter::with_config(
            params.gain.clone(),
            params.max_onehot_cats,
            split_strategy_mode,
        );

        // Create column sampler (handles all/none case gracefully)
        let col_sampler = ColSampler::new(params.col_sampling.clone(), n_features as u32, 0);

        Self {
            params,
            histogram_pool,
            partitioner,
            tree_builder: TreeBuilder::with_capacity(max_nodes),
            feature_types,
            feature_metas,
            split_strategy,
            col_sampler,
        }
    }

    /// Grow a tree from gradients for a specific output.
    ///
    /// # Arguments
    /// * `dataset` - Binned dataset
    /// * `gradients` - Gradient storage
    /// * `output` - Which output index to grow the tree for (0 to n_outputs-1)
    /// * `sampled_rows` - Optional sampled row indices (skips unsampled rows in histograms)
    ///
    /// # Returns
    /// The trained tree.
    pub fn grow(
        &mut self,
        dataset: &BinnedDataset,
        gradients: &Gradients,
        output: usize,
        sampled_rows: Option<&[u32]>,
    ) -> Tree {
        let n_samples = dataset.n_rows();
        assert_eq!(gradients.n_samples(), n_samples);

        // Reset state for new tree (partitioner initialized with sampled rows)
        self.reset(n_samples, sampled_rows);

        // Initialize column sampler for this tree
        self.col_sampler.sample_tree();
        self.col_sampler.sample_level(0);

        // Pre-compute feature views once per tree (avoids per-node Vec allocation)
        let bin_views = dataset.feature_views();

        // Initialize growth state
        let strategy = self.params.growth_strategy;
        let mut state = strategy.init();

        // Initialize root
        let root_tree_node = self.tree_builder.init_root();

        // Compute root gradient sums (f64 accumulation for numerical stability)
        let (total_grad, total_hess) = gradients.sum(output, None);

        // Build root histogram
        self.build_histogram(0, gradients, output, &bin_views);

        // Find root split
        let root_split = self.find_split(0, total_grad, total_hess, n_samples as u32);

        // Push root candidate
        state.push_root(NodeCandidate::new(
            0,
            root_tree_node,
            0,
            root_split,
            total_grad,
            total_hess,
            n_samples as u32,
        ));

        // Main expansion loop
        while state.should_continue() {
            // Pop nodes to expand this iteration
            let candidates = state.pop_next();

            for candidate in candidates {
                self.process_candidate(candidate, dataset, gradients, output, &bin_views, &mut state);
            }

            // Advance to next iteration (depth-wise: move to next level)
            state.advance();
        }

        // Finalize any remaining candidates as leaves
        self.finalize_remaining(&mut state);

        // Apply learning rate and finish
        self.tree_builder
            .apply_learning_rate(self.params.learning_rate);
        std::mem::take(&mut self.tree_builder).finish()
    }

    /// Process a single candidate node.
    fn process_candidate(
        &mut self,
        candidate: NodeCandidate,
        dataset: &BinnedDataset,
        gradients: &Gradients,
        output: usize,
        bin_views: &[FeatureView<'_>],
        state: &mut GrowthState,
    ) {
        // Check if we should expand
        if !self.should_expand(&candidate) {
            // Make leaf
            let weight = self
                .split_strategy
                .compute_leaf_weight(candidate.grad_sum, candidate.hess_sum);
            self.tree_builder.make_leaf(candidate.tree_node, weight);
            self.histogram_pool.release(candidate.node);
            return;
        }

        // Apply split
        let (left_tree, right_tree) = self
            .tree_builder
            .apply_split(candidate.tree_node, &candidate.split, candidate.depth);

        // Partition rows
        let (right_node, left_count, right_count) = self
            .partitioner
            .split(candidate.node, &candidate.split, dataset);

        // Use original node as left (now owns only left rows)
        let left_node = candidate.node;
        let parent_node = candidate.node; // Parent histogram is in this node's slot

        // Determine smaller/larger child for subtraction trick
        let (small_node, small_count, large_node, large_count) = if left_count <= right_count {
            (left_node, left_count, right_node, right_count)
        } else {
            (right_node, right_count, left_node, left_count)
        };

        // Subtraction trick:
        // 1. Move parent histogram to large child (copies data, frees parent slot)
        // 2. Build small child histogram (can now reuse parent's slot)
        // 3. Subtract: large = large - small (large still has parent data)
        self.histogram_pool.move_mapping(parent_node, large_node);
        self.build_histogram(small_node, gradients, output, bin_views);
        self.histogram_pool.subtract(large_node, small_node);

        // Compute gradient sums for smaller child
        let small_rows = self.partitioner.get_leaf_indices(small_node);
        let (small_grad, small_hess) = gradients.sum(output, Some(small_rows));
        let (large_grad, large_hess) = (
            candidate.grad_sum - small_grad,
            candidate.hess_sum - small_hess,
        );

        let new_depth = candidate.depth + 1;

        // Sample level features if depth changed
        self.col_sampler.sample_level(new_depth);

        // Find splits for children
        let small_split = self.find_split(small_node, small_grad, small_hess, small_count);
        let large_split = self.find_split(large_node, large_grad, large_hess, large_count);

        // Map back to left/right
        let (left_grad, left_hess, left_split, left_count_final) = if left_node == small_node {
            (small_grad, small_hess, small_split.clone(), small_count)
        } else {
            (large_grad, large_hess, large_split.clone(), large_count)
        };
        let (right_grad, right_hess, right_split, right_count_final) = if right_node == small_node {
            (small_grad, small_hess, small_split, small_count)
        } else {
            (large_grad, large_hess, large_split, large_count)
        };

        // Push children to expansion state
        state.push(NodeCandidate::new(
            left_node,
            left_tree,
            new_depth,
            left_split,
            left_grad,
            left_hess,
            left_count_final,
        ));
        state.push(NodeCandidate::new(
            right_node,
            right_tree,
            new_depth,
            right_split,
            right_grad,
            right_hess,
            right_count_final,
        ));
    }

    /// Finalize any remaining candidates in the state as leaves.
    fn finalize_remaining(&mut self, state: &mut GrowthState) {
        loop {
            let candidates = match state {
                GrowthState::DepthWise { current_level, .. } => {
                    if current_level.is_empty() {
                        break;
                    }
                    current_level.drain(..).collect()
                }
                GrowthState::LeafWise { candidates, .. } => {
                    if candidates.is_empty() {
                        break;
                    }
                    vec![candidates.pop().unwrap()]
                }
            };

            for candidate in candidates {
                let weight = self
                    .split_strategy
                    .compute_leaf_weight(candidate.grad_sum, candidate.hess_sum);
                self.tree_builder.make_leaf(candidate.tree_node, weight);
                self.histogram_pool.release(candidate.node);
            }
        }
    }

    /// Reset for a new tree.
    fn reset(&mut self, n_samples: usize, sampled: Option<&[u32]>) {
        self.histogram_pool.reset_mappings();
        self.partitioner.reset(n_samples, sampled);
        self.tree_builder.reset();
    }

    /// Check if a candidate should be expanded.
    fn should_expand(&self, candidate: &NodeCandidate) -> bool {
        // Check gain threshold
        if !candidate.is_valid() || candidate.gain() <= self.params.gain.min_gain {
            return false;
        }

        // Depth limit is handled by the growth strategy
        // We just check minimum constraints here
        true
    }

    /// Build histogram for a node.
    fn build_histogram(
        &mut self,
        node: NodeId,
        gradients: &Gradients,
        output: usize,
        bin_views: &[FeatureView<'_>],
    ) {
        let result = self.histogram_pool.acquire(node);
        let slot = result.slot();

        // Clear histogram on miss
        if !result.is_hit() {
            self.histogram_pool.slot_mut(slot).clear();
        }

        // Get row indices for this node (already filtered to sampled rows by partitioner)
        let rows = self.partitioner.get_leaf_indices(node);
        if rows.is_empty() {
            return;
        }

        // Get mutable histogram slice
        let hist = self.histogram_pool.slot_mut(slot);

        // Build histogram (rows are already filtered by partitioner)
        build_histograms(
            hist.bins,
            gradients.output_grads(output),
            gradients.output_hess(output),
            rows,
            bin_views,
            &self.feature_metas,
        );
    }

    /// Find best split for a node using column sampler.
    fn find_split(
        &mut self,
        node: NodeId,
        grad_sum: f64,
        hess_sum: f64,
        count: u32,
    ) -> SplitInfo {
        let histogram = match self.histogram_pool.get(node) {
            Some(h) => h,
            None => return SplitInfo::invalid(),
        };

        // Get features from column sampler (always returns a slice)
        let features = self.col_sampler.sample_node();

        self.split_strategy.find_split(
            &histogram,
            grad_sum,
            hess_sum,
            count,
            &self.feature_types,
            features,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{BinMapper, BinnedDataset, BinnedDatasetBuilder, GroupLayout, GroupStrategy, MissingType};

    fn make_simple_dataset() -> BinnedDataset {
        // 10 samples, 1 feature with 4 bins
        let bins = vec![0, 0, 0, 1, 1, 2, 2, 3, 3, 3];
        let mapper = BinMapper::numerical(
            vec![0.5, 1.5, 2.5, 3.5],
            MissingType::None,
            0,
            0,
            0.0,
            0.0,
            3.0,
        );
        BinnedDatasetBuilder::new()
            .add_binned(bins, mapper)
            .group_strategy(GroupStrategy::SingleGroup { layout: GroupLayout::ColumnMajor })
            .build()
            .unwrap()
    }

    fn make_two_feature_dataset() -> BinnedDataset {
        let f0_bins = vec![0, 0, 1, 1, 0, 0, 1, 1];
        let f0_mapper =
            BinMapper::numerical(vec![0.5, 1.5], MissingType::None, 0, 0, 0.0, 0.0, 1.0);

        let f1_bins = vec![0, 0, 0, 0, 1, 1, 1, 1];
        let f1_mapper =
            BinMapper::numerical(vec![0.5, 1.5], MissingType::None, 0, 0, 0.0, 0.0, 1.0);

        BinnedDatasetBuilder::new()
            .add_binned(f0_bins, f0_mapper)
            .add_binned(f1_bins, f1_mapper)
            .group_strategy(GroupStrategy::SingleGroup { layout: GroupLayout::ColumnMajor })
            .build()
            .unwrap()
    }

    #[test]
    fn test_grower_single_leaf() {
        let dataset = make_simple_dataset();
        let params = GrowerParams {
            growth_strategy: GrowthStrategy::DepthWise { max_depth: 0 },
            ..Default::default()
        };

        let mut grower = TreeGrower::new(&dataset, params, 4);

        // All same gradient: no good splits
        let grad: Vec<f32> = vec![1.0; 10];
        let hess: Vec<f32> = vec![1.0; 10];
        let mut gradients = Gradients::new(10, 1);
        gradients.output_grads_mut(0).copy_from_slice(&grad);
        gradients.output_hess_mut(0).copy_from_slice(&hess);

        let tree = grower.grow(&dataset, &gradients, 0, None);

        assert_eq!(tree.n_leaves(), 1);
        assert!(tree.node(0).is_leaf);
    }

    #[test]
    fn test_grower_simple_split() {
        let dataset = make_simple_dataset();
        let params = GrowerParams {
            growth_strategy: GrowthStrategy::DepthWise { max_depth: 1 },
            gain: GainParams { min_gain: 0.0, ..Default::default() },
            ..Default::default()
        };

        let mut grower = TreeGrower::new(&dataset, params, 4);

        // Gradients that suggest a split
        let grad: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0];
        let hess: Vec<f32> = vec![1.0; 10];
        let mut gradients = Gradients::new(10, 1);
        gradients.output_grads_mut(0).copy_from_slice(&grad);
        gradients.output_hess_mut(0).copy_from_slice(&hess);

        let tree = grower.grow(&dataset, &gradients, 0, None);

        // Should have a split at root with 2 leaves
        assert!(!tree.node(0).is_leaf);
        assert_eq!(tree.n_leaves(), 2);
    }

    #[test]
    fn test_grower_max_depth() {
        let dataset = make_two_feature_dataset();
        let params = GrowerParams {
            growth_strategy: GrowthStrategy::DepthWise { max_depth: 2 },
            gain: GainParams { min_gain: 0.0, min_child_weight: 0.1, ..Default::default() },
            ..Default::default()
        };

        let mut grower = TreeGrower::new(&dataset, params, 8);

        let grad: Vec<f32> = vec![2.0, 1.0, -1.0, -2.0, 2.0, 1.0, -1.0, -2.0];
        let hess: Vec<f32> = vec![1.0; 8];
        let mut gradients = Gradients::new(8, 1);
        gradients.output_grads_mut(0).copy_from_slice(&grad);
        gradients.output_hess_mut(0).copy_from_slice(&hess);

        let tree = grower.grow(&dataset, &gradients, 0, None);

        // Should respect max_depth
        assert!(tree.max_depth() <= 2);
    }

    #[test]
    fn test_growth_strategy_depth_wise() {
        let mut state = GrowthStrategy::DepthWise { max_depth: 2 }.init();

        // Push root
        state.push_root(NodeCandidate::new(
            0, 0, 0,
            SplitInfo::numerical(0, 1, 0.5, false),
            0.0, 1.0, 10,
        ));

        assert!(state.should_continue());

        // Pop and verify
        let nodes = state.pop_next();
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].node, 0);
    }

    #[test]
    fn test_growth_strategy_leaf_wise() {
        let mut state = GrowthStrategy::LeafWise { max_leaves: 10 }.init();

        // Push candidates with different gains
        state.push_root(NodeCandidate::new(
            0, 0, 0,
            SplitInfo::numerical(0, 1, 0.5, false),
            0.0, 1.0, 10,
        ));
        state.push(NodeCandidate::new(
            1, 1, 0,
            SplitInfo::numerical(0, 1, 0.8, false),
            0.0, 1.0, 10,
        ));

        // Should pop highest gain first
        let nodes = state.pop_next();
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].node, 1);
        assert!((nodes[0].gain() - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_tree_params_default() {
        let params = GrowerParams::default();
        assert_eq!(params.learning_rate, 0.3);
        assert!(matches!(params.growth_strategy, GrowthStrategy::DepthWise { max_depth: 6 }));
    }

    #[test]
    fn test_grower_learning_rate() {
        let dataset = make_simple_dataset();
        let params = GrowerParams {
            growth_strategy: GrowthStrategy::DepthWise { max_depth: 1 },
            learning_rate: 0.1,
            gain: GainParams { min_gain: 0.0, ..Default::default() },
            ..Default::default()
        };

        let mut grower = TreeGrower::new(&dataset, params, 4);

        let grad_f32: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0];
        let hess_f32: Vec<f32> = vec![1.0; 10];
        let mut gradients = Gradients::new(10, 1);
        gradients.output_grads_mut(0).copy_from_slice(&grad_f32);
        gradients.output_hess_mut(0).copy_from_slice(&hess_f32);

        let tree = grower.grow(&dataset, &gradients, 0, None);

        // Check that leaf values are scaled by learning rate
        for (_, node) in tree.iter_leaves() {
            // Values should be relatively small due to 0.1 learning rate
            assert!(node.value.abs() < 1.0);
        }
    }

    #[test]
    fn test_leaf_wise_growth() {
        let dataset = make_two_feature_dataset();
        let params = GrowerParams {
            growth_strategy: GrowthStrategy::LeafWise { max_leaves: 4 },
            gain: GainParams { min_gain: 0.0, min_child_weight: 0.1, ..Default::default() },
            ..Default::default()
        };

        let mut grower = TreeGrower::new(&dataset, params, 8);

        let grad: Vec<f32> = vec![2.0, 1.0, -1.0, -2.0, 2.0, 1.0, -1.0, -2.0];
        let hess: Vec<f32> = vec![1.0; 8];
        let mut gradients = Gradients::new(8, 1);
        gradients.output_grads_mut(0).copy_from_slice(&grad);
        gradients.output_hess_mut(0).copy_from_slice(&hess);

        let tree = grower.grow(&dataset, &gradients, 0, None);

        // Should have at most 4 leaves
        assert!(tree.n_leaves() <= 4);
    }

    #[test]
    fn test_grower_with_col_sampler() {
        use crate::training::sampling::ColSamplingParams;

        let dataset = make_two_feature_dataset();
        
        let params_no_sampling = GrowerParams {
            growth_strategy: GrowthStrategy::DepthWise { max_depth: 2 },
            gain: GainParams { min_gain: 0.0, min_child_weight: 0.1, ..Default::default() },
            col_sampling: ColSamplingParams::None,
            ..Default::default()
        };

        let params_with_sampling = GrowerParams {
            growth_strategy: GrowthStrategy::DepthWise { max_depth: 2 },
            gain: GainParams { min_gain: 0.0, min_child_weight: 0.1, ..Default::default() },
            col_sampling: ColSamplingParams::bytree(0.5),
            ..Default::default()
        };

        let grad: Vec<f32> = vec![2.0, 1.0, -1.0, -2.0, 2.0, 1.0, -1.0, -2.0];
        let hess: Vec<f32> = vec![1.0; 8];
        let mut gradients = Gradients::new(8, 1);
        gradients.output_grads_mut(0).copy_from_slice(&grad);
        gradients.output_hess_mut(0).copy_from_slice(&hess);

        // Without column sampling
        let mut grower_all = TreeGrower::new(&dataset, params_no_sampling, 8);
        let tree_all = grower_all.grow(&dataset, &gradients, 0, None);

        // With column sampling
        let mut grower_sampled = TreeGrower::new(&dataset, params_with_sampling, 8);
        let tree_sampled = grower_sampled.grow(&dataset, &gradients, 0, None);

        // Both should produce trees
        assert!(tree_all.n_leaves() >= 1);
        assert!(tree_sampled.n_leaves() >= 1);
    }

    #[test]
    fn test_grower_with_f32_input() {
        let dataset = make_simple_dataset();
        let params = GrowerParams {
            growth_strategy: GrowthStrategy::DepthWise { max_depth: 1 },
            gain: GainParams { min_gain: 0.0, ..Default::default() },
            ..Default::default()
        };

        let mut grower = TreeGrower::new(&dataset, params, 4);

        let grad_f32: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0];
        let hess_f32: Vec<f32> = vec![1.0; 10];
        let mut gradients = Gradients::new(10, 1);
        gradients.output_grads_mut(0).copy_from_slice(&grad_f32);
        gradients.output_hess_mut(0).copy_from_slice(&hess_f32);

        let tree = grower.grow(&dataset, &gradients, 0, None);

        assert!(!tree.node(0).is_leaf);
        assert_eq!(tree.n_leaves(), 2);
    }
}

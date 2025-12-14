//! Tree grower for gradient boosting.
//!
//! Orchestrates tree training using histogram-based split finding, row partitioning,
//! and the subtraction trick to reduce computation.

use crate::data::BinnedDataset;
use crate::repr::gbdt::{categories_to_bitset, MutableTree, ScalarLeaf, Tree};
use crate::training::Gradients;
use crate::training::sampling::{ColSampler, ColSamplingParams};

use super::expansion::{GrowthState, GrowthStrategy, NodeCandidate};
use super::histograms::{build_histograms_ordered, FeatureMeta, FeatureView, HistogramPool};
use super::optimization::OptimizationProfile;
use super::partition::RowPartitioner;
use super::split::{GainParams, GreedySplitter, SplitInfo, SplitType};

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
/// Uses ordered gradients for cache-efficient histogram building.
pub struct TreeGrower {
    /// Growth parameters.
    params: GrowerParams,
    /// Histogram pool with LRU caching.
    histogram_pool: HistogramPool,
    /// Row partitioner.
    partitioner: RowPartitioner,
    /// Tree builder (builds inference-ready Tree directly).
    tree_builder: MutableTree<ScalarLeaf>,
    /// Feature types (true = categorical).
    feature_types: Vec<bool>,
    /// Feature metadata for histogram building.
    feature_metas: Vec<FeatureMeta>,
    /// Split strategy.
    split_strategy: GreedySplitter,
    /// Column sampler for feature subsampling.
    col_sampler: ColSampler,
    /// Per-node leaf values (scaled by learning rate) for the last grown tree.
    /// Indexed by training node id used by the partitioner/histogram pool.
    last_leaf_values: Vec<f32>,
    /// Buffer for ordered (pre-gathered) gradients.
    /// Reused across histogram builds to avoid allocation.
    ordered_grad: Vec<f32>,
    /// Buffer for ordered (pre-gathered) hessians.
    ordered_hess: Vec<f32>,
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
            tree_builder: MutableTree::with_capacity(max_nodes),
            feature_types,
            feature_metas,
            split_strategy,
            col_sampler,
            last_leaf_values: Vec::new(),
            ordered_grad: Vec::with_capacity(n_samples),
            ordered_hess: Vec::with_capacity(n_samples),
        }
    }

    /// Grow a tree and (optionally) update predictions using training-time leaf assignments.
    ///
    /// When `sampled_rows` is `None`, the partitioner contains **all** rows, so we can update
    /// predictions by iterating leaf ranges instead of traversing the tree per row.
    pub fn grow_and_update_predictions(
        &mut self,
        dataset: &BinnedDataset,
        gradients: &Gradients,
        output: usize,
        sampled_rows: Option<&[u32]>,
        predictions: &mut [f32],
    ) -> Tree<ScalarLeaf> {
        debug_assert_eq!(predictions.len(), dataset.n_rows());

        let tree = self.grow(dataset, gradients, output, sampled_rows);

        // Fast path only valid when the partitioner includes all rows.
        // After tree growth, partitioner leaves contain the final leaf assignment.
        if sampled_rows.is_none() {
            self.update_predictions_from_last_tree(predictions);
        }

        tree
    }

    /// Update predictions using the partitioner's leaf assignments and cached leaf values.
    ///
    /// After tree growth, the partitioner's leaves correspond to the final tree leaves.
    /// Instead of traversing the tree for each row, we iterate over each leaf's row range
    /// and apply the cached leaf value.
    ///
    /// This is O(n_rows) with sequential writes to predictions, which is much faster than
    /// O(n_rows × tree_depth) random tree traversals.
    fn update_predictions_from_last_tree(&self, predictions: &mut [f32]) {
        for (leaf_node, leaf_value) in self.last_leaf_values.iter().enumerate() {
            if leaf_value.is_nan() {
                continue;
            }
            let (begin, end) = self.partitioner.leaf_range(leaf_node as u32);
            let indices = self.partitioner.indices();
            for &row_idx in &indices[begin..end] {
                predictions[row_idx as usize] += *leaf_value;
            }
        }
    }

    /// Record a leaf value for fast prediction updates.
    fn record_leaf_value(&mut self, node: usize, weight: f32) {
        // Ensure we have room for this node
        if node >= self.last_leaf_values.len() {
            self.last_leaf_values.resize(node + 1, f32::NAN);
        }
        // Apply learning rate since tree.predict returns the post-learning-rate values
        self.last_leaf_values[node] = weight * self.params.learning_rate;
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
    /// The trained tree, ready for inference.
    pub fn grow(
        &mut self,
        dataset: &BinnedDataset,
        gradients: &Gradients,
        output: usize,
        sampled_rows: Option<&[u32]>,
    ) -> Tree<ScalarLeaf> {
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
        std::mem::take(&mut self.tree_builder).freeze()
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
            self.tree_builder
                .make_leaf(candidate.tree_node, ScalarLeaf(weight));
            self.record_leaf_value(candidate.node as usize, weight);
            self.histogram_pool.release(candidate.node);
            return;
        }

        // Apply split (translating bin thresholds to float values)
        let (left_tree, right_tree) =
            Self::apply_split_to_builder(&mut self.tree_builder, candidate.tree_node, &candidate.split, dataset);

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

        // Histogram building for children.
        //
        // Fast path: subtraction trick.
        // This requires the parent's histogram to still be cached, so we pin the parent
        // slot to prevent eviction during the subtract path.
        //
        // If the parent histogram is missing (should be rare), fall back to building
        // both child histograms from scratch.
        if self.histogram_pool.get(parent_node).is_some() {
            // Subtraction trick:
            // 1. Move parent histogram to large child (copies data, frees parent slot)
            // 2. Build small child histogram (can now reuse parent's slot)
            // 3. Subtract: large = large - small (large still has parent data)
            self.histogram_pool.pin(parent_node);
            self.histogram_pool.move_mapping(parent_node, large_node);
            self.build_histogram(small_node, gradients, output, bin_views);
            self.histogram_pool.subtract(large_node, small_node);
            self.histogram_pool.unpin(large_node);
        } else {
            // Fallback: build both histograms directly.
            self.build_histogram(left_node, gradients, output, bin_views);
            self.build_histogram(right_node, gradients, output, bin_views);
        }

        // Compute gradient sums for smaller child from histogram (O(n_bins) instead of O(n_rows))
        let (small_grad, small_hess) = self.histogram_pool
            .get(small_node)
            .expect("small_node histogram should exist after build")
            .sum_gradients();
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
                self.tree_builder
                    .make_leaf(candidate.tree_node, ScalarLeaf(weight));
                self.record_leaf_value(candidate.node as usize, weight);
                self.histogram_pool.release(candidate.node);
            }
        }
    }

    /// Reset for a new tree.
    fn reset(&mut self, n_samples: usize, sampled: Option<&[u32]>) {
        self.histogram_pool.reset_mappings();
        self.partitioner.reset(n_samples, sampled);
        self.tree_builder.reset();
        // Clear leaf values from previous tree
        for v in &mut self.last_leaf_values {
            *v = f32::NAN;
        }
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

    /// Helper to compute next_up for f32 (mirrors f32::next_up without MSRV dependency).
    #[inline]
    fn next_up_f32(x: f32) -> f32 {
        if x.is_nan() || x == f32::INFINITY {
            return x;
        }
        if x == 0.0 {
            return f32::from_bits(1);
        }
        let bits = x.to_bits();
        if x.is_sign_positive() {
            f32::from_bits(bits + 1)
        } else {
            f32::from_bits(bits - 1)
        }
    }

    /// Apply a split to the tree builder, translating bin thresholds to float values.
    ///
    /// For numerical splits:
    /// - Training uses `bin <= threshold` (go left)
    /// - Inference uses `value < threshold` (go left)
    /// - We use `next_up(bin_upper_bound)` to make the semantics match
    ///
    /// For categorical splits:
    /// - Training: categories in `left_cats` go LEFT
    /// - Inference: categories in bitset go RIGHT
    /// - We swap children and invert default_left to preserve semantics
    ///
    /// Important: categorical split categories are treated as **category indices** (0..K-1),
    /// i.e. the same domain as categorical bin indices in `BinnedDataset`.
    fn apply_split_to_builder(
        builder: &mut MutableTree<ScalarLeaf>,
        node: u32,
        split: &SplitInfo,
        dataset: &BinnedDataset,
    ) -> (u32, u32) {
        let feature = split.feature;
        let mapper = dataset.bin_mapper(feature as usize);

        match &split.split_type {
            SplitType::Numerical { bin } => {
                // Convert bin threshold to float value
                let bin_upper = mapper.bin_to_value(*bin as u32) as f32;
                let threshold = Self::next_up_f32(bin_upper);
                builder.apply_numeric_split(node, feature, threshold, split.default_left)
            }
            SplitType::Categorical { left_cats } => {
                // Convert training categorical bitset to inference format.
                // Training: categories in left_cats go LEFT
                // Inference: categories in bitset go RIGHT
                // We swap children at the inference level by storing left_cats in the bitset
                // and swapping left/right children when applying the split.
                //
                // Note: `left_cats` already contains categorical **bin indices**, which we
                // treat as canonical category indices (0..K-1). Do NOT convert via
                // `bin_to_value()` here: categorical bin mappers store original category
                // values, which may be large and would explode bitset size.
                let mut categories: Vec<u32> = left_cats.iter().map(|c| c as u32).collect();
                categories.sort_unstable();
                categories.dedup();
                let bitset = categories_to_bitset(&categories);

                // Apply split with swapped children and inverted default.
                // The inference builder's apply_categorical_split returns (left, right),
                // but since we're inverting the semantics, we swap them here.
                let (inf_left, inf_right) =
                    builder.apply_categorical_split(node, feature, bitset, !split.default_left);
                // Swap children to match training semantics
                (inf_right, inf_left)
            }
        }
    }

    /// Check if indices represent a contiguous range [first, first + n).
    /// Only returns true if indices are strictly sequential: [k, k+1, k+2, ...].
    #[inline]
    fn is_contiguous_range(indices: &[u32]) -> bool {
        if indices.is_empty() {
            return true;
        }
        let first = indices[0];
        let last = indices[indices.len() - 1];
        // Check: last >= first (sorted) and range equals length
        last >= first && (last - first) as usize == indices.len() - 1
    }

    /// Build histogram for a node.
    fn build_histogram(
        &mut self,
        node: u32,
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

        // Get gradient slices for this output
        let grad_slice = gradients.output_grads(output);
        let hess_slice = gradients.output_hess(output);

        // Fast path: if indices are contiguous (common for root and early nodes),
        // we can use the original gradient slices directly without gathering.
        // The ordered histogram function will iterate sequentially over both.
        if Self::is_contiguous_range(rows) {
            let start = rows[0] as usize;
            let end = start + rows.len();
            // Use the contiguous slice directly - ordered grad access matches row access
            build_histograms_ordered(
                hist.bins,
                &grad_slice[start..end],
                &hess_slice[start..end],
                rows,
                bin_views,
                &self.feature_metas,
            );
            return;
        }

        // Non-contiguous indices: gather gradients into ordered buffers
        let n_rows = rows.len();

        // Ensure buffers have capacity (reuses allocation across builds)
        if self.ordered_grad.capacity() < n_rows {
            self.ordered_grad.reserve(n_rows - self.ordered_grad.capacity());
        }
        if self.ordered_hess.capacity() < n_rows {
            self.ordered_hess.reserve(n_rows - self.ordered_hess.capacity());
        }

        // Gather gradients in partition order with direct writes (no capacity checks)
        // SAFETY: We just ensured capacity >= n_rows, and row indices are valid
        unsafe {
            self.ordered_grad.set_len(n_rows);
            self.ordered_hess.set_len(n_rows);
            let grad_ptr = self.ordered_grad.as_mut_ptr();
            let hess_ptr = self.ordered_hess.as_mut_ptr();
            for i in 0..n_rows {
                let row = *rows.get_unchecked(i) as usize;
                *grad_ptr.add(i) = *grad_slice.get_unchecked(row);
                *hess_ptr.add(i) = *hess_slice.get_unchecked(row);
            }
        }

        // Build histogram with ordered gradients (sequential gradient access)
        build_histograms_ordered(
            hist.bins,
            &self.ordered_grad,
            &self.ordered_hess,
            rows,
            bin_views,
            &self.feature_metas,
        );
    }

    /// Find best split for a node using column sampler.
    fn find_split(
        &mut self,
        node: u32,
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

    /// Helper to count leaves in a Tree.
    fn count_leaves<L: crate::repr::gbdt::LeafValue>(tree: &Tree<L>) -> usize {
        (0..tree.n_nodes() as u32).filter(|&i| tree.is_leaf(i)).count()
    }

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

    fn make_numeric_boundary_dataset() -> BinnedDataset {
        // 2 samples, 1 feature, 2 bins.
        // Bin 0 upper bound is 0.5, bin 1 upper bound is 1.5.
        let bins = vec![0, 1];
        let mapper = BinMapper::numerical(
            vec![0.5, 1.5],
            MissingType::None,
            0,
            0,
            0.0,
            0.0,
            1.0,
        );
        BinnedDatasetBuilder::new()
            .add_binned(bins, mapper)
            .group_strategy(GroupStrategy::SingleGroup { layout: GroupLayout::ColumnMajor })
            .build()
            .unwrap()
    }

    fn make_categorical_domain_dataset() -> BinnedDataset {
        // 4 samples, 1 categorical feature with 4 categories.
        // The raw category values are intentionally large to catch accidental use of
        // raw values as bitset indices.
        let bins = vec![0, 1, 2, 3];
        let mapper = BinMapper::categorical(
            vec![1000, 2000, 3000, 4000],
            MissingType::None,
            0,
            0,
            0.0,
        );
        BinnedDatasetBuilder::new()
            .add_binned(bins, mapper)
            .group_strategy(GroupStrategy::SingleGroup { layout: GroupLayout::ColumnMajor })
            .build()
            .unwrap()
    }

    #[test]
    fn phase4_numeric_threshold_translation_preserves_boundary() {
        let dataset = make_numeric_boundary_dataset();

        let mut builder = MutableTree::<ScalarLeaf>::with_capacity(3);
        let root = builder.init_root();

        // Training semantics: bin <= 0 goes left.
        let split = SplitInfo::numerical(0, 0, 1.0, true);
        let (left, right) = TreeGrower::apply_split_to_builder(&mut builder, root, &split, &dataset);
        builder.make_leaf(left, ScalarLeaf(10.0));
        builder.make_leaf(right, ScalarLeaf(20.0));
        let tree = builder.freeze();

        let upper = dataset.bin_mapper(0).bin_to_value(0) as f32;
        let threshold = tree.split_threshold(0);

        assert!(threshold > upper);
        // Value exactly on the bin boundary should still go LEFT.
        assert_eq!(tree.predict_row(&[upper]).0, 10.0);
        // Value exactly equal to the stored threshold should go RIGHT.
        assert_eq!(tree.predict_row(&[threshold]).0, 20.0);

        // And the binned path must match training semantics.
        let row0 = dataset.row_view(0).unwrap();
        let row1 = dataset.row_view(1).unwrap();
        assert_eq!(tree.predict_binned(&row0, &dataset).0, 10.0);
        assert_eq!(tree.predict_binned(&row1, &dataset).0, 20.0);
    }

    #[test]
    fn phase5_categorical_domain_is_bin_indices_and_semantics_match() {
        let dataset = make_categorical_domain_dataset();

        let mut builder = MutableTree::<ScalarLeaf>::with_capacity(3);
        let root = builder.init_root();

        // Training semantics: categories {1, 3} go LEFT.
        let mut left_cats = crate::training::gbdt::categorical::CatBitset::empty();
        left_cats.insert(1);
        left_cats.insert(3);
        let split = SplitInfo::categorical(0, left_cats, 1.0, false);
        let (left, right) = TreeGrower::apply_split_to_builder(&mut builder, root, &split, &dataset);
        builder.make_leaf(left, ScalarLeaf(10.0));
        builder.make_leaf(right, ScalarLeaf(20.0));
        let tree = builder.freeze();

        // Category indices are in 0..=3, so the stored bitset should fit in one word.
        assert!(tree.has_categorical());
        assert_eq!(tree.categories().bitset_for_node(0).len(), 1);

        // Inference path expects category *indices* in f32 form.
        assert_eq!(tree.predict_row(&[1.0]).0, 10.0);
        assert_eq!(tree.predict_row(&[2.0]).0, 20.0);

        // Binned path should match too (bins are the category indices).
        let row1 = dataset.row_view(1).unwrap(); // bin 1
        let row2 = dataset.row_view(2).unwrap(); // bin 2
        assert_eq!(tree.predict_binned(&row1, &dataset).0, 10.0);
        assert_eq!(tree.predict_binned(&row2, &dataset).0, 20.0);
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

        assert_eq!(count_leaves(&tree), 1);
        assert!(tree.is_leaf(0));
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
        assert!(!tree.is_leaf(0));
        assert_eq!(count_leaves(&tree), 2);
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

        // Should produce a tree with multiple nodes (max_depth=2 allows up to 4 leaves)
        assert!(tree.n_nodes() >= 1);
        assert!(count_leaves(&tree) <= 4);
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
        for i in 0..tree.n_nodes() as u32 {
            if tree.is_leaf(i) {
                // Values should be relatively small due to 0.1 learning rate
                assert!(tree.leaf_value(i).0.abs() < 1.0);
            }
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
        assert!(count_leaves(&tree) <= 4);
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
        assert!(count_leaves(&tree_all) >= 1);
        assert!(count_leaves(&tree_sampled) >= 1);
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

        assert!(!tree.is_leaf(0));
        assert_eq!(count_leaves(&tree), 2);
    }
}

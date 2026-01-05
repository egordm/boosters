//! Tree grower for gradient boosting.
//!
//! Orchestrates tree training using histogram-based split finding, row partitioning,
//! and the subtraction trick to reduce computation.

use ndarray::ArrayViewMut1;

use crate::data::BinnedDataset;
use crate::repr::gbdt::{MutableTree, NodeId, ScalarLeaf, categories_to_bitset};
use crate::training::sampling::{ColSampler, ColSamplingParams};
use crate::training::{Gradients, GradsTuple};

use super::expansion::{GrowthState, GrowthStrategy, NodeCandidate};
use super::histograms::{FeatureView, HistogramBuilder, HistogramLayout, HistogramPool};
use super::partition::RowPartitioner;
use super::split::{GainParams, GreedySplitter, SplitInfo, SplitType};
use crate::utils::Parallelism;

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
    /// Feature types (true = categorical) for original features.
    feature_types: Vec<bool>,
    /// Feature has missing values (true = has missing) for original features.
    feature_has_missing: Vec<bool>,
    /// Whether each effective feature is a bundle (true = bundle column).
    /// Bundles require special handling in split finding: bin 0 is reserved
    /// for "all defaults" and cannot be used as a split point.
    feature_is_bundle: Vec<bool>,
    /// Feature metadata for histogram building (always original features).
    feature_metas: Vec<HistogramLayout>,
    /// Split strategy.
    split_strategy: GreedySplitter,
    /// Column sampler for feature subsampling.
    col_sampler: ColSampler,
    /// Per-node leaf values (scaled by learning rate) for the last grown tree.
    /// Indexed by training node id used by the partitioner/histogram pool.
    last_leaf_values: Vec<f32>,
    /// Buffer for ordered (pre-gathered) gradients.
    /// Reused across histogram builds to avoid allocation.
    ordered_grad_hess: Vec<GradsTuple>,
    /// Mapping from tree_node to partitioner_node for leaves.
    /// Populated during tree growth, cleared on reset.
    leaf_node_map: Vec<(NodeId, NodeId)>,

    /// Histogram builder (encapsulates parallelism and kernel dispatch).
    histogram_builder: HistogramBuilder,
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
        parallelism: Parallelism,
    ) -> Self {
        let n_samples = dataset.n_samples();

        // Get effective feature layout (bundles + standalone features)
        let effective = dataset.feature_views();
        let n_effective_columns = effective.n_columns();

        // Build histogram layout from effective bin counts
        let mut offset = 0u32;
        let feature_metas: Vec<HistogramLayout> = effective
            .bin_counts
            .iter()
            .map(|&n_bins| {
                let layout = HistogramLayout { offset, n_bins };
                offset += n_bins;
                layout
            })
            .collect();

        // Use effective feature metadata directly
        let feature_types = effective.is_categorical.clone();
        let feature_has_missing = effective.has_missing.clone();
        let feature_is_bundle: Vec<bool> = (0..n_effective_columns)
            .map(|col| effective.is_bundle(col))
            .collect();

        // Maximum nodes: 2*max_leaves - 1 for a full binary tree
        let max_leaves = params.max_leaves();
        let max_nodes = (2 * max_leaves as usize).saturating_sub(1).max(63);

        let histogram_pool =
            HistogramPool::new(feature_metas.clone(), cache_size.max(2), max_nodes);
        let partitioner = RowPartitioner::new(n_samples, max_nodes);

        // Build split strategy with gain params encapsulated
        // Parallelism self-corrects based on workload
        let split_strategy =
            GreedySplitter::with_config(params.gain.clone(), params.max_onehot_cats, parallelism);

        // Column sampler uses effective columns for sampling
        let col_sampler =
            ColSampler::new(params.col_sampling.clone(), n_effective_columns as u32, 0);

        Self {
            params,
            histogram_pool,
            partitioner,
            tree_builder: MutableTree::with_capacity(max_nodes),
            feature_types,
            feature_has_missing,
            feature_is_bundle,
            feature_metas,
            split_strategy,
            col_sampler,
            last_leaf_values: Vec::new(),
            ordered_grad_hess: Vec::with_capacity(n_samples),
            leaf_node_map: Vec::with_capacity(max_nodes / 2 + 1),
            histogram_builder: HistogramBuilder::new(parallelism),
        }
    }

    /// Get read-only access to the row partitioner.
    ///
    /// After `grow()` returns, the partitioner contains the final leaf assignments.
    /// Use `get_leaf_indices(leaf_id)` to get row indices assigned to each leaf.
    ///
    /// This is useful for fitting linear leaves or any post-processing that needs
    /// to know which rows ended up in which leaf.
    #[inline]
    pub fn partitioner(&self) -> &RowPartitioner {
        &self.partitioner
    }

    /// Get the leaf node mapping (tree_node -> partitioner_node).
    ///
    /// After `grow()` returns, this contains a mapping for each leaf in the tree.
    /// Use with `partitioner().get_leaf_indices(partitioner_node)` to get row indices.
    ///
    /// Returns tuples of `(tree_node_id, partitioner_node_id)`.
    #[inline]
    pub fn leaf_node_mapping(&self) -> &[(NodeId, NodeId)] {
        &self.leaf_node_map
    }

    /// Update predictions using the partitioner's leaf assignments and cached leaf values.
    ///
    /// After tree growth, the partitioner's leaves correspond to the final tree leaves.
    /// Instead of traversing the tree for each row, we iterate over each leaf's row range
    /// and apply the cached leaf value.
    ///
    /// This is O(n_rows) with sequential writes to predictions, which is much faster than
    /// O(n_rows × tree_depth) random tree traversals.
    ///
    /// # Note
    ///
    /// This method uses only the base scalar leaf values (not linear coefficients).
    /// When linear leaves are enabled, the trainer should use `tree.predict_into()`
    /// instead to correctly compute linear predictions for gradient updates.
    pub fn update_predictions_from_last_tree(&self, mut predictions: ArrayViewMut1<f32>) {
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

    /// Update a cached leaf value for fast prediction updates after leaf renewing.
    ///
    /// When leaf values are renewed (e.g., for quantile objectives), the cached
    /// values in `last_leaf_values` become stale. This method updates them so that
    /// `update_predictions_from_last_tree` can be used instead of tree traversal.
    ///
    /// # Arguments
    /// * `partitioner_node` - The partitioner node ID (from `leaf_node_mapping()`)
    /// * `new_value` - The new leaf value (already scaled by learning rate)
    #[inline]
    pub fn update_cached_leaf_value(&mut self, partitioner_node: NodeId, new_value: f32) {
        let node = partitioner_node as usize;
        if node < self.last_leaf_values.len() {
            self.last_leaf_values[node] = new_value;
        }
    }

    /// Record a leaf value for fast prediction updates.
    /// Also records the mapping from tree node to partitioner node.
    fn record_leaf_value(&mut self, tree_node: NodeId, partitioner_node: NodeId, weight: f32) {
        // Ensure we have room for this partitioner node
        let node = partitioner_node as usize;
        if node >= self.last_leaf_values.len() {
            self.last_leaf_values.resize(node + 1, f32::NAN);
        }
        // Apply learning rate since tree.predict returns the post-learning-rate values
        self.last_leaf_values[node] = weight * self.params.learning_rate;
        // Record tree_node -> partitioner_node mapping for linear leaf fitting
        self.leaf_node_map.push((tree_node, partitioner_node));
    }

    /// Grow a tree from gradients for a specific output.
    ///
    /// Returns a `MutableTree` which allows post-processing (e.g., fitting linear leaves)
    /// before calling `freeze()` to get the final `Tree`.
    ///
    /// # Arguments
    /// * `dataset` - Binned dataset
    /// * `gradients` - Gradient storage
    /// * `output` - Which output index to grow the tree for (0 to n_outputs-1)
    /// * `sampled_rows` - Optional sampled row indices (skips unsampled rows in histograms)
    ///
    /// # Returns
    /// The trained mutable tree. Call `freeze()` to convert to inference-ready `Tree`.
    pub fn grow(
        &mut self,
        dataset: &BinnedDataset,
        gradients: &Gradients,
        output: usize,
        sampled_rows: Option<&[u32]>,
    ) -> MutableTree<ScalarLeaf> {
        let n_samples = dataset.n_samples();
        assert_eq!(gradients.n_samples(), n_samples);

        // Reset state for new tree (partitioner initialized with sampled rows)
        self.reset(n_samples, sampled_rows);

        // Initialize column sampler for this tree
        self.col_sampler.sample_tree();
        self.col_sampler.sample_level(0);

        // Pre-compute effective feature views once per tree
        // This includes bundle views (if any) followed by standalone feature views
        let effective = dataset.feature_views();
        let bin_views = &effective.views;

        // Initialize growth state
        let strategy = self.params.growth_strategy;
        let mut state = strategy.init();

        // Initialize root
        let root_tree_node = self.tree_builder.init_root();

        // Build root histogram first
        self.build_histogram(0, gradients, output, bin_views);

        // Compute root gradient sums from histogram (O(n_bins) instead of O(n_samples))
        let (total_grad, total_hess) = self
            .histogram_pool
            .get(0)
            .expect("root histogram must exist")
            .sum_gradients();

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
                self.process_candidate(
                    candidate, dataset, &effective, gradients, output, bin_views, &mut state,
                );
            }

            // Advance to next iteration (depth-wise: move to next level)
            state.advance();
        }

        // Finalize any remaining candidates as leaves
        self.finalize_remaining(&mut state);

        // Apply learning rate
        self.tree_builder
            .apply_learning_rate(self.params.learning_rate);

        // Return mutable tree (caller should call freeze() after any post-processing)
        std::mem::take(&mut self.tree_builder)
    }

    /// Process a single candidate node.
    #[allow(clippy::too_many_arguments)]
    fn process_candidate(
        &mut self,
        candidate: NodeCandidate,
        dataset: &BinnedDataset,
        effective: &crate::data::BinnedFeatureViews<'_>,
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
            // Record cover for explainability (leaves have gain=0)
            self.tree_builder
                .set_cover(candidate.tree_node, candidate.hess_sum as f32);
            self.record_leaf_value(candidate.tree_node, candidate.node, weight);
            self.histogram_pool.release(candidate.node);
            return;
        }

        // Record node stats for explainability before applying split
        self.tree_builder.set_node_stats(
            candidate.tree_node,
            candidate.split.gain,
            candidate.hess_sum as f32,
        );

        // Apply split (translating bin thresholds to float values)
        // For bundled columns, decode the effective column index to original feature
        let (left_tree, right_tree) = Self::apply_split_to_builder(
            &mut self.tree_builder,
            candidate.tree_node,
            &candidate.split,
            dataset,
            effective,
        );

        // Partition rows based on the split.
        // For bundled columns, the partitioner needs effective views to decode the split
        let (right_node, left_count, right_count) =
            self.partitioner
                .split(candidate.node, &candidate.split, dataset, effective);

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
        let (small_grad, small_hess) = self
            .histogram_pool
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
                    std::mem::take(current_level)
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
                // Record cover for explainability (leaves have gain=0)
                self.tree_builder
                    .set_cover(candidate.tree_node, candidate.hess_sum as f32);
                self.record_leaf_value(candidate.tree_node, candidate.node, weight);
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
        // Clear leaf node mapping from previous tree
        self.leaf_node_map.clear();
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
    ///
    /// Note: With LightGBM-style bundling, `split.feature` is an effective column index.
    /// We decode it to the original feature before storing in the tree, so predictions
    /// work without bundling context.
    fn apply_split_to_builder(
        builder: &mut MutableTree<ScalarLeaf>,
        node: u32,
        split: &SplitInfo,
        dataset: &BinnedDataset,
        effective: &crate::data::BinnedFeatureViews<'_>,
    ) -> (u32, u32) {
        // Decode effective column index to original feature and bin
        let effective_col = split.feature as usize;

        match &split.split_type {
            SplitType::Numerical { bin } => {
                // Decode effective column + bin to original feature + bin
                let (orig_feature, orig_bin) =
                    dataset.decode_split_to_original(effective, effective_col, *bin as u32);

                let mapper = dataset.bin_mapper(orig_feature);

                // Debug: ensure bin is in valid range
                debug_assert!(
                    orig_bin < mapper.n_bins(),
                    "split bin {} but feature {} has only {} bins",
                    orig_bin,
                    orig_feature,
                    mapper.n_bins(),
                );

                let bin_upper = mapper.bin_to_value(orig_bin) as f32;
                let threshold = Self::next_up_f32(bin_upper);
                builder.apply_numeric_split(
                    node,
                    orig_feature as u32,
                    threshold,
                    split.default_left,
                )
            }
            SplitType::Categorical { left_cats } => {
                // For categorical splits on bundles, we need to decode the effective column.
                // But categorical splits accumulate bins, not individual thresholds.
                // For now, assume categorical features are not bundled (TODO: support this).
                let (orig_feature, _) = dataset.decode_split_to_original(
                    effective,
                    effective_col,
                    0, // bin doesn't matter for categorical - we just need the feature
                );

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
                let mut categories: Vec<u32> = left_cats.iter().collect();
                categories.sort_unstable();
                categories.dedup();
                let bitset = categories_to_bitset(&categories);

                // Apply split with swapped children and inverted default.
                // The inference builder's apply_categorical_split returns (left, right),
                // but since we're inverting the semantics, we swap them here.
                let (inf_left, inf_right) = builder.apply_categorical_split(
                    node,
                    orig_feature as u32,
                    bitset,
                    !split.default_left,
                );
                // Swap children to match training semantics
                (inf_right, inf_left)
            }
        }
    }

    /// Build histogram for a node.
    ///
    /// When bundling is active (`use_bundling == true`):
    /// - `bin_views` are bundled column views (fewer columns, efficient access)
    /// - Uses unbundled histogram building to decode bins to original features
    /// - Histograms are indexed by original feature, not bundled column
    ///
    /// When bundling is inactive:
    /// - `bin_views` are original feature views
    /// - Uses standard histogram building
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
        let grad_hess_slice = gradients.output_pairs(output);

        if let Some(start) = self.partitioner.leaf_sequential_start(node) {
            // Contiguous rows: direct access without gathering
            let start = start as usize;
            let end = start + rows.len();
            self.histogram_builder.build_contiguous(
                hist.bins,
                &grad_hess_slice[start..end],
                start,
                bin_views,
                &self.feature_metas,
            );
        } else {
            // Non-sequential indices: gather gradients into partition order.
            // Invariant: `ordered_grad_hess[i]` must correspond to `rows[i]`.
            let n_rows = rows.len();

            // Ensure buffers have capacity (reuses allocation across builds)
            if self.ordered_grad_hess.capacity() < n_rows {
                self.ordered_grad_hess
                    .reserve(n_rows - self.ordered_grad_hess.capacity());
            }

            // Gather gradients in partition order with direct writes (no capacity checks)
            // SAFETY: We just ensured capacity >= n_rows, and row indices are valid
            unsafe {
                self.ordered_grad_hess.set_len(n_rows);
                let gh_ptr = self.ordered_grad_hess.as_mut_ptr();
                for i in 0..n_rows {
                    let row = *rows.get_unchecked(i) as usize;
                    *gh_ptr.add(i) = *grad_hess_slice.get_unchecked(row);
                }
            }

            self.histogram_builder.build_gathered(
                hist.bins,
                &self.ordered_grad_hess,
                rows,
                bin_views,
                &self.feature_metas,
            );
        }
    }

    /// Find best split for a node using column sampler.
    fn find_split(&mut self, node: u32, grad_sum: f64, hess_sum: f64, count: u32) -> SplitInfo {
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
            &self.feature_has_missing,
            &self.feature_is_bundle,
            features,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{BinnedDataset, BinningConfig, Dataset};
    use crate::repr::gbdt::{Tree, TreeView};
    use ndarray::{Array1, array};

    /// Helper to count leaves in a Tree.
    fn count_leaves<L: crate::repr::gbdt::LeafValue>(tree: &Tree<L>) -> usize {
        (0..tree.n_nodes() as u32)
            .filter(|&i| tree.is_leaf(i))
            .count()
    }

    fn make_simple_dataset() -> BinnedDataset {
        // 10 samples, 1 feature
        // Raw values 0,0,0,1,1,2,2,3,3,3 bin to bins 0,0,0,1,1,2,2,3,3,3
        let raw = Dataset::builder()
            .add_feature_unnamed(array![0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0])
            .build()
            .unwrap();
        BinnedDataset::from_dataset(&raw, &BinningConfig::default()).unwrap()
    }

    fn make_two_feature_dataset() -> BinnedDataset {
        // 8 samples, 2 features
        // Feature 0: [0,0,1,1,0,0,1,1] → bins based on 0/1 values
        // Feature 1: [0,0,0,0,1,1,1,1] → bins based on 0/1 values
        let raw = Dataset::builder()
            .add_feature_unnamed(array![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0])
            .add_feature_unnamed(array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
            .build()
            .unwrap();
        BinnedDataset::from_dataset(&raw, &BinningConfig::default()).unwrap()
    }

    fn make_numeric_boundary_dataset() -> BinnedDataset {
        // 2 samples, 1 feature, targeting 2 bins
        // Values 0 and 1 should bin to bins 0 and 1
        let raw = Dataset::builder()
            .add_feature_unnamed(array![0.0, 1.0])
            .build()
            .unwrap();
        BinnedDataset::from_dataset(&raw, &BinningConfig::default()).unwrap()
    }

    fn make_categorical_domain_dataset() -> BinnedDataset {
        // 4 samples, 1 categorical feature with 4 categories
        // Category values 0, 1, 2, 3 map to bins 0, 1, 2, 3
        let raw = Dataset::builder()
            .add_categorical_unnamed(array![0.0, 1.0, 2.0, 3.0])
            .build()
            .unwrap();

        BinnedDataset::from_dataset(&raw, &BinningConfig::default()).unwrap()
    }

    #[test]
    fn phase4_numeric_threshold_translation_preserves_boundary() {
        let dataset = make_numeric_boundary_dataset();
        let effective = dataset.feature_views();

        let mut builder = MutableTree::<ScalarLeaf>::with_capacity(3);
        let root = builder.init_root();

        // Training semantics: bin <= 0 goes left.
        let split = SplitInfo::numerical(0, 0, 1.0, true);
        let (left, right) =
            TreeGrower::apply_split_to_builder(&mut builder, root, &split, &dataset, &effective);
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
        let x0 = dataset.bin_mapper(0).bin_to_value(dataset.bin(0, 0)) as f32;
        let x1 = dataset.bin_mapper(0).bin_to_value(dataset.bin(1, 0)) as f32;
        assert_eq!(tree.predict_row(&[x0]).0, 10.0);
        assert_eq!(tree.predict_row(&[x1]).0, 20.0);
    }

    #[test]
    fn phase5_categorical_domain_is_bin_indices_and_semantics_match() {
        let dataset = make_categorical_domain_dataset();
        let effective = dataset.feature_views();

        let mut builder = MutableTree::<ScalarLeaf>::with_capacity(3);
        let root = builder.init_root();

        // Training semantics: categories {1, 3} go LEFT.
        let mut left_cats = crate::training::gbdt::categorical::CatBitset::empty();
        left_cats.insert(1);
        left_cats.insert(3);
        let split = SplitInfo::categorical(0, left_cats, 1.0, false);
        let (left, right) =
            TreeGrower::apply_split_to_builder(&mut builder, root, &split, &dataset, &effective);
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
        let x1 = dataset.bin(1, 0) as f32; // bin 1
        let x2 = dataset.bin(2, 0) as f32; // bin 2
        assert_eq!(tree.predict_row(&[x1]).0, 10.0);
        assert_eq!(tree.predict_row(&[x2]).0, 20.0);
    }

    #[test]
    fn test_grower_single_leaf() {
        let dataset = make_simple_dataset();
        let params = GrowerParams {
            growth_strategy: GrowthStrategy::DepthWise { max_depth: 0 },
            ..Default::default()
        };

        let mut grower = TreeGrower::new(&dataset, params, 4, Parallelism::Sequential);

        // All same gradient: no good splits
        let grad: Vec<f32> = vec![1.0; 10];
        let hess: Vec<f32> = vec![1.0; 10];
        let mut gradients = Gradients::new(10, 1);
        for (dst, (&g, &h)) in gradients
            .output_pairs_mut(0)
            .iter_mut()
            .zip(grad.iter().zip(hess.iter()))
        {
            dst.grad = g;
            dst.hess = h;
        }

        let tree = grower.grow(&dataset, &gradients, 0, None).freeze();

        assert_eq!(count_leaves(&tree), 1);
        assert!(tree.is_leaf(0));
    }

    #[test]
    fn test_grower_simple_split() {
        let dataset = make_simple_dataset();
        let params = GrowerParams {
            growth_strategy: GrowthStrategy::DepthWise { max_depth: 1 },
            gain: GainParams {
                min_gain: 0.0,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut grower = TreeGrower::new(&dataset, params, 4, Parallelism::Sequential);

        // Gradients that suggest a split
        let grad: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0];
        let hess: Vec<f32> = vec![1.0; 10];
        let mut gradients = Gradients::new(10, 1);
        for (dst, (&g, &h)) in gradients
            .output_pairs_mut(0)
            .iter_mut()
            .zip(grad.iter().zip(hess.iter()))
        {
            dst.grad = g;
            dst.hess = h;
        }

        let tree = grower.grow(&dataset, &gradients, 0, None).freeze();

        // Should have a split at root with 2 leaves
        assert!(!tree.is_leaf(0));
        assert_eq!(count_leaves(&tree), 2);
    }

    #[test]
    fn test_grower_max_depth() {
        let dataset = make_two_feature_dataset();
        let params = GrowerParams {
            growth_strategy: GrowthStrategy::DepthWise { max_depth: 2 },
            gain: GainParams {
                min_gain: 0.0,
                min_child_weight: 0.1,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut grower = TreeGrower::new(&dataset, params, 8, Parallelism::Sequential);

        let grad: Vec<f32> = vec![2.0, 1.0, -1.0, -2.0, 2.0, 1.0, -1.0, -2.0];
        let hess: Vec<f32> = vec![1.0; 8];
        let mut gradients = Gradients::new(8, 1);
        for (dst, (&g, &h)) in gradients
            .output_pairs_mut(0)
            .iter_mut()
            .zip(grad.iter().zip(hess.iter()))
        {
            dst.grad = g;
            dst.hess = h;
        }

        let tree = grower.grow(&dataset, &gradients, 0, None).freeze();

        // Should produce a tree with multiple nodes (max_depth=2 allows up to 4 leaves)
        assert!(tree.n_nodes() >= 1);
        assert!(count_leaves(&tree) <= 4);
    }

    #[test]
    fn test_growth_strategy_depth_wise() {
        let mut state = GrowthStrategy::DepthWise { max_depth: 2 }.init();

        // Push root
        state.push_root(NodeCandidate::new(
            0,
            0,
            0,
            SplitInfo::numerical(0, 1, 0.5, false),
            0.0,
            1.0,
            10,
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
            0,
            0,
            0,
            SplitInfo::numerical(0, 1, 0.5, false),
            0.0,
            1.0,
            10,
        ));
        state.push(NodeCandidate::new(
            1,
            1,
            0,
            SplitInfo::numerical(0, 1, 0.8, false),
            0.0,
            1.0,
            10,
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
            gain: GainParams {
                min_gain: 0.0,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut grower = TreeGrower::new(&dataset, params, 4, Parallelism::Sequential);

        let grad_f32: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0];
        let hess_f32: Vec<f32> = vec![1.0; 10];
        let mut gradients = Gradients::new(10, 1);
        for (dst, (&g, &h)) in gradients
            .output_pairs_mut(0)
            .iter_mut()
            .zip(grad_f32.iter().zip(hess_f32.iter()))
        {
            dst.grad = g;
            dst.hess = h;
        }

        let tree = grower.grow(&dataset, &gradients, 0, None).freeze();

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
            gain: GainParams {
                min_gain: 0.0,
                min_child_weight: 0.1,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut grower = TreeGrower::new(&dataset, params, 8, Parallelism::Sequential);

        let grad: Vec<f32> = vec![2.0, 1.0, -1.0, -2.0, 2.0, 1.0, -1.0, -2.0];
        let hess: Vec<f32> = vec![1.0; 8];
        let mut gradients = Gradients::new(8, 1);
        for (dst, (&g, &h)) in gradients
            .output_pairs_mut(0)
            .iter_mut()
            .zip(grad.iter().zip(hess.iter()))
        {
            dst.grad = g;
            dst.hess = h;
        }

        let tree = grower.grow(&dataset, &gradients, 0, None).freeze();

        // Should have at most 4 leaves
        assert!(count_leaves(&tree) <= 4);
    }

    #[test]
    fn test_grower_with_col_sampler() {
        use crate::training::sampling::ColSamplingParams;

        let dataset = make_two_feature_dataset();

        let params_no_sampling = GrowerParams {
            growth_strategy: GrowthStrategy::DepthWise { max_depth: 2 },
            gain: GainParams {
                min_gain: 0.0,
                min_child_weight: 0.1,
                ..Default::default()
            },
            col_sampling: ColSamplingParams::None,
            ..Default::default()
        };

        let params_with_sampling = GrowerParams {
            growth_strategy: GrowthStrategy::DepthWise { max_depth: 2 },
            gain: GainParams {
                min_gain: 0.0,
                min_child_weight: 0.1,
                ..Default::default()
            },
            col_sampling: ColSamplingParams::bytree(0.5),
            ..Default::default()
        };

        let grad: Vec<f32> = vec![2.0, 1.0, -1.0, -2.0, 2.0, 1.0, -1.0, -2.0];
        let hess: Vec<f32> = vec![1.0; 8];
        let mut gradients = Gradients::new(8, 1);
        for (dst, (&g, &h)) in gradients
            .output_pairs_mut(0)
            .iter_mut()
            .zip(grad.iter().zip(hess.iter()))
        {
            dst.grad = g;
            dst.hess = h;
        }

        // Without column sampling
        let mut grower_all =
            TreeGrower::new(&dataset, params_no_sampling, 8, Parallelism::Sequential);
        let tree_all = grower_all.grow(&dataset, &gradients, 0, None).freeze();

        // With column sampling
        let mut grower_sampled =
            TreeGrower::new(&dataset, params_with_sampling, 8, Parallelism::Sequential);
        let tree_sampled = grower_sampled.grow(&dataset, &gradients, 0, None).freeze();

        // Both should produce trees
        assert!(count_leaves(&tree_all) >= 1);
        assert!(count_leaves(&tree_sampled) >= 1);
    }

    #[test]
    fn test_grower_with_f32_input() {
        let dataset = make_simple_dataset();
        let params = GrowerParams {
            growth_strategy: GrowthStrategy::DepthWise { max_depth: 1 },
            gain: GainParams {
                min_gain: 0.0,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut grower = TreeGrower::new(&dataset, params, 4, Parallelism::Sequential);

        let grad_f32: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0];
        let hess_f32: Vec<f32> = vec![1.0; 10];
        let mut gradients = Gradients::new(10, 1);
        for (dst, (&g, &h)) in gradients
            .output_pairs_mut(0)
            .iter_mut()
            .zip(grad_f32.iter().zip(hess_f32.iter()))
        {
            dst.grad = g;
            dst.hess = h;
        }

        let tree = grower.grow(&dataset, &gradients, 0, None).freeze();

        assert!(!tree.is_leaf(0));
        assert_eq!(count_leaves(&tree), 2);
    }

    #[test]
    fn test_partitioner_exposes_leaf_indices() {
        let dataset = make_simple_dataset();
        let params = GrowerParams {
            growth_strategy: GrowthStrategy::DepthWise { max_depth: 1 },
            gain: GainParams {
                min_gain: 0.0,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut grower = TreeGrower::new(&dataset, params, 4, Parallelism::Sequential);

        // Create gradients that produce a simple left/right split
        let grad: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0];
        let hess: Vec<f32> = vec![1.0; 10];
        let mut gradients = Gradients::new(10, 1);
        for (dst, (&g, &h)) in gradients
            .output_pairs_mut(0)
            .iter_mut()
            .zip(grad.iter().zip(hess.iter()))
        {
            dst.grad = g;
            dst.hess = h;
        }

        let _tree = grower.grow(&dataset, &gradients, 0, None);

        // Partitioner should have 2 leaves after the split
        let partitioner = grower.partitioner();
        assert!(partitioner.n_leaves() >= 2, "Should have at least 2 leaves");

        // Total samples across leaves should equal dataset size
        let total_samples: usize = (0..partitioner.n_leaves())
            .map(|i| partitioner.get_leaf_indices(i as u32).len())
            .sum();
        assert_eq!(
            total_samples, 10,
            "All 10 rows should be assigned to leaves"
        );

        // Each leaf should have non-overlapping row indices
        let mut all_indices: Vec<u32> = Vec::new();
        for leaf in 0..partitioner.n_leaves() {
            let indices = partitioner.get_leaf_indices(leaf as u32);
            all_indices.extend(indices);
        }
        all_indices.sort();
        all_indices.dedup();
        assert_eq!(all_indices.len(), 10, "Row indices should be unique");
    }

    #[test]
    fn test_grower_with_bundling_enabled() {
        // Create sparse categorical features that should trigger bundling
        // Feature 0: non-zero only for rows 0-4 (5% density - well under 10%)
        // Feature 1: non-zero only for rows 50-54 (5% density - well under 10%)
        // These are mutually exclusive (no overlap) so should bundle perfectly
        let n_samples = 100;

        // Create just 2 sparse categorical features (no numeric)
        let mut f0_values = Vec::with_capacity(n_samples);
        let mut f1_values = Vec::with_capacity(n_samples);
        for sample in 0..n_samples {
            // Feature 0: sparse categorical, non-zero for rows 0-4
            let v0 = if sample < 5 {
                (sample % 3 + 1) as f32
            } else {
                0.0
            };
            // Feature 1: sparse categorical, non-zero for rows 50-54
            let v1 = if (50..55).contains(&sample) {
                ((sample - 50) % 3 + 1) as f32
            } else {
                0.0
            };
            f0_values.push(v0);
            f1_values.push(v1);
        }

        let config = BinningConfig::builder()
            .enable_bundling(true)
            .sparsity_threshold(0.9)
            .build();

        let raw = Dataset::builder()
            .add_categorical_unnamed(Array1::from_vec(f0_values))
            .add_categorical_unnamed(Array1::from_vec(f1_values))
            .build()
            .unwrap();

        let dataset = BinnedDataset::from_dataset(&raw, &config).unwrap();

        // Verify bundling occurred
        let effective = dataset.feature_views();
        assert!(
            effective.has_bundles(),
            "Should have bundles with sparse categorical features (n_bundles={})",
            effective.n_bundles
        );
        assert_eq!(
            effective.n_bundles, 1,
            "Should have exactly 1 bundle (2 sparse cats bundled)"
        );
        // 1 bundle = 1 effective column
        assert_eq!(
            effective.n_columns(),
            1,
            "Should have 1 effective column (bundle only)"
        );

        // Create gradients: positive for first half, negative for second half
        let mut gradients = Gradients::new(n_samples, 1);
        for (i, pair) in gradients.output_pairs_mut(0).iter_mut().enumerate() {
            pair.grad = if i < 50 { 1.0 } else { -1.0 };
            pair.hess = 1.0;
        }

        // Configure grower
        let params = GrowerParams {
            growth_strategy: GrowthStrategy::DepthWise { max_depth: 2 },
            gain: GainParams {
                min_gain: 0.0,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut grower = TreeGrower::new(&dataset, params, 8, Parallelism::Sequential);

        // Grow the tree
        let tree = grower.grow(&dataset, &gradients, 0, None).freeze();

        // Tree should have been grown successfully
        assert!(count_leaves(&tree) >= 1, "Tree should have at least 1 leaf");

        // With only bundle features and sparse data, splits may or may not occur
        // The key test is that the grower handles bundled features without panicking
        // and produces a valid tree structure
        assert!(tree.n_nodes() >= 1, "Tree should have at least 1 node");
    }
}

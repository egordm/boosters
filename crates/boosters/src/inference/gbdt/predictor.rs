//! Unified predictor for tree ensemble inference.
//!
//! This module provides [`Predictor`], a single generic predictor that works with
//! any [`TreeTraversal`] strategy. This consolidates what were previously separate
//! implementations (`Predictor`, `BlockPredictor`, `UnrolledPredictor`) into one
//! flexible type.
//!
//! # Usage
//!
//! ```ignore
//! use boosters::inference::gbdt::{Predictor, StandardTraversal, UnrolledTraversal6};
//! use boosters::data::DenseMatrix;
//!
//! // Simple predictor (no pre-computation)
//! let predictor = Predictor::<StandardTraversal>::new(&forest);
//!

// Allow many arguments for internal prediction functions.
// Allow range loops when we need indices to access multiple arrays.
#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]
//!
//! // Unrolled predictor (pre-computes layouts for speed)
//! let fast_predictor = Predictor::<UnrolledTraversal6>::new(&forest);
//!
//! // Both use the same API
//! let output = predictor.predict(&features);
//! let output = fast_predictor.predict(&features);
//! ```
//!
//! # Block Size
//!
//! All predictors use block-based processing for cache efficiency. The default
//! block size is 64 rows (matching XGBoost). Use [`Predictor::with_block_size`]
//! to customize.
//!
//! # Choosing a Traversal Strategy
//!
//! - [`StandardTraversal`]: Simple, no setup cost. Best for single rows or small batches.
//! - [`UnrolledTraversal6`]: Pre-computes tree layouts. 2-3x faster for large batches (100+ rows).
//!
//! See [`TreeTraversal`] for implementing custom strategies.

use crate::Parallelism;
use crate::data::SamplesView;
use crate::data::{Dataset, axis};
use crate::repr::gbdt::{Forest, ScalarLeaf, TreeView};
use ndarray::{Array2, ArrayView1, ArrayViewMut2};

use super::TreeTraversal;

/// Default block size for batch processing (matches XGBoost).
pub const DEFAULT_BLOCK_SIZE: usize = 64;

/// Unified predictor for tree ensemble inference.
///
/// Generic over [`TreeTraversal`] strategy, allowing different optimization
/// techniques while sharing the prediction orchestration logic.
///
/// # Type Parameters
///
/// - `T`: Traversal strategy (e.g., [`StandardTraversal`], [`UnrolledTraversal6`])
///
/// # Example
///
/// ```ignore
/// use boosters::inference::gbdt::{Predictor, UnrolledTraversal6};
///
/// let predictor = Predictor::<UnrolledTraversal6>::new(&forest);
/// let output = predictor.predict(&features);
/// ```
///
/// [`StandardTraversal`]: super::StandardTraversal
/// [`UnrolledTraversal6`]: super::UnrolledTraversal6
#[derive(Debug)]
pub struct Predictor<'f, T: TreeTraversal<ScalarLeaf>> {
    forest: &'f Forest<ScalarLeaf>,
    /// Pre-computed state for each tree (e.g., unrolled layouts)
    tree_states: Box<[T::TreeState]>,
    /// Number of rows to process together
    block_size: usize,
}

impl<'f, T: TreeTraversal<ScalarLeaf>> Predictor<'f, T> {
    /// Create a new predictor for the given forest.
    ///
    /// Builds per-tree state (e.g., unrolled layouts) upfront.
    #[inline]
    pub fn new(forest: &'f Forest<ScalarLeaf>) -> Self {
        let tree_states: Box<[_]> = forest
            .trees()
            .map(|tree| T::build_tree_state(tree))
            .collect();

        Self {
            forest,
            tree_states,
            block_size: DEFAULT_BLOCK_SIZE,
        }
    }

    /// Create a predictor with custom block size.
    ///
    /// Block size controls how many rows are processed together for cache efficiency.
    /// Default is 64 (matching XGBoost).
    #[inline]
    pub fn with_block_size(mut self, block_size: usize) -> Self {
        self.block_size = block_size;
        self
    }

    /// Get the block size.
    #[inline]
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Get a reference to the underlying forest.
    #[inline]
    pub fn forest(&self) -> &Forest<ScalarLeaf> {
        self.forest
    }

    /// Number of output groups.
    #[inline]
    pub fn n_groups(&self) -> usize {
        self.forest.n_groups() as usize
    }

    /// Predict a single row and write to output buffer.
    ///
    /// # Arguments
    ///
    /// * `features` - Feature values for one sample (length = n_features)
    /// * `tree_weights` - Optional per-tree weights for DART (length = n_trees).
    ///   These are tree weights from DART's dropout mechanism, NOT sample weights.
    /// * `output` - Output buffer (length = n_groups)
    pub fn predict_row_into(
        &self,
        sample: ArrayView1<f32>,
        tree_weights: Option<&[f32]>,
        output: &mut [f32],
    ) {
        assert_eq!(
            output.len(),
            self.n_groups(),
            "output length must equal n_groups"
        );
        if let Some(w) = tree_weights {
            assert_eq!(
                w.len(),
                self.forest.n_trees(),
                "tree_weights length must match number of trees"
            );
        }

        // Initialize with base scores
        output.copy_from_slice(self.forest.base_score());

        // Accumulate tree contributions
        for (tree_idx, (tree, group)) in self.forest.trees_with_groups().enumerate() {
            let state = &self.tree_states[tree_idx];

            let leaf_idx = T::traverse_tree(tree, state, sample);

            let leaf_value = if tree.has_linear_leaves() {
                tree.compute_leaf_value(leaf_idx, sample)
            } else {
                tree.leaf_value(leaf_idx).0
            };

            let weighted_value = tree_weights
                .map(|w| leaf_value * w[tree_idx])
                .unwrap_or(leaf_value);

            output[group as usize] += weighted_value;
        }
    }

    /// Predict batch into provided buffer.
    ///
    /// Takes a Dataset and handles the transpose internally using
    /// block buffers for cache efficiency.
    ///
    /// # Arguments
    ///
    /// * `dataset` - Dataset containing feature-major data `[n_features, n_samples]`
    /// * `tree_weights` - Optional per-tree weights for DART dropout (length = n_trees).
    ///   These are NOT sample weights - they are tree scaling factors from DART.
    /// * `parallelism` - Whether to use parallel execution
    /// * `output` - Output buffer `[n_groups, n_samples]`
    ///
    /// # Block Buffering
    ///
    /// Uses per-thread `Array2` buffers sized at `[block_size, n_features]`.
    /// For 100 features and block_size=64, this is ~25KB per thread, fitting
    /// comfortably in L2 cache.
    pub fn predict_into(
        &self,
        dataset: &Dataset,
        tree_weights: Option<&[f32]>,
        parallelism: Parallelism,
        mut output: ArrayViewMut2<f32>,
    ) {
        let n_features = dataset.n_features();
        let n_samples = dataset.n_samples();
        let n_groups = self.n_groups();
        let block_size = self.block_size;

        assert_eq!(
            output.shape(),
            &[n_groups, n_samples],
            "output shape must match (n_groups, n_samples)"
        );
        if let Some(w) = tree_weights {
            assert_eq!(
                w.len(),
                self.forest.n_trees(),
                "tree_weights length must match number of trees"
            );
        }

        if n_samples == 0 {
            return;
        }

        // Initialize with base scores
        ndarray::Zip::from(output.axis_iter_mut(axis::ROWS))
            .and(self.forest.base_score())
            .for_each(|mut row, &score| row.fill(score));

        // Process output chunks in parallel with per-thread buffer reuse
        let output_chunks = output.axis_chunks_iter_mut(axis::COLS, block_size);

        parallelism.maybe_par_bridge_for_each_init(
            output_chunks.enumerate(),
            || Array2::<f32>::zeros((block_size, n_features)),
            |buffer, (block_idx, output_chunk)| {
                let start_sample = block_idx * block_size;

                // Fill buffer with samples from dataset and get a SamplesView
                let samples = dataset.buffer_samples(buffer, start_sample);

                // Predict block into output chunk
                self.predict_block_into(samples, tree_weights, output_chunk);
            },
        );
    }

    /// Convenience method: predict with allocation.
    ///
    /// Allocates output buffer and calls `predict_into`. For maximum performance
    /// when reusing buffers, use `predict_into` directly.
    ///
    /// # Arguments
    ///
    /// * `dataset` - Dataset containing feature-major data `[n_features, n_samples]`
    /// * `parallelism` - Whether to use parallel execution
    ///
    /// # Returns
    ///
    /// Prediction matrix `[n_groups, n_samples]`
    pub fn predict(&self, dataset: &Dataset, parallelism: Parallelism) -> Array2<f32> {
        let n_samples = dataset.n_samples();
        let n_groups = self.n_groups();
        let mut output = Array2::<f32>::zeros((n_groups, n_samples));
        self.predict_into(dataset, None, parallelism, output.view_mut());
        output
    }

    /// Internal: predict a block using SamplesView (optimized for contiguous data).
    fn predict_block_into(
        &self,
        features: SamplesView<'_>,
        tree_weights: Option<&[f32]>,
        mut output: ArrayViewMut2<f32>,
    ) {
        let n_rows = features.n_samples();
        let mut leaf_indices = vec![0u32; n_rows];

        for (tree_idx, (tree, group)) in self.forest.trees_with_groups().enumerate() {
            let state = &self.tree_states[tree_idx];
            let group_idx = group as usize;
            let tree_weight = tree_weights.map(|w| w[tree_idx]).unwrap_or(1.0);

            // Step 1: Get leaf indices
            T::traverse_block(tree, state, &features, &mut leaf_indices);

            // Step 2: Extract values and accumulate into group row
            let mut group_row = output.row_mut(group_idx);
            if tree.has_linear_leaves() {
                for i in 0..n_rows {
                    let sample = features.sample_view(i);
                    let value = tree.compute_leaf_value(leaf_indices[i], sample);
                    group_row[i] += value * tree_weight;
                }
            } else {
                for i in 0..n_rows {
                    group_row[i] += tree.leaf_value(leaf_indices[i]).0 * tree_weight;
                }
            }
        }
    }
}

// =============================================================================
// Type Aliases for Convenience
// =============================================================================

use super::traversal::{StandardTraversal, UnrolledTraversal};
use super::{Depth4, Depth6, Depth8};

/// Simple predictor using standard traversal (no pre-computation).
///
/// Good for single rows or small batches where setup cost matters.
pub type SimplePredictor<'f> = Predictor<'f, StandardTraversal>;

/// Unrolled predictor with depth 4 (15 nodes, 16 exits).
pub type UnrolledPredictor4<'f> = Predictor<'f, UnrolledTraversal<Depth4>>;

/// Unrolled predictor with depth 6 (63 nodes, 64 exits) - default, matches XGBoost.
pub type UnrolledPredictor6<'f> = Predictor<'f, UnrolledTraversal<Depth6>>;

/// Unrolled predictor with depth 8 (255 nodes, 256 exits).
pub type UnrolledPredictor8<'f> = Predictor<'f, UnrolledTraversal<Depth8>>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repr::gbdt::{Forest, ScalarLeaf, Tree};
    use approx::assert_abs_diff_eq;
    use ndarray::Array2;

    fn build_simple_tree(left_val: f32, right_val: f32, threshold: f32) -> Tree<ScalarLeaf> {
        crate::scalar_tree! {
            0 => num(0, threshold, L) -> 1, 2,
            1 => leaf(left_val),
            2 => leaf(right_val),
        }
    }

    fn build_deeper_tree() -> Tree<ScalarLeaf> {
        crate::scalar_tree! {
            0 => num(0, 0.5, L) -> 1, 2,
            1 => num(1, 0.3, L) -> 3, 4,
            2 => num(1, 0.7, L) -> 5, 6,
            3 => leaf(1.0),
            4 => leaf(2.0),
            5 => leaf(3.0),
            6 => leaf(4.0),
        }
    }

    /// Helper: create Dataset from a flat slice in feature-major layout
    fn make_dataset(data: &[f32], n_features: usize, n_samples: usize) -> Dataset {
        let arr = Array2::from_shape_vec((n_features, n_samples), data.to_vec()).unwrap();
        Dataset::from_array(arr.view(), None, None)
    }

    // =========================================================================
    // Single-row prediction tests
    // =========================================================================

    #[test]
    fn simple_predictor_single_row() {
        let mut forest = Forest::for_regression();
        forest.push_tree(build_simple_tree(1.0, 2.0, 0.5), 0);

        let predictor = SimplePredictor::new(&forest);

        let mut output = vec![0.0];
        predictor.predict_row_into(ndarray::array![0.3f32].view(), None, &mut output);
        assert_eq!(output, vec![1.0]);

        predictor.predict_row_into(ndarray::array![0.7f32].view(), None, &mut output);
        assert_eq!(output, vec![2.0]);
    }

    #[test]
    fn simple_predictor_with_base_score() {
        let mut forest = Forest::for_regression().with_base_score(vec![0.5]);
        forest.push_tree(build_simple_tree(1.0, 2.0, 0.5), 0);

        let predictor = SimplePredictor::new(&forest);

        let mut output = vec![0.0];
        predictor.predict_row_into(ndarray::array![0.3f32].view(), None, &mut output);
        assert_eq!(output, vec![1.5]);
    }

    #[test]
    fn simple_predictor_weighted() {
        let mut forest = Forest::for_regression();
        forest.push_tree(build_simple_tree(1.0, 2.0, 0.5), 0);
        forest.push_tree(build_simple_tree(0.5, 1.5, 0.5), 0);

        let predictor = SimplePredictor::new(&forest);

        let mut output = vec![0.0];
        predictor.predict_row_into(
            ndarray::array![0.3f32].view(),
            Some(&[1.0, 0.5]),
            &mut output,
        );
        assert!((output[0] - 1.25).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "tree_weights length must match number of trees")]
    fn weighted_wrong_length_panics() {
        let mut forest = Forest::for_regression();
        forest.push_tree(build_simple_tree(1.0, 2.0, 0.5), 0);
        forest.push_tree(build_simple_tree(0.5, 1.5, 0.5), 0);

        let predictor = SimplePredictor::new(&forest);
        let mut output = vec![0.0];
        predictor.predict_row_into(ndarray::array![0.3f32].view(), Some(&[1.0]), &mut output); // only 1 tree_weight
    }

    // =========================================================================
    // Batch prediction tests
    // =========================================================================

    #[test]
    fn simple_predictor_batch() {
        let mut forest = Forest::for_regression();
        forest.push_tree(build_simple_tree(1.0, 2.0, 0.5), 0);

        let predictor = SimplePredictor::new(&forest);

        // Feature-major: [n_features=1, n_samples=3]
        // Data: feature 0 values for samples [0, 1, 2]
        let dataset = make_dataset(&[0.3, 0.7, 0.5], 1, 3);
        let output = predictor.predict(&dataset, Parallelism::Sequential);

        // Output shape: [n_groups, n_samples] = [1, 3]
        assert_eq!(output.shape(), &[1, 3]);
        assert_eq!(output[[0, 0]], 1.0); // 0.3 < 0.5 -> left
        assert_eq!(output[[0, 1]], 2.0); // 0.7 >= 0.5 -> right
        assert_eq!(output[[0, 2]], 2.0); // 0.5 >= 0.5 -> right
    }

    #[test]
    fn simple_predictor_multiclass() {
        let mut forest = Forest::new(3);
        forest.push_tree(build_simple_tree(0.1, 0.9, 0.5), 0);
        forest.push_tree(build_simple_tree(0.2, 0.8, 0.5), 1);
        forest.push_tree(build_simple_tree(0.3, 0.7, 0.5), 2);

        let predictor = SimplePredictor::new(&forest);

        // Feature-major: [n_features=1, n_samples=2]
        let dataset = make_dataset(&[0.3, 0.7], 1, 2);
        let output = predictor.predict(&dataset, Parallelism::Sequential);

        // Output shape: [n_groups, n_samples] = [3, 2]
        assert_eq!(output.shape(), &[3, 2]);
        // Sample 0 (feature=0.3, goes left)
        assert_abs_diff_eq!(output[[0, 0]], 0.1, epsilon = 1e-6);
        assert_abs_diff_eq!(output[[1, 0]], 0.2, epsilon = 1e-6);
        assert_abs_diff_eq!(output[[2, 0]], 0.3, epsilon = 1e-6);
        // Sample 1 (feature=0.7, goes right)
        assert_abs_diff_eq!(output[[0, 1]], 0.9, epsilon = 1e-6);
        assert_abs_diff_eq!(output[[1, 1]], 0.8, epsilon = 1e-6);
        assert_abs_diff_eq!(output[[2, 1]], 0.7, epsilon = 1e-6);
    }

    #[test]
    fn multiple_trees_sum() {
        let mut forest = Forest::for_regression();
        forest.push_tree(build_simple_tree(1.0, 2.0, 0.5), 0);
        forest.push_tree(build_simple_tree(0.5, 1.5, 0.5), 0);

        let predictor = SimplePredictor::new(&forest);

        // Feature-major: [n_features=1, n_samples=2]
        let dataset = make_dataset(&[0.3, 0.7], 1, 2);
        let output = predictor.predict(&dataset, Parallelism::Sequential);

        assert_eq!(output[[0, 0]], 1.5); // 1.0 + 0.5
        assert_eq!(output[[0, 1]], 3.5); // 2.0 + 1.5
    }

    // =========================================================================
    // UnrolledPredictor tests
    // =========================================================================

    #[test]
    fn unrolled_predictor_matches_simple() {
        let mut forest = Forest::for_regression();
        forest.push_tree(build_simple_tree(1.0, 2.0, 0.5), 0);
        forest.push_tree(build_simple_tree(0.5, 1.5, 0.5), 0);

        let simple = SimplePredictor::new(&forest);
        let unrolled = UnrolledPredictor6::new(&forest);

        for n_samples in [1, 10, 64, 100, 128, 200] {
            let data: Vec<f32> = (0..n_samples)
                .map(|i| (i as f32) / (n_samples as f32))
                .collect();
            // Feature-major: [n_features=1, n_samples]
            let dataset = make_dataset(&data, 1, n_samples);

            let simple_output = simple.predict(&dataset, Parallelism::Sequential);
            let unrolled_output = unrolled.predict(&dataset, Parallelism::Sequential);

            assert_abs_diff_eq!(simple_output, unrolled_output, epsilon = 1e-6);
        }
    }

    #[test]
    fn unrolled_predictor_deeper_tree() {
        let mut forest = Forest::for_regression();
        forest.push_tree(build_deeper_tree(), 0);

        let simple = SimplePredictor::new(&forest);
        let unrolled = UnrolledPredictor4::new(&forest);

        // Feature-major: [n_features=2, n_samples=4]
        // Feature 0: [0.2, 0.2, 0.6, 0.6]  (split threshold 0.5)
        // Feature 1: [0.1, 0.5, 0.5, 0.9]  (split threshold 0.3 for left, 0.7 for right)
        let data = vec![
            0.2, 0.2, 0.6, 0.6, // feature 0 values
            0.1, 0.5, 0.5, 0.9, // feature 1 values
        ];
        let dataset = make_dataset(&data, 2, 4);

        let simple_output = simple.predict(&dataset, Parallelism::Sequential);
        let unrolled_output = unrolled.predict(&dataset, Parallelism::Sequential);

        // Expected leaves: [1.0, 2.0, 3.0, 4.0]
        assert_abs_diff_eq!(simple_output, unrolled_output, epsilon = 1e-6);
    }

    #[test]
    fn unrolled_predictor_weighted() {
        let mut forest = Forest::for_regression();
        forest.push_tree(build_simple_tree(1.0, 2.0, 0.5), 0);
        forest.push_tree(build_simple_tree(0.5, 1.5, 0.5), 0);

        let simple = SimplePredictor::new(&forest);
        let unrolled = UnrolledPredictor6::new(&forest);

        // Feature-major: [n_features=1, n_samples=2]
        let data = vec![0.3, 0.7];
        let dataset = make_dataset(&data, 1, 2);
        let weights = &[1.0, 0.5];

        let mut simple_output = Array2::<f32>::zeros((1, 2));
        let mut unrolled_output = Array2::<f32>::zeros((1, 2));

        simple.predict_into(
            &dataset,
            Some(weights),
            Parallelism::Sequential,
            simple_output.view_mut(),
        );
        unrolled.predict_into(
            &dataset,
            Some(weights),
            Parallelism::Sequential,
            unrolled_output.view_mut(),
        );

        assert_abs_diff_eq!(simple_output, unrolled_output, epsilon = 1e-6);
    }

    // =========================================================================
    // Block size tests
    // =========================================================================

    #[test]
    fn custom_block_size() {
        let mut forest = Forest::for_regression();
        forest.push_tree(build_simple_tree(1.0, 2.0, 0.5), 0);

        let predictor = SimplePredictor::new(&forest).with_block_size(16);
        assert_eq!(predictor.block_size(), 16);

        // Feature-major: [n_features=1, n_samples=100]
        let data = vec![0.3f32; 100];
        let dataset = make_dataset(&data, 1, 100);
        let output = predictor.predict(&dataset, Parallelism::Sequential);

        assert_eq!(output.shape(), &[1, 100]);
        for sample in 0..100 {
            assert_eq!(output[[0, sample]], 1.0);
        }
    }

    #[test]
    fn different_block_sizes_same_result() {
        let mut forest = Forest::for_regression();
        forest.push_tree(build_simple_tree(1.0, 2.0, 0.5), 0);
        forest.push_tree(build_simple_tree(0.5, 1.5, 0.5), 0);

        // Feature-major: [n_features=1, n_samples=200]
        let data: Vec<f32> = (0..200).map(|i| (i as f32) / 200.0).collect();
        let dataset = make_dataset(&data, 1, 200);

        let p16 = SimplePredictor::new(&forest).with_block_size(16);
        let p64 = SimplePredictor::new(&forest).with_block_size(64);
        let p128 = SimplePredictor::new(&forest).with_block_size(128);

        let o16 = p16.predict(&dataset, Parallelism::Sequential);
        let o64 = p64.predict(&dataset, Parallelism::Sequential);
        let o128 = p128.predict(&dataset, Parallelism::Sequential);

        assert_abs_diff_eq!(o16, o64, epsilon = 1e-6);
        assert_abs_diff_eq!(o64, o128, epsilon = 1e-6);
    }

    // =========================================================================
    // Edge cases
    // =========================================================================

    #[test]
    fn empty_input() {
        let mut forest = Forest::for_regression();
        forest.push_tree(build_simple_tree(1.0, 2.0, 0.5), 0);

        let predictor = SimplePredictor::new(&forest);

        // Feature-major: [n_features=1, n_samples=0]
        let dataset = make_dataset(&[], 1, 0);
        let output = predictor.predict(&dataset, Parallelism::Sequential);

        assert_eq!(output.shape(), &[1, 0]);
    }

    // =========================================================================
    // Parallel prediction tests
    // =========================================================================

    #[test]
    fn parallel_matches_sequential() {
        let mut forest = Forest::for_regression();
        forest.push_tree(build_simple_tree(1.0, 2.0, 0.5), 0);
        forest.push_tree(build_simple_tree(0.5, 1.5, 0.5), 0);
        forest.push_tree(build_deeper_tree(), 0);

        let simple = SimplePredictor::new(&forest);
        let unrolled = UnrolledPredictor6::new(&forest);

        for n_samples in [1, 10, 64, 100, 128, 200, 1000] {
            // Feature-major: [n_features=2, n_samples]
            let data: Vec<f32> = (0..n_samples * 2)
                .map(|i| (i as f32) / (n_samples as f32 * 2.0))
                .collect();
            let dataset = make_dataset(&data, 2, n_samples);

            let seq_simple = simple.predict(&dataset, Parallelism::Sequential);
            let par_simple = simple.predict(&dataset, Parallelism::Parallel);

            let seq_unrolled = unrolled.predict(&dataset, Parallelism::Sequential);
            let par_unrolled = unrolled.predict(&dataset, Parallelism::Parallel);

            assert_abs_diff_eq!(seq_simple, par_simple, epsilon = 1e-6);
            assert_abs_diff_eq!(seq_unrolled, par_unrolled, epsilon = 1e-6);
        }
    }

    #[test]
    fn parallel_weighted_matches_sequential() {
        let mut forest = Forest::for_regression();
        forest.push_tree(build_simple_tree(1.0, 2.0, 0.5), 0);
        forest.push_tree(build_simple_tree(0.5, 1.5, 0.5), 0);

        let predictor = UnrolledPredictor6::new(&forest);
        let weights = &[1.0, 0.5];

        // Feature-major: [n_features=1, n_samples=100]
        let data: Vec<f32> = (0..100).map(|i| (i as f32) / 100.0).collect();
        let dataset = make_dataset(&data, 1, 100);

        let mut seq_output = Array2::<f32>::zeros((1, 100));
        let mut par_output = Array2::<f32>::zeros((1, 100));

        predictor.predict_into(
            &dataset,
            Some(weights),
            Parallelism::Sequential,
            seq_output.view_mut(),
        );
        predictor.predict_into(
            &dataset,
            Some(weights),
            Parallelism::Parallel,
            par_output.view_mut(),
        );

        assert_abs_diff_eq!(seq_output, par_output, epsilon = 1e-6);
    }

    #[test]
    fn parallel_multiclass() {
        let mut forest = Forest::new(3);
        forest.push_tree(build_simple_tree(0.1, 0.9, 0.5), 0);
        forest.push_tree(build_simple_tree(0.2, 0.8, 0.5), 1);
        forest.push_tree(build_simple_tree(0.3, 0.7, 0.5), 2);

        let predictor = UnrolledPredictor6::new(&forest);

        // Feature-major: [n_features=1, n_samples=4]
        let dataset = make_dataset(&[0.3, 0.7, 0.4, 0.6], 1, 4);

        let seq_output = predictor.predict(&dataset, Parallelism::Sequential);
        let par_output = predictor.predict(&dataset, Parallelism::Parallel);

        assert_abs_diff_eq!(seq_output, par_output, epsilon = 1e-6);
    }

    #[test]
    fn parallel_empty_input() {
        let mut forest = Forest::for_regression();
        forest.push_tree(build_simple_tree(1.0, 2.0, 0.5), 0);

        let predictor = UnrolledPredictor6::new(&forest);

        // Feature-major: [n_features=1, n_samples=0]
        let dataset = make_dataset(&[], 1, 0);
        let output = predictor.predict(&dataset, Parallelism::Parallel);

        assert_eq!(output.shape(), &[1, 0]);
    }
}

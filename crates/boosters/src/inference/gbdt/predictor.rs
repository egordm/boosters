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
use crate::data::axis;
use crate::repr::gbdt::{Forest, ScalarLeaf, Tree, TreeView};
use ndarray::{Array2, ArrayView2, ArrayViewMut2};

use super::TreeTraversal;

/// Default block size for batch processing (matches XGBoost).
pub const DEFAULT_BLOCK_SIZE: usize = 64;

/// Compute the prediction value for a linear leaf.
///
/// Returns `intercept + Σ(coef × feature)`, or the base leaf value if any
/// linear feature is NaN.
#[inline]
fn compute_linear_leaf_value(tree: &Tree<ScalarLeaf>, leaf_idx: u32, features: &[f32]) -> f32 {
    let base = tree.leaf_value(leaf_idx).0;

    if let Some((feat_indices, coefs)) = tree.leaf_terms(leaf_idx) {
        // Check for NaN in any linear feature
        for &feat_idx in feat_indices {
            let val = features.get(feat_idx as usize).copied().unwrap_or(f32::NAN);
            if val.is_nan() {
                return base; // Fall back to constant leaf
            }
        }

        // Compute linear prediction: intercept + Σ(coef × feature)
        let intercept = tree.leaf_intercept(leaf_idx);
        let linear_sum: f32 = feat_indices
            .iter()
            .zip(coefs.iter())
            .map(|(&f, &c)| c * features.get(f as usize).copied().unwrap_or(0.0))
            .sum();

        intercept + linear_sum
    } else {
        base
    }
}

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

    pub fn predict_row_into(
        &self,
        features: &[f32],
        weights: Option<&[f32]>,
        output: &mut [f32],
    ) {
        assert_eq!(output.len(), self.n_groups(), "output length must equal n_groups");
        if let Some(w) = weights {
            assert_eq!(w.len(), self.forest.n_trees(), "weights length must match number of trees");
        }

        // Initialize with base scores
        output.copy_from_slice(self.forest.base_score());

        // Accumulate tree contributions
        for (tree_idx, (tree, group)) in self.forest.trees_with_groups().enumerate() {
            let state = &self.tree_states[tree_idx];

            let leaf_idx = T::traverse_tree(tree, state, features);

            let leaf_value = if tree.has_linear_leaves() {
                compute_linear_leaf_value(tree, leaf_idx, features)
            } else {
                tree.leaf_value(leaf_idx).0
            };

            let weighted_value = weights
                .map(|w| leaf_value * w[tree_idx])
                .unwrap_or(leaf_value);
            
            output[group as usize] += weighted_value;
        }
    }

    pub fn predict_into(
        &self,
        features: ArrayView2<f32>,
        weights: Option<&[f32]>,
        parallelism: Parallelism,
        mut output: ArrayViewMut2<f32>,
    ) {
        // features: [n_samples, n_features]
        // output: [n_groups, n_samples]
        let n_samples = features.nrows();
        let n_groups = self.n_groups();
        assert_eq!(output.shape(), &[n_groups, n_samples], "output shape must match (n_groups, n_samples)");
        if let Some(w) = weights {
            assert_eq!(w.len(), self.forest.n_trees(), "weights length must match number of trees");
        }

        if n_samples == 0 {
            return;
        }

        // Initialize with base scores (TODO: can create a helper for this pattern)
        ndarray::Zip::from(output.axis_iter_mut(axis::ROWS))
            .and(self.forest.base_score())
            .for_each(|mut col, &score| col.fill(score));

        // Process in blocks
        let feature_chunks = features.axis_chunks_iter(axis::ROWS, self.block_size);
        let output_chunks = output.axis_chunks_iter_mut(axis::COLS, self.block_size);
        let chunks_iter = feature_chunks.zip(output_chunks);

        parallelism.maybe_par_bridge_for_each(chunks_iter, |(feat_chunk, output_chunk)| {
            self.predict_block_into(feat_chunk, weights, output_chunk);
        });
    }

    /// Convenience method: predict with allocation.
    ///
    /// Allocates output buffer and calls `predict_into`. For maximum performance
    /// when reusing buffers, use `predict_into` directly.
    ///
    /// # Arguments
    ///
    /// * `features` - Feature matrix [n_samples, n_features]
    /// * `parallelism` - Whether to use parallel execution
    ///
    /// # Returns
    ///
    /// Prediction matrix [n_groups, n_samples]
    pub fn predict(&self, features: ArrayView2<f32>, parallelism: Parallelism) -> Array2<f32> {
        let n_samples = features.nrows();
        let n_groups = self.n_groups();
        let mut output = Array2::<f32>::zeros((n_groups, n_samples));
        self.predict_into(features, None, parallelism, output.view_mut());
        output
    }


    fn predict_block_into(
        &self,
        features: ArrayView2<f32>,
        weights: Option<&[f32]>,
        mut output: ArrayViewMut2<f32>,
    ) {
        let feature_data = features.as_slice().expect("features must be contiguous");
        let n_features = features.ncols();
        let n_rows = features.nrows();
        let mut leaf_indices = vec![0u32; n_rows];

        for (tree_idx, (tree, group)) in self.forest.trees_with_groups().enumerate() {
            let state = &self.tree_states[tree_idx];
            let group_idx = group as usize;
            let weight = weights.map(|w| w[tree_idx]).unwrap_or(1.0);

            // Step 1: Get leaf indices
            T::traverse_block(tree, state, feature_data, n_features, &mut leaf_indices);

            // Step 2: Extract values and accumulate into group row
            let mut group_row = output.row_mut(group_idx);
            if tree.has_linear_leaves() {
                for (i, feat_row) in features.axis_iter(axis::ROWS).enumerate() {
                    let value = compute_linear_leaf_value(
                        tree,
                        leaf_indices[i],
                        feat_row.as_slice().unwrap(),
                    );
                    group_row[i] += value * weight;
                }
            } else {
                for i in 0..n_rows {
                    group_row[i] += tree.leaf_value(leaf_indices[i]).0 * weight;
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
    use approx::assert_abs_diff_eq;
    use ndarray::Array2;
    use crate::repr::gbdt::{Forest, ScalarLeaf, Tree};

    fn build_simple_tree(
        left_val: f32,
        right_val: f32,
        threshold: f32,
    ) -> Tree<ScalarLeaf> {
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

    /// Helper: create ArrayView2 from row-major data
    fn features_view(data: &[f32], n_rows: usize, n_cols: usize) -> ArrayView2<'_, f32> {
        ArrayView2::from_shape((n_rows, n_cols), data).unwrap()
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
        predictor.predict_row_into(&[0.3], None, &mut output);
        assert_eq!(output, vec![1.0]);

        predictor.predict_row_into(&[0.7], None, &mut output);
        assert_eq!(output, vec![2.0]);
    }

    #[test]
    fn simple_predictor_with_base_score() {
        let mut forest = Forest::for_regression().with_base_score(vec![0.5]);
        forest.push_tree(build_simple_tree(1.0, 2.0, 0.5), 0);

        let predictor = SimplePredictor::new(&forest);

        let mut output = vec![0.0];
        predictor.predict_row_into(&[0.3], None, &mut output);
        assert_eq!(output, vec![1.5]);
    }

    #[test]
    fn simple_predictor_weighted() {
        let mut forest = Forest::for_regression();
        forest.push_tree(build_simple_tree(1.0, 2.0, 0.5), 0);
        forest.push_tree(build_simple_tree(0.5, 1.5, 0.5), 0);

        let predictor = SimplePredictor::new(&forest);

        let mut output = vec![0.0];
        predictor.predict_row_into(&[0.3], Some(&[1.0, 0.5]), &mut output);
        assert!((output[0] - 1.25).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "weights length must match number of trees")]
    fn weighted_wrong_length_panics() {
        let mut forest = Forest::for_regression();
        forest.push_tree(build_simple_tree(1.0, 2.0, 0.5), 0);
        forest.push_tree(build_simple_tree(0.5, 1.5, 0.5), 0);

        let predictor = SimplePredictor::new(&forest);
        let mut output = vec![0.0];
        predictor.predict_row_into(&[0.3], Some(&[1.0]), &mut output); // only 1 weight
    }

    // =========================================================================
    // Batch prediction tests
    // =========================================================================

    #[test]
    fn simple_predictor_batch() {
        let mut forest = Forest::for_regression();
        forest.push_tree(build_simple_tree(1.0, 2.0, 0.5), 0);

        let predictor = SimplePredictor::new(&forest);

        let features = features_view(&[0.3, 0.7, 0.5], 3, 1);
        let output = predictor.predict(features, Parallelism::Sequential);

        // Output shape: [n_groups, n_samples] = [1, 3]
        assert_eq!(output.shape(), &[1, 3]);
        assert_eq!(output[[0, 0]], 1.0);
        assert_eq!(output[[0, 1]], 2.0);
        assert_eq!(output[[0, 2]], 2.0);
    }

    #[test]
    fn simple_predictor_multiclass() {
        let mut forest = Forest::new(3);
        forest.push_tree(build_simple_tree(0.1, 0.9, 0.5), 0);
        forest.push_tree(build_simple_tree(0.2, 0.8, 0.5), 1);
        forest.push_tree(build_simple_tree(0.3, 0.7, 0.5), 2);

        let predictor = SimplePredictor::new(&forest);

        let features = features_view(&[0.3, 0.7], 2, 1);
        let output = predictor.predict(features, Parallelism::Sequential);

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

        let features = features_view(&[0.3, 0.7], 2, 1);
        let output = predictor.predict(features, Parallelism::Sequential);

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

        for num_rows in [1, 10, 64, 100, 128, 200] {
            let data: Vec<f32> = (0..num_rows)
                .map(|i| (i as f32) / (num_rows as f32))
                .collect();
            let features = features_view(&data, num_rows, 1);

            let simple_output = simple.predict(features, Parallelism::Sequential);
            let unrolled_output = unrolled.predict(features, Parallelism::Sequential);

            assert_abs_diff_eq!(simple_output, unrolled_output, epsilon = 1e-6);
        }
    }

    #[test]
    fn unrolled_predictor_deeper_tree() {
        let mut forest = Forest::for_regression();
        forest.push_tree(build_deeper_tree(), 0);

        let simple = SimplePredictor::new(&forest);
        let unrolled = UnrolledPredictor4::new(&forest);

        let data = vec![
            0.2, 0.1, // leaf 1.0
            0.2, 0.5, // leaf 2.0
            0.6, 0.5, // leaf 3.0
            0.6, 0.9, // leaf 4.0
        ];
        let features = features_view(&data, 4, 2);

        let simple_output = simple.predict(features, Parallelism::Sequential);
        let unrolled_output = unrolled.predict(features, Parallelism::Sequential);

        for sample in 0..4 {
            assert_abs_diff_eq!(
                simple_output[[0, sample]],
                unrolled_output[[0, sample]],
                epsilon = 1e-6
            );
        }
    }

    #[test]
    fn unrolled_predictor_weighted() {
        let mut forest = Forest::for_regression();
        forest.push_tree(build_simple_tree(1.0, 2.0, 0.5), 0);
        forest.push_tree(build_simple_tree(0.5, 1.5, 0.5), 0);

        let simple = SimplePredictor::new(&forest);
        let unrolled = UnrolledPredictor6::new(&forest);

        let data = vec![0.3, 0.7];
        let features = features_view(&data, 2, 1);
        let weights = &[1.0, 0.5];

        let mut simple_output = Array2::<f32>::zeros((1, 2));
        let mut unrolled_output = Array2::<f32>::zeros((1, 2));

        simple.predict_into(features, Some(weights), Parallelism::Sequential, simple_output.view_mut());
        unrolled.predict_into(features, Some(weights), Parallelism::Sequential, unrolled_output.view_mut());

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

        let data = vec![0.3f32; 100];
        let features = features_view(&data, 100, 1);
        let output = predictor.predict(features, Parallelism::Sequential);

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

        let data: Vec<f32> = (0..200).map(|i| (i as f32) / 200.0).collect();
        let features = features_view(&data, 200, 1);

        let p16 = SimplePredictor::new(&forest).with_block_size(16);
        let p64 = SimplePredictor::new(&forest).with_block_size(64);
        let p128 = SimplePredictor::new(&forest).with_block_size(128);

        let o16 = p16.predict(features, Parallelism::Sequential);
        let o64 = p64.predict(features, Parallelism::Sequential);
        let o128 = p128.predict(features, Parallelism::Sequential);

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

        let features = features_view(&[], 0, 1);
        let output = predictor.predict(features, Parallelism::Sequential);

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

        for num_rows in [1, 10, 64, 100, 128, 200, 1000] {
            let data: Vec<f32> = (0..num_rows * 2)
                .map(|i| (i as f32) / (num_rows as f32 * 2.0))
                .collect();
            let features = features_view(&data, num_rows, 2);

            let seq_simple = simple.predict(features, Parallelism::Sequential);
            let par_simple = simple.predict(features, Parallelism::Parallel);

            let seq_unrolled = unrolled.predict(features, Parallelism::Sequential);
            let par_unrolled = unrolled.predict(features, Parallelism::Parallel);

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

        let data: Vec<f32> = (0..100).map(|i| (i as f32) / 100.0).collect();
        let features = features_view(&data, 100, 1);

        let mut seq_output = Array2::<f32>::zeros((1, 100));
        let mut par_output = Array2::<f32>::zeros((1, 100));

        predictor.predict_into(features, Some(weights), Parallelism::Sequential, seq_output.view_mut());
        predictor.predict_into(features, Some(weights), Parallelism::Parallel, par_output.view_mut());

        assert_abs_diff_eq!(seq_output, par_output, epsilon = 1e-6);
    }

    #[test]
    fn parallel_multiclass() {
        let mut forest = Forest::new(3);
        forest.push_tree(build_simple_tree(0.1, 0.9, 0.5), 0);
        forest.push_tree(build_simple_tree(0.2, 0.8, 0.5), 1);
        forest.push_tree(build_simple_tree(0.3, 0.7, 0.5), 2);

        let predictor = UnrolledPredictor6::new(&forest);

        let features = features_view(&[0.3, 0.7, 0.4, 0.6], 4, 1);

        let seq_output = predictor.predict(features, Parallelism::Sequential);
        let par_output = predictor.predict(features, Parallelism::Parallel);

        assert_abs_diff_eq!(seq_output, par_output, epsilon = 1e-6);
    }

    #[test]
    fn parallel_empty_input() {
        let mut forest = Forest::for_regression();
        forest.push_tree(build_simple_tree(1.0, 2.0, 0.5), 0);

        let predictor = UnrolledPredictor6::new(&forest);

        let features = features_view(&[], 0, 1);
        let output = predictor.predict(features, Parallelism::Parallel);

        assert_eq!(output.shape(), &[1, 0]);
    }
}

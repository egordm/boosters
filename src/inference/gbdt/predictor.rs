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
//! use booste_rs::inference::gbdt::{Predictor, StandardTraversal, UnrolledTraversal6};
//! use booste_rs::data::DenseMatrix;
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

use crate::data::DataMatrix;
use super::Forest;
use super::ScalarLeaf;
use rayon::prelude::*;

use crate::inference::common::PredictionOutput;
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
    /// use booste_rs::inference::gbdt::{Predictor, UnrolledTraversal6};
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
            .map(|tree| T::build_tree_state(tree.into_storage()))
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
    pub fn num_groups(&self) -> usize {
        self.forest.num_groups() as usize
    }

    /// Predict for a batch of features.
    ///
    /// Returns a [`PredictionOutput`] with shape `(num_rows, num_groups)`.
    #[inline]
    pub fn predict<M: DataMatrix<Element = f32>>(&self, features: &M) -> PredictionOutput {
        self.predict_internal(features, None)
    }

    /// Predict with per-tree weights (for DART).
    ///
    /// Each tree's contribution is multiplied by its corresponding weight.
    /// This matches XGBoost's DART inference where `weight_drop[i]` scales tree `i`.
    ///
    /// # Panics
    ///
    /// Panics if `weights.len() != forest.num_trees()`.
    #[inline]
    pub fn predict_weighted<M: DataMatrix<Element = f32>>(
        &self,
        features: &M,
        weights: &[f32],
    ) -> PredictionOutput {
        assert_eq!(
            weights.len(),
            self.forest.num_trees(),
            "weights length must match number of trees"
        );
        self.predict_internal(features, Some(weights))
    }

    /// Parallel prediction for a batch of features.
    ///
    /// Uses Rayon to parallelize block processing across available CPU cores.
    /// Each block is processed independently, enabling work-stealing load balancing.
    ///
    /// Returns a [`PredictionOutput`] with shape `(num_rows, num_groups)`.
    ///
    /// # Performance
    ///
    /// Best for large batches (1000+ rows) on multi-core systems. For small batches
    /// or single-core systems, [`predict`](Self::predict) may be faster due to lower overhead.
    #[inline]
    pub fn par_predict<M: DataMatrix<Element = f32> + Sync>(
        &self,
        features: &M,
    ) -> PredictionOutput {
        self.par_predict_internal(features, None)
    }

    /// Parallel prediction with per-tree weights (for DART).
    ///
    /// Uses Rayon to parallelize block processing. Each tree's contribution
    /// is multiplied by its corresponding weight.
    ///
    /// # Panics
    ///
    /// Panics if `weights.len() != forest.num_trees()`.
    #[inline]
    pub fn par_predict_weighted<M: DataMatrix<Element = f32> + Sync>(
        &self,
        features: &M,
        weights: &[f32],
    ) -> PredictionOutput {
        assert_eq!(
            weights.len(),
            self.forest.num_trees(),
            "weights length must match number of trees"
        );
        self.par_predict_internal(features, Some(weights))
    }

    /// Internal parallel prediction with optional weights.
    fn par_predict_internal<M: DataMatrix<Element = f32> + Sync>(
        &self,
        features: &M,
        weights: Option<&[f32]>,
    ) -> PredictionOutput {
        let num_rows = features.num_rows();
        let num_groups = self.num_groups();
        let num_features = features.num_features();

        if num_rows == 0 {
            return PredictionOutput::zeros(0, num_groups);
        }

        let base_score = self.forest.base_score();

        // Split rows into blocks and process in parallel
        let blocks: Vec<_> = (0..num_rows)
            .step_by(self.block_size)
            .map(|block_start| {
                let block_end = (block_start + self.block_size).min(num_rows);
                (block_start, block_end)
            })
            .collect();

        // Process each block in parallel
        let block_outputs: Vec<_> = blocks
            .par_iter()
            .map(|&(block_start, block_end)| {
                let current_block_size = block_end - block_start;
                self.process_block_parallel(
                    features,
                    block_start,
                    current_block_size,
                    num_features,
                    num_groups,
                    base_score,
                    weights,
                )
            })
            .collect();

        // Combine results
        let mut output = PredictionOutput::zeros(num_rows, num_groups);
        for (block_idx, &(block_start, block_end)) in blocks.iter().enumerate() {
            let block_output = &block_outputs[block_idx];
            for i in 0..(block_end - block_start) {
                output.row_mut(block_start + i).copy_from_slice(block_output.row(i));
            }
        }

        output
    }

    /// Process a single block of rows for parallel prediction.
    ///
    /// Uses block-optimized traversal (`traverse_block`), which is most efficient for
    /// `UnrolledTraversal`. For `StandardTraversal`, the default `traverse_block`
    /// implementation falls back to per-row traversal.
    fn process_block_parallel<M: DataMatrix<Element = f32>>(
        &self,
        features: &M,
        block_start: usize,
        block_size: usize,
        num_features: usize,
        num_groups: usize,
        base_score: &[f32],
        weights: Option<&[f32]>,
    ) -> PredictionOutput {
        let mut block_output = PredictionOutput::zeros(block_size, num_groups);

        // Initialize with base scores
        for row_idx in 0..block_size {
            block_output.row_mut(row_idx).copy_from_slice(base_score);
        }

        // Load features for this block into contiguous buffer
        let mut feature_buffer = vec![f32::NAN; block_size * num_features];
        for i in 0..block_size {
            let buf_offset = i * num_features;
            features.copy_row(
                block_start + i,
                &mut feature_buffer[buf_offset..][..num_features],
            );
        }

        // Use block-optimized traversal
        let mut group_buffer = vec![0.0f32; block_size];

        for (tree_idx, (tree, group)) in self.forest.trees_with_groups().enumerate() {
            let state = &self.tree_states[tree_idx];
            let group_idx = group as usize;
            let weight = weights.map(|w| w[tree_idx]).unwrap_or(1.0);

            group_buffer[..block_size].fill(0.0);

            T::traverse_block(
                &tree,
                state,
                &feature_buffer[..block_size * num_features],
                num_features,
                &mut group_buffer[..block_size],
                weight,
            );

            for i in 0..block_size {
                block_output.row_mut(i)[group_idx] += group_buffer[i];
            }
        }

        block_output
    }

    /// Internal prediction with optional weights.
    #[inline]
    fn predict_internal<M: DataMatrix<Element = f32>>(
        &self,
        features: &M,
        weights: Option<&[f32]>,
    ) -> PredictionOutput {
        // Use block-optimized path only for traversals that benefit from it
        if T::USES_BLOCK_OPTIMIZATION {
            self.predict_block_optimized(features, weights)
        } else {
            self.predict_simple(features, weights)
        }
    }

    /// Simple per-row prediction (for StandardTraversal).
    #[inline]
    fn predict_simple<M: DataMatrix<Element = f32>>(
        &self,
        features: &M,
        weights: Option<&[f32]>,
    ) -> PredictionOutput {
        let num_rows = features.num_rows();
        let num_groups = self.num_groups();
        let num_features = features.num_features();

        let mut output = PredictionOutput::zeros(num_rows, num_groups);

        // Initialize with base scores
        let base_score = self.forest.base_score();
        for row_idx in 0..num_rows {
            output.row_mut(row_idx).copy_from_slice(base_score);
        }

        if num_rows == 0 {
            return output;
        }

        // Pre-allocate feature buffer for block processing
        let actual_block_size = self.block_size.min(num_rows);
        let mut feature_buffer = vec![f32::NAN; actual_block_size * num_features];

        // Process in blocks for cache efficiency
        for block_start in (0..num_rows).step_by(self.block_size) {
            let block_end = (block_start + self.block_size).min(num_rows);
            let current_block_size = block_end - block_start;

            // Load features for this block into contiguous buffer
            for i in 0..current_block_size {
                let buf_offset = i * num_features;
                features.copy_row(
                    block_start + i,
                    &mut feature_buffer[buf_offset..][..num_features],
                );
            }

            // Process all trees for this block - simple per-row accumulation
            for (tree_idx, (tree, group)) in self.forest.trees_with_groups().enumerate() {
                let state = &self.tree_states[tree_idx];
                let group_idx = group as usize;

                for i in 0..current_block_size {
                    let buf_offset = i * num_features;
                    let row_features = &feature_buffer[buf_offset..][..num_features];

                    let leaf_value = T::traverse_tree(&tree, state, row_features);

                    let value = match weights {
                        Some(w) => leaf_value.0 * w[tree_idx],
                        None => leaf_value.0,
                    };
                    output.row_mut(block_start + i)[group_idx] += value;
                }
            }
        }

        output
    }

    /// Block-optimized prediction (for UnrolledTraversal).
    #[inline]
    fn predict_block_optimized<M: DataMatrix<Element = f32>>(
        &self,
        features: &M,
        weights: Option<&[f32]>,
    ) -> PredictionOutput {
        let num_rows = features.num_rows();
        let num_groups = self.num_groups();
        let num_features = features.num_features();

        let mut output = PredictionOutput::zeros(num_rows, num_groups);

        // Initialize with base scores
        let base_score = self.forest.base_score();
        for row_idx in 0..num_rows {
            output.row_mut(row_idx).copy_from_slice(base_score);
        }

        if num_rows == 0 {
            return output;
        }

        // Pre-allocate buffers for block processing
        let actual_block_size = self.block_size.min(num_rows);
        let mut feature_buffer = vec![f32::NAN; actual_block_size * num_features];
        // Temporary buffer for accumulating one group's results
        let mut group_buffer = vec![0.0f32; actual_block_size];

        // Process in blocks for cache efficiency
        for block_start in (0..num_rows).step_by(self.block_size) {
            let block_end = (block_start + self.block_size).min(num_rows);
            let current_block_size = block_end - block_start;

            // Load features for this block into contiguous buffer
            for i in 0..current_block_size {
                let buf_offset = i * num_features;
                features.copy_row(
                    block_start + i,
                    &mut feature_buffer[buf_offset..][..num_features],
                );
            }

            // Process all trees for this block using traverse_block
            for (tree_idx, (tree, group)) in self.forest.trees_with_groups().enumerate() {
                let state = &self.tree_states[tree_idx];
                let group_idx = group as usize;
                let weight = weights.map(|w| w[tree_idx]).unwrap_or(1.0);

                // Reset group buffer
                group_buffer[..current_block_size].fill(0.0);

                // Traverse all rows through this tree
                T::traverse_block(
                    &tree,
                    state,
                    &feature_buffer[..current_block_size * num_features],
                    num_features,
                    &mut group_buffer[..current_block_size],
                    weight,
                );

                // Scatter results into output (row-major layout)
                for i in 0..current_block_size {
                    output.row_mut(block_start + i)[group_idx] += group_buffer[i];
                }
            }
        }

        output
    }

    /// Predict for a single row of features.
    ///
    /// Returns a vector with one value per output group.
    #[inline]
    pub fn predict_row(&self, features: &[f32]) -> Vec<f32> {
        let mut output: Vec<f32> = self.forest.base_score().to_vec();

        for (tree_idx, (tree, group)) in self.forest.trees_with_groups().enumerate() {
            let state = &self.tree_states[tree_idx];
            let leaf_value = T::traverse_tree(&tree, state, features);
            output[group as usize] += leaf_value.0;
        }

        output
    }

    /// Predict for a single row with per-tree weights (for DART).
    ///
    /// # Panics
    ///
    /// Panics if `weights.len() != forest.num_trees()`.
    #[inline]
    pub fn predict_row_weighted(&self, features: &[f32], weights: &[f32]) -> Vec<f32> {
        assert_eq!(
            weights.len(),
            self.forest.num_trees(),
            "weights length must match number of trees"
        );

        let mut output: Vec<f32> = self.forest.base_score().to_vec();

        for (tree_idx, (tree, group)) in self.forest.trees_with_groups().enumerate() {
            let state = &self.tree_states[tree_idx];
            let leaf_value = T::traverse_tree(&tree, state, features);
            output[group as usize] += leaf_value.0 * weights[tree_idx];
        }

        output
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
    use crate::data::RowMatrix;
    use crate::inference::gbdt::{Forest, TreeStorage, TreeBuilder, ScalarLeaf};

    fn build_simple_tree(
        left_val: f32,
        right_val: f32,
        threshold: f32,
    ) -> TreeStorage<ScalarLeaf> {
        let mut builder = TreeBuilder::new();
        builder.add_split(0, threshold, true, 1, 2);
        builder.add_leaf(ScalarLeaf(left_val));
        builder.add_leaf(ScalarLeaf(right_val));
        builder.build()
    }

    fn build_deeper_tree() -> TreeStorage<ScalarLeaf> {
        let mut builder = TreeBuilder::new();
        builder.add_split(0, 0.5, true, 1, 2);
        builder.add_split(1, 0.3, true, 3, 4);
        builder.add_split(1, 0.7, true, 5, 6);
        builder.add_leaf(ScalarLeaf(1.0));
        builder.add_leaf(ScalarLeaf(2.0));
        builder.add_leaf(ScalarLeaf(3.0));
        builder.add_leaf(ScalarLeaf(4.0));
        builder.build()
    }

    // =========================================================================
    // SimplePredictor tests
    // =========================================================================

    #[test]
    fn simple_predictor_single_row() {
        let mut forest = Forest::for_regression();
        forest.push_tree(build_simple_tree(1.0, 2.0, 0.5), 0);

        let predictor = SimplePredictor::new(&forest);

        assert_eq!(predictor.predict_row(&[0.3]), vec![1.0]);
        assert_eq!(predictor.predict_row(&[0.7]), vec![2.0]);
    }

    #[test]
    fn simple_predictor_batch() {
        let mut forest = Forest::for_regression();
        forest.push_tree(build_simple_tree(1.0, 2.0, 0.5), 0);

        let predictor = SimplePredictor::new(&forest);

        let features = RowMatrix::from_vec(vec![0.3, 0.7, 0.5], 3, 1);
        let output = predictor.predict(&features);

        assert_eq!(output.shape(), (3, 1));
        assert_eq!(output.row(0), &[1.0]);
        assert_eq!(output.row(1), &[2.0]);
        assert_eq!(output.row(2), &[2.0]);
    }

    #[test]
    fn simple_predictor_with_base_score() {
        let mut forest = Forest::for_regression().with_base_score(vec![0.5]);
        forest.push_tree(build_simple_tree(1.0, 2.0, 0.5), 0);

        let predictor = SimplePredictor::new(&forest);

        assert_eq!(predictor.predict_row(&[0.3]), vec![1.5]);
    }

    #[test]
    fn simple_predictor_multiclass() {
        let mut forest = Forest::new(3);
        forest.push_tree(build_simple_tree(0.1, 0.9, 0.5), 0);
        forest.push_tree(build_simple_tree(0.2, 0.8, 0.5), 1);
        forest.push_tree(build_simple_tree(0.3, 0.7, 0.5), 2);

        let predictor = SimplePredictor::new(&forest);

        let features = RowMatrix::from_vec(vec![0.3, 0.7], 2, 1);
        let output = predictor.predict(&features);

        assert_eq!(output.shape(), (2, 3));
        assert_eq!(output.row(0), &[0.1, 0.2, 0.3]);
        assert_eq!(output.row(1), &[0.9, 0.8, 0.7]);
    }

    #[test]
    fn simple_predictor_weighted() {
        let mut forest = Forest::for_regression();
        forest.push_tree(build_simple_tree(1.0, 2.0, 0.5), 0);
        forest.push_tree(build_simple_tree(0.5, 1.5, 0.5), 0);

        let predictor = SimplePredictor::new(&forest);

        let result = predictor.predict_row_weighted(&[0.3], &[1.0, 0.5]);
        assert!((result[0] - 1.25).abs() < 1e-6);
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
            let features = RowMatrix::from_vec(data, num_rows, 1);

            let simple_output = simple.predict(&features);
            let unrolled_output = unrolled.predict(&features);

            assert_abs_diff_eq!(
                simple_output,
                unrolled_output,
                epsilon = 1e-6,
            );
        }
    }

    #[test]
    fn unrolled_predictor_deeper_tree() {
        let mut forest = Forest::for_regression();
        forest.push_tree(build_deeper_tree(), 0);

        let simple = SimplePredictor::new(&forest);
        let unrolled = UnrolledPredictor4::new(&forest);

        let features = RowMatrix::from_vec(
            vec![
                0.2, 0.1, // leaf 1.0
                0.2, 0.5, // leaf 2.0
                0.6, 0.5, // leaf 3.0
                0.6, 0.9, // leaf 4.0
            ],
            4,
            2,
        );

        let simple_output = simple.predict(&features);
        let unrolled_output = unrolled.predict(&features);

        for row_idx in 0..4 {
            assert_eq!(
                simple_output.row(row_idx),
                unrolled_output.row(row_idx),
                "Mismatch at row {}",
                row_idx
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

        let features = RowMatrix::from_vec(vec![0.3, 0.7], 2, 1);
        let weights = &[1.0, 0.5];

        let simple_output = simple.predict_weighted(&features, weights);
        let unrolled_output = unrolled.predict_weighted(&features, weights);

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

        let features = RowMatrix::from_vec(vec![0.3; 100], 100, 1);
        let output = predictor.predict(&features);

        assert_eq!(output.shape(), (100, 1));
        for row_idx in 0..100 {
            assert_eq!(output.row(row_idx), &[1.0]);
        }
    }

    #[test]
    fn different_block_sizes_same_result() {
        let mut forest = Forest::for_regression();
        forest.push_tree(build_simple_tree(1.0, 2.0, 0.5), 0);
        forest.push_tree(build_simple_tree(0.5, 1.5, 0.5), 0);

        let data: Vec<f32> = (0..200).map(|i| (i as f32) / 200.0).collect();
        let features = RowMatrix::from_vec(data, 200, 1);

        let p16 = SimplePredictor::new(&forest).with_block_size(16);
        let p64 = SimplePredictor::new(&forest).with_block_size(64);
        let p128 = SimplePredictor::new(&forest).with_block_size(128);

        let o16 = p16.predict(&features);
        let o64 = p64.predict(&features);
        let o128 = p128.predict(&features);

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

        let features = RowMatrix::from_vec(vec![], 0, 1);
        let output = predictor.predict(&features);

        assert_eq!(output.shape(), (0, 1));
    }

    #[test]
    fn multiple_trees_sum() {
        let mut forest = Forest::for_regression();
        forest.push_tree(build_simple_tree(1.0, 2.0, 0.5), 0);
        forest.push_tree(build_simple_tree(0.5, 1.5, 0.5), 0);

        let predictor = SimplePredictor::new(&forest);

        let features = RowMatrix::from_vec(vec![0.3, 0.7], 2, 1);
        let output = predictor.predict(&features);

        assert_eq!(output.row(0), &[1.5]); // 1.0 + 0.5
        assert_eq!(output.row(1), &[3.5]); // 2.0 + 1.5
    }

    #[test]
    #[should_panic(expected = "weights length must match number of trees")]
    fn weighted_wrong_length_panics() {
        let mut forest = Forest::for_regression();
        forest.push_tree(build_simple_tree(1.0, 2.0, 0.5), 0);
        forest.push_tree(build_simple_tree(0.5, 1.5, 0.5), 0);

        let predictor = SimplePredictor::new(&forest);
        predictor.predict_row_weighted(&[0.3], &[1.0]); // only 1 weight for 2 trees
    }

    // =========================================================================
    // Parallel prediction tests
    // =========================================================================

    #[test]
    fn par_predict_matches_sequential() {
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
            let features = RowMatrix::from_vec(data, num_rows, 2);

            let seq_simple = simple.predict(&features);
            let par_simple = simple.par_predict(&features);

            let seq_unrolled = unrolled.predict(&features);
            let par_unrolled = unrolled.par_predict(&features);

            assert_abs_diff_eq!(seq_simple, par_simple, epsilon = 1e-6);
            assert_abs_diff_eq!(seq_unrolled, par_unrolled, epsilon = 1e-6);
        }
    }

    #[test]
    fn par_predict_weighted_matches_sequential() {
        let mut forest = Forest::for_regression();
        forest.push_tree(build_simple_tree(1.0, 2.0, 0.5), 0);
        forest.push_tree(build_simple_tree(0.5, 1.5, 0.5), 0);

        let predictor = UnrolledPredictor6::new(&forest);
        let weights = &[1.0, 0.5];

        let data: Vec<f32> = (0..100).map(|i| (i as f32) / 100.0).collect();
        let features = RowMatrix::from_vec(data, 100, 1);

        let seq_output = predictor.predict_weighted(&features, weights);
        let par_output = predictor.par_predict_weighted(&features, weights);

        assert_abs_diff_eq!(seq_output, par_output, epsilon = 1e-6);
    }

    #[test]
    fn par_predict_multiclass() {
        let mut forest = Forest::new(3);
        forest.push_tree(build_simple_tree(0.1, 0.9, 0.5), 0);
        forest.push_tree(build_simple_tree(0.2, 0.8, 0.5), 1);
        forest.push_tree(build_simple_tree(0.3, 0.7, 0.5), 2);

        let predictor = UnrolledPredictor6::new(&forest);

        let features = RowMatrix::from_vec(vec![0.3, 0.7, 0.4, 0.6], 4, 1);

        let seq_output = predictor.predict(&features);
        let par_output = predictor.par_predict(&features);

        assert_abs_diff_eq!(seq_output, par_output, epsilon = 1e-6);
    }

    #[test]
    fn par_predict_empty_input() {
        let mut forest = Forest::for_regression();
        forest.push_tree(build_simple_tree(1.0, 2.0, 0.5), 0);

        let predictor = UnrolledPredictor6::new(&forest);

        let features = RowMatrix::from_vec(vec![], 0, 1);
        let output = predictor.par_predict(&features);

        assert_eq!(output.shape(), (0, 1));
    }
}

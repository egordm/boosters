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

use crate::data::DataMatrix;
use crate::repr::gbdt::{Forest, ScalarLeaf, Tree, TreeView};
use rayon::prelude::*;

use crate::inference::common::PredictionOutput;
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
    /// Panics if `weights.len() != forest.n_trees()`.
    #[inline]
    pub fn predict_weighted<M: DataMatrix<Element = f32>>(
        &self,
        features: &M,
        weights: &[f32],
    ) -> PredictionOutput {
        assert_eq!(
            weights.len(),
            self.forest.n_trees(),
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
    /// # Arguments
    /// * `features` - Input feature matrix
    /// * `n_threads` - Number of threads to use:
    ///   - `0`: Auto-detect (use Rayon default based on available cores)
    ///   - `1`: Serial execution (no parallelism, avoids Rayon overhead)
    ///   - `>1`: Use exactly this many threads
    ///
    /// # Performance
    ///
    /// Best for large batches (1000+ rows) on multi-core systems. For small batches
    /// or single-core systems, use `n_threads=1` to avoid parallelism overhead.
    #[inline]
    pub fn par_predict<M: DataMatrix<Element = f32> + Sync>(
        &self,
        features: &M,
        n_threads: usize,
    ) -> PredictionOutput {
        self.par_predict_internal(features, None, n_threads)
    }

    /// Parallel prediction with per-tree weights (for DART).
    ///
    /// Uses Rayon to parallelize block processing. Each tree's contribution
    /// is multiplied by its corresponding weight.
    ///
    /// # Arguments
    /// * `features` - Input feature matrix
    /// * `weights` - Per-tree weights (length must equal number of trees)
    /// * `n_threads` - Number of threads (0=auto, 1=serial, >1=exact)
    ///
    /// # Panics
    ///
    /// Panics if `weights.len() != forest.n_trees()`.
    #[inline]
    pub fn par_predict_weighted<M: DataMatrix<Element = f32> + Sync>(
        &self,
        features: &M,
        weights: &[f32],
        n_threads: usize,
    ) -> PredictionOutput {
        assert_eq!(
            weights.len(),
            self.forest.n_trees(),
            "weights length must match number of trees"
        );
        self.par_predict_internal(features, Some(weights), n_threads)
    }

    /// Internal parallel prediction with optional weights and thread control.
    fn par_predict_internal<M: DataMatrix<Element = f32> + Sync>(
        &self,
        features: &M,
        weights: Option<&[f32]>,
        n_threads: usize,
    ) -> PredictionOutput {
        let num_rows = features.num_rows();
        let num_groups = self.n_groups();
        let num_features = features.num_features();

        if num_rows == 0 {
            return PredictionOutput::zeros(0, num_groups);
        }

        let base_score = self.forest.base_score();

        // n_threads == 1 means serial execution
        if n_threads == 1 {
            return self.predict_internal(features, weights);
        }

        // Split rows into blocks and process in parallel
        let blocks: Vec<_> = (0..num_rows)
            .step_by(self.block_size)
            .map(|block_start| {
                let block_end = (block_start + self.block_size).min(num_rows);
                (block_start, block_end)
            })
            .collect();

        // Closure to process blocks in parallel
        let process_blocks = || {
            blocks
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
                .collect::<Vec<_>>()
        };

        // Process blocks with optional thread pool
        // n_threads == 0 means auto (use Rayon default)
        let block_outputs = if n_threads == 0 {
            process_blocks()
        } else {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n_threads)
                .build()
                .expect("Failed to create thread pool");
            pool.install(process_blocks)
        };

        // Combine results
        let mut output = PredictionOutput::zeros(num_rows, num_groups);
        for (block_idx, &(block_start, block_end)) in blocks.iter().enumerate() {
            let block_output = &block_outputs[block_idx];
            let block_size = block_end - block_start;
            for group_idx in 0..num_groups {
                let out_col = output.column_mut(group_idx);
                let blk_col = block_output.column(group_idx);
                out_col[block_start..block_end].copy_from_slice(&blk_col[..block_size]);
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

        // Initialize with base scores (column-major: efficient fill per group)
        for (group_idx, &score) in base_score.iter().enumerate() {
            block_output.column_mut(group_idx).fill(score);
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

            // Column-major: add to output column (contiguous in memory)
            let out_col = block_output.column_mut(group_idx);
            for i in 0..block_size {
                out_col[i] += group_buffer[i];
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
        let num_groups = self.n_groups();
        let num_features = features.num_features();

        let mut output = PredictionOutput::zeros(num_rows, num_groups);

        // Initialize with base scores (column-major: efficient fill per group)
        let base_score = self.forest.base_score();
        for (group_idx, &score) in base_score.iter().enumerate() {
            output.column_mut(group_idx).fill(score);
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

                    let leaf_value = if tree.has_linear_leaves() {
                        // Linear tree: compute linear value
                        let leaf_idx = super::traversal::traverse_from_node(&tree, 0, row_features);
                        compute_linear_leaf_value(&tree, leaf_idx, row_features)
                    } else {
                        // Standard: use traversal result
                        T::traverse_tree(&tree, state, row_features).0
                    };

                    let value = match weights {
                        Some(w) => leaf_value * w[tree_idx],
                        None => leaf_value,
                    };
                    output.add(block_start + i, group_idx, value);
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
        let num_groups = self.n_groups();
        let num_features = features.num_features();

        let mut output = PredictionOutput::zeros(num_rows, num_groups);

        // Initialize with base scores (column-major: efficient fill per group)
        let base_score = self.forest.base_score();
        for (group_idx, &score) in base_score.iter().enumerate() {
            output.column_mut(group_idx).fill(score);
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

                if tree.has_linear_leaves() {
                    // Linear tree path: traverse to get leaf index, then compute linear value
                    for i in 0..current_block_size {
                        let row_offset = i * num_features;
                        let row_features = &feature_buffer[row_offset..][..num_features];
                        let leaf_idx = super::traversal::traverse_from_node(&tree, 0, row_features);
                        let value = compute_linear_leaf_value(&tree, leaf_idx, row_features);
                        group_buffer[i] = value * weight;
                    }
                } else {
                    // Standard path: use optimized traversal
                    T::traverse_block(
                        &tree,
                        state,
                        &feature_buffer[..current_block_size * num_features],
                        num_features,
                        &mut group_buffer[..current_block_size],
                        weight,
                    );
                }

                // Scatter results into output (column-major: contiguous writes)
                let out_col = output.column_mut(group_idx);
                for i in 0..current_block_size {
                    out_col[block_start + i] += group_buffer[i];
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
            let leaf_value = if tree.has_linear_leaves() {
                let leaf_idx = super::traversal::traverse_from_node(&tree, 0, features);
                compute_linear_leaf_value(&tree, leaf_idx, features)
            } else {
                T::traverse_tree(&tree, state, features).0
            };
            output[group as usize] += leaf_value;
        }

        output
    }

    /// Predict for a single row with per-tree weights (for DART).
    ///
    /// # Panics
    ///
    /// Panics if `weights.len() != forest.n_trees()`.
    #[inline]
    pub fn predict_row_weighted(&self, features: &[f32], weights: &[f32]) -> Vec<f32> {
        assert_eq!(
            weights.len(),
            self.forest.n_trees(),
            "weights length must match number of trees"
        );

        let mut output: Vec<f32> = self.forest.base_score().to_vec();

        for (tree_idx, (tree, group)) in self.forest.trees_with_groups().enumerate() {
            let state = &self.tree_states[tree_idx];
            let leaf_value = if tree.has_linear_leaves() {
                let leaf_idx = super::traversal::traverse_from_node(&tree, 0, features);
                compute_linear_leaf_value(&tree, leaf_idx, features)
            } else {
                T::traverse_tree(&tree, state, features).0
            };
            output[group as usize] += leaf_value * weights[tree_idx];
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
        assert_eq!(output.row_vec(0), vec![1.0]);
        assert_eq!(output.row_vec(1), vec![2.0]);
        assert_eq!(output.row_vec(2), vec![2.0]);
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
        assert_eq!(output.row_vec(0), vec![0.1, 0.2, 0.3]);
        assert_eq!(output.row_vec(1), vec![0.9, 0.8, 0.7]);
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
                simple_output.row_vec(row_idx),
                unrolled_output.row_vec(row_idx),
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
            assert_eq!(output.get(row_idx, 0), 1.0);
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

        assert_eq!(output.row_vec(0), vec![1.5]); // 1.0 + 0.5
        assert_eq!(output.row_vec(1), vec![3.5]); // 2.0 + 1.5
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
            let par_simple = simple.par_predict(&features, 0); // 0 = auto threads

            let seq_unrolled = unrolled.predict(&features);
            let par_unrolled = unrolled.par_predict(&features, 0);

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
        let par_output = predictor.par_predict_weighted(&features, weights, 0);

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
        let par_output = predictor.par_predict(&features, 0);

        assert_abs_diff_eq!(seq_output, par_output, epsilon = 1e-6);
    }

    #[test]
    fn par_predict_empty_input() {
        let mut forest = Forest::for_regression();
        forest.push_tree(build_simple_tree(1.0, 2.0, 0.5), 0);

        let predictor = UnrolledPredictor6::new(&forest);

        let features = RowMatrix::from_vec(vec![], 0, 1);
        let output = predictor.par_predict(&features, 0);

        assert_eq!(output.shape(), (0, 1));
    }

    #[test]
    fn par_predict_serial_matches_parallel() {
        let mut forest = Forest::for_regression();
        forest.push_tree(build_deeper_tree(), 0);

        let predictor = UnrolledPredictor6::new(&forest);
        let data: Vec<f32> = (0..200).map(|i| (i as f32) / 200.0).collect();
        let features = RowMatrix::from_vec(data, 100, 2);

        let parallel = predictor.par_predict(&features, 0);  // auto
        let serial = predictor.par_predict(&features, 1);    // serial
        let two_threads = predictor.par_predict(&features, 2); // exact

        assert_abs_diff_eq!(parallel, serial, epsilon = 1e-6);
        assert_abs_diff_eq!(parallel, two_threads, epsilon = 1e-6);
    }
}

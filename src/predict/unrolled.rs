//! Unrolled-layout based predictor for optimized batch prediction.
//!
//! This module provides `UnrolledPredictor` which uses `UnrolledTreeLayout` to
//! accelerate batch prediction by:
//!
//! 1. Unrolling top tree levels for cache-friendly traversal
//! 2. Processing all rows through the same tree level together
//! 3. Only falling back to regular traversal for deep nodes
//!
//! # When to Use
//!
//! - **Use `UnrolledPredictor`**: For batch prediction (100+ rows)
//! - **Use `Predictor`**: For single-row or very small batches
//!
//! The overhead of building `UnrolledTreeLayout` is amortized over many rows.

use crate::data::DataMatrix;
use crate::forest::SoAForest;
use crate::trees::{UnrolledTreeLayout, LeafValue, ScalarLeaf, MAX_UNROLL_DEPTH};

use super::output::PredictionOutput;

/// Predictor using unrolled-layout optimization for batch processing.
///
/// Builds `UnrolledTreeLayout` for each tree on first prediction, then
/// uses level-by-level traversal for the unrolled portion.
#[derive(Debug)]
pub struct UnrolledPredictor<'f, L: LeafValue = ScalarLeaf> {
    forest: &'f SoAForest<L>,
    /// Cached unrolled layouts for each tree (built lazily or eagerly)
    layouts: Vec<UnrolledTreeLayout>,
    /// Unroll depth used for layouts
    unroll_depth: usize,
}

impl<'f, L: LeafValue> UnrolledPredictor<'f, L> {
    /// Create a new unrolled predictor with default unroll depth (6 levels).
    pub fn new(forest: &'f SoAForest<L>) -> Self {
        Self::with_depth(forest, MAX_UNROLL_DEPTH)
    }

    /// Create a new unrolled predictor with custom unroll depth.
    pub fn with_depth(forest: &'f SoAForest<L>, unroll_depth: usize) -> Self {
        // Build layouts for all trees upfront
        let layouts = forest
            .trees()
            .map(|tree| UnrolledTreeLayout::from_tree(&tree.into_storage(), unroll_depth))
            .collect();

        Self {
            forest,
            layouts,
            unroll_depth,
        }
    }

    /// Get the unroll depth.
    #[inline]
    pub fn unroll_depth(&self) -> usize {
        self.unroll_depth
    }

    /// Get reference to the underlying forest.
    #[inline]
    pub fn forest(&self) -> &SoAForest<L> {
        self.forest
    }

    /// Number of output groups.
    #[inline]
    pub fn num_groups(&self) -> usize {
        self.forest.num_groups() as usize
    }
}

impl<'f> UnrolledPredictor<'f, ScalarLeaf> {
    /// Predict for a batch of features using unrolled-layout optimization.
    ///
    /// Returns a [`PredictionOutput`] with shape `(num_rows, num_groups)`.
    pub fn predict<M: DataMatrix<Element = f32>>(&self, features: &M) -> PredictionOutput {
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

        // Pre-allocate buffers
        let mut feature_buffer = vec![f32::NAN; num_rows * num_features];
        let mut exit_indices = vec![0usize; num_rows];

        // Load all features into contiguous buffer
        for row_idx in 0..num_rows {
            let buf_offset = row_idx * num_features;
            features.copy_row(row_idx, &mut feature_buffer[buf_offset..][..num_features]);
        }

        // Process each tree
        for (tree_idx, (tree, group)) in self.forest.trees_with_groups().enumerate() {
            let layout = &self.layouts[tree_idx];
            let group_idx = group as usize;

            // Phase 1: Traverse unrolled levels for all rows
            layout.process_block(&feature_buffer, num_features, &mut exit_indices);

            // Phase 2: Continue from exit nodes to leaves and accumulate
            for row_idx in 0..num_rows {
                let exit_idx = exit_indices[row_idx];
                let mut node_idx = layout.exit_node_idx(exit_idx);

                // Get features for this row
                let row_offset = row_idx * num_features;
                let row_features = &feature_buffer[row_offset..][..num_features];

                // Continue traversal from exit node if not already at a leaf
                while !tree.is_leaf(node_idx) {
                    let feat_idx = tree.split_index(node_idx) as usize;
                    let fvalue = row_features.get(feat_idx).copied().unwrap_or(f32::NAN);

                    node_idx = if fvalue.is_nan() {
                        if tree.default_left(node_idx) {
                            tree.left_child(node_idx)
                        } else {
                            tree.right_child(node_idx)
                        }
                    } else if fvalue < tree.split_threshold(node_idx) {
                        tree.left_child(node_idx)
                    } else {
                        tree.right_child(node_idx)
                    };
                }

                // Accumulate leaf value
                output.row_mut(row_idx)[group_idx] += tree.leaf_value(node_idx).0;
            }
        }

        output
    }

    /// Predict with per-tree weights (for DART).
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

        // Pre-allocate buffers
        let mut feature_buffer = vec![f32::NAN; num_rows * num_features];
        let mut exit_indices = vec![0usize; num_rows];

        // Load all features into contiguous buffer
        for row_idx in 0..num_rows {
            let buf_offset = row_idx * num_features;
            features.copy_row(row_idx, &mut feature_buffer[buf_offset..][..num_features]);
        }

        // Process each tree
        for (tree_idx, (tree, group)) in self.forest.trees_with_groups().enumerate() {
            let layout = &self.layouts[tree_idx];
            let weight = weights[tree_idx];
            let group_idx = group as usize;

            // Phase 1: Traverse unrolled levels for all rows
            layout.process_block(&feature_buffer, num_features, &mut exit_indices);

            // Phase 2: Continue from exit nodes to leaves and accumulate
            for row_idx in 0..num_rows {
                let exit_idx = exit_indices[row_idx];
                let mut node_idx = layout.exit_node_idx(exit_idx);

                let row_offset = row_idx * num_features;
                let row_features = &feature_buffer[row_offset..][..num_features];

                while !tree.is_leaf(node_idx) {
                    let feat_idx = tree.split_index(node_idx) as usize;
                    let fvalue = row_features.get(feat_idx).copied().unwrap_or(f32::NAN);

                    node_idx = if fvalue.is_nan() {
                        if tree.default_left(node_idx) {
                            tree.left_child(node_idx)
                        } else {
                            tree.right_child(node_idx)
                        }
                    } else if fvalue < tree.split_threshold(node_idx) {
                        tree.left_child(node_idx)
                    } else {
                        tree.right_child(node_idx)
                    };
                }

                output.row_mut(row_idx)[group_idx] += tree.leaf_value(node_idx).0 * weight;
            }
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::DenseMatrix;
    use crate::forest::SoAForest;
    use crate::predict::Predictor;
    use crate::trees::{ScalarLeaf, TreeBuilder};

    fn build_simple_tree(
        left_val: f32,
        right_val: f32,
        threshold: f32,
    ) -> crate::trees::SoATreeStorage<ScalarLeaf> {
        let mut builder = TreeBuilder::new();
        builder.add_split(0, threshold, true, 1, 2);
        builder.add_leaf(ScalarLeaf(left_val));
        builder.add_leaf(ScalarLeaf(right_val));
        builder.build()
    }

    fn build_deeper_tree() -> crate::trees::SoATreeStorage<ScalarLeaf> {
        // Build a tree with depth 4 to test continuation after array layout
        //           [0] f0 < 0.5
        //          /           \
        //      [1] f1<0.3    [2] f1<0.7
        //      /    \        /    \
        //    [3]   [4]     [5]   [6]
        //   leaf  leaf    leaf  leaf
        let mut builder = TreeBuilder::new();
        builder.add_split(0, 0.5, true, 1, 2); // root
        builder.add_split(1, 0.3, true, 3, 4); // left subtree
        builder.add_split(1, 0.7, true, 5, 6); // right subtree
        builder.add_leaf(ScalarLeaf(1.0)); // node 3
        builder.add_leaf(ScalarLeaf(2.0)); // node 4
        builder.add_leaf(ScalarLeaf(3.0)); // node 5
        builder.add_leaf(ScalarLeaf(4.0)); // node 6
        builder.build()
    }

    #[test]
    fn unrolled_predictor_matches_regular() {
        let mut forest = SoAForest::for_regression();
        forest.push_tree(build_simple_tree(1.0, 2.0, 0.5), 0);
        forest.push_tree(build_simple_tree(0.5, 1.5, 0.5), 0);

        let regular = Predictor::new(&forest);
        let unrolled = UnrolledPredictor::new(&forest);

        // Test with various batch sizes
        for num_rows in [1, 10, 64, 100, 128, 200] {
            let data: Vec<f32> = (0..num_rows).map(|i| (i as f32) / (num_rows as f32)).collect();
            let features = DenseMatrix::from_vec(data, num_rows, 1);

            let regular_output = regular.predict(&features);
            let unrolled_output = unrolled.predict(&features);

            assert_eq!(regular_output.shape(), unrolled_output.shape());
            for row_idx in 0..num_rows {
                let r = regular_output.row(row_idx);
                let u = unrolled_output.row(row_idx);
                assert!(
                    (r[0] - u[0]).abs() < 1e-6,
                    "Mismatch at row {} with {} total rows: {:?} vs {:?}",
                    row_idx,
                    num_rows,
                    r,
                    u
                );
            }
        }
    }

    #[test]
    fn unrolled_predictor_deeper_tree() {
        let mut forest = SoAForest::for_regression();
        forest.push_tree(build_deeper_tree(), 0);

        let regular = Predictor::new(&forest);
        // Use depth 2 to test continuation after unrolled layout
        let unrolled = UnrolledPredictor::with_depth(&forest, 2);

        let features = DenseMatrix::from_vec(
            vec![
                0.2, 0.1, // row 0: f0<0.5 (left), f1<0.3 (left) → leaf 1.0
                0.2, 0.5, // row 1: f0<0.5 (left), f1>=0.3 (right) → leaf 2.0
                0.6, 0.5, // row 2: f0>=0.5 (right), f1<0.7 (left) → leaf 3.0
                0.6, 0.9, // row 3: f0>=0.5 (right), f1>=0.7 (right) → leaf 4.0
            ],
            4,
            2,
        );

        let regular_output = regular.predict(&features);
        let unrolled_output = unrolled.predict(&features);

        assert_eq!(regular_output.shape(), unrolled_output.shape());
        for row_idx in 0..4 {
            assert_eq!(
                regular_output.row(row_idx),
                unrolled_output.row(row_idx),
                "Mismatch at row {}",
                row_idx
            );
        }
    }

    #[test]
    fn unrolled_predictor_multiclass() {
        let mut forest = SoAForest::new(3);
        forest.push_tree(build_simple_tree(0.1, 0.9, 0.5), 0);
        forest.push_tree(build_simple_tree(0.2, 0.8, 0.5), 1);
        forest.push_tree(build_simple_tree(0.3, 0.7, 0.5), 2);

        let unrolled = UnrolledPredictor::new(&forest);

        let features = DenseMatrix::from_vec(vec![0.3, 0.7], 2, 1);
        let output = unrolled.predict(&features);

        assert_eq!(output.shape(), (2, 3));
        assert_eq!(output.row(0), &[0.1, 0.2, 0.3]); // all go left
        assert_eq!(output.row(1), &[0.9, 0.8, 0.7]); // all go right
    }

    #[test]
    fn unrolled_predictor_weighted() {
        let mut forest = SoAForest::for_regression();
        forest.push_tree(build_simple_tree(1.0, 2.0, 0.5), 0);
        forest.push_tree(build_simple_tree(0.5, 1.5, 0.5), 0);

        let regular = Predictor::new(&forest);
        let unrolled = UnrolledPredictor::new(&forest);

        let features = DenseMatrix::from_vec(vec![0.3, 0.7], 2, 1);
        let weights = &[1.0, 0.5];

        let regular_output = regular.predict_weighted(&features, weights);
        let unrolled_output = unrolled.predict_weighted(&features, weights);

        for row_idx in 0..2 {
            let r = regular_output.row(row_idx);
            let u = unrolled_output.row(row_idx);
            assert!(
                (r[0] - u[0]).abs() < 1e-6,
                "Mismatch at row {}: {:?} vs {:?}",
                row_idx,
                r,
                u
            );
        }
    }

    #[test]
    fn unrolled_predictor_empty_input() {
        let mut forest = SoAForest::for_regression();
        forest.push_tree(build_simple_tree(1.0, 2.0, 0.5), 0);

        let unrolled = UnrolledPredictor::new(&forest);

        let features = DenseMatrix::from_vec(vec![], 0, 1);
        let output = unrolled.predict(&features);

        assert_eq!(output.shape(), (0, 1));
    }
}

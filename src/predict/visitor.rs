//! Tree visitor traits and implementations.

use crate::data::DataMatrix;
use crate::forest::{SoAForest, SoATreeView};
use crate::trees::{LeafValue, ScalarLeaf};

use super::output::PredictionOutput;

/// Visitor for traversing a single tree with a single row of features.
///
/// This is the core abstraction for tree traversal. Implementations can
/// specialize for different leaf types or traversal strategies.
pub trait TreeVisitor<L: LeafValue> {
    /// Output type from visiting a tree.
    type Output;

    /// Visit a tree with given features, return result.
    fn visit_tree(&self, tree: &SoATreeView<'_, L>, features: &[f32]) -> Self::Output;
}

/// Visitor for forests with scalar leaves.
///
/// Traverses from root to leaf, handling missing values via default direction.
/// Uses const generics for specialization (future: categorical support).
#[derive(Debug, Clone, Copy, Default)]
pub struct ScalarVisitor;

impl TreeVisitor<ScalarLeaf> for ScalarVisitor {
    type Output = f32;

    #[inline]
    fn visit_tree(&self, tree: &SoATreeView<'_, ScalarLeaf>, features: &[f32]) -> f32 {
        let mut idx = 0u32;

        while !tree.is_leaf(idx) {
            let feat_idx = tree.split_index(idx) as usize;
            let threshold = tree.split_threshold(idx);
            let fvalue = features.get(feat_idx).copied().unwrap_or(f32::NAN);

            idx = if fvalue.is_nan() {
                // Missing value: use default direction
                if tree.default_left(idx) {
                    tree.left_child(idx)
                } else {
                    tree.right_child(idx)
                }
            } else if fvalue < threshold {
                tree.left_child(idx)
            } else {
                tree.right_child(idx)
            };
        }

        tree.leaf_value(idx).0
    }
}

/// Predictor that orchestrates batch prediction over a forest.
///
/// Wraps a forest and provides batch prediction with proper output formatting.
///
/// # Example
///
/// ```ignore
/// use booste_rs::predict::Predictor;
/// use booste_rs::data::DenseMatrix;
///
/// let predictor = Predictor::new(&forest);
/// let features = DenseMatrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
/// let output = predictor.predict(&features);
/// ```
#[derive(Debug, Clone)]
pub struct Predictor<'f, L: LeafValue = ScalarLeaf> {
    forest: &'f SoAForest<L>,
}

impl<'f, L: LeafValue> Predictor<'f, L> {
    /// Create a new predictor for the given forest.
    pub fn new(forest: &'f SoAForest<L>) -> Self {
        Self { forest }
    }

    /// Get a reference to the underlying forest.
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

impl<'f> Predictor<'f, ScalarLeaf> {
    /// Predict for a batch of features.
    ///
    /// Returns a [`PredictionOutput`] with shape `(num_rows, num_groups)`.
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

    /// Internal prediction with optional weights.
    fn predict_internal<M: DataMatrix<Element = f32>>(
        &self,
        features: &M,
        weights: Option<&[f32]>,
    ) -> PredictionOutput {
        let num_rows = features.num_rows();
        let num_groups = self.num_groups();
        let num_features = features.num_features();

        let mut output = PredictionOutput::zeros(num_rows, num_groups);
        let visitor = ScalarVisitor;

        // Add base scores first
        let base_score = self.forest.base_score();
        for row_idx in 0..num_rows {
            let out_row = output.row_mut(row_idx);
            out_row.copy_from_slice(base_score);
        }

        // Buffer for copying row features
        let mut row_buf = vec![f32::NAN; num_features];

        // Accumulate tree contributions
        for row_idx in 0..num_rows {
            // Copy row into buffer
            features.copy_row(row_idx, &mut row_buf);

            for (tree_idx, (tree, group)) in self.forest.trees_with_groups().enumerate() {
                let leaf_value = visitor.visit_tree(&tree, &row_buf);
                let weighted_value = match weights {
                    Some(w) => leaf_value * w[tree_idx],
                    None => leaf_value,
                };
                output.row_mut(row_idx)[group as usize] += weighted_value;
            }
        }

        output
    }

    /// Predict for a single row of features.
    ///
    /// Returns a vector with one value per output group.
    pub fn predict_row(&self, features: &[f32]) -> Vec<f32> {
        let mut output: Vec<f32> = self.forest.base_score().to_vec();
        let visitor = ScalarVisitor;

        for (tree, group) in self.forest.trees_with_groups() {
            let leaf_value = visitor.visit_tree(&tree, features);
            output[group as usize] += leaf_value;
        }

        output
    }

    /// Predict for a single row with per-tree weights (for DART).
    ///
    /// Each tree's contribution is multiplied by its corresponding weight.
    ///
    /// # Panics
    ///
    /// Panics if `weights.len() != forest.num_trees()`.
    pub fn predict_row_weighted(&self, features: &[f32], weights: &[f32]) -> Vec<f32> {
        assert_eq!(
            weights.len(),
            self.forest.num_trees(),
            "weights length must match number of trees"
        );

        let mut output: Vec<f32> = self.forest.base_score().to_vec();
        let visitor = ScalarVisitor;

        for (tree_idx, (tree, group)) in self.forest.trees_with_groups().enumerate() {
            let leaf_value = visitor.visit_tree(&tree, features);
            output[group as usize] += leaf_value * weights[tree_idx];
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::DenseMatrix;
    use crate::trees::TreeBuilder;

    /// Build a simple tree:
    ///        [0] feat0 < threshold
    ///        /              \
    ///    [1] leaf=left   [2] leaf=right
    fn build_simple_tree(left_val: f32, right_val: f32, threshold: f32) -> crate::trees::SoATreeStorage<ScalarLeaf> {
        let mut builder = TreeBuilder::new();
        builder.add_split(0, threshold, true, 1, 2);
        builder.add_leaf(ScalarLeaf(left_val));
        builder.add_leaf(ScalarLeaf(right_val));
        builder.build()
    }

    #[test]
    fn scalar_visitor_goes_left() {
        let tree_storage = build_simple_tree(1.0, 2.0, 0.5);
        let mut forest = SoAForest::for_regression();
        forest.push_tree(tree_storage, 0);
        let tree = forest.tree(0);

        let visitor = ScalarVisitor;
        // feat0 = 0.3 < 0.5 → go left
        let result = visitor.visit_tree(&tree, &[0.3]);
        assert_eq!(result, 1.0);
    }

    #[test]
    fn scalar_visitor_goes_right() {
        let tree_storage = build_simple_tree(1.0, 2.0, 0.5);
        let mut forest = SoAForest::for_regression();
        forest.push_tree(tree_storage, 0);
        let tree = forest.tree(0);

        let visitor = ScalarVisitor;
        // feat0 = 0.7 >= 0.5 → go right
        let result = visitor.visit_tree(&tree, &[0.7]);
        assert_eq!(result, 2.0);
    }

    #[test]
    fn scalar_visitor_missing_default_left() {
        let tree_storage = build_simple_tree(1.0, 2.0, 0.5);
        let mut forest = SoAForest::for_regression();
        forest.push_tree(tree_storage, 0);
        let tree = forest.tree(0);

        let visitor = ScalarVisitor;
        // NaN → default left (tree built with default_left=true)
        let result = visitor.visit_tree(&tree, &[f32::NAN]);
        assert_eq!(result, 1.0);
    }

    #[test]
    fn predictor_single_row() {
        let mut forest = SoAForest::for_regression();
        forest.push_tree(build_simple_tree(1.0, 2.0, 0.5), 0);

        let predictor = Predictor::new(&forest);

        assert_eq!(predictor.predict_row(&[0.3]), vec![1.0]);
        assert_eq!(predictor.predict_row(&[0.7]), vec![2.0]);
    }

    #[test]
    fn predictor_with_base_score() {
        let mut forest = SoAForest::for_regression().with_base_score(vec![0.5]);
        forest.push_tree(build_simple_tree(1.0, 2.0, 0.5), 0);

        let predictor = Predictor::new(&forest);

        // base_score + leaf = 0.5 + 1.0 = 1.5
        assert_eq!(predictor.predict_row(&[0.3]), vec![1.5]);
    }

    #[test]
    fn predictor_batch() {
        let mut forest = SoAForest::for_regression();
        forest.push_tree(build_simple_tree(1.0, 2.0, 0.5), 0);

        let predictor = Predictor::new(&forest);

        // 3 rows, 1 feature each
        let features = DenseMatrix::from_vec(vec![0.3, 0.7, 0.5], 3, 1);
        let output = predictor.predict(&features);

        assert_eq!(output.shape(), (3, 1));
        assert_eq!(output.row(0), &[1.0]); // 0.3 < 0.5 → left
        assert_eq!(output.row(1), &[2.0]); // 0.7 >= 0.5 → right
        assert_eq!(output.row(2), &[2.0]); // 0.5 >= 0.5 → right
    }

    #[test]
    fn predictor_multiclass() {
        let mut forest = SoAForest::new(3);
        forest.push_tree(build_simple_tree(0.1, 0.9, 0.5), 0);
        forest.push_tree(build_simple_tree(0.2, 0.8, 0.5), 1);
        forest.push_tree(build_simple_tree(0.3, 0.7, 0.5), 2);

        let predictor = Predictor::new(&forest);

        // 2 rows, 1 feature each
        let features = DenseMatrix::from_vec(vec![0.3, 0.7], 2, 1);
        let output = predictor.predict(&features);

        assert_eq!(output.shape(), (2, 3));
        assert_eq!(output.row(0), &[0.1, 0.2, 0.3]); // all go left
        assert_eq!(output.row(1), &[0.9, 0.8, 0.7]); // all go right
    }

    #[test]
    fn predictor_multiple_trees_sum() {
        let mut forest = SoAForest::for_regression();
        forest.push_tree(build_simple_tree(1.0, 2.0, 0.5), 0);
        forest.push_tree(build_simple_tree(0.5, 1.5, 0.5), 0);

        let predictor = Predictor::new(&forest);

        let features = DenseMatrix::from_vec(vec![0.3, 0.7], 2, 1);
        let output = predictor.predict(&features);

        assert_eq!(output.row(0), &[1.5]); // 1.0 + 0.5
        assert_eq!(output.row(1), &[3.5]); // 2.0 + 1.5
    }

    #[test]
    fn predictor_to_nested() {
        let mut forest = SoAForest::for_regression();
        forest.push_tree(build_simple_tree(1.0, 2.0, 0.5), 0);

        let predictor = Predictor::new(&forest);

        let features = DenseMatrix::from_vec(vec![0.3, 0.7], 2, 1);
        let output = predictor.predict(&features);

        let nested = output.to_nested();
        assert_eq!(nested, vec![vec![1.0], vec![2.0]]);
    }

    // ==========================================================================
    // Weighted prediction tests (DART support)
    // ==========================================================================

    #[test]
    fn predictor_weighted_single_tree() {
        let mut forest = SoAForest::for_regression();
        forest.push_tree(build_simple_tree(1.0, 2.0, 0.5), 0);

        let predictor = Predictor::new(&forest);

        // Weight of 0.5 halves the tree contribution
        assert_eq!(predictor.predict_row_weighted(&[0.3], &[0.5]), vec![0.5]);
        assert_eq!(predictor.predict_row_weighted(&[0.7], &[0.5]), vec![1.0]);
    }

    #[test]
    fn predictor_weighted_multiple_trees() {
        let mut forest = SoAForest::for_regression();
        forest.push_tree(build_simple_tree(1.0, 2.0, 0.5), 0); // tree 0
        forest.push_tree(build_simple_tree(0.5, 1.5, 0.5), 0); // tree 1

        let predictor = Predictor::new(&forest);

        // Weight tree 0 at 1.0, tree 1 at 0.5
        // For row with feature 0.3 (go left): 1.0*1.0 + 0.5*0.5 = 1.25
        let result = predictor.predict_row_weighted(&[0.3], &[1.0, 0.5]);
        assert!((result[0] - 1.25).abs() < 1e-6);

        // For row with feature 0.7 (go right): 2.0*1.0 + 1.5*0.5 = 2.75
        let result = predictor.predict_row_weighted(&[0.7], &[1.0, 0.5]);
        assert!((result[0] - 2.75).abs() < 1e-6);
    }

    #[test]
    fn predictor_weighted_batch() {
        let mut forest = SoAForest::for_regression();
        forest.push_tree(build_simple_tree(1.0, 2.0, 0.5), 0);
        forest.push_tree(build_simple_tree(0.5, 1.5, 0.5), 0);

        let predictor = Predictor::new(&forest);
        let features = DenseMatrix::from_vec(vec![0.3, 0.7], 2, 1);
        let output = predictor.predict_weighted(&features, &[1.0, 0.5]);

        assert_eq!(output.shape(), (2, 1));
        // Row 0 (0.3 < 0.5): 1.0*1.0 + 0.5*0.5 = 1.25
        assert!((output.row(0)[0] - 1.25).abs() < 1e-6);
        // Row 1 (0.7 >= 0.5): 2.0*1.0 + 1.5*0.5 = 2.75
        assert!((output.row(1)[0] - 2.75).abs() < 1e-6);
    }

    #[test]
    fn predictor_weighted_with_base_score() {
        let mut forest = SoAForest::for_regression().with_base_score(vec![0.5]);
        forest.push_tree(build_simple_tree(1.0, 2.0, 0.5), 0);

        let predictor = Predictor::new(&forest);

        // base_score + weighted leaf = 0.5 + 1.0*0.5 = 1.0
        let result = predictor.predict_row_weighted(&[0.3], &[0.5]);
        assert!((result[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn predictor_weighted_multiclass() {
        let mut forest = SoAForest::new(3);
        forest.push_tree(build_simple_tree(1.0, 2.0, 0.5), 0);
        forest.push_tree(build_simple_tree(3.0, 4.0, 0.5), 1);
        forest.push_tree(build_simple_tree(5.0, 6.0, 0.5), 2);

        let predictor = Predictor::new(&forest);

        // Weights: [0.5, 1.0, 2.0]
        // Row 0.3 (go left): [1.0*0.5, 3.0*1.0, 5.0*2.0] = [0.5, 3.0, 10.0]
        let result = predictor.predict_row_weighted(&[0.3], &[0.5, 1.0, 2.0]);
        assert!((result[0] - 0.5).abs() < 1e-6);
        assert!((result[1] - 3.0).abs() < 1e-6);
        assert!((result[2] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn predictor_weighted_zero_weights() {
        let mut forest = SoAForest::for_regression();
        forest.push_tree(build_simple_tree(1.0, 2.0, 0.5), 0);
        forest.push_tree(build_simple_tree(100.0, 200.0, 0.5), 0);

        let predictor = Predictor::new(&forest);

        // Weight of 0 for tree 1 means it doesn't contribute
        let result = predictor.predict_row_weighted(&[0.3], &[1.0, 0.0]);
        assert!((result[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "weights length must match number of trees")]
    fn predictor_weighted_wrong_weights_length() {
        let mut forest = SoAForest::for_regression();
        forest.push_tree(build_simple_tree(1.0, 2.0, 0.5), 0);
        forest.push_tree(build_simple_tree(0.5, 1.5, 0.5), 0);

        let predictor = Predictor::new(&forest);

        // Should panic: only 1 weight for 2 trees
        predictor.predict_row_weighted(&[0.3], &[1.0]);
    }
}

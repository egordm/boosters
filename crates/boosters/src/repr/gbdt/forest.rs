//! Canonical forest representation (collection of trees).

use crate::data::SamplesView;

use super::{tree::TreeValidationError, LeafValue, ScalarLeaf, Tree, TreeView};

/// Structural validation errors for [`Forest`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ForestValidationError {
    BaseScoreLenMismatch { n_groups: u32, len: usize },
    TreeGroupsLenMismatch { n_trees: usize, len: usize },
    TreeGroupOutOfRange { tree_idx: usize, group: u32, n_groups: u32 },
    InvalidTree { tree_idx: usize, error: TreeValidationError },
}

/// Forest of decision trees.
///
/// Stores multiple trees with their group assignments for multi-class support.
#[derive(Debug, Clone)]
pub struct Forest<L: LeafValue = ScalarLeaf> {
    trees: Vec<Tree<L>>,
    tree_groups: Vec<u32>,
    n_groups: u32,
    base_score: Vec<f32>,
}

impl<L: LeafValue> Forest<L> {
    /// Create a new forest with the given number of groups.
    pub fn new(n_groups: u32) -> Self {
        Self {
            trees: Vec::new(),
            tree_groups: Vec::new(),
            n_groups,
            base_score: vec![0.0; n_groups as usize],
        }
    }

    /// Create a forest for regression (single output group).
    pub fn for_regression() -> Self {
        Self::new(1)
    }

    /// Set the base score for all groups.
    pub fn with_base_score(mut self, base_score: Vec<f32>) -> Self {
        debug_assert_eq!(base_score.len(), self.n_groups as usize);
        self.base_score = base_score;
        self
    }

    /// Add a tree to the forest.
    pub fn push_tree(&mut self, tree: Tree<L>, group: u32) {
        debug_assert!(group < self.n_groups, "group out of range");
        self.trees.push(tree);
        self.tree_groups.push(group);
    }

    /// Number of trees.
    #[inline]
    pub fn n_trees(&self) -> usize {
        self.trees.len()
    }

    /// Number of output groups.
    #[inline]
    pub fn n_groups(&self) -> u32 {
        self.n_groups
    }

    /// Get the base score for each group.
    #[inline]
    pub fn base_score(&self) -> &[f32] {
        &self.base_score
    }

    /// Get a reference to a specific tree.
    #[inline]
    pub fn tree(&self, idx: usize) -> &Tree<L> {
        &self.trees[idx]
    }

    /// Get all tree group assignments as a slice.
    #[inline]
    pub fn tree_groups(&self) -> &[u32] {
        &self.tree_groups
    }

    /// Iterate over trees.
    pub fn trees(&self) -> impl Iterator<Item = &Tree<L>> {
        self.trees.iter()
    }

    /// Iterate over trees with their group assignments.
    pub fn trees_with_groups(&self) -> impl Iterator<Item = (&Tree<L>, u32)> {
        self.trees
            .iter()
            .zip(self.tree_groups.iter())
            .map(|(t, &g)| (t, g))
    }

    /// Validate structural invariants for this forest (trees, group assignments, base score).
    ///
    /// Intended for debug checks and tests (e.g., model conversion invariants).
    pub fn validate(&self) -> Result<(), ForestValidationError> {
        if self.base_score.len() != self.n_groups as usize {
            return Err(ForestValidationError::BaseScoreLenMismatch {
                n_groups: self.n_groups,
                len: self.base_score.len(),
            });
        }
        if self.tree_groups.len() != self.trees.len() {
            return Err(ForestValidationError::TreeGroupsLenMismatch {
                n_trees: self.trees.len(),
                len: self.tree_groups.len(),
            });
        }

        for (i, &g) in self.tree_groups.iter().enumerate() {
            if g >= self.n_groups {
                return Err(ForestValidationError::TreeGroupOutOfRange {
                    tree_idx: i,
                    group: g,
                    n_groups: self.n_groups,
                });
            }
        }

        for (i, tree) in self.trees.iter().enumerate() {
            tree.validate()
                .map_err(|e| ForestValidationError::InvalidTree { tree_idx: i, error: e })?;
        }

        Ok(())
    }
}

/// Prediction methods for forests with scalar leaves.
impl Forest<ScalarLeaf> {
    /// Predict for a single row of features.
    ///
    /// Handles linear leaf coefficients if present, computing `intercept + Σ(coef × feature)`.
    pub fn predict_row(&self, features: &[f32]) -> Vec<f32> {
        let mut output = self.base_score.clone();

        let view = SamplesView::from_slice(features, 1, features.len())
            .expect("features slice must be valid");

        for (tree, group) in self.trees_with_groups() {
            let leaf_idx = tree.traverse_to_leaf(&view, 0);
            let leaf_val = tree.compute_leaf_value(leaf_idx, &view, 0);
            output[group as usize] += leaf_val;
        }

        output
    }

    /// Predict for a batch of rows.
    pub fn predict_batch(&self, features: &[&[f32]]) -> Vec<Vec<f32>> {
        features.iter().map(|row| self.predict_row(row)).collect()
    }

    /// Batch predict using any feature accessor, writing into a flat output buffer.
    ///
    /// This is the unified batch prediction method that works with any data source
    /// implementing `FeatureAccessor`.
    ///
    /// # Note on Linear Leaves
    ///
    /// This method does **not** support linear leaf coefficients. For trees with
    /// linear leaves, use [`Predictor`](crate::inference::gbdt::Predictor) instead.
    ///
    /// # Arguments
    /// * `accessor` - Feature value source (SamplesView, FeaturesView, BinnedAccessor, etc.)
    /// * `output` - Pre-allocated output buffer, must have length `n_rows * n_groups`.
    ///   Layout: row-major `[row0_g0, row0_g1, ..., row1_g0, row1_g1, ...]`
    ///
    /// # Panics
    /// Panics if `output.len() != accessor.num_rows() * self.n_groups()`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use boosters::repr::gbdt::{Forest, ScalarLeaf};
    /// use boosters::data::SamplesView;
    ///
    /// let forest: Forest<ScalarLeaf> = /* ... */;
    /// let data = [0.1f32, 0.2, 0.3, 0.4];
    /// let view = SamplesView::from_slice(&data, 2, 2).unwrap();
    /// let mut output = vec![0.0; 2 * forest.n_groups()];
    /// forest.predict_into(&view, &mut output);
    /// ```
    pub fn predict_into<A: crate::data::FeatureAccessor>(
        &self,
        accessor: &A,
        output: &mut [f32],
    ) {
        // Debug check: this method doesn't support linear leaves
        debug_assert!(
            !self.trees.iter().any(|t| t.has_linear_leaves()),
            "predict_into does not support linear leaves; use Predictor instead"
        );

        let n_rows = accessor.num_rows();
        let n_groups = self.n_groups() as usize;
        assert_eq!(
            output.len(),
            n_rows * n_groups,
            "output buffer must have length n_rows * n_groups"
        );

        // Initialize with base scores
        for row in 0..n_rows {
            for (group, &base) in self.base_score.iter().enumerate() {
                output[row * n_groups + group] = base;
            }
        }

        // Accumulate tree predictions
        for (tree, group) in self.trees_with_groups() {
            let group_idx = group as usize;
            for row in 0..n_rows {
                let leaf_idx = tree.traverse_to_leaf(accessor, row);
                let leaf_val = tree.leaf_value(leaf_idx).0;
                output[row * n_groups + group_idx] += leaf_val;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repr::gbdt::ScalarLeaf;

    fn build_simple_tree(left_val: f32, right_val: f32, threshold: f32) -> Tree<ScalarLeaf> {
        crate::scalar_tree! {
            0 => num(0, threshold, L) -> 1, 2,
            1 => leaf(left_val),
            2 => leaf(right_val),
        }
    }

    #[test]
    fn forest_single_tree_regression() {
        let mut forest = Forest::for_regression();
        forest.push_tree(build_simple_tree(1.0, 2.0, 0.5), 0);

        let pred = forest.predict_row(&[0.3]);
        assert_eq!(pred, vec![1.0]);

        let pred = forest.predict_row(&[0.7]);
        assert_eq!(pred, vec![2.0]);
    }

    #[test]
    fn forest_multiple_trees_sum() {
        let mut forest = Forest::for_regression();
        forest.push_tree(build_simple_tree(1.0, 2.0, 0.5), 0);
        forest.push_tree(build_simple_tree(0.5, 1.5, 0.5), 0);

        let pred = forest.predict_row(&[0.3]);
        assert_eq!(pred, vec![1.5]);

        let pred = forest.predict_row(&[0.7]);
        assert_eq!(pred, vec![3.5]);
    }

    #[test]
    fn forest_with_base_score() {
        let mut forest = Forest::for_regression().with_base_score(vec![0.5]);
        forest.push_tree(build_simple_tree(1.0, 2.0, 0.5), 0);

        let pred = forest.predict_row(&[0.3]);
        assert_eq!(pred, vec![1.5]);
    }

    #[test]
    fn test_predict_into_matches_predict_row() {
        use crate::data::SamplesView;

        let mut forest = Forest::for_regression().with_base_score(vec![0.1]);
        forest.push_tree(build_simple_tree(1.0, 2.0, 0.5), 0);
        forest.push_tree(build_simple_tree(0.5, 1.0, 0.5), 0);

        // Test data: 3 rows, 1 feature
        let data = [0.3f32, 0.7, 0.5];
        let view = SamplesView::from_slice(&data, 3, 1).unwrap();
        
        // predict_into
        let mut batch_output = vec![0.0; 3];
        forest.predict_into(&view, &mut batch_output);

        // predict_row for comparison
        let row0 = forest.predict_row(&[0.3])[0];
        let row1 = forest.predict_row(&[0.7])[0];
        let row2 = forest.predict_row(&[0.5])[0];

        assert!((batch_output[0] - row0).abs() < 1e-6);
        assert!((batch_output[1] - row1).abs() < 1e-6);
        assert!((batch_output[2] - row2).abs() < 1e-6);
    }
}

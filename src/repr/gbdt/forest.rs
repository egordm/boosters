//! Canonical forest representation (collection of trees).

use super::{LeafValue, ScalarLeaf, Tree};

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

    /// Get the group assignment for a tree.
    #[inline]
    pub fn tree_group(&self, idx: usize) -> u32 {
        self.tree_groups[idx]
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
}

/// Prediction methods for forests with scalar leaves.
impl Forest<ScalarLeaf> {
    /// Predict for a single row of features.
    pub fn predict_row(&self, features: &[f32]) -> Vec<f32> {
        let mut output = self.base_score.clone();

        for (tree, group) in self.trees_with_groups() {
            let leaf = tree.predict_row(features);
            output[group as usize] += leaf.0;
        }

        output
    }

    /// Predict for a batch of rows.
    pub fn predict_batch(&self, features: &[&[f32]]) -> Vec<Vec<f32>> {
        features.iter().map(|row| self.predict_row(row)).collect()
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
}

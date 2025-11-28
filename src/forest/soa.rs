//! Structure-of-Arrays forest implementation.

use crate::trees::{CategoriesStorage, LeafValue, ScalarLeaf, SoATreeStorage};

/// Structure-of-Arrays forest for efficient inference.
///
/// Stores multiple trees with their group assignments for multi-class support.
#[derive(Debug, Clone)]
pub struct SoAForest<L: LeafValue = ScalarLeaf> {
    /// Individual tree storage
    trees: Vec<SoATreeStorage<L>>,
    /// Which output group each tree belongs to (for multi-class)
    tree_groups: Vec<u32>,
    /// Number of output groups (1 for regression, K for K-class)
    num_groups: u32,
    /// Base score per group (added to predictions)
    base_score: Vec<f32>,
}

impl<L: LeafValue> SoAForest<L> {
    /// Create a new forest with the given number of groups.
    pub fn new(num_groups: u32) -> Self {
        Self {
            trees: Vec::new(),
            tree_groups: Vec::new(),
            num_groups,
            base_score: vec![0.0; num_groups as usize],
        }
    }

    /// Create a forest for regression (single output group).
    pub fn for_regression() -> Self {
        Self::new(1)
    }

    /// Set the base score for all groups.
    pub fn with_base_score(mut self, base_score: Vec<f32>) -> Self {
        debug_assert_eq!(base_score.len(), self.num_groups as usize);
        self.base_score = base_score;
        self
    }

    /// Add a tree to the forest.
    pub fn push_tree(&mut self, tree: SoATreeStorage<L>, group: u32) {
        debug_assert!(group < self.num_groups, "group out of range");
        self.trees.push(tree);
        self.tree_groups.push(group);
    }

    /// Number of trees in the forest.
    #[inline]
    pub fn num_trees(&self) -> usize {
        self.trees.len()
    }

    /// Number of output groups.
    #[inline]
    pub fn num_groups(&self) -> u32 {
        self.num_groups
    }

    /// Get the base score for each group.
    #[inline]
    pub fn base_score(&self) -> &[f32] {
        &self.base_score
    }

    /// Get a view into a specific tree.
    #[inline]
    pub fn tree(&self, idx: usize) -> SoATreeView<'_, L> {
        SoATreeView {
            storage: &self.trees[idx],
        }
    }

    /// Get the group assignment for a tree.
    #[inline]
    pub fn tree_group(&self, idx: usize) -> u32 {
        self.tree_groups[idx]
    }

    /// Iterate over trees as views.
    pub fn trees(&self) -> impl Iterator<Item = SoATreeView<'_, L>> {
        self.trees.iter().map(|t| SoATreeView { storage: t })
    }

    /// Iterate over trees with their group assignments.
    pub fn trees_with_groups(&self) -> impl Iterator<Item = (SoATreeView<'_, L>, u32)> {
        self.trees
            .iter()
            .zip(self.tree_groups.iter())
            .map(|(t, &g)| (SoATreeView { storage: t }, g))
    }
}

/// Prediction methods for forests with scalar leaves.
impl SoAForest<ScalarLeaf> {
    /// Predict for a single row of features.
    ///
    /// Returns one value per output group.
    pub fn predict_row(&self, features: &[f32]) -> Vec<f32> {
        let mut output = self.base_score.clone();

        for (tree, group) in self.trees_with_groups() {
            let leaf = tree.predict_row(features);
            output[group as usize] += leaf.0;
        }

        output
    }

    /// Predict for a batch of rows.
    ///
    /// Returns `[num_rows][num_groups]` predictions.
    pub fn predict_batch(&self, features: &[&[f32]]) -> Vec<Vec<f32>> {
        features.iter().map(|row| self.predict_row(row)).collect()
    }
}

/// Borrowed view into a single tree within a forest.
///
/// Zero-copy reference to tree data for traversal.
#[derive(Debug, Clone, Copy)]
pub struct SoATreeView<'a, L: LeafValue> {
    storage: &'a SoATreeStorage<L>,
}

impl<'a, L: LeafValue> SoATreeView<'a, L> {
    /// Number of nodes in this tree.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.storage.num_nodes()
    }

    /// Check if a node is a leaf.
    #[inline]
    pub fn is_leaf(&self, node_idx: u32) -> bool {
        self.storage.is_leaf(node_idx)
    }

    /// Get split feature index for a node.
    #[inline]
    pub fn split_index(&self, node_idx: u32) -> u32 {
        self.storage.split_index(node_idx)
    }

    /// Get split threshold for a node.
    #[inline]
    pub fn split_threshold(&self, node_idx: u32) -> f32 {
        self.storage.split_threshold(node_idx)
    }

    /// Get left child index.
    #[inline]
    pub fn left_child(&self, node_idx: u32) -> u32 {
        self.storage.left_child(node_idx)
    }

    /// Get right child index.
    #[inline]
    pub fn right_child(&self, node_idx: u32) -> u32 {
        self.storage.right_child(node_idx)
    }

    /// Get default direction for missing values.
    #[inline]
    pub fn default_left(&self, node_idx: u32) -> bool {
        self.storage.default_left(node_idx)
    }

    /// Get leaf value for a node.
    #[inline]
    pub fn leaf_value(&self, node_idx: u32) -> &L {
        self.storage.leaf_value(node_idx)
    }

    /// Get split type for a node.
    #[inline]
    pub fn split_type(&self, node_idx: u32) -> crate::trees::node::SplitType {
        self.storage.split_type(node_idx)
    }

    /// Check if this tree has any categorical splits.
    #[inline]
    pub fn has_categorical(&self) -> bool {
        self.storage.has_categorical()
    }

    /// Get reference to categories storage.
    #[inline]
    pub fn categories(&self) -> &CategoriesStorage {
        self.storage.categories()
    }

    /// Traverse the tree to find the leaf for given features.
    #[inline]
    pub fn predict_row(&self, features: &[f32]) -> &L {
        self.storage.predict_row(features)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trees::TreeBuilder;

    /// Build a simple tree:
    ///        [0] feat0 < 0.5
    ///        /          \
    ///    [1] leaf=1.0   [2] leaf=2.0
    fn build_simple_tree(left_val: f32, right_val: f32, threshold: f32) -> SoATreeStorage<ScalarLeaf> {
        let mut builder = TreeBuilder::new();
        builder.add_split(0, threshold, true, 1, 2);
        builder.add_leaf(ScalarLeaf(left_val));
        builder.add_leaf(ScalarLeaf(right_val));
        builder.build()
    }

    #[test]
    fn forest_single_tree_regression() {
        let mut forest = SoAForest::for_regression();
        forest.push_tree(build_simple_tree(1.0, 2.0, 0.5), 0);

        // Go left: feat0 = 0.3 < 0.5
        let pred = forest.predict_row(&[0.3]);
        assert_eq!(pred, vec![1.0]);

        // Go right: feat0 = 0.7 >= 0.5
        let pred = forest.predict_row(&[0.7]);
        assert_eq!(pred, vec![2.0]);
    }

    #[test]
    fn forest_multiple_trees_sum() {
        let mut forest = SoAForest::for_regression();
        forest.push_tree(build_simple_tree(1.0, 2.0, 0.5), 0);
        forest.push_tree(build_simple_tree(0.5, 1.5, 0.5), 0);

        // Both go left: 1.0 + 0.5 = 1.5
        let pred = forest.predict_row(&[0.3]);
        assert_eq!(pred, vec![1.5]);

        // Both go right: 2.0 + 1.5 = 3.5
        let pred = forest.predict_row(&[0.7]);
        assert_eq!(pred, vec![3.5]);
    }

    #[test]
    fn forest_with_base_score() {
        let mut forest = SoAForest::for_regression().with_base_score(vec![0.5]);
        forest.push_tree(build_simple_tree(1.0, 2.0, 0.5), 0);

        // base_score + leaf = 0.5 + 1.0 = 1.5
        let pred = forest.predict_row(&[0.3]);
        assert_eq!(pred, vec![1.5]);
    }

    #[test]
    fn forest_multiclass() {
        // 3-class classification with one tree per class
        let mut forest = SoAForest::new(3);
        forest.push_tree(build_simple_tree(0.1, 0.9, 0.5), 0); // class 0
        forest.push_tree(build_simple_tree(0.2, 0.8, 0.5), 1); // class 1
        forest.push_tree(build_simple_tree(0.3, 0.7, 0.5), 2); // class 2

        // All go left
        let pred = forest.predict_row(&[0.3]);
        assert_eq!(pred, vec![0.1, 0.2, 0.3]);

        // All go right
        let pred = forest.predict_row(&[0.7]);
        assert_eq!(pred, vec![0.9, 0.8, 0.7]);
    }

    #[test]
    fn forest_predict_batch() {
        let mut forest = SoAForest::for_regression();
        forest.push_tree(build_simple_tree(1.0, 2.0, 0.5), 0);

        let features: Vec<&[f32]> = vec![&[0.3], &[0.7], &[0.5]];
        let preds = forest.predict_batch(&features);

        assert_eq!(preds, vec![vec![1.0], vec![2.0], vec![2.0]]);
    }

    #[test]
    fn tree_view_delegates_to_storage() {
        let tree = build_simple_tree(1.0, 2.0, 0.5);
        let forest = {
            let mut f = SoAForest::for_regression();
            f.push_tree(tree, 0);
            f
        };

        let view = forest.tree(0);
        assert_eq!(view.num_nodes(), 3);
        assert!(!view.is_leaf(0));
        assert!(view.is_leaf(1));
        assert!(view.is_leaf(2));
        assert_eq!(view.split_index(0), 0);
        assert_eq!(view.split_threshold(0), 0.5);
        assert_eq!(view.left_child(0), 1);
        assert_eq!(view.right_child(0), 2);
    }

    #[test]
    fn forest_tree_iteration() {
        let mut forest = SoAForest::for_regression();
        forest.push_tree(build_simple_tree(1.0, 2.0, 0.5), 0);
        forest.push_tree(build_simple_tree(3.0, 4.0, 0.5), 0);

        let tree_count: usize = forest.trees().count();
        assert_eq!(tree_count, 2);

        let groups: Vec<u32> = forest.trees_with_groups().map(|(_, g)| g).collect();
        assert_eq!(groups, vec![0, 0]);
    }
}

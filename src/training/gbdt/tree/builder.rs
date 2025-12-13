//! Tree builder for constructing trees during training.

use super::super::split::{SplitInfo, SplitType};
use super::node::{CategoricalSplit, NodeId, Tree, TreeNode, NO_CHILD};

/// Mutable tree builder for use during training.
///
/// Provides methods to construct the tree incrementally during the growth process.
#[derive(Clone, Debug)]
pub struct TreeBuilder {
    /// Nodes being built.
    nodes: Vec<TreeNode>,
    /// Categorical splits.
    categorical_splits: Vec<CategoricalSplit>,
    /// Next node ID to allocate.
    next_id: NodeId,
    /// Current number of leaves.
    n_leaves: u32,
    /// Maximum depth seen so far.
    max_depth: u16,
}

impl TreeBuilder {
    /// Create a new tree builder.
    pub fn new() -> Self {
        Self {
            nodes: Vec::with_capacity(64),
            categorical_splits: Vec::new(),
            next_id: 0,
            n_leaves: 0,
            max_depth: 0,
        }
    }

    /// Create a builder with capacity hint.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(capacity),
            categorical_splits: Vec::new(),
            next_id: 0,
            n_leaves: 0,
            max_depth: 0,
        }
    }

    /// Initialize the root node.
    ///
    /// Returns the root node ID (always 0).
    pub fn init_root(&mut self) -> NodeId {
        self.nodes.clear();
        self.categorical_splits.clear();
        self.next_id = 0;
        self.n_leaves = 0;
        self.max_depth = 0;

        // Allocate root as placeholder (will be set as split or leaf later)
        self.nodes.push(TreeNode::default());
        self.next_id = 1;

        0
    }

    /// Apply a split to a node, allocating child nodes.
    ///
    /// Returns (left_id, right_id).
    pub fn apply_split(&mut self, node: NodeId, split: &SplitInfo, depth: u16) -> (NodeId, NodeId) {
        // Allocate children
        let left_id = self.next_id;
        let right_id = self.next_id + 1;
        self.next_id += 2;

        // Extend nodes vector
        while self.nodes.len() <= right_id as usize {
            self.nodes.push(TreeNode::default());
        }

        // Set up the split node
        let node_ref = &mut self.nodes[node as usize];
        node_ref.is_leaf = false;
        node_ref.feature = split.feature;
        node_ref.default_left = split.default_left;
        node_ref.left = left_id;
        node_ref.right = right_id;

        match &split.split_type {
            SplitType::Numerical { bin } => {
                node_ref.threshold = *bin;
            }
            SplitType::Categorical { left_cats } => {
                // Store threshold as 0 for categorical (unused)
                node_ref.threshold = 0;
                // Store categorical info separately
                self.categorical_splits.push(CategoricalSplit {
                    node,
                    left_cats: left_cats.clone(),
                });
            }
        }

        // Update max depth
        self.max_depth = self.max_depth.max(depth + 1);

        (left_id, right_id)
    }

    /// Set a node as a leaf with the given value.
    pub fn make_leaf(&mut self, node: NodeId, value: f32) {
        let node_ref = &mut self.nodes[node as usize];
        node_ref.is_leaf = true;
        node_ref.value = value;
        node_ref.left = NO_CHILD;
        node_ref.right = NO_CHILD;

        self.n_leaves += 1;
    }

    /// Apply learning rate to all leaf values.
    pub fn apply_learning_rate(&mut self, learning_rate: f32) {
        for node in &mut self.nodes {
            if node.is_leaf {
                node.value *= learning_rate;
            }
        }
    }

    /// Finalize the tree.
    ///
    /// Consumes the builder and returns an immutable Tree.
    pub fn finish(self) -> Tree {
        Tree::new(
            self.nodes,
            self.categorical_splits,
            self.n_leaves,
            self.max_depth,
        )
    }

    /// Get current number of nodes.
    #[inline]
    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Get current number of leaves.
    #[inline]
    pub fn n_leaves(&self) -> u32 {
        self.n_leaves
    }

    /// Reset the builder for reuse.
    pub fn reset(&mut self) {
        self.nodes.clear();
        self.categorical_splits.clear();
        self.next_id = 0;
        self.n_leaves = 0;
        self.max_depth = 0;
    }
}

impl Default for TreeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::super::categorical::CatBitset;

    #[test]
    fn test_tree_builder_simple() {
        let mut builder = TreeBuilder::new();
        let root = builder.init_root();

        // Make root a leaf
        builder.make_leaf(root, 1.0);

        let tree = builder.finish();
        assert_eq!(tree.n_leaves(), 1);
        assert_eq!(tree.n_nodes(), 1);
        assert!(tree.node(0).is_leaf);
        assert_eq!(tree.node(0).value, 1.0);
    }

    #[test]
    fn test_tree_builder_with_split() {
        let mut builder = TreeBuilder::new();
        let root = builder.init_root();

        // Apply split at root
        let split = SplitInfo::numerical(0, 5, 0.5, false);
        let (left, right) = builder.apply_split(root, &split, 0);

        // Make children leaves
        builder.make_leaf(left, -0.5);
        builder.make_leaf(right, 0.5);

        let tree = builder.finish();

        assert_eq!(tree.n_nodes(), 3);
        assert_eq!(tree.n_leaves(), 2);
        assert_eq!(tree.max_depth(), 1);

        // Check root
        let root_node = tree.node(0);
        assert!(!root_node.is_leaf);
        assert_eq!(root_node.feature, 0);
        assert_eq!(root_node.threshold, 5);
        assert_eq!(root_node.left, left);
        assert_eq!(root_node.right, right);

        // Check leaves
        assert!(tree.node(left).is_leaf);
        assert_eq!(tree.node(left).value, -0.5);
        assert!(tree.node(right).is_leaf);
        assert_eq!(tree.node(right).value, 0.5);
    }

    #[test]
    fn test_tree_builder_categorical() {
        let mut builder = TreeBuilder::new();
        let root = builder.init_root();

        let mut cats = CatBitset::empty();
        cats.insert(0);
        cats.insert(2);
        let split = SplitInfo::categorical(1, cats, 0.8, false);
        let (left, right) = builder.apply_split(root, &split, 0);

        builder.make_leaf(left, -1.0);
        builder.make_leaf(right, 1.0);

        let tree = builder.finish();

        // Check categorical split info
        let cat_split = tree.categorical_split(0).unwrap();
        assert!(cat_split.left_cats.contains(0));
        assert!(cat_split.left_cats.contains(2));
        assert!(!cat_split.left_cats.contains(1));
    }

    #[test]
    fn test_tree_predict_numerical() {
        use crate::data::{BinMapper, BinStorage, BinnedDataset, FeatureGroup, FeatureMeta, GroupLayout, MissingType};

        let mut builder = TreeBuilder::new();
        let root = builder.init_root();

        let split = SplitInfo::numerical(0, 3, 0.5, false);
        let (left, right) = builder.apply_split(root, &split, 0);
        builder.make_leaf(left, -1.0);
        builder.make_leaf(right, 1.0);

        let tree = builder.finish();

        // Create a simple dataset with bins [2, 5] for testing
        let storage = BinStorage::from_u8(vec![2, 5]); // row 0: bin 2, row 1: bin 5
        let group = FeatureGroup::new(vec![0], GroupLayout::ColumnMajor, 2, storage, vec![8]);
        let mapper = BinMapper::numerical(
            (0..8).map(|i| i as f64 + 0.5).collect(),
            MissingType::None, 0, 0, 0.0, 7.0, 7.0,
        );
        let features = vec![FeatureMeta::new(mapper, 0, 0)];
        let dataset = BinnedDataset::new(2, features, vec![group]);

        // Bin 2 (<=3) should go left
        let view0 = dataset.row_view(0).unwrap();
        let pred1 = tree.predict(&view0);
        assert_eq!(pred1, -1.0);

        // Bin 5 (>3) should go right
        let view1 = dataset.row_view(1).unwrap();
        let pred2 = tree.predict(&view1);
        assert_eq!(pred2, 1.0);
    }

    #[test]
    fn test_tree_predict_missing() {
        use crate::data::{BinMapper, BinStorage, BinnedDataset, FeatureGroup, FeatureMeta, GroupLayout, MissingType};

        let mut builder = TreeBuilder::new();
        let root = builder.init_root();

        let split = SplitInfo::numerical(0, 3, 0.5, true); // default_left=true
        let (left, right) = builder.apply_split(root, &split, 0);
        builder.make_leaf(left, -1.0);
        builder.make_leaf(right, 1.0);

        let tree = builder.finish();

        // Create a sparse dataset where row 0 is missing (not in indices)
        let storage = BinStorage::from_sparse_u8(vec![1], vec![5], 8); // Only row 1 has a value
        let group = FeatureGroup::new(vec![0], GroupLayout::ColumnMajor, 2, storage, vec![8]);
        let mapper = BinMapper::numerical(
            (0..8).map(|i| i as f64 + 0.5).collect(),
            MissingType::NaN, 0, 0, 0.0, 7.0, 7.0,
        );
        let features = vec![FeatureMeta::new(mapper, 0, 0)];
        let dataset = BinnedDataset::new(2, features, vec![group]);

        // Row 0 has missing value (sparse), should go left (default_left=true)
        let view0 = dataset.row_view(0).unwrap();
        let pred = tree.predict(&view0);
        assert_eq!(pred, -1.0);
    }

    #[test]
    fn test_tree_learning_rate() {
        let mut builder = TreeBuilder::new();
        let root = builder.init_root();

        let split = SplitInfo::numerical(0, 3, 0.5, false);
        let (left, right) = builder.apply_split(root, &split, 0);
        builder.make_leaf(left, -1.0);
        builder.make_leaf(right, 1.0);
        builder.apply_learning_rate(0.1);

        let tree = builder.finish();

        assert!((tree.node(left).value - -0.1).abs() < 1e-6);
        assert!((tree.node(right).value - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_iter_leaves() {
        let mut builder = TreeBuilder::new();
        let root = builder.init_root();

        let split = SplitInfo::numerical(0, 3, 0.5, false);
        let (left, right) = builder.apply_split(root, &split, 0);

        let split2 = SplitInfo::numerical(1, 2, 0.3, false);
        let (ll, lr) = builder.apply_split(left, &split2, 1);

        builder.make_leaf(ll, 0.1);
        builder.make_leaf(lr, 0.2);
        builder.make_leaf(right, 0.3);

        let tree = builder.finish();

        let leaves: Vec<_> = tree.iter_leaves().collect();
        assert_eq!(leaves.len(), 3);

        // Check leaf values
        let values: Vec<f32> = leaves.iter().map(|(_, n)| n.value).collect();
        assert!(values.contains(&0.1));
        assert!(values.contains(&0.2));
        assert!(values.contains(&0.3));
    }
}

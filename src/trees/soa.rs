//! Structure-of-Arrays tree storage implementation.

use super::leaf::LeafValue;

/// Structure-of-Arrays tree storage for efficient inference.
///
/// Stores tree nodes in flat arrays for cache-friendly traversal.
/// Child indices are local to this tree (0 = root).
#[derive(Debug, Clone)]
pub struct SoATreeStorage<L: LeafValue> {
    /// Split feature index per node
    split_indices: Box<[u32]>,
    /// Split threshold per node
    split_thresholds: Box<[f32]>,
    /// Left child index per node (only valid for non-leaf nodes)
    left_children: Box<[u32]>,
    /// Right child index per node (only valid for non-leaf nodes)
    right_children: Box<[u32]>,
    /// Default direction for missing values (true = left)
    default_left: Box<[bool]>,
    /// Whether each node is a leaf
    is_leaf: Box<[bool]>,
    /// Leaf values (indexed by node index, only valid for leaf nodes)
    leaf_values: Box<[L]>,
}

impl<L: LeafValue> SoATreeStorage<L> {
    /// Create a new tree from parallel arrays.
    ///
    /// All arrays must have the same length (number of nodes).
    pub fn new(
        split_indices: Vec<u32>,
        split_thresholds: Vec<f32>,
        left_children: Vec<u32>,
        right_children: Vec<u32>,
        default_left: Vec<bool>,
        is_leaf: Vec<bool>,
        leaf_values: Vec<L>,
    ) -> Self {
        debug_assert_eq!(split_indices.len(), split_thresholds.len());
        debug_assert_eq!(split_indices.len(), left_children.len());
        debug_assert_eq!(split_indices.len(), right_children.len());
        debug_assert_eq!(split_indices.len(), default_left.len());
        debug_assert_eq!(split_indices.len(), is_leaf.len());
        debug_assert_eq!(split_indices.len(), leaf_values.len());

        Self {
            split_indices: split_indices.into_boxed_slice(),
            split_thresholds: split_thresholds.into_boxed_slice(),
            left_children: left_children.into_boxed_slice(),
            right_children: right_children.into_boxed_slice(),
            default_left: default_left.into_boxed_slice(),
            is_leaf: is_leaf.into_boxed_slice(),
            leaf_values: leaf_values.into_boxed_slice(),
        }
    }

    /// Number of nodes in this tree.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.is_leaf.len()
    }

    /// Check if a node is a leaf.
    #[inline]
    pub fn is_leaf(&self, node_idx: u32) -> bool {
        self.is_leaf[node_idx as usize]
    }

    /// Get split feature index for a node.
    #[inline]
    pub fn split_index(&self, node_idx: u32) -> u32 {
        self.split_indices[node_idx as usize]
    }

    /// Get split threshold for a node.
    #[inline]
    pub fn split_threshold(&self, node_idx: u32) -> f32 {
        self.split_thresholds[node_idx as usize]
    }

    /// Get left child index.
    #[inline]
    pub fn left_child(&self, node_idx: u32) -> u32 {
        self.left_children[node_idx as usize]
    }

    /// Get right child index.
    #[inline]
    pub fn right_child(&self, node_idx: u32) -> u32 {
        self.right_children[node_idx as usize]
    }

    /// Get default direction for missing values.
    #[inline]
    pub fn default_left(&self, node_idx: u32) -> bool {
        self.default_left[node_idx as usize]
    }

    /// Get leaf value for a node.
    #[inline]
    pub fn leaf_value(&self, node_idx: u32) -> &L {
        &self.leaf_values[node_idx as usize]
    }

    /// Traverse the tree to find the leaf for given features.
    pub fn predict_row(&self, features: &[f32]) -> &L {
        let mut idx = 0u32; // Start at root

        while !self.is_leaf(idx) {
            let feat_idx = self.split_index(idx) as usize;
            let threshold = self.split_threshold(idx);
            let fvalue = features.get(feat_idx).copied().unwrap_or(f32::NAN);

            idx = if fvalue.is_nan() {
                if self.default_left(idx) {
                    self.left_child(idx)
                } else {
                    self.right_child(idx)
                }
            } else if fvalue < threshold {
                self.left_child(idx)
            } else {
                self.right_child(idx)
            };
        }

        self.leaf_value(idx)
    }
}

/// Builder for constructing SoATreeStorage from individual nodes.
#[derive(Debug, Default)]
pub struct TreeBuilder<L: LeafValue> {
    split_indices: Vec<u32>,
    split_thresholds: Vec<f32>,
    left_children: Vec<u32>,
    right_children: Vec<u32>,
    default_left: Vec<bool>,
    is_leaf: Vec<bool>,
    leaf_values: Vec<L>,
}

impl<L: LeafValue> TreeBuilder<L> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a split node. Returns the node index.
    pub fn add_split(
        &mut self,
        feature_index: u32,
        threshold: f32,
        default_left: bool,
        left_child: u32,
        right_child: u32,
    ) -> u32 {
        let idx = self.split_indices.len() as u32;
        self.split_indices.push(feature_index);
        self.split_thresholds.push(threshold);
        self.left_children.push(left_child);
        self.right_children.push(right_child);
        self.default_left.push(default_left);
        self.is_leaf.push(false);
        self.leaf_values.push(L::default());
        idx
    }

    /// Add a leaf node. Returns the node index.
    pub fn add_leaf(&mut self, value: L) -> u32 {
        let idx = self.split_indices.len() as u32;
        self.split_indices.push(0);
        self.split_thresholds.push(0.0);
        self.left_children.push(0);
        self.right_children.push(0);
        self.default_left.push(false);
        self.is_leaf.push(true);
        self.leaf_values.push(value);
        idx
    }

    /// Build the tree storage.
    pub fn build(self) -> SoATreeStorage<L> {
        SoATreeStorage::new(
            self.split_indices,
            self.split_thresholds,
            self.left_children,
            self.right_children,
            self.default_left,
            self.is_leaf,
            self.leaf_values,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trees::leaf::ScalarLeaf;

    /// Build a simple tree:
    ///        [0] feat0 < 0.5
    ///        /          \
    ///    [1] leaf=1.0   [2] feat1 < 0.3
    ///                    /          \
    ///               [3] leaf=2.0   [4] leaf=3.0
    fn build_test_tree() -> SoATreeStorage<ScalarLeaf> {
        let mut builder = TreeBuilder::new();

        // Add nodes in BFS order
        // Node 0: split on feature 0 at 0.5, default left
        builder.add_split(0, 0.5, true, 1, 2);
        // Node 1: leaf with value 1.0
        builder.add_leaf(ScalarLeaf(1.0));
        // Node 2: split on feature 1 at 0.3, default right
        builder.add_split(1, 0.3, false, 3, 4);
        // Node 3: leaf with value 2.0
        builder.add_leaf(ScalarLeaf(2.0));
        // Node 4: leaf with value 3.0
        builder.add_leaf(ScalarLeaf(3.0));

        builder.build()
    }

    #[test]
    fn tree_structure() {
        let tree = build_test_tree();

        assert_eq!(tree.num_nodes(), 5);

        // Root is a split
        assert!(!tree.is_leaf(0));
        assert_eq!(tree.split_index(0), 0);
        assert_eq!(tree.split_threshold(0), 0.5);
        assert_eq!(tree.left_child(0), 1);
        assert_eq!(tree.right_child(0), 2);

        // Node 1 is a leaf
        assert!(tree.is_leaf(1));
        assert_eq!(tree.leaf_value(1).0, 1.0);

        // Node 2 is a split
        assert!(!tree.is_leaf(2));
        assert_eq!(tree.split_index(2), 1);

        // Nodes 3 and 4 are leaves
        assert!(tree.is_leaf(3));
        assert!(tree.is_leaf(4));
    }

    #[test]
    fn predict_goes_left() {
        let tree = build_test_tree();
        // feat0 = 0.3 < 0.5 → go left to node 1 (leaf=1.0)
        let features = [0.3, 0.5];
        assert_eq!(tree.predict_row(&features).0, 1.0);
    }

    #[test]
    fn predict_goes_right_then_left() {
        let tree = build_test_tree();
        // feat0 = 0.7 >= 0.5 → go right to node 2
        // feat1 = 0.2 < 0.3 → go left to node 3 (leaf=2.0)
        let features = [0.7, 0.2];
        assert_eq!(tree.predict_row(&features).0, 2.0);
    }

    #[test]
    fn predict_goes_right_then_right() {
        let tree = build_test_tree();
        // feat0 = 0.7 >= 0.5 → go right to node 2
        // feat1 = 0.5 >= 0.3 → go right to node 4 (leaf=3.0)
        let features = [0.7, 0.5];
        assert_eq!(tree.predict_row(&features).0, 3.0);
    }

    #[test]
    fn predict_missing_at_root_default_left() {
        let tree = build_test_tree();
        // feat0 = NaN → default left (node 0 has default_left=true) → node 1 (leaf=1.0)
        let features = [f32::NAN, 0.5];
        assert_eq!(tree.predict_row(&features).0, 1.0);
    }

    #[test]
    fn predict_missing_at_node2_default_right() {
        let tree = build_test_tree();
        // feat0 = 0.7 → go right to node 2
        // feat1 = NaN → default right (node 2 has default_left=false) → node 4 (leaf=3.0)
        let features = [0.7, f32::NAN];
        assert_eq!(tree.predict_row(&features).0, 3.0);
    }

    #[test]
    fn predict_missing_feature_out_of_bounds() {
        let tree = build_test_tree();
        // Only provide feat0, feat1 is out of bounds → treated as NaN
        // feat0 = 0.7 → go right to node 2
        // feat1 missing → default right → node 4 (leaf=3.0)
        let features = [0.7];
        assert_eq!(tree.predict_row(&features).0, 3.0);
    }
}

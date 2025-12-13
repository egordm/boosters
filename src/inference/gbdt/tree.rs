//! Structure-of-Arrays tree storage implementation.

// Allow many constructor arguments for creating trees with all their fields.
#![allow(clippy::too_many_arguments)]

use super::categories::{float_to_category, CategoriesStorage};
use super::leaf::LeafValue;
use super::node::SplitType;

// =============================================================================
// SoA Tree Storage
// =============================================================================

/// Structure-of-Arrays tree storage for efficient inference.
///
/// Stores tree nodes in flat arrays for cache-friendly traversal.
/// Child indices are local to this tree (0 = root).
#[derive(Debug, Clone)]
pub struct TreeStorage<L: LeafValue> {
    /// Split feature index per node
    split_indices: Box<[u32]>,
    /// Split threshold per node (also used for one-hot categorical)
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
    /// Split type per node (Numeric or Categorical)
    split_types: Box<[SplitType]>,
    /// Categorical split data (bitsets for partition-based categorical splits)
    categories: CategoriesStorage,
}

impl<L: LeafValue> TreeStorage<L> {
    /// Create a new tree from parallel arrays.
    ///
    /// All arrays must have the same length (number of nodes).
    /// Creates a tree with only numeric splits (no categorical).
    pub fn new(
        split_indices: Vec<u32>,
        split_thresholds: Vec<f32>,
        left_children: Vec<u32>,
        right_children: Vec<u32>,
        default_left: Vec<bool>,
        is_leaf: Vec<bool>,
        leaf_values: Vec<L>,
    ) -> Self {
        let num_nodes = split_indices.len();
        debug_assert_eq!(num_nodes, split_thresholds.len());
        debug_assert_eq!(num_nodes, left_children.len());
        debug_assert_eq!(num_nodes, right_children.len());
        debug_assert_eq!(num_nodes, default_left.len());
        debug_assert_eq!(num_nodes, is_leaf.len());
        debug_assert_eq!(num_nodes, leaf_values.len());

        Self {
            split_indices: split_indices.into_boxed_slice(),
            split_thresholds: split_thresholds.into_boxed_slice(),
            left_children: left_children.into_boxed_slice(),
            right_children: right_children.into_boxed_slice(),
            default_left: default_left.into_boxed_slice(),
            is_leaf: is_leaf.into_boxed_slice(),
            leaf_values: leaf_values.into_boxed_slice(),
            split_types: vec![SplitType::Numeric; num_nodes].into_boxed_slice(),
            categories: CategoriesStorage::empty(),
        }
    }

    /// Create a new tree with categorical split support.
    ///
    /// All arrays must have the same length (number of nodes).
    pub fn with_categories(
        split_indices: Vec<u32>,
        split_thresholds: Vec<f32>,
        left_children: Vec<u32>,
        right_children: Vec<u32>,
        default_left: Vec<bool>,
        is_leaf: Vec<bool>,
        leaf_values: Vec<L>,
        split_types: Vec<SplitType>,
        categories: CategoriesStorage,
    ) -> Self {
        let num_nodes = split_indices.len();
        debug_assert_eq!(num_nodes, split_thresholds.len());
        debug_assert_eq!(num_nodes, left_children.len());
        debug_assert_eq!(num_nodes, right_children.len());
        debug_assert_eq!(num_nodes, default_left.len());
        debug_assert_eq!(num_nodes, is_leaf.len());
        debug_assert_eq!(num_nodes, leaf_values.len());
        debug_assert_eq!(num_nodes, split_types.len());

        Self {
            split_indices: split_indices.into_boxed_slice(),
            split_thresholds: split_thresholds.into_boxed_slice(),
            left_children: left_children.into_boxed_slice(),
            right_children: right_children.into_boxed_slice(),
            default_left: default_left.into_boxed_slice(),
            is_leaf: is_leaf.into_boxed_slice(),
            leaf_values: leaf_values.into_boxed_slice(),
            split_types: split_types.into_boxed_slice(),
            categories,
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

    /// Get split type for a node.
    #[inline]
    pub fn split_type(&self, node_idx: u32) -> SplitType {
        self.split_types[node_idx as usize]
    }

    /// Check if this tree has any categorical splits.
    #[inline]
    pub fn has_categorical(&self) -> bool {
        !self.categories.is_empty()
    }

    /// Get reference to categories storage.
    #[inline]
    pub fn categories(&self) -> &CategoriesStorage {
        &self.categories
    }

    /// Traverse the tree to find the leaf for given features.
    pub fn predict_row(&self, features: &[f32]) -> &L {
        let mut idx = 0u32; // Start at root

        while !self.is_leaf(idx) {
            let feat_idx = self.split_index(idx) as usize;
            let fvalue = features.get(feat_idx).copied().unwrap_or(f32::NAN);

            idx = if fvalue.is_nan() {
                // Missing value: use default direction
                if self.default_left(idx) {
                    self.left_child(idx)
                } else {
                    self.right_child(idx)
                }
            } else {
                match self.split_type(idx) {
                    SplitType::Numeric => {
                        // Numeric split: go left if value < threshold
                        if fvalue < self.split_threshold(idx) {
                            self.left_child(idx)
                        } else {
                            self.right_child(idx)
                        }
                    }
                    SplitType::Categorical => {
                        // Categorical split: go right if category is in the set
                        let category = float_to_category(fvalue);
                        if self.categories.category_goes_right(idx, category) {
                            self.right_child(idx)
                        } else {
                            self.left_child(idx)
                        }
                    }
                }
            };
        }

        self.leaf_value(idx)
    }
}

/// Builder for constructing TreeStorage from individual nodes.
#[derive(Debug, Default)]
pub struct TreeBuilder<L: LeafValue> {
    split_indices: Vec<u32>,
    split_thresholds: Vec<f32>,
    left_children: Vec<u32>,
    right_children: Vec<u32>,
    default_left: Vec<bool>,
    is_leaf: Vec<bool>,
    leaf_values: Vec<L>,
    split_types: Vec<SplitType>,
    // Categorical data: (node_idx, category_bitset)
    categorical_nodes: Vec<(u32, Vec<u32>)>,
}

impl<L: LeafValue> TreeBuilder<L> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a numeric split node. Returns the node index.
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
        self.split_types.push(SplitType::Numeric);
        idx
    }

    /// Add a categorical split node. Returns the node index.
    ///
    /// The `category_bitset` contains the packed u32 words for the category bitset.
    /// Categories in this bitset go RIGHT, categories not in the set go LEFT.
    pub fn add_categorical_split(
        &mut self,
        feature_index: u32,
        category_bitset: Vec<u32>,
        default_left: bool,
        left_child: u32,
        right_child: u32,
    ) -> u32 {
        let idx = self.split_indices.len() as u32;
        self.split_indices.push(feature_index);
        self.split_thresholds.push(0.0); // Not used for categorical
        self.left_children.push(left_child);
        self.right_children.push(right_child);
        self.default_left.push(default_left);
        self.is_leaf.push(false);
        self.leaf_values.push(L::default());
        self.split_types.push(SplitType::Categorical);
        self.categorical_nodes.push((idx, category_bitset));
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
        self.split_types.push(SplitType::Numeric); // Default, not relevant for leaves
        idx
    }

    /// Build the tree storage.
    pub fn build(self) -> TreeStorage<L> {
        // Build categories storage from categorical nodes
        let categories = if self.categorical_nodes.is_empty() {
            CategoriesStorage::empty()
        } else {
            // Sort by node index for binary search during lookup
            let mut cat_nodes = self.categorical_nodes;
            cat_nodes.sort_by_key(|(idx, _)| *idx);

            // Build per-node segments. For nodes with categorical splits,
            // we store (start_index, size). Non-categorical nodes get (0, 0).
            let num_nodes = self.split_indices.len();
            let mut segments = vec![(0u32, 0u32); num_nodes];
            let mut bitsets = Vec::new();

            for (node_idx, bitset) in cat_nodes {
                let start = bitsets.len() as u32;
                let size = bitset.len() as u32;
                segments[node_idx as usize] = (start, size);
                bitsets.extend(bitset);
            }

            CategoriesStorage::new(bitsets, segments)
        };

        // Build split_types, defaulting to Numeric if not explicitly set
        let split_types = if self.split_types.is_empty() {
            vec![SplitType::Numeric; self.split_indices.len()]
        } else {
            self.split_types
        };

        TreeStorage::with_categories(
            self.split_indices,
            self.split_thresholds,
            self.left_children,
            self.right_children,
            self.default_left,
            self.is_leaf,
            self.leaf_values,
            split_types,
            categories,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::gbdt::leaf::ScalarLeaf;

    /// Build a simple tree:
    ///        [0] feat0 < 0.5
    ///        /          \
    ///    [1] leaf=1.0   [2] feat1 < 0.3
    ///                    /          \
    ///               [3] leaf=2.0   [4] leaf=3.0
    fn build_test_tree() -> TreeStorage<ScalarLeaf> {
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

    // ==========================================================================
    // Categorical split tests
    // ==========================================================================

    /// Build a tree with a categorical split at root.
    /// Categories {1, 3} go RIGHT, others go LEFT.
    ///
    ///        [0] feat0 categorical {1,3}
    ///        /                  \
    ///    [1] leaf=-1.0      [2] leaf=1.0
    fn build_categorical_tree() -> TreeStorage<ScalarLeaf> {
        let mut builder = TreeBuilder::new();

        // Bitset for categories {1, 3}: bits 1 and 3 set = 0b1010 = 10
        let bitset = vec![0b1010u32];

        // Node 0: categorical split on feature 0
        builder.add_categorical_split(0, bitset, true, 1, 2);
        // Node 1: left leaf (categories NOT in set)
        builder.add_leaf(ScalarLeaf(-1.0));
        // Node 2: right leaf (categories IN set)
        builder.add_leaf(ScalarLeaf(1.0));

        builder.build()
    }

    #[test]
    fn categorical_tree_structure() {
        let tree = build_categorical_tree();

        assert_eq!(tree.num_nodes(), 3);
        assert!(!tree.is_leaf(0));
        assert_eq!(tree.split_type(0), SplitType::Categorical);
        assert!(tree.has_categorical());
    }

    #[test]
    fn categorical_category_in_set_goes_right() {
        let tree = build_categorical_tree();
        // Category 1 is in set {1, 3} → go right → leaf=1.0
        let features = [1.0];
        assert_eq!(tree.predict_row(&features).0, 1.0);

        // Category 3 is in set {1, 3} → go right → leaf=1.0
        let features = [3.0];
        assert_eq!(tree.predict_row(&features).0, 1.0);
    }

    #[test]
    fn categorical_category_not_in_set_goes_left() {
        let tree = build_categorical_tree();
        // Category 0 is NOT in set {1, 3} → go left → leaf=-1.0
        let features = [0.0];
        assert_eq!(tree.predict_row(&features).0, -1.0);

        // Category 2 is NOT in set {1, 3} → go left → leaf=-1.0
        let features = [2.0];
        assert_eq!(tree.predict_row(&features).0, -1.0);

        // Category 5 is NOT in set {1, 3} → go left → leaf=-1.0
        let features = [5.0];
        assert_eq!(tree.predict_row(&features).0, -1.0);
    }

    #[test]
    fn categorical_missing_uses_default() {
        let tree = build_categorical_tree();
        // Missing value → default left (set to true) → leaf=-1.0
        let features = [f32::NAN];
        assert_eq!(tree.predict_row(&features).0, -1.0);
    }

    #[test]
    fn categorical_large_category_outside_bitset() {
        let tree = build_categorical_tree();
        // Category 100 is way beyond our 1-word bitset → treated as not in set → left
        let features = [100.0];
        assert_eq!(tree.predict_row(&features).0, -1.0);
    }

    /// Build a tree with categorical split using multiple u32 words.
    /// Categories {35, 64} go RIGHT (spans multiple words).
    fn build_multi_word_categorical_tree() -> TreeStorage<ScalarLeaf> {
        let mut builder = TreeBuilder::new();

        // Category 35: word 1 (35/32=1), bit 3 (35%32=3) → 0b1000 in word 1
        // Category 64: word 2 (64/32=2), bit 0 (64%32=0) → 0b1 in word 2
        let bitset = vec![0u32, 0b1000u32, 0b1u32]; // words 0, 1, 2

        builder.add_categorical_split(0, bitset, false, 1, 2);
        builder.add_leaf(ScalarLeaf(-1.0));
        builder.add_leaf(ScalarLeaf(1.0));

        builder.build()
    }

    #[test]
    fn categorical_multi_word_bitset() {
        let tree = build_multi_word_categorical_tree();

        // Category 35 is in set → go right
        let features = [35.0];
        assert_eq!(tree.predict_row(&features).0, 1.0);

        // Category 64 is in set → go right
        let features = [64.0];
        assert_eq!(tree.predict_row(&features).0, 1.0);

        // Category 0 is NOT in set → go left
        let features = [0.0];
        assert_eq!(tree.predict_row(&features).0, -1.0);

        // Category 32 is NOT in set → go left
        let features = [32.0];
        assert_eq!(tree.predict_row(&features).0, -1.0);
    }

    /// Build a tree with mixed numeric and categorical splits.
    fn build_mixed_tree() -> TreeStorage<ScalarLeaf> {
        let mut builder = TreeBuilder::new();

        //        [0] feat0 < 0.5 (numeric)
        //        /               \
        //    [1] feat1 cat {0,2}  [2] leaf=3.0
        //    /           \
        // [3] leaf=1.0  [4] leaf=2.0

        // Node 0: numeric split
        builder.add_split(0, 0.5, true, 1, 2);
        // Node 1: categorical split - categories {0, 2} go right
        builder.add_categorical_split(1, vec![0b101], true, 3, 4);
        // Node 2: leaf
        builder.add_leaf(ScalarLeaf(3.0));
        // Node 3: leaf
        builder.add_leaf(ScalarLeaf(1.0));
        // Node 4: leaf
        builder.add_leaf(ScalarLeaf(2.0));

        builder.build()
    }

    #[test]
    fn mixed_numeric_and_categorical() {
        let tree = build_mixed_tree();

        // feat0=0.3 < 0.5 → node 1
        // feat1=0 in {0,2} → right → leaf=2.0
        assert_eq!(tree.predict_row(&[0.3, 0.0]).0, 2.0);

        // feat0=0.3 < 0.5 → node 1
        // feat1=2 in {0,2} → right → leaf=2.0
        assert_eq!(tree.predict_row(&[0.3, 2.0]).0, 2.0);

        // feat0=0.3 < 0.5 → node 1
        // feat1=1 NOT in {0,2} → left → leaf=1.0
        assert_eq!(tree.predict_row(&[0.3, 1.0]).0, 1.0);

        // feat0=0.7 >= 0.5 → right → leaf=3.0
        assert_eq!(tree.predict_row(&[0.7, 0.0]).0, 3.0);
    }
}

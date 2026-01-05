//! Mutable tree construction API for training.

use super::NodeId;
use super::categories::CategoriesStorage;
use super::coefficients::LeafCoefficientsBuilder;
use super::tree::Tree;
use super::tree_view::TreeView;
use super::types::LeafValue;
use super::types::SplitType;

/// Mutable tree for use during training.
///
/// Supports the training pattern where nodes are allocated first (placeholders)
/// and filled in later when splits/leaves are determined.
#[derive(Debug, Clone)]
pub struct MutableTree<L: LeafValue> {
    split_indices: Vec<u32>,
    split_thresholds: Vec<f32>,
    left_children: Vec<u32>,
    right_children: Vec<u32>,
    default_left: Vec<bool>,
    is_leaf: Vec<bool>,
    leaf_values: Vec<L>,
    split_types: Vec<SplitType>,
    /// Categorical data: (node_idx, category_bitset)
    categorical_nodes: Vec<(NodeId, Vec<u32>)>,
    /// Linear leaf coefficients: (node_idx, feature_indices, coefficients, intercept)
    linear_leaves: Vec<(NodeId, Vec<u32>, Vec<f32>, f32)>,
    /// Split gains (one per node, 0.0 for leaves).
    gains: Vec<f32>,
    /// Node covers/hessian sums (one per node).
    covers: Vec<f32>,
    next_id: NodeId,
}

impl<L: LeafValue> Default for MutableTree<L> {
    fn default() -> Self {
        Self::new()
    }
}

impl<L: LeafValue> MutableTree<L> {
    /// Create a new mutable tree.
    pub fn new() -> Self {
        Self {
            split_indices: Vec::with_capacity(64),
            split_thresholds: Vec::with_capacity(64),
            left_children: Vec::with_capacity(64),
            right_children: Vec::with_capacity(64),
            default_left: Vec::with_capacity(64),
            is_leaf: Vec::with_capacity(64),
            leaf_values: Vec::with_capacity(64),
            split_types: Vec::with_capacity(64),
            categorical_nodes: Vec::new(),
            linear_leaves: Vec::new(),
            gains: Vec::with_capacity(64),
            covers: Vec::with_capacity(64),
            next_id: 0,
        }
    }

    /// Create a tree with capacity hint.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            split_indices: Vec::with_capacity(capacity),
            split_thresholds: Vec::with_capacity(capacity),
            left_children: Vec::with_capacity(capacity),
            right_children: Vec::with_capacity(capacity),
            default_left: Vec::with_capacity(capacity),
            is_leaf: Vec::with_capacity(capacity),
            leaf_values: Vec::with_capacity(capacity),
            split_types: Vec::with_capacity(capacity),
            categorical_nodes: Vec::new(),
            linear_leaves: Vec::new(),
            gains: Vec::with_capacity(capacity),
            covers: Vec::with_capacity(capacity),
            next_id: 0,
        }
    }

    /// Initialize the root node as a placeholder.
    ///
    /// Returns the root node ID (always 0).
    pub fn init_root(&mut self) -> NodeId {
        self.reset();
        self.allocate_node();
        0
    }

    /// Initialize the tree with a fixed number of placeholder nodes.
    ///
    /// This is useful for model loaders where node indices and child references
    /// are already known (e.g. XGBoost JSON). Returns the root node ID (0).
    pub fn init_root_with_n_nodes(&mut self, n_nodes: usize) -> NodeId {
        self.reset();
        for _ in 0..n_nodes {
            self.allocate_node();
        }
        0
    }

    /// Apply a numeric split to a node, allocating child nodes.
    ///
    /// Returns `(left_id, right_id)`.
    pub fn apply_numeric_split(
        &mut self,
        node: NodeId,
        feature: u32,
        threshold: f32,
        default_left: bool,
    ) -> (NodeId, NodeId) {
        let left_id = self.allocate_node();
        let right_id = self.allocate_node();

        let idx = node as usize;
        self.split_indices[idx] = feature;
        self.split_thresholds[idx] = threshold;
        self.left_children[idx] = left_id;
        self.right_children[idx] = right_id;
        self.default_left[idx] = default_left;
        self.is_leaf[idx] = false;
        self.split_types[idx] = SplitType::Numeric;

        (left_id, right_id)
    }

    /// Set a numeric split on an existing node, with explicit child indices.
    ///
    /// Intended for conversion code where the full node set is pre-allocated.
    pub fn set_numeric_split(
        &mut self,
        node: NodeId,
        feature: u32,
        threshold: f32,
        default_left: bool,
        left_child: NodeId,
        right_child: NodeId,
    ) {
        let idx = node as usize;
        self.split_indices[idx] = feature;
        self.split_thresholds[idx] = threshold;
        self.left_children[idx] = left_child;
        self.right_children[idx] = right_child;
        self.default_left[idx] = default_left;
        self.is_leaf[idx] = false;
        self.split_types[idx] = SplitType::Numeric;
    }

    /// Apply a categorical split to a node, allocating child nodes.
    ///
    /// The `category_bitset` contains the packed u32 words for the category bitset.
    /// Categories in this bitset go RIGHT, categories not in the set go LEFT.
    ///
    /// Returns `(left_id, right_id)`.
    pub fn apply_categorical_split(
        &mut self,
        node: NodeId,
        feature: u32,
        category_bitset: Vec<u32>,
        default_left: bool,
    ) -> (NodeId, NodeId) {
        let left_id = self.allocate_node();
        let right_id = self.allocate_node();

        let idx = node as usize;
        self.split_indices[idx] = feature;
        self.split_thresholds[idx] = 0.0;
        self.left_children[idx] = left_id;
        self.right_children[idx] = right_id;
        self.default_left[idx] = default_left;
        self.is_leaf[idx] = false;
        self.split_types[idx] = SplitType::Categorical;
        self.categorical_nodes.push((node, category_bitset));

        (left_id, right_id)
    }

    /// Set a categorical split on an existing node, with explicit child indices.
    ///
    /// Intended for conversion code where the full node set is pre-allocated.
    pub fn set_categorical_split(
        &mut self,
        node: NodeId,
        feature: u32,
        category_bitset: Vec<u32>,
        default_left: bool,
        left_child: NodeId,
        right_child: NodeId,
    ) {
        let idx = node as usize;
        self.split_indices[idx] = feature;
        self.split_thresholds[idx] = 0.0;
        self.left_children[idx] = left_child;
        self.right_children[idx] = right_child;
        self.default_left[idx] = default_left;
        self.is_leaf[idx] = false;
        self.split_types[idx] = SplitType::Categorical;

        if let Some(pos) = self.categorical_nodes.iter().position(|(n, _)| *n == node) {
            self.categorical_nodes.remove(pos);
        }
        self.categorical_nodes.push((node, category_bitset));
    }

    /// Set a node as a leaf with the given value.
    pub fn make_leaf(&mut self, node: NodeId, value: L) {
        let idx = node as usize;
        self.is_leaf[idx] = true;
        self.leaf_values[idx] = value;
    }

    /// Update an existing leaf's value.
    ///
    /// This is used for leaf value renewing after tree structure is finalized.
    /// For quantile objectives, the optimal leaf value is the α-quantile of
    /// residuals, not the Newton-step from gradient sums.
    ///
    /// # Panics
    ///
    /// Debug-panics if the node is not a leaf.
    #[inline]
    pub fn set_leaf_value(&mut self, node: NodeId, value: L) {
        let idx = node as usize;
        debug_assert!(
            self.is_leaf[idx],
            "set_leaf_value called on non-leaf node {}",
            node
        );
        self.leaf_values[idx] = value;
    }

    /// Set linear coefficients for a leaf node.
    ///
    /// The leaf should already be marked as a leaf via `make_leaf`.
    /// The `feature_indices` identify which features have non-zero coefficients,
    /// and `coefficients` are the corresponding weights.
    ///
    /// # Panics
    ///
    /// Debug-panics if `feature_indices.len() != coefficients.len()`.
    pub fn set_linear_leaf(
        &mut self,
        node: NodeId,
        feature_indices: Vec<u32>,
        intercept: f32,
        coefficients: Vec<f32>,
    ) {
        debug_assert_eq!(
            feature_indices.len(),
            coefficients.len(),
            "feature_indices and coefficients must have same length"
        );

        // Remove any existing entry for this node
        if let Some(pos) = self
            .linear_leaves
            .iter()
            .position(|(n, _, _, _)| *n == node)
        {
            self.linear_leaves.remove(pos);
        }
        self.linear_leaves
            .push((node, feature_indices, coefficients, intercept));
    }

    /// Apply learning rate to all leaf values.
    pub fn apply_learning_rate(&mut self, learning_rate: f32) {
        for (is_leaf, value) in self.is_leaf.iter().zip(self.leaf_values.iter_mut()) {
            if *is_leaf {
                value.scale(learning_rate);
            }
        }
    }

    /// Current number of allocated nodes.
    #[inline]
    pub fn n_nodes(&self) -> usize {
        self.split_indices.len()
    }

    /// Reset the tree for reuse.
    pub fn reset(&mut self) {
        self.split_indices.clear();
        self.split_thresholds.clear();
        self.left_children.clear();
        self.right_children.clear();
        self.default_left.clear();
        self.is_leaf.clear();
        self.leaf_values.clear();
        self.split_types.clear();
        self.categorical_nodes.clear();
        self.linear_leaves.clear();
        self.gains.clear();
        self.covers.clear();
        self.next_id = 0;
    }

    /// Finalize the tree and return immutable storage.
    pub fn freeze(self) -> Tree<L> {
        let num_nodes = self.split_indices.len();

        // Build categories storage
        let categories = if self.categorical_nodes.is_empty() {
            CategoriesStorage::empty()
        } else {
            let mut cat_nodes = self.categorical_nodes;
            cat_nodes.sort_by_key(|(idx, _)| *idx);

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

        // Build linear coefficients storage
        let leaf_coefficients = if self.linear_leaves.is_empty() {
            super::coefficients::LeafCoefficients::empty()
        } else {
            let mut builder = LeafCoefficientsBuilder::new();
            for (node, features, coefs, intercept) in self.linear_leaves {
                builder.add(node, &features, intercept, &coefs);
            }
            builder.build(num_nodes)
        };

        // Check if gains/covers were populated (any non-zero values).
        // During training, the grower always populates stats. But loaders
        // (XGBoost/LightGBM) may not have this data, so we check.
        let has_stats =
            self.gains.iter().any(|&g| g != 0.0) || self.covers.iter().any(|&c| c != 0.0);

        let mut tree = Tree::new(
            self.split_indices,
            self.split_thresholds,
            self.left_children,
            self.right_children,
            self.default_left,
            self.is_leaf,
            self.leaf_values,
            self.split_types,
            categories,
            leaf_coefficients,
        );

        // Attach gains/covers if they were populated
        if has_stats {
            tree = tree.with_stats(self.gains, self.covers);
        }

        tree
    }

    fn allocate_node(&mut self) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;

        self.split_indices.push(0);
        self.split_thresholds.push(0.0);
        self.left_children.push(0);
        self.right_children.push(0);
        self.default_left.push(false);
        self.is_leaf.push(false);
        self.leaf_values.push(L::default());
        self.split_types.push(SplitType::Numeric);
        self.gains.push(0.0);
        self.covers.push(0.0);

        id
    }

    /// Set gain and cover for a node (for explainability).
    ///
    /// Should be called after applying a split with the split's gain
    /// and the node's hessian sum (cover).
    #[inline]
    pub fn set_node_stats(&mut self, node: NodeId, gain: f32, cover: f32) {
        if let Some(g) = self.gains.get_mut(node as usize) {
            *g = gain;
        }
        if let Some(c) = self.covers.get_mut(node as usize) {
            *c = cover;
        }
    }

    /// Set only cover for a node (e.g., for leaves).
    #[inline]
    pub fn set_cover(&mut self, node: NodeId, cover: f32) {
        if let Some(c) = self.covers.get_mut(node as usize) {
            *c = cover;
        }
    }
}

// =============================================================================
// TreeView for MutableTree
// =============================================================================

impl<L: LeafValue> TreeView for MutableTree<L> {
    type LeafValue = L;

    #[inline]
    fn n_nodes(&self) -> usize {
        self.split_indices.len()
    }

    #[inline]
    fn is_leaf(&self, node: NodeId) -> bool {
        self.is_leaf[node as usize]
    }

    #[inline]
    fn split_index(&self, node: NodeId) -> u32 {
        self.split_indices[node as usize]
    }

    #[inline]
    fn split_threshold(&self, node: NodeId) -> f32 {
        self.split_thresholds[node as usize]
    }

    #[inline]
    fn left_child(&self, node: NodeId) -> NodeId {
        self.left_children[node as usize]
    }

    #[inline]
    fn right_child(&self, node: NodeId) -> NodeId {
        self.right_children[node as usize]
    }

    #[inline]
    fn default_left(&self, node: NodeId) -> bool {
        self.default_left[node as usize]
    }

    #[inline]
    fn split_type(&self, node: NodeId) -> SplitType {
        self.split_types[node as usize]
    }

    #[inline]
    fn categories(&self) -> &CategoriesStorage {
        // During construction, categorical splits are stored per-node
        // and not yet packed. Returns a static empty storage.
        // Categorical traversal is only supported after freeze().
        CategoriesStorage::empty_ref()
    }

    #[inline]
    fn leaf_value(&self, node: NodeId) -> &L {
        &self.leaf_values[node as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::SamplesView;
    use crate::repr::gbdt::ScalarLeaf;
    use ndarray::Array2;

    #[test]
    fn test_mutable_tree_linear_leaf_freeze() {
        let mut tree: MutableTree<ScalarLeaf> = MutableTree::new();

        // Create a simple tree with one split
        let root = tree.init_root();
        let (left, right) = tree.apply_numeric_split(root, 0, 0.5, false);
        tree.make_leaf(left, ScalarLeaf(1.0));
        tree.make_leaf(right, ScalarLeaf(2.0));

        // Set linear coefficients on the left leaf only
        tree.set_linear_leaf(left, vec![0], 0.5, vec![1.0]);

        // Freeze and verify
        let frozen = tree.freeze();

        // Left leaf (node 1) should have linear terms
        let terms = frozen.leaf_terms(1);
        assert!(terms.is_some());
        let (features, coefs) = terms.unwrap();
        assert_eq!(features, &[0u32]);
        assert_eq!(coefs, &[1.0f32]);
        assert_eq!(frozen.leaf_intercept(1), 0.5);

        // Right leaf (node 2) should not have linear terms
        assert!(frozen.leaf_terms(2).is_none());
        assert_eq!(frozen.leaf_intercept(2), 0.0);

        // Tree should report having linear leaves
        assert!(frozen.has_linear_leaves());
    }

    #[test]
    fn test_mutable_tree_reset_clears_linear() {
        let mut tree: MutableTree<ScalarLeaf> = MutableTree::new();

        // Create tree with linear leaf
        let root = tree.init_root();
        tree.make_leaf(root, ScalarLeaf(1.0));
        tree.set_linear_leaf(root, vec![0], 0.5, vec![1.0]);

        // Reset should clear linear leaves
        tree.reset();

        // Freeze an empty tree
        let frozen = tree.freeze();
        assert!(!frozen.has_linear_leaves());
    }

    #[test]
    fn test_traverse_on_mutable_tree() {
        let mut tree = MutableTree::<ScalarLeaf>::new();
        let _root = tree.init_root();

        // Apply a split at root: feature 0, threshold 0.5
        let (left, right) = tree.apply_numeric_split(0, 0, 0.5, true);
        tree.make_leaf(left, ScalarLeaf(-1.0));
        tree.make_leaf(right, ScalarLeaf(1.0));

        // Verify TreeView works on MutableTree
        assert!(!TreeView::is_leaf(&tree, 0));
        assert!(TreeView::is_leaf(&tree, left));
        assert!(TreeView::is_leaf(&tree, right));
        assert_eq!(TreeView::split_index(&tree, 0), 0);
        assert_eq!(TreeView::split_threshold(&tree, 0), 0.5);

        // Test traversal with MutableTree
        // 2 samples, 1 feature each: [[0.3], [0.7]]
        let arr = Array2::from_shape_vec((2, 1), vec![0.3, 0.7]).unwrap();
        let row_data = SamplesView::from_array(arr.view());
        let sample_0 = row_data.sample_view(0);
        let sample_1 = row_data.sample_view(1);
        let leaf_0 = tree.traverse_to_leaf(sample_0);
        let leaf_1 = tree.traverse_to_leaf(sample_1);

        assert_eq!(leaf_0, left);
        assert_eq!(leaf_1, right);
    }
}

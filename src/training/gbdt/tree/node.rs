//! Tree node and immutable tree structures.
//!
//! This module provides the core tree data structures for prediction:
//! - [`TreeNode`]: A single tree node (split or leaf)
//! - [`Tree`]: An immutable trained decision tree
//! - [`CategoricalSplit`]: Categorical split information
//! - [`NodeId`]: Type alias for tree node indices

use super::super::categorical::CatBitset;

/// Type alias for tree node indices.
pub type NodeId = u32;

/// Sentinel value for "no child" (leaf nodes).
pub const NO_CHILD: NodeId = u32::MAX;

/// A single tree node.
///
/// Uses struct layout (not enum) for cache-friendly fixed-size nodes.
/// The `is_leaf` flag distinguishes split vs leaf nodes.
#[derive(Clone, Debug)]
pub struct TreeNode {
    // Split information (only valid when is_leaf=false)
    /// Feature index for split.
    pub feature: u32,
    /// Bin threshold (numerical: bin <= threshold goes left).
    pub threshold: u16,
    /// Direction for missing values.
    pub default_left: bool,
    /// Left child node index.
    pub left: NodeId,
    /// Right child node index.
    pub right: NodeId,

    // Leaf information (only valid when is_leaf=true)
    /// Leaf prediction value.
    pub value: f32,

    /// Whether this is a leaf node.
    pub is_leaf: bool,
}

impl Default for TreeNode {
    fn default() -> Self {
        Self {
            feature: 0,
            threshold: 0,
            default_left: false,
            left: NO_CHILD,
            right: NO_CHILD,
            value: 0.0,
            is_leaf: true, // Default to leaf
        }
    }
}

impl TreeNode {
    /// Create a leaf node.
    #[inline]
    pub fn leaf(value: f32) -> Self {
        Self {
            is_leaf: true,
            value,
            ..Default::default()
        }
    }

    /// Create a numerical split node.
    #[inline]
    pub fn numerical_split(feature: u32, threshold: u16, default_left: bool) -> Self {
        Self {
            feature,
            threshold,
            default_left,
            is_leaf: false,
            ..Default::default()
        }
    }
}

/// Categorical split information (stored separately from TreeNode).
///
/// Only allocated for nodes that have categorical splits to avoid
/// bloating the common case.
#[derive(Clone, Debug)]
pub struct CategoricalSplit {
    /// Node ID this split belongs to.
    pub node: NodeId,
    /// Categories that go left.
    pub left_cats: CatBitset,
}

/// An immutable trained decision tree.
///
/// Nodes are stored in a contiguous array with depth-first ordering.
/// Root is always at index 0.
#[derive(Clone, Debug)]
pub struct Tree {
    /// Tree nodes.
    pub(crate) nodes: Vec<TreeNode>,
    /// Categorical splits (only for nodes with categorical features).
    pub(crate) categorical_splits: Vec<CategoricalSplit>,
    /// Number of leaf nodes.
    pub(crate) n_leaves: u32,
    /// Maximum depth reached.
    pub(crate) max_depth: u16,
}

impl Tree {
    /// Create a tree from components (used by TreeBuilder).
    pub(crate) fn new(
        nodes: Vec<TreeNode>,
        categorical_splits: Vec<CategoricalSplit>,
        n_leaves: u32,
        max_depth: u16,
    ) -> Self {
        Self { nodes, categorical_splits, n_leaves, max_depth }
    }

    /// Get the root node ID (always 0).
    #[inline]
    pub fn root(&self) -> NodeId {
        0
    }

    /// Get a node by ID.
    #[inline]
    pub fn node(&self, id: NodeId) -> &TreeNode {
        &self.nodes[id as usize]
    }

    /// Get all nodes.
    #[inline]
    pub fn nodes(&self) -> &[TreeNode] {
        &self.nodes
    }

    /// Number of leaf nodes.
    #[inline]
    pub fn n_leaves(&self) -> u32 {
        self.n_leaves
    }

    /// Total number of nodes.
    #[inline]
    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Maximum depth reached.
    #[inline]
    pub fn max_depth(&self) -> u16 {
        self.max_depth
    }

    /// Get categorical split info for a node, if any.
    pub fn categorical_split(&self, node: NodeId) -> Option<&CategoricalSplit> {
        self.categorical_splits.iter().find(|s| s.node == node)
    }

    /// Predict for a validated row view.
    ///
    /// # Arguments
    /// * `row` - A validated row view from `BinnedDataset::row_view()`
    ///
    /// # Example
    ///
    /// ```ignore
    /// for row in 0..dataset.n_rows() {
    ///     if let Some(view) = dataset.row_view(row) {
    ///         let pred = tree.predict(&view);
    ///     }
    /// }
    /// ```
    pub fn predict(&self, row: &crate::data::binned::RowView<'_>) -> f32 {
        let mut node_id = self.root();

        loop {
            let node = &self.nodes[node_id as usize];

            if node.is_leaf {
                return node.value;
            }

            let feature = node.feature as usize;
            let bin_opt = row.get_bin(feature);

            // Handle missing values
            let go_left = match bin_opt {
                None => node.default_left,
                Some(bin) => {
                    if let Some(cat_split) = self.categorical_split(node_id) {
                        // Categorical split
                        cat_split.left_cats.contains(bin)
                    } else {
                        // Numerical split
                        bin <= node.threshold as u32
                    }
                }
            };

            node_id = if go_left { node.left } else { node.right };
        }
    }

    /// Iterate over all leaf nodes.
    pub fn iter_leaves(&self) -> impl Iterator<Item = (NodeId, &TreeNode)> {
        self.nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.is_leaf)
            .map(|(i, n)| (i as NodeId, n))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tree_node_leaf() {
        let node = TreeNode::leaf(0.5);
        assert!(node.is_leaf);
        assert_eq!(node.value, 0.5);
        assert_eq!(node.left, NO_CHILD);
        assert_eq!(node.right, NO_CHILD);
    }

    #[test]
    fn test_tree_node_split() {
        let node = TreeNode::numerical_split(2, 5, true);
        assert!(!node.is_leaf);
        assert_eq!(node.feature, 2);
        assert_eq!(node.threshold, 5);
        assert!(node.default_left);
    }
}

//! Tree data structures for building.
//!
//! This module contains the mutable tree structures used during training:
//!
//! - [`BuildingNode`]: A single node in a tree being constructed
//! - [`BuildingTree`]: The tree structure with expand operations
//! - [`NodeCandidate`]: A candidate node for expansion with its best split

use super::super::split::SplitInfo;

// ============================================================================
// BuildingNode
// ============================================================================

/// A node in a tree being constructed.
///
/// During building, nodes start as leaves and can be expanded into split nodes.
/// This structure maintains parent/child links for tree traversal.
#[derive(Debug, Clone)]
pub struct BuildingNode {
    /// Split information (None if this is a leaf)
    pub split: Option<SplitInfo>,
    /// Left child index (u32::MAX if leaf)
    pub left: u32,
    /// Right child index (u32::MAX if leaf)
    pub right: u32,
    /// Parent index (u32::MAX if root)
    pub parent: u32,
    /// Depth in tree (root = 0)
    pub depth: u32,
    /// Leaf weight value
    pub weight: f32,
    /// Whether this node is currently a leaf
    pub is_leaf: bool,
    /// Partition node ID (for accessing rows)
    pub partition_id: u32,
}

impl BuildingNode {
    /// Create a new leaf node.
    pub(crate) fn new_leaf(weight: f32, parent: u32, depth: u32, partition_id: u32) -> Self {
        Self {
            split: None,
            left: u32::MAX,
            right: u32::MAX,
            parent,
            depth,
            weight,
            is_leaf: true,
            partition_id,
        }
    }
}

// ============================================================================
// BuildingTree
// ============================================================================

/// A tree being constructed during training.
///
/// Maintains a mutable tree structure that can be expanded node by node.
/// Once complete, can be converted to an immutable inference tree.
#[derive(Debug, Clone)]
pub struct BuildingTree {
    /// All nodes in the tree
    nodes: Vec<BuildingNode>,
    /// Current number of leaves
    num_leaves: u32,
    /// Maximum depth reached
    max_depth: u32,
}

impl BuildingTree {
    /// Create a new tree with just a root leaf.
    ///
    /// The root node starts as a leaf with the given base weight.
    pub fn new(base_weight: f32) -> Self {
        Self {
            nodes: vec![BuildingNode::new_leaf(base_weight, u32::MAX, 0, 0)],
            num_leaves: 1,
            max_depth: 0,
        }
    }

    /// Number of nodes in the tree.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Number of leaves in the tree.
    #[inline]
    pub fn num_leaves(&self) -> u32 {
        self.num_leaves
    }

    /// Maximum depth of the tree.
    #[inline]
    pub fn max_depth(&self) -> u32 {
        self.max_depth
    }

    /// Get a node by index.
    #[inline]
    pub fn node(&self, index: u32) -> &BuildingNode {
        &self.nodes[index as usize]
    }

    /// Get a mutable node by index.
    #[inline]
    pub fn node_mut(&mut self, index: u32) -> &mut BuildingNode {
        &mut self.nodes[index as usize]
    }

    /// Expand a leaf node into a split node with two children.
    ///
    /// # Arguments
    ///
    /// * `node_id` - Index of the leaf node to expand
    /// * `split` - Split information (feature, threshold, etc.)
    /// * `left_partition` - Partition node ID for left child rows
    /// * `right_partition` - Partition node ID for right child rows
    ///
    /// # Returns
    ///
    /// `(left_child_id, right_child_id)` - Indices of the new child nodes
    ///
    /// # Panics
    ///
    /// Panics if `node_id` is not a leaf.
    pub fn expand(
        &mut self,
        node_id: u32,
        split: SplitInfo,
        left_partition: u32,
        right_partition: u32,
    ) -> (u32, u32) {
        debug_assert!(
            self.nodes[node_id as usize].is_leaf,
            "Cannot expand non-leaf node"
        );

        let node = &self.nodes[node_id as usize];
        let depth = node.depth;

        let left_id = self.nodes.len() as u32;
        let right_id = left_id + 1;

        // Create left and right children
        self.nodes.push(BuildingNode::new_leaf(
            split.weight_left,
            node_id,
            depth + 1,
            left_partition,
        ));
        self.nodes.push(BuildingNode::new_leaf(
            split.weight_right,
            node_id,
            depth + 1,
            right_partition,
        ));

        // Convert parent to split node
        let node = &mut self.nodes[node_id as usize];
        node.split = Some(split);
        node.left = left_id;
        node.right = right_id;
        node.is_leaf = false;

        self.num_leaves += 1; // net +1 (remove 1, add 2)
        self.max_depth = self.max_depth.max(depth + 1);

        (left_id, right_id)
    }

    /// Iterate over all current leaf node indices.
    pub fn leaves(&self) -> impl Iterator<Item = u32> + '_ {
        (0..self.nodes.len() as u32).filter(|&i| self.nodes[i as usize].is_leaf)
    }

    /// Apply learning rate to all leaf weights.
    pub fn apply_learning_rate(&mut self, learning_rate: f32) {
        for node in &mut self.nodes {
            if node.is_leaf {
                node.weight *= learning_rate;
            }
        }
    }
}

// ============================================================================
// NodeCandidate
// ============================================================================

/// A candidate node for expansion.
///
/// Tracks the best split found for a node along with metadata for the
/// growth policy to make decisions.
#[derive(Debug, Clone)]
pub struct NodeCandidate {
    /// Index in the BuildingTree
    pub node_id: u32,
    /// Best split found for this node
    pub split: SplitInfo,
    /// Depth of this node
    pub depth: u32,
    /// Number of samples in this node
    pub num_samples: u32,
}

impl NodeCandidate {
    /// Create a new candidate.
    pub fn new(node_id: u32, split: SplitInfo, depth: u32, num_samples: u32) -> Self {
        Self {
            node_id,
            split,
            depth,
            num_samples,
        }
    }

    /// Get the gain of the best split.
    #[inline]
    pub fn gain(&self) -> f32 {
        self.split.gain
    }

    /// Check if this candidate has a valid split.
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.split.gain > 0.0
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a test split with common defaults.
    fn test_split(gain: f32, weight_left: f32, weight_right: f32) -> SplitInfo {
        SplitInfo {
            feature: 0,
            split_bin: 5,
            threshold: 0.5,
            gain,
            grad_left: 1.0,
            hess_left: 1.0,
            grad_right: 1.0,
            hess_right: 1.0,
            weight_left,
            weight_right,
            default_left: true,
            is_categorical: false,
            categories_left: vec![],
        }
    }

    #[test]
    fn test_building_node_new_leaf() {
        let node = BuildingNode::new_leaf(0.5, 0, 1, 2);
        assert!(node.is_leaf);
        assert!(node.split.is_none());
        assert_eq!(node.weight, 0.5);
        assert_eq!(node.parent, 0);
        assert_eq!(node.depth, 1);
        assert_eq!(node.partition_id, 2);
        assert_eq!(node.left, u32::MAX);
        assert_eq!(node.right, u32::MAX);
    }

    #[test]
    fn test_building_tree_new() {
        let tree = BuildingTree::new(0.1);
        assert_eq!(tree.num_nodes(), 1);
        assert_eq!(tree.num_leaves(), 1);
        assert_eq!(tree.max_depth(), 0);

        let root = tree.node(0);
        assert!(root.is_leaf);
        assert_eq!(root.weight, 0.1);
        assert_eq!(root.depth, 0);
    }

    #[test]
    fn test_building_tree_expand() {
        let mut tree = BuildingTree::new(0.0);

        let split = test_split(1.0, -0.2, 0.3);
        let (left, right) = tree.expand(0, split, 1, 2);

        assert_eq!(left, 1);
        assert_eq!(right, 2);
        assert_eq!(tree.num_nodes(), 3);
        assert_eq!(tree.num_leaves(), 2);
        assert_eq!(tree.max_depth(), 1);

        // Root is no longer a leaf
        let root = tree.node(0);
        assert!(!root.is_leaf);
        assert!(root.split.is_some());
        assert_eq!(root.left, 1);
        assert_eq!(root.right, 2);

        // Children are leaves
        let left_node = tree.node(1);
        assert!(left_node.is_leaf);
        assert_eq!(left_node.weight, -0.2);
        assert_eq!(left_node.parent, 0);
        assert_eq!(left_node.depth, 1);
        assert_eq!(left_node.partition_id, 1);

        let right_node = tree.node(2);
        assert!(right_node.is_leaf);
        assert_eq!(right_node.weight, 0.3);
        assert_eq!(right_node.parent, 0);
        assert_eq!(right_node.depth, 1);
        assert_eq!(right_node.partition_id, 2);
    }

    #[test]
    fn test_building_tree_leaves() {
        let mut tree = BuildingTree::new(0.0);

        // Initially just root
        let leaves: Vec<u32> = tree.leaves().collect();
        assert_eq!(leaves, vec![0]);

        // Expand root
        let split = test_split(1.0, 0.0, 0.0);
        tree.expand(0, split.clone(), 1, 2);

        let leaves: Vec<u32> = tree.leaves().collect();
        assert_eq!(leaves, vec![1, 2]);

        // Expand left child
        tree.expand(1, split.clone(), 3, 4);

        let leaves: Vec<u32> = tree.leaves().collect();
        assert_eq!(leaves, vec![2, 3, 4]);
    }

    #[test]
    fn test_building_tree_apply_learning_rate() {
        let mut tree = BuildingTree::new(0.0);

        let split = test_split(1.0, 1.0, 2.0);
        tree.expand(0, split, 1, 2);

        tree.apply_learning_rate(0.1);

        // Only leaf weights are scaled
        let left = tree.node(1);
        let right = tree.node(2);
        assert!((left.weight - 0.1).abs() < 1e-6);
        assert!((right.weight - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_node_candidate() {
        let split = test_split(1.5, 0.0, 0.0);
        let candidate = NodeCandidate::new(0, split, 2, 100);

        assert_eq!(candidate.node_id, 0);
        assert_eq!(candidate.depth, 2);
        assert_eq!(candidate.num_samples, 100);
        assert!((candidate.gain() - 1.5).abs() < 1e-6);
        assert!(candidate.is_valid());
    }

    #[test]
    fn test_node_candidate_invalid() {
        let split = test_split(0.0, 0.0, 0.0); // No gain
        let candidate = NodeCandidate::new(0, split, 0, 100);
        assert!(!candidate.is_valid());
    }
}

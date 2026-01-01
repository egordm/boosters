//! Read-only tree interface and generic tree validation.
//!
//! This module provides:
//! - [`TreeView`]: Read-only trait for unified tree access
//! - [`TreeValidationError`]: Structural validation errors  
//! - [`validate_tree`]: Generic tree structure validation

use ndarray::ArrayView1;

use super::NodeId;
use super::categories::{CategoriesStorage, float_to_category};
use super::types::LeafValue;
use super::types::SplitType;

// ============================================================================
// TreeView Trait
// ============================================================================

/// Read-only view of a tree for traversal.
///
/// Provides the minimal interface needed to traverse a tree from root to leaf.
/// Implemented for both [`Tree`](super::Tree) and [`MutableTree`](super::MutableTree).
///
/// # Design
///
/// This trait abstracts tree structure access, enabling generic prediction code
/// that works with both immutable trees (inference) and mutable trees (training).
///
/// # Example
///
/// ```ignore
/// use boosters::repr::gbdt::TreeView;
///
/// fn count_leaves<T: TreeView>(tree: &T) -> usize {
///     (0..tree.n_nodes())
///         .filter(|&n| tree.is_leaf(n as u32))
///         .count()
/// }
/// ```
pub trait TreeView {
    /// The leaf value type (e.g., `ScalarLeaf`).
    type LeafValue: LeafValue;

    /// Number of nodes in the tree.
    fn n_nodes(&self) -> usize;

    /// Check if a node is a leaf.
    fn is_leaf(&self, node: NodeId) -> bool;

    /// Get the feature index for a split node.
    fn split_index(&self, node: NodeId) -> u32;

    /// Get the split threshold for a numeric split.
    fn split_threshold(&self, node: NodeId) -> f32;

    /// Get the left child node index.
    fn left_child(&self, node: NodeId) -> NodeId;

    /// Get the right child node index.
    fn right_child(&self, node: NodeId) -> NodeId;

    /// Get the default direction for missing values.
    fn default_left(&self, node: NodeId) -> bool;

    /// Get the split type (numeric or categorical).
    fn split_type(&self, node: NodeId) -> SplitType;

    /// Get reference to categories storage for categorical splits.
    fn categories(&self) -> &CategoriesStorage;

    /// Get the leaf value at a leaf node.
    fn leaf_value(&self, node: NodeId) -> &Self::LeafValue;

    /// Check if the tree has any categorical splits.
    fn has_categorical(&self) -> bool {
        !self.categories().is_empty()
    }

    /// Traverse the tree to find the leaf node for a sample.
    ///
    /// This is the primary traversal method that works with a feature slice.
    /// The traversal handles NaN values using the tree's default direction,
    /// and supports both numeric and categorical splits.
    ///
    /// # Arguments
    ///
    /// * `sample` - Feature values for the sample
    ///
    /// # Returns
    ///
    /// The `NodeId` of the reached leaf node.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use boosters::repr::gbdt::TreeView;
    /// let features: &[f32] = &[0.5, 1.0, 2.3];
    /// let leaf_id = tree.traverse_to_leaf(ndarray::ArrayView1::from(features));
    /// ```
    #[inline]
    fn traverse_to_leaf(&self, sample: ArrayView1<'_, f32>) -> NodeId {
        self.traverse_to_leaf_from(0, sample)
    }

    /// Traverse the tree starting from a specific node.
    ///
    /// This is useful for resuming traversal after unrolled levels or
    /// partial tree traversal.
    ///
    /// # Arguments
    ///
    /// * `start_node` - Node ID to start traversal from
    /// * `sample` - Feature values for the sample
    ///
    /// # Returns
    ///
    /// The `NodeId` of the reached leaf node.
    #[inline]
    fn traverse_to_leaf_from(&self, start_node: NodeId, sample: ArrayView1<'_, f32>) -> NodeId {
        let mut node = start_node;

        while !self.is_leaf(node) {
            let feat_idx = self.split_index(node) as usize;
            debug_assert!(
                feat_idx < sample.len(),
                "sample has {} features but split needs feature {feat_idx}",
                sample.len()
            );
            let fvalue = sample[feat_idx];

            node = if fvalue.is_nan() {
                // Missing value: use default direction
                if self.default_left(node) {
                    self.left_child(node)
                } else {
                    self.right_child(node)
                }
            } else {
                match self.split_type(node) {
                    SplitType::Numeric => {
                        if fvalue < self.split_threshold(node) {
                            self.left_child(node)
                        } else {
                            self.right_child(node)
                        }
                    }
                    SplitType::Categorical => {
                        let category = float_to_category(fvalue);
                        if self.categories().category_goes_right(node, category) {
                            self.right_child(node)
                        } else {
                            self.left_child(node)
                        }
                    }
                }
            };
        }

        node
    }
}

// ============================================================================
// TreeValidationError
// ============================================================================

/// Structural validation errors for trees.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TreeValidationError {
    /// Tree has no nodes.
    EmptyTree,
    /// A child pointer references an out-of-bounds node.
    ChildOutOfBounds {
        node: NodeId,
        side: &'static str,
        child: NodeId,
        n_nodes: usize,
    },
    /// A node references itself as a child.
    SelfLoop { node: NodeId },
    /// A node was reached by more than one path (DAG) or due to a cycle.
    DuplicateVisit { node: NodeId },
    /// A cycle was detected during traversal.
    CycleDetected { node: NodeId },
    /// A node exists in storage but is unreachable from the root.
    UnreachableNode { node: NodeId },
    /// Tree contains categorical splits but the category segments array is not sized to nodes.
    CategoricalSegmentsLenMismatch { segments_len: usize, n_nodes: usize },
}

// ============================================================================
// Generic Validation
// ============================================================================

/// Validate basic structural invariants for any tree.
///
/// This function validates properties that can be checked via the [`TreeView`] trait:
/// - Tree is not empty
/// - All child pointers are within bounds
/// - No self-loops
/// - No cycles (proper tree structure)
/// - All nodes are reachable from root
///
/// For implementation-specific validation (like categorical segments), use
/// the type's own `validate()` method which calls this function.
///
/// # Example
///
/// ```ignore
/// use boosters::repr::gbdt::{TreeView, validate_tree};
///
/// fn check_tree<T: TreeView>(tree: &T) -> bool {
///     validate_tree(tree).is_ok()
/// }
/// ```
pub fn validate_tree<T: TreeView>(tree: &T) -> Result<(), TreeValidationError> {
    let n_nodes = tree.n_nodes();
    if n_nodes == 0 {
        return Err(TreeValidationError::EmptyTree);
    }

    // Iterative DFS with color marking.
    // 0 = unvisited, 1 = visiting, 2 = done
    let mut color = vec![0u8; n_nodes];
    let mut stack: Vec<(NodeId, u8)> = vec![(0, 0)];

    while let Some((node, phase)) = stack.pop() {
        let node_usize = node as usize;
        if node_usize >= n_nodes {
            return Err(TreeValidationError::ChildOutOfBounds {
                node,
                side: "root",
                child: node,
                n_nodes,
            });
        }

        match phase {
            0 => {
                match color[node_usize] {
                    0 => {}
                    1 => return Err(TreeValidationError::CycleDetected { node }),
                    2 => return Err(TreeValidationError::DuplicateVisit { node }),
                    _ => unreachable!(),
                }

                color[node_usize] = 1;
                stack.push((node, 1));

                if !tree.is_leaf(node) {
                    let left = tree.left_child(node);
                    let right = tree.right_child(node);

                    if left == node || right == node {
                        return Err(TreeValidationError::SelfLoop { node });
                    }

                    let left_usize = left as usize;
                    if left_usize >= n_nodes {
                        return Err(TreeValidationError::ChildOutOfBounds {
                            node,
                            side: "left",
                            child: left,
                            n_nodes,
                        });
                    }
                    let right_usize = right as usize;
                    if right_usize >= n_nodes {
                        return Err(TreeValidationError::ChildOutOfBounds {
                            node,
                            side: "right",
                            child: right,
                            n_nodes,
                        });
                    }

                    // Visit children
                    stack.push((right, 0));
                    stack.push((left, 0));
                }
            }
            1 => {
                color[node_usize] = 2;
            }
            _ => unreachable!(),
        }
    }

    for (i, &c) in color.iter().enumerate() {
        if c == 0 {
            return Err(TreeValidationError::UnreachableNode { node: i as u32 });
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repr::gbdt::coefficients::LeafCoefficients;
    use crate::repr::gbdt::types::SplitType;
    use crate::repr::gbdt::{ScalarLeaf, Tree};

    #[test]
    fn validate_tree_empty_tree_fails() {
        // A tree with 0 nodes should fail
        let tree: Tree<ScalarLeaf> = Tree::new(
            vec![], // split_indices
            vec![], // split_thresholds
            vec![], // left_children
            vec![], // right_children
            vec![], // default_left
            vec![], // is_leaf
            vec![], // leaf_values
            vec![], // split_types
            CategoriesStorage::empty(),
            LeafCoefficients::empty(),
        );
        assert_eq!(validate_tree(&tree), Err(TreeValidationError::EmptyTree));
    }

    #[test]
    fn validate_tree_single_leaf_succeeds() {
        // Single leaf node is valid
        let tree: Tree<ScalarLeaf> = Tree::new(
            vec![0],               // split_indices (unused for leaf)
            vec![0.0],             // split_thresholds (unused for leaf)
            vec![0],               // left_children (self-referencing, but it's a leaf)
            vec![0],               // right_children
            vec![true],            // default_left
            vec![true],            // is_leaf - this node IS a leaf
            vec![ScalarLeaf(1.0)], // leaf_values
            vec![SplitType::Numeric],
            CategoriesStorage::empty(),
            LeafCoefficients::empty(),
        );
        assert!(validate_tree(&tree).is_ok());
    }

    #[test]
    fn validate_tree_simple_tree_succeeds() {
        // Root with two leaf children
        // Node 0: root (split), Node 1: left leaf, Node 2: right leaf
        let tree: Tree<ScalarLeaf> = Tree::new(
            vec![0, 0, 0],           // split_indices (feature 0 at root)
            vec![0.5, 0.0, 0.0],     // split_thresholds (0.5 at root)
            vec![1, 1, 2],           // left_children (root->1, leaves point to self)
            vec![2, 1, 2],           // right_children (root->2, leaves point to self)
            vec![true, true, true],  // default_left
            vec![false, true, true], // is_leaf: root is NOT a leaf, children ARE leaves
            vec![ScalarLeaf(0.0), ScalarLeaf(1.0), ScalarLeaf(2.0)],
            vec![SplitType::Numeric, SplitType::Numeric, SplitType::Numeric],
            CategoriesStorage::empty(),
            LeafCoefficients::empty(),
        );
        assert!(validate_tree(&tree).is_ok());
    }
}

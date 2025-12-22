//! Tree traversal strategies for prediction.
//!
//! This module provides the [`TreeTraversal`] trait and implementations that
//! abstract how trees are traversed during prediction.
//!
//! # Available Strategies
//!
//! - [`StandardTraversal`]: Direct node-by-node traversal (simple, good for single rows)
//! - [`UnrolledTraversal`]: Uses [`UnrolledTreeLayout`] for cache-friendly batch traversal

use crate::repr::gbdt::{LeafValue, NodeId, ScalarLeaf, SplitType, Tree, TreeView, float_to_category};
use super::{Depth6, UnrollDepth, UnrolledTreeLayout};

// =============================================================================
// TreeTraversal Trait
// =============================================================================

/// Strategy for traversing a tree during prediction.
///
/// Implementations define how features are traversed through a tree to reach
/// leaf values. The trait supports both single-row and batch traversal.
///
/// # Type Parameters
///
/// - `L`: Leaf value type (e.g., [`ScalarLeaf`])
pub trait TreeTraversal<L: LeafValue>: Clone {
    /// State held per-tree for this traversal strategy.
    ///
    /// For simple traversal, this is `()`. For unrolled traversal, this is
    /// `UnrolledTreeLayout<D>`.
    type TreeState: Clone + Send + Sync;

    /// Whether this traversal benefits from block-level optimization.
    ///
    /// When true, the predictor will use `traverse_block` for batch prediction.
    /// When false, it will use the simpler per-row `traverse_tree` approach.
    const USES_BLOCK_OPTIMIZATION: bool = false;

    /// Build traversal state for a tree.
    ///
    /// Called once per tree when creating a predictor.
    fn build_tree_state(tree: &Tree<L>) -> Self::TreeState;

    /// Traverse a tree with given features, returning the leaf value.
    fn traverse_tree(tree: &Tree<L>, state: &Self::TreeState, features: &[f32]) -> NodeId;

    /// Traverse a tree for a block of rows, accumulating results.
    ///
    /// Default implementation calls `traverse_tree` per-row. Specialized
    /// implementations (like `UnrolledTraversal`) override this for
    /// better cache efficiency by processing all rows level-by-level.
    ///
    /// # Arguments
    ///
    /// - `tree`: The tree to traverse
    /// - `state`: Pre-computed state for this tree
    /// - `feature_buffer`: Contiguous buffer of features, `block_size * num_features`
    /// - `num_features`: Number of features per row
    #[inline]
    fn traverse_block(
        tree: &Tree<L>,
        state: &Self::TreeState,
        feature_buffer: &[f32],
        num_features: usize,
        output: &mut [NodeId],
    ) where
        L: Into<f32>,
    {
        // Default: per-row traversal
        for (row_idx, out) in output.iter_mut().enumerate() {
            let row_offset = row_idx * num_features;
            let row_features = &feature_buffer[row_offset..][..num_features];
            *out = Self::traverse_tree(tree, state, row_features);
        }
    }
}

// =============================================================================
// Shared Traversal Helper
// =============================================================================

/// Continue traversal from a given node to a leaf.
///
/// This is the core traversal loop used by multiple implementations.
/// Handles numeric splits and missing values. For categorical splits,
/// use the full tree visitor.
///
/// # Arguments
///
/// - `tree`: The tree view
/// - `start_node`: Node index to start from
/// - `features`: Feature values
///
/// # Returns
///
/// The leaf node index reached.
#[inline]
pub fn traverse_from_node(tree: &Tree<ScalarLeaf>, start_node: u32, features: &[f32]) -> NodeId {
    let mut idx = start_node;

    while !tree.is_leaf(idx) {
        let feat_idx = tree.split_index(idx) as usize;
        let fvalue = features.get(feat_idx).copied().unwrap_or(f32::NAN);

        idx = if fvalue.is_nan() {
            // Missing value: use default direction
            if tree.default_left(idx) {
                tree.left_child(idx)
            } else {
                tree.right_child(idx)
            }
        } else {
            match tree.split_type(idx) {
                SplitType::Numeric => {
                    if fvalue < tree.split_threshold(idx) {
                        tree.left_child(idx)
                    } else {
                        tree.right_child(idx)
                    }
                }
                SplitType::Categorical => {
                    let category = float_to_category(fvalue);
                    if tree.categories().category_goes_right(idx, category) {
                        tree.right_child(idx)
                    } else {
                        tree.left_child(idx)
                    }
                }
            }
        };
    }

    idx
}

// =============================================================================
// StandardTraversal
// =============================================================================

/// Standard node-by-node tree traversal.
///
/// Traverses from root to leaf following split conditions. Handles:
/// - Numeric splits: `feature < threshold`
/// - Categorical splits: category in bitset
/// - Missing values: follow default direction
///
/// This is the simplest traversal strategy and works well for single-row
/// predictions or small batches.
#[derive(Debug, Clone, Copy, Default)]
pub struct StandardTraversal;

impl TreeTraversal<ScalarLeaf> for StandardTraversal {
    type TreeState = ();

    #[inline]
    fn build_tree_state(_tree: &Tree<ScalarLeaf>) -> Self::TreeState {
        // No pre-computation needed
    }

    #[inline]
    fn traverse_tree(
        tree: &Tree<ScalarLeaf>,
        _state: &Self::TreeState,
        features: &[f32],
    ) -> NodeId {
        let mut idx = 0u32;

        while !tree.is_leaf(idx) {
            let feat_idx = tree.split_index(idx) as usize;
            let fvalue = features.get(feat_idx).copied().unwrap_or(f32::NAN);

            idx = if fvalue.is_nan() {
                // Missing value: use default direction
                if tree.default_left(idx) {
                    tree.left_child(idx)
                } else {
                    tree.right_child(idx)
                }
            } else {
                match tree.split_type(idx) {
                    SplitType::Numeric => {
                        if fvalue < tree.split_threshold(idx) {
                            tree.left_child(idx)
                        } else {
                            tree.right_child(idx)
                        }
                    }
                    SplitType::Categorical => {
                        let category = float_to_category(fvalue);
                        if tree.categories().category_goes_right(idx, category) {
                            tree.right_child(idx)
                        } else {
                            tree.left_child(idx)
                        }
                    }
                }
            };
        }

       idx
    }
}

// =============================================================================
// UnrolledTraversal
// =============================================================================

/// Unrolled tree traversal using [`UnrolledTreeLayout`].
///
/// Pre-computes a flat array layout for the top `D::DEPTH` levels of each tree.
/// During traversal:
///
/// 1. Traverse unrolled levels using simple index arithmetic
/// 2. Fall back to standard traversal for deeper nodes
///
/// This provides significant speedups for batch prediction (2-3x) because:
/// - Top tree levels stay in L1/L2 cache
/// - Index computation is branchless
/// - Enables future SIMD optimization
///
/// # Type Parameters
///
/// - `D`: Depth marker type ([`Depth4`], [`Depth6`], [`Depth8`])
///
/// [`Depth4`]: super::Depth4
/// [`Depth6`]: super::Depth6
/// [`Depth8`]: super::Depth8
#[derive(Debug, Clone, Copy, Default)]
pub struct UnrolledTraversal<D: UnrollDepth = Depth6> {
    _marker: std::marker::PhantomData<D>,
}

impl<D: UnrollDepth> UnrolledTraversal<D> {
    /// Create a new unrolled traversal strategy.
    pub fn new() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

impl<D: UnrollDepth> TreeTraversal<ScalarLeaf> for UnrolledTraversal<D> {
    type TreeState = UnrolledTreeLayout<D>;

    const USES_BLOCK_OPTIMIZATION: bool = true;

    #[inline]
    fn build_tree_state(tree: &Tree<ScalarLeaf>) -> Self::TreeState {
        UnrolledTreeLayout::from_tree(tree)
    }

    #[inline]
    fn traverse_tree(
        tree: &Tree<ScalarLeaf>,
        state: &Self::TreeState,
        features: &[f32],
    ) -> NodeId {
        // Phase 1: Traverse unrolled levels
        let exit_idx = state.traverse_to_exit(features);
        let node_idx = state.exit_node_idx(exit_idx);

        // Phase 2: Continue to leaf if not already there
        traverse_from_node(tree, node_idx, features)
    }

    /// Optimized block traversal using level-by-level processing.
    ///
    /// All rows traverse the same tree level together, keeping level data in cache.
    #[inline]
    fn traverse_block(
        tree: &Tree<ScalarLeaf>,
        state: &Self::TreeState,
        feature_buffer: &[f32],
        num_features: usize,
        output: &mut [NodeId],
    ) {
        let block_size = output.len();

        // Phase 2 logic: continue from exit nodes to leaves
        let mut traverse_from_exits = |indices: &[usize]| {
            for (row_idx, &exit_idx) in indices.iter().enumerate() {
                let node_idx = state.exit_node_idx(exit_idx);
                let row_offset = row_idx * num_features;
                let row_features = &feature_buffer[row_offset..][..num_features];
                let leaf_idx = traverse_from_node(tree, node_idx, row_features);
                output[row_idx] = leaf_idx;
            }
        };

        // Use stack for small blocks, heap for large
        if block_size <= 256 {
            let mut indices = [0usize; 256];
            let indices = &mut indices[..block_size];
            state.process_block(feature_buffer, num_features, indices);
            traverse_from_exits(indices);
        } else {
            let mut indices = vec![0usize; block_size];
            state.process_block(feature_buffer, num_features, &mut indices);
            traverse_from_exits(&indices);
        }
    }
}

// =============================================================================
// Type Aliases
// =============================================================================

/// Unrolled traversal with depth 4 (15 nodes, 16 exits).
pub type UnrolledTraversal4 = UnrolledTraversal<super::Depth4>;

/// Unrolled traversal with depth 6 (63 nodes, 64 exits) - default.
pub type UnrolledTraversal6 = UnrolledTraversal<super::Depth6>;

/// Unrolled traversal with depth 8 (255 nodes, 256 exits).
pub type UnrolledTraversal8 = UnrolledTraversal<super::Depth8>;

#[cfg(test)]
mod tests {
    #![allow(clippy::let_unit_value)]

    use super::*;
    use crate::repr::gbdt::Forest;

    fn build_simple_tree(
        left_val: f32,
        right_val: f32,
        threshold: f32,
    ) -> Tree<ScalarLeaf> {
        crate::scalar_tree! {
            0 => num(0, threshold, L) -> 1, 2,
            1 => leaf(left_val),
            2 => leaf(right_val),
        }
    }

    fn build_simple_tree_default_right(
        left_val: f32,
        right_val: f32,
        threshold: f32,
    ) -> Tree<ScalarLeaf> {
        crate::scalar_tree! {
            0 => num(0, threshold, R) -> 1, 2,
            1 => leaf(left_val),
            2 => leaf(right_val),
        }
    }

    fn build_categorical_root_tree(default_missing_left: bool) -> Tree<ScalarLeaf> {
        if default_missing_left {
            crate::scalar_tree! {
                0 => cat(0, [1, 3], L) -> 1, 2,
                1 => leaf(10.0),
                2 => leaf(20.0),
            }
        } else {
            crate::scalar_tree! {
                0 => cat(0, [1, 3], R) -> 1, 2,
                1 => leaf(10.0),
                2 => leaf(20.0),
            }
        }
    }

    #[test]
    fn standard_traversal_left() {
        let tree_storage = build_simple_tree(1.0, 2.0, 0.5);
        let mut forest = Forest::for_regression();
        forest.push_tree(tree_storage.clone(), 0);
        let tree = forest.tree(0);

        let state = StandardTraversal::build_tree_state(&tree_storage);
        let result = StandardTraversal::traverse_tree(&tree, &state, &[0.3]);

        assert_eq!(tree.leaf_value(result).0, 1.0);
    }

    #[test]
    fn standard_traversal_right() {
        let tree_storage = build_simple_tree(1.0, 2.0, 0.5);
        let mut forest = Forest::for_regression();
        forest.push_tree(tree_storage.clone(), 0);
        let tree = forest.tree(0);

        let state = StandardTraversal::build_tree_state(&tree_storage);
        let result = StandardTraversal::traverse_tree(&tree, &state, &[0.7]);

        assert_eq!(tree.leaf_value(result).0, 2.0);
    }

    #[test]
    fn standard_traversal_missing() {
        let tree_storage = build_simple_tree(1.0, 2.0, 0.5);
        let mut forest = Forest::for_regression();
        forest.push_tree(tree_storage.clone(), 0);
        let tree = forest.tree(0);

        let state = StandardTraversal::build_tree_state(&tree_storage);
        let result = StandardTraversal::traverse_tree(&tree, &state, &[f32::NAN]);

        // default_left=true, so goes left
        assert_eq!(tree.leaf_value(result).0, 1.0);
    }

    #[test]
    fn standard_traversal_threshold_equality_goes_right() {
        let tree_storage = build_simple_tree(1.0, 2.0, 0.5);
        let state = StandardTraversal::build_tree_state(&tree_storage);
        let result = StandardTraversal::traverse_tree(&tree_storage, &state, &[0.5]);

        // Spec: numeric split goes left iff value < threshold; equality goes right.
        assert_eq!(tree_storage.leaf_value(result).0, 2.0);
    }

    #[test]
    fn standard_traversal_missing_default_right() {
        let tree_storage = build_simple_tree_default_right(1.0, 2.0, 0.5);
        let state = StandardTraversal::build_tree_state(&tree_storage);
        let result = StandardTraversal::traverse_tree(&tree_storage, &state, &[f32::NAN]);

        assert_eq!(tree_storage.leaf_value(result).0, 2.0);
    }

    #[test]
    fn standard_traversal_categorical_membership_and_unknown_go_left() {
        let tree_storage = build_categorical_root_tree(true);
        let state = StandardTraversal::build_tree_state(&tree_storage);

        // In-set categories go RIGHT.
        let in_set = StandardTraversal::traverse_tree(&tree_storage, &state, &[1.0]);
        assert_eq!(tree_storage.leaf_value(in_set).0, 20.0);

        // Not-in-set goes LEFT.
        let not_in_set = StandardTraversal::traverse_tree(&tree_storage, &state, &[2.0]);
        assert_eq!(tree_storage.leaf_value(not_in_set).0, 10.0);

        // Beyond stored bitset defaults to not-in-set => LEFT.
        let unknown = StandardTraversal::traverse_tree(&tree_storage, &state, &[64.0]);
        assert_eq!(tree_storage.leaf_value(unknown).0, 10.0);
    }

    #[test]
    fn standard_traversal_categorical_missing_uses_default_direction() {
        let tree_storage = build_categorical_root_tree(false);
        let state = StandardTraversal::build_tree_state(&tree_storage);
        let result = StandardTraversal::traverse_tree(&tree_storage, &state, &[f32::NAN]);

        // default_left=false, so missing goes right.
        assert_eq!(tree_storage.leaf_value(result).0, 20.0);
    }

    #[test]
    fn unrolled_traversal_matches_standard() {
        let tree_storage = build_simple_tree(1.0, 2.0, 0.5);
        let mut forest = Forest::for_regression();
        forest.push_tree(tree_storage.clone(), 0);
        let tree = forest.tree(0);

        let std_state = StandardTraversal::build_tree_state(&tree_storage);
        let unrolled_state = UnrolledTraversal6::build_tree_state(&tree_storage);

        for fval in [0.1, 0.3, 0.5, 0.7, 0.9, f32::NAN] {
            let std_result = StandardTraversal::traverse_tree(&tree, &std_state, &[fval]);
            let unrolled_result =
                <UnrolledTraversal6 as TreeTraversal<ScalarLeaf>>::traverse_tree(
                    &tree,
                    &unrolled_state,
                    &[fval],
                );

            // Both traversals should return the same leaf index
            assert_eq!(
                std_result, unrolled_result,
                "Mismatch for feature value {:?}",
                fval
            );
        }
    }

    #[test]
    fn unrolled_traversal_handles_categorical_below_unroll_section() {
        // Build a deeper tree where the node at depth 4 is categorical.
        // This specifically exercises the "continue from exit node" path.
        let tree_storage: Tree<ScalarLeaf> = crate::scalar_tree! {
            0 => num(0, 0.5, L) -> 1, 2,
            2 => leaf(200.0),

            1 => num(0, 0.5, L) -> 3, 4,
            4 => leaf(150.0),

            3 => num(0, 0.5, L) -> 7, 8,
            8 => leaf(120.0),

            7 => num(0, 0.5, L) -> 15, 16,
            16 => leaf(110.0),

            // Depth 4 node (along the left-most path): categorical split on feature 1.
            15 => cat(1, [1], L) -> 31, 32,
            31 => leaf(10.0),
            32 => leaf(20.0),
        };

        let std_state = StandardTraversal::build_tree_state(&tree_storage);
        let unrolled_state = UnrolledTraversal4::build_tree_state(&tree_storage);

        // Reach node 15 by taking left at all numeric splits (feature 0 < 0.5),
        // then resolve categorical split by feature 1.
        let left_cat_left = [0.1, 2.0];
        let left_cat_right = [0.1, 1.0];

        let std_left = StandardTraversal::traverse_tree(&tree_storage, &std_state, &left_cat_left);
        let unrolled_left = <UnrolledTraversal4 as TreeTraversal<ScalarLeaf>>::traverse_tree(
            &tree_storage,
            &unrolled_state,
            &left_cat_left,
        );
        // Traversal returns leaf index, get value from tree
        assert_eq!(tree_storage.leaf_value(std_left).0, 10.0);
        assert_eq!(std_left, unrolled_left);

        let std_right = StandardTraversal::traverse_tree(&tree_storage, &std_state, &left_cat_right);
        let unrolled_right = <UnrolledTraversal4 as TreeTraversal<ScalarLeaf>>::traverse_tree(
            &tree_storage,
            &unrolled_state,
            &left_cat_right,
        );
        assert_eq!(tree_storage.leaf_value(std_right).0, 20.0);
        assert_eq!(std_right, unrolled_right);
    }

    #[test]
    fn traverse_from_node_basic() {
        let tree_storage = build_simple_tree(1.0, 2.0, 0.5);
        let mut forest = Forest::for_regression();
        forest.push_tree(tree_storage, 0);
        let tree = forest.tree(0);

        // Starting from root (node 0)
        let leaf_left = traverse_from_node(&tree, 0, &[0.3]);
        let leaf_right = traverse_from_node(&tree, 0, &[0.7]);

        assert_eq!(tree.leaf_value(leaf_left).0, 1.0);
        assert_eq!(tree.leaf_value(leaf_right).0, 2.0);
    }
}

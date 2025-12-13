//! Tree traversal strategies for prediction.
//!
//! This module provides the [`TreeTraversal`] trait and implementations that
//! abstract how trees are traversed during prediction.
//!
//! # Available Strategies
//!
//! - [`StandardTraversal`]: Direct node-by-node traversal (simple, good for single rows)
//! - [`UnrolledTraversal`]: Uses [`UnrolledTreeLayout`] for cache-friendly batch traversal

use super::SplitType;
use super::{
    float_to_category, Depth6, LeafValue, ScalarLeaf, Tree, UnrollDepth,
    UnrolledTreeLayout,
};

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
    fn traverse_tree(tree: &Tree<L>, state: &Self::TreeState, features: &[f32]) -> L;

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
    /// - `output`: Slice to accumulate leaf values into (one per row)
    /// - `weight`: Optional weight to multiply leaf values by
    #[inline]
    fn traverse_block(
        tree: &Tree<L>,
        state: &Self::TreeState,
        feature_buffer: &[f32],
        num_features: usize,
        output: &mut [f32],
        weight: f32,
    ) where
        L: Into<f32>,
    {
        // Default: per-row traversal
        for (row_idx, out) in output.iter_mut().enumerate() {
            let row_offset = row_idx * num_features;
            let row_features = &feature_buffer[row_offset..][..num_features];
            let leaf_value: f32 = Self::traverse_tree(tree, state, row_features).into();
            *out += leaf_value * weight;
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
pub fn traverse_from_node(tree: &Tree<ScalarLeaf>, start_node: u32, features: &[f32]) -> u32 {
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
        } else if fvalue < tree.split_threshold(idx) {
            tree.left_child(idx)
        } else {
            tree.right_child(idx)
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
    ) -> ScalarLeaf {
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

        *tree.leaf_value(idx)
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
    ) -> ScalarLeaf {
        // Phase 1: Traverse unrolled levels
        let exit_idx = state.traverse_to_exit(features);
        let node_idx = state.exit_node_idx(exit_idx);

        // Phase 2: Continue to leaf if not already there
        let leaf_idx = traverse_from_node(tree, node_idx, features);

        *tree.leaf_value(leaf_idx)
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
        output: &mut [f32],
        weight: f32,
    ) {
        let block_size = output.len();

        // Phase 2 logic: continue from exit nodes to leaves
        let mut accumulate_from_exits = |indices: &[usize]| {
            for (row_idx, &exit_idx) in indices.iter().enumerate() {
                let node_idx = state.exit_node_idx(exit_idx);
                let row_offset = row_idx * num_features;
                let row_features = &feature_buffer[row_offset..][..num_features];
                let leaf_idx = traverse_from_node(tree, node_idx, row_features);
                output[row_idx] += tree.leaf_value(leaf_idx).0 * weight;
            }
        };

        // Use stack for small blocks, heap for large
        if block_size <= 256 {
            let mut indices = [0usize; 256];
            let indices = &mut indices[..block_size];
            state.process_block(feature_buffer, num_features, indices);
            accumulate_from_exits(indices);
        } else {
            let mut indices = vec![0usize; block_size];
            state.process_block(feature_buffer, num_features, &mut indices);
            accumulate_from_exits(&indices);
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
    use crate::inference::gbdt::Forest;

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

    #[test]
    fn standard_traversal_left() {
        let tree_storage = build_simple_tree(1.0, 2.0, 0.5);
        let mut forest = Forest::for_regression();
        forest.push_tree(tree_storage.clone(), 0);
        let tree = forest.tree(0);

        let state = StandardTraversal::build_tree_state(&tree_storage);
        let result = StandardTraversal::traverse_tree(&tree, &state, &[0.3]);

        assert_eq!(result.0, 1.0);
    }

    #[test]
    fn standard_traversal_right() {
        let tree_storage = build_simple_tree(1.0, 2.0, 0.5);
        let mut forest = Forest::for_regression();
        forest.push_tree(tree_storage.clone(), 0);
        let tree = forest.tree(0);

        let state = StandardTraversal::build_tree_state(&tree_storage);
        let result = StandardTraversal::traverse_tree(&tree, &state, &[0.7]);

        assert_eq!(result.0, 2.0);
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
        assert_eq!(result.0, 1.0);
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

            assert_eq!(
                std_result.0, unrolled_result.0,
                "Mismatch for feature value {:?}",
                fval
            );
        }
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

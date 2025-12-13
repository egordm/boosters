//! SIMD-enabled traversal strategies.
//!
//! The intent of this module is to provide traversal implementations that can
//! evaluate multiple rows in parallel.
//!
//! NOTE: This is currently a correctness-first implementation that reuses the
//! unrolled tree layout and scalar traversal logic. It exists primarily to keep
//! the `simd` feature functional and provide stable public type names.

use super::{
    Depth4, Depth6, Depth8, ScalarLeaf, Tree, UnrollDepth, UnrolledTreeLayout,
};
use crate::inference::gbdt::traversal::{traverse_from_node, TreeTraversal};

/// Number of lanes the SIMD traversal is expected to operate on.
///
/// This is a public constant so callers can tune batch/block sizes.
///
/// The current implementation is scalar-correctness-first, so this is best
/// treated as a hint.
pub const SIMD_WIDTH: usize = 8;

/// Traversal strategy intended for SIMD execution.
///
/// Currently uses the same unrolled tree layout as [`super::UnrolledTraversal`]
/// for correctness and cache efficiency.
#[derive(Debug, Clone, Copy, Default)]
pub struct SimdTraversal<D: UnrollDepth = Depth6> {
    _marker: core::marker::PhantomData<D>,
}

impl<D: UnrollDepth> SimdTraversal<D> {
    /// Create a new SIMD traversal strategy.
    pub fn new() -> Self {
        Self {
            _marker: core::marker::PhantomData,
        }
    }
}

/// SIMD traversal with 4-level unrolling.
pub type SimdTraversal4 = SimdTraversal<Depth4>;
/// SIMD traversal with 6-level unrolling.
pub type SimdTraversal6 = SimdTraversal<Depth6>;
/// SIMD traversal with 8-level unrolling.
pub type SimdTraversal8 = SimdTraversal<Depth8>;

impl<D: UnrollDepth> TreeTraversal<ScalarLeaf> for SimdTraversal<D> {
    type TreeState = UnrolledTreeLayout<D>;

    const USES_BLOCK_OPTIMIZATION: bool = true;

    #[inline]
    fn build_tree_state(tree: &Tree<ScalarLeaf>) -> Self::TreeState {
        UnrolledTreeLayout::from_tree(tree)
    }

    #[inline]
    fn traverse_tree(tree: &Tree<ScalarLeaf>, state: &Self::TreeState, features: &[f32]) -> ScalarLeaf {
        // Phase 1: Traverse unrolled levels
        let exit_idx = state.traverse_to_exit(features);
        let node_idx = state.exit_node_idx(exit_idx);

        // Phase 2: Continue to leaf if not already there
        let leaf_idx = traverse_from_node(tree, node_idx, features);

        *tree.leaf_value(leaf_idx)
    }
}

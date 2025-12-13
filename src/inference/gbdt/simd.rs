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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::gbdt::traversal::{StandardTraversal, UnrolledTraversal4};

    fn categorical_under_unroll() -> Tree<ScalarLeaf> {
        crate::scalar_tree! {
            0 => num(0, 0.5, L) -> 1, 2,
            2 => leaf(200.0),

            1 => num(0, 0.5, L) -> 3, 4,
            4 => leaf(150.0),

            3 => num(0, 0.5, L) -> 7, 8,
            8 => leaf(120.0),

            7 => num(0, 0.5, L) -> 15, 16,
            16 => leaf(110.0),

            15 => cat(1, [1], L) -> 31, 32,
            31 => leaf(10.0),
            32 => leaf(20.0),
        }
    }

    #[test]
    fn simd_traversal_matches_standard_and_unrolled() {
        let tree = categorical_under_unroll();
        let std_state = StandardTraversal::build_tree_state(&tree);
        let unrolled_state = UnrolledTraversal4::build_tree_state(&tree);
        let simd_state = SimdTraversal4::build_tree_state(&tree);

        let cases: Vec<Vec<f32>> = vec![
            vec![0.1, 2.0],
            vec![0.1, 1.0],
            vec![0.9, 1.0],
            vec![f32::NAN, 1.0],
        ];

        for features in cases {
            let std = StandardTraversal::traverse_tree(&tree, &std_state, &features);
            let unrolled = <UnrolledTraversal4 as TreeTraversal<ScalarLeaf>>::traverse_tree(
                &tree,
                &unrolled_state,
                &features,
            );
            let simd = <SimdTraversal4 as TreeTraversal<ScalarLeaf>>::traverse_tree(
                &tree,
                &simd_state,
                &features,
            );

            assert_eq!(std.0, unrolled.0, "unrolled mismatch for {features:?}");
            assert_eq!(std.0, simd.0, "simd mismatch for {features:?}");
        }
    }
}

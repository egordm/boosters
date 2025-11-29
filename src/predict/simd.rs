//! SIMD-accelerated tree traversal.
//!
//! This module provides [`SimdTraversal`], which processes 8 rows simultaneously
//! using SIMD instructions during the level-by-level unrolled traversal.
//!
//! # Performance Note
//!
//! **Current Status**: The row-parallel SIMD approach is ~12% slower than
//! [`UnrolledTraversal`] due to gather overhead. The bottleneck is that:
//!
//! 1. Features are stored row-major, but SIMD needs column values
//! 2. The `wide` crate doesn't expose hardware gather instructions
//! 3. Scalar loops for gathering negate SIMD comparison benefits
//!
//! This implementation is preserved for experimentation and as a foundation
//! for future optimizations (column-major layout, tree-parallel SIMD, etc.).
//!
//! For production use, prefer [`UnrolledTraversal`] which provides 2.9x speedup.
//!
//! # Algorithm
//!
//! SIMD traversal attempts to parallelize by:
//!
//! - Comparing 8 feature values against thresholds in parallel
//! - Computing 8 next-node indices simultaneously
//! - Reducing branch mispredictions through branchless SIMD operations
//!
//! # Requirements
//!
//! Requires the `simd` feature flag and works on stable Rust via the `wide` crate.
//!
//! [`UnrolledTraversal`]: crate::predict::UnrolledTraversal

use wide::{f32x8, i32x8, CmpLt};

use crate::forest::SoATreeView;
use crate::trees::{
    nodes_at_depth, Depth6, ScalarLeaf, SoATreeStorage, UnrollDepth, UnrolledTreeLayout,
};

use super::traversal::{traverse_from_node, TreeTraversal};

/// SIMD lane width - process 8 rows at a time.
pub const SIMD_WIDTH: usize = 8;

// =============================================================================
// SimdTraversal
// =============================================================================

/// SIMD-accelerated tree traversal using [`UnrolledTreeLayout`].
///
/// Processes 8 rows simultaneously through the unrolled tree levels using
/// SIMD comparison operations. This combines the benefits of:
///
/// 1. **Unrolled traversal**: Level-by-level processing keeps data in cache
/// 2. **SIMD parallelism**: 8 comparisons execute in a single instruction
///
/// # Type Parameters
///
/// - `D`: Depth marker type ([`Depth4`], [`Depth6`], [`Depth8`])
///
/// # Example
///
/// ```ignore
/// use booste_rs::predict::{Predictor, SimdTraversal6};
///
/// let predictor = Predictor::<SimdTraversal6>::new(&forest);
/// let output = predictor.predict(&features);
/// ```
///
/// [`Depth4`]: crate::trees::Depth4
/// [`Depth6`]: crate::trees::Depth6
/// [`Depth8`]: crate::trees::Depth8
#[derive(Debug, Clone, Copy, Default)]
pub struct SimdTraversal<D: UnrollDepth = Depth6> {
    _marker: std::marker::PhantomData<D>,
}

impl<D: UnrollDepth> SimdTraversal<D> {
    /// Create a new SIMD traversal strategy.
    pub fn new() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

impl<D: UnrollDepth> TreeTraversal<ScalarLeaf> for SimdTraversal<D> {
    type TreeState = UnrolledTreeLayout<D>;

    const USES_BLOCK_OPTIMIZATION: bool = true;

    #[inline]
    fn build_tree_state(tree: &SoATreeStorage<ScalarLeaf>) -> Self::TreeState {
        UnrolledTreeLayout::from_tree(tree)
    }

    #[inline]
    fn traverse_tree(
        tree: &SoATreeView<'_, ScalarLeaf>,
        state: &Self::TreeState,
        features: &[f32],
    ) -> ScalarLeaf {
        // Single-row fallback to non-SIMD path
        let exit_idx = state.traverse_to_exit(features);
        let node_idx = state.exit_node_idx(exit_idx);
        let leaf_idx = traverse_from_node(tree, node_idx, features);
        tree.leaf_value(leaf_idx).clone()
    }

    /// SIMD-optimized block traversal.
    ///
    /// Processes rows in groups of 8 using SIMD instructions.
    #[inline]
    fn traverse_block(
        tree: &SoATreeView<'_, ScalarLeaf>,
        state: &Self::TreeState,
        feature_buffer: &[f32],
        num_features: usize,
        output: &mut [f32],
        weight: f32,
    ) {
        let block_size = output.len();

        // Process full SIMD groups of 8
        let simd_groups = block_size / SIMD_WIDTH;
        let remainder = block_size % SIMD_WIDTH;

        for group_idx in 0..simd_groups {
            let row_start = group_idx * SIMD_WIDTH;

            // SIMD traversal for 8 rows
            let exit_indices =
                traverse_simd::<D>(state, feature_buffer, num_features, row_start);

            // Accumulate results
            for lane in 0..SIMD_WIDTH {
                let row_idx = row_start + lane;
                let exit_idx = exit_indices[lane] as usize;
                let node_idx = state.exit_node_idx(exit_idx);

                let row_offset = row_idx * num_features;
                let row_features = &feature_buffer[row_offset..][..num_features];
                let leaf_idx = traverse_from_node(tree, node_idx, row_features);
                output[row_idx] += tree.leaf_value(leaf_idx).0 * weight;
            }
        }

        // Handle remainder rows (< 8) with scalar traversal
        if remainder > 0 {
            let start_idx = simd_groups * SIMD_WIDTH;
            for row_idx in start_idx..block_size {
                let row_offset = row_idx * num_features;
                let row_features = &feature_buffer[row_offset..][..num_features];
                let exit_idx = state.traverse_to_exit(row_features);
                let node_idx = state.exit_node_idx(exit_idx);
                let leaf_idx = traverse_from_node(tree, node_idx, row_features);
                output[row_idx] += tree.leaf_value(leaf_idx).0 * weight;
            }
        }
    }
}

// =============================================================================
// SIMD Traversal Core
// =============================================================================

/// Traverse 8 rows through the unrolled layout using SIMD.
///
/// Returns an array of 8 exit indices.
#[inline]
fn traverse_simd<D: UnrollDepth>(
    layout: &UnrolledTreeLayout<D>,
    features: &[f32],
    num_features: usize,
    row_start: usize,
) -> [i32; SIMD_WIDTH] {
    // Position within level (0..2^level) for each row, as i32 for SIMD
    let mut positions = i32x8::splat(0);

    // Traverse level by level
    for level in 0..D::DEPTH {
        let level_start = nodes_at_depth(level) as i32;

        // Compute array indices: level_start + position
        let array_indices = positions + i32x8::splat(level_start);

        // Extract indices for gather (wide doesn't have gather, so we extract)
        let indices = array_indices.to_array();

        // Gather split info for all 8 rows
        let mut split_indices = [0u32; SIMD_WIDTH];
        let mut thresholds = [0.0f32; SIMD_WIDTH];
        let mut default_left_flags = [false; SIMD_WIDTH];

        for lane in 0..SIMD_WIDTH {
            let array_idx = indices[lane] as usize;
            split_indices[lane] = layout.split_index(array_idx);
            thresholds[lane] = layout.split_threshold(array_idx);
            default_left_flags[lane] = layout.default_left(array_idx);
        }

        // Gather feature values for all 8 rows
        let mut fvalues = [f32::NAN; SIMD_WIDTH];
        for lane in 0..SIMD_WIDTH {
            let row_idx = row_start + lane;
            let feat_idx = split_indices[lane] as usize;
            let row_offset = row_idx * num_features;
            fvalues[lane] = features
                .get(row_offset + feat_idx)
                .copied()
                .unwrap_or(f32::NAN);
        }

        // SIMD comparison: fvalue < threshold
        let fvalues_simd = f32x8::from(fvalues);
        let thresholds_simd = f32x8::from(thresholds);
        let cmp_result = fvalues_simd.cmp_lt(thresholds_simd);

        // Handle NaN (missing values): use default direction
        // For NaN: cmp_result is false, but we need default_left
        // For now, we handle NaN in scalar fallback
        let mut go_left = [false; SIMD_WIDTH];
        let cmp_mask = cmp_result.to_array();

        for lane in 0..SIMD_WIDTH {
            if fvalues[lane].is_nan() {
                go_left[lane] = default_left_flags[lane];
            } else {
                // cmp_mask is -1 (all bits set) for true, 0 for false
                go_left[lane] = cmp_mask[lane] != 0.0;
            }
        }

        // Compute next positions: 2 * pos + (go_right as i32)
        // go_right = !go_left
        let mut go_right_int = [0i32; SIMD_WIDTH];
        for lane in 0..SIMD_WIDTH {
            go_right_int[lane] = !go_left[lane] as i32;
        }

        let doubled = positions + positions; // 2 * pos
        let go_right_simd = i32x8::from(go_right_int);
        positions = doubled + go_right_simd;
    }

    positions.to_array()
}

// =============================================================================
// Type Aliases
// =============================================================================

/// SIMD traversal with depth 4 (15 nodes, 16 exits).
pub type SimdTraversal4 = SimdTraversal<crate::trees::Depth4>;

/// SIMD traversal with depth 6 (63 nodes, 64 exits) - default.
pub type SimdTraversal6 = SimdTraversal<crate::trees::Depth6>;

/// SIMD traversal with depth 8 (255 nodes, 256 exits).
pub type SimdTraversal8 = SimdTraversal<crate::trees::Depth8>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forest::SoAForest;
    use crate::predict::traversal::{StandardTraversal, UnrolledTraversal6};
    use crate::trees::TreeBuilder;

    fn build_simple_tree(
        left_val: f32,
        right_val: f32,
        threshold: f32,
    ) -> SoATreeStorage<ScalarLeaf> {
        let mut builder = TreeBuilder::new();
        builder.add_split(0, threshold, true, 1, 2);
        builder.add_leaf(ScalarLeaf(left_val));
        builder.add_leaf(ScalarLeaf(right_val));
        builder.build()
    }

    #[test]
    fn simd_traversal_matches_standard() {
        let tree_storage = build_simple_tree(1.0, 2.0, 0.5);
        let mut forest = SoAForest::for_regression();
        forest.push_tree(tree_storage.clone(), 0);
        let tree = forest.tree(0);

        let std_state = StandardTraversal::build_tree_state(&tree_storage);
        let simd_state = SimdTraversal6::build_tree_state(&tree_storage);

        for fval in [0.1, 0.3, 0.5, 0.7, 0.9, f32::NAN] {
            let std_result = StandardTraversal::traverse_tree(&tree, &std_state, &[fval]);
            let simd_result =
                <SimdTraversal6 as TreeTraversal<ScalarLeaf>>::traverse_tree(
                    &tree,
                    &simd_state,
                    &[fval],
                );

            assert_eq!(
                std_result.0, simd_result.0,
                "Mismatch for feature value {:?}",
                fval
            );
        }
    }

    #[test]
    fn simd_traversal_matches_unrolled() {
        let tree_storage = build_simple_tree(1.0, 2.0, 0.5);
        let mut forest = SoAForest::for_regression();
        forest.push_tree(tree_storage.clone(), 0);
        let tree = forest.tree(0);

        let unrolled_state = UnrolledTraversal6::build_tree_state(&tree_storage);
        let simd_state = SimdTraversal6::build_tree_state(&tree_storage);

        for fval in [0.1, 0.3, 0.5, 0.7, 0.9, f32::NAN] {
            let unrolled_result =
                <crate::predict::traversal::UnrolledTraversal6 as TreeTraversal<ScalarLeaf>>::traverse_tree(
                    &tree,
                    &unrolled_state,
                    &[fval],
                );
            let simd_result =
                <SimdTraversal6 as TreeTraversal<ScalarLeaf>>::traverse_tree(
                    &tree,
                    &simd_state,
                    &[fval],
                );

            assert_eq!(
                unrolled_result.0, simd_result.0,
                "Mismatch for feature value {:?}",
                fval
            );
        }
    }

    #[test]
    fn simd_block_traversal_basic() {
        let tree_storage = build_simple_tree(1.0, 2.0, 0.5);
        let mut forest = SoAForest::for_regression();
        forest.push_tree(tree_storage.clone(), 0);
        let tree = forest.tree(0);

        let simd_state = SimdTraversal6::build_tree_state(&tree_storage);

        // 8 rows with 1 feature each
        let features: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9];
        let mut output = vec![0.0f32; 8];

        SimdTraversal6::traverse_block(&tree, &simd_state, &features, 1, &mut output, 1.0);

        // First 4 go left (< 0.5), last 4 go right (>= 0.5)
        assert_eq!(output[0], 1.0);
        assert_eq!(output[1], 1.0);
        assert_eq!(output[2], 1.0);
        assert_eq!(output[3], 1.0);
        assert_eq!(output[4], 2.0);
        assert_eq!(output[5], 2.0);
        assert_eq!(output[6], 2.0);
        assert_eq!(output[7], 2.0);
    }

    #[test]
    fn simd_block_traversal_with_remainder() {
        let tree_storage = build_simple_tree(1.0, 2.0, 0.5);
        let mut forest = SoAForest::for_regression();
        forest.push_tree(tree_storage.clone(), 0);
        let tree = forest.tree(0);

        let simd_state = SimdTraversal6::build_tree_state(&tree_storage);

        // 11 rows (8 + 3 remainder)
        let features: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.25, 0.75, 0.35];
        let mut output = vec![0.0f32; 11];

        SimdTraversal6::traverse_block(&tree, &simd_state, &features, 1, &mut output, 1.0);

        assert_eq!(output[0], 1.0); // 0.1 < 0.5
        assert_eq!(output[4], 2.0); // 0.6 >= 0.5
        assert_eq!(output[8], 1.0); // 0.25 < 0.5 (remainder)
        assert_eq!(output[9], 2.0); // 0.75 >= 0.5 (remainder)
        assert_eq!(output[10], 1.0); // 0.35 < 0.5 (remainder)
    }
}

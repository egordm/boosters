//! SIMD-enabled traversal strategies.
//!
//! This module provides vectorized tree traversal that processes multiple rows
//! simultaneously using SIMD instructions. The implementation uses the `wide`
//! crate for portable SIMD across x86, ARM, and other architectures.
//!
//! # Architecture
//!
//! The SIMD traversal processes 8 rows at a time (8xf32 lanes):
//!
//! 1. **Level-by-level processing**: All 8 rows traverse the same tree level together
//! 2. **Vectorized comparisons**: Feature values compared to thresholds using `cmp_lt`
//! 3. **Masked child selection**: SIMD mask selects left/right child per lane
//!
//! For trees deeper than the unroll depth, the implementation falls back to
//! scalar traversal for remaining levels (since deep paths diverge).
//!
//! # Performance
//!
//! Expected ~2-4x speedup for batch inference on numeric-only trees.
//! Categorical splits and missing values reduce speedup due to scalar fallback.

use super::{
    Depth4, Depth6, Depth8, ScalarLeaf, Tree, UnrollDepth, UnrolledTreeLayout,
};
use crate::inference::gbdt::traversal::{traverse_from_node, TreeTraversal};
use wide::{f32x8, i32x8, CmpLt, CmpNe};

/// Number of lanes the SIMD traversal operates on.
///
/// This matches the width of `f32x8` (8 lanes). Block sizes should be
/// multiples of this for optimal performance.
pub const SIMD_WIDTH: usize = 8;

/// Traversal strategy using SIMD vectorization.
///
/// Processes 8 rows simultaneously using `f32x8` SIMD vectors. The unrolled
/// tree layout keeps node data contiguous for efficient cache access.
///
/// # When to use
///
/// - Batch inference with many rows
/// - Numeric-only or mostly-numeric trees
/// - When `simd` feature is enabled
///
/// For trees with many categorical splits or single-row inference,
/// prefer [`super::UnrolledTraversal`].
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
        // Single-row: use scalar traversal
        let exit_idx = state.traverse_to_exit(features);
        let node_idx = state.exit_node_idx(exit_idx);
        let leaf_idx = traverse_from_node(tree, node_idx, features);
        *tree.leaf_value(leaf_idx)
    }

    /// Vectorized block traversal using SIMD.
    ///
    /// Processes 8 rows at a time through the unrolled tree levels.
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
        
        // Process full SIMD-width chunks (8 rows at a time)
        let simd_chunks = block_size / SIMD_WIDTH;
        let remainder = block_size % SIMD_WIDTH;

        for chunk_idx in 0..simd_chunks {
            let chunk_start = chunk_idx * SIMD_WIDTH;
            let chunk_features = &feature_buffer[chunk_start * num_features..];
            let chunk_output = &mut output[chunk_start..chunk_start + SIMD_WIDTH];
            
            traverse_simd_chunk::<D>(tree, state, chunk_features, num_features, chunk_output, weight);
        }

        // Handle remaining rows with scalar traversal
        if remainder > 0 {
            let start = simd_chunks * SIMD_WIDTH;
            for row_idx in start..block_size {
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

/// Process exactly 8 rows through the tree using SIMD.
#[inline]
fn traverse_simd_chunk<D: UnrollDepth>(
    tree: &Tree<ScalarLeaf>,
    state: &UnrolledTreeLayout<D>,
    features: &[f32],
    num_features: usize,
    output: &mut [f32],
    weight: f32,
) {
    // Track position within each level for all 8 rows
    // position[lane] = offset within current level (0 = leftmost)
    let mut positions = i32x8::splat(0);
    
    // Precompute row feature pointers
    let row_offsets: [usize; 8] = core::array::from_fn(|i| i * num_features);
    
    // Get references to unrolled arrays
    let split_indices = state.split_indices_slice();
    let split_thresholds = state.split_thresholds_slice();
    let default_left = state.default_left_slice();
    
    // Traverse level by level through unrolled section
    let mut level_start = 0usize;
    for _level in 0..D::DEPTH {
        // Gather feature values for all 8 rows
        let feat_values = gather_features_simd(features, &row_offsets, split_indices, level_start, &positions);
        
        // Gather thresholds for current positions
        let thresholds = gather_thresholds_simd(split_thresholds, level_start, &positions);
        
        // Vectorized comparison: feat < threshold
        let go_left_mask = feat_values.cmp_lt(thresholds);
        
        // Handle NaN (missing values): check if value is NaN
        let is_nan_mask = feat_values.cmp_ne(feat_values); // NaN != NaN
        
        // For NaN values, use default direction
        let default_dirs = gather_defaults_simd(default_left, level_start, &positions);
        
        // go_left = is_nan ? default_left : (feat < threshold)
        let go_left_i32 = blend_mask_i32(is_nan_mask, default_dirs, mask_to_i32(go_left_mask));
        
        // Update positions: pos = 2 * pos + (go_left ? 0 : 1)
        let two = i32x8::splat(2);
        let one = i32x8::splat(1);
        let zero = i32x8::splat(0);
        let offset = blend_i32(go_left_i32, zero, one);
        positions = two * positions + offset;
        
        level_start += 1 << _level;
    }
    
    // Convert final positions to exit indices
    let positions_arr: [i32; 8] = positions.into();
    
    // Finish traversal and accumulate results
    for lane in 0..8 {
        let exit_idx = positions_arr[lane] as usize;
        let node_idx = state.exit_node_idx(exit_idx);
        
        let row_features = &features[row_offsets[lane]..][..num_features];
        let leaf_idx = traverse_from_node(tree, node_idx, row_features);
        output[lane] += tree.leaf_value(leaf_idx).0 * weight;
    }
}

/// Gather feature values for 8 rows based on their current positions.
#[inline]
fn gather_features_simd(
    features: &[f32],
    row_offsets: &[usize; 8],
    split_indices: &[u32],
    level_start: usize,
    positions: &i32x8,
) -> f32x8 {
    let positions_arr: [i32; 8] = (*positions).into();
    let mut values = [0.0f32; 8];
    
    for lane in 0..8 {
        let array_idx = level_start + positions_arr[lane] as usize;
        let feat_idx = split_indices[array_idx] as usize;
        values[lane] = features.get(row_offsets[lane] + feat_idx)
            .copied()
            .unwrap_or(f32::NAN);
    }
    
    f32x8::from(values)
}

/// Gather thresholds for 8 positions.
#[inline]
fn gather_thresholds_simd(
    split_thresholds: &[f32],
    level_start: usize,
    positions: &i32x8,
) -> f32x8 {
    let positions_arr: [i32; 8] = (*positions).into();
    let mut thresholds = [0.0f32; 8];
    
    for lane in 0..8 {
        let array_idx = level_start + positions_arr[lane] as usize;
        thresholds[lane] = split_thresholds[array_idx];
    }
    
    f32x8::from(thresholds)
}

/// Gather default_left flags for 8 positions.
#[inline]
fn gather_defaults_simd(
    default_left: &[bool],
    level_start: usize,
    positions: &i32x8,
) -> i32x8 {
    let positions_arr: [i32; 8] = (*positions).into();
    let mut defaults = [0i32; 8];
    
    for lane in 0..8 {
        let array_idx = level_start + positions_arr[lane] as usize;
        defaults[lane] = if default_left[array_idx] { 1 } else { 0 };
    }
    
    i32x8::from(defaults)
}

/// Convert f32x8 comparison mask to i32x8 (1 for true, 0 for false).
#[inline]
fn mask_to_i32(mask: f32x8) -> i32x8 {
    // f32x8 comparison results in f32 with all bits set (NaN-like) for true
    // We convert to array and check for non-zero
    let mask_arr: [f32; 8] = mask.into();
    let result: [i32; 8] = core::array::from_fn(|i| if mask_arr[i].to_bits() != 0 { 1 } else { 0 });
    i32x8::from(result)
}

/// Blend two i32x8 values based on a f32x8 mask.
#[inline]
fn blend_mask_i32(mask: f32x8, if_true: i32x8, if_false: i32x8) -> i32x8 {
    let mask_arr: [f32; 8] = mask.into();
    let a: [i32; 8] = if_true.into();
    let b: [i32; 8] = if_false.into();
    let result: [i32; 8] = core::array::from_fn(|i| if mask_arr[i].to_bits() != 0 { a[i] } else { b[i] });
    i32x8::from(result)
}

/// Blend two i32x8 values based on an i32x8 mask (non-zero = true).
#[inline]
fn blend_i32(mask: i32x8, if_true: i32x8, if_false: i32x8) -> i32x8 {
    let m: [i32; 8] = mask.into();
    let a: [i32; 8] = if_true.into();
    let b: [i32; 8] = if_false.into();
    let result: [i32; 8] = core::array::from_fn(|i| if m[i] != 0 { a[i] } else { b[i] });
    i32x8::from(result)
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

    fn simple_numeric_tree() -> Tree<ScalarLeaf> {
        crate::scalar_tree! {
            0 => num(0, 0.5, L) -> 1, 2,
            1 => num(1, 0.3, L) -> 3, 4,
            2 => num(1, 0.7, L) -> 5, 6,
            3 => leaf(1.0),
            4 => leaf(2.0),
            5 => leaf(3.0),
            6 => leaf(4.0),
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

    #[test]
    fn simd_block_traversal_matches_scalar() {
        let tree = simple_numeric_tree();
        let simd_state = SimdTraversal4::build_tree_state(&tree);
        let std_state = StandardTraversal::build_tree_state(&tree);

        // Create test rows (each row has 2 features)
        let num_features = 2;
        let test_rows: Vec<[f32; 2]> = vec![
            [0.1, 0.1],  // -> leaf 1.0 (left-left)
            [0.1, 0.5],  // -> leaf 2.0 (left-right)
            [0.9, 0.5],  // -> leaf 3.0 (right-left)
            [0.9, 0.9],  // -> leaf 4.0 (right-right)
            [0.3, 0.2],  // -> leaf 1.0
            [0.3, 0.4],  // -> leaf 2.0
            [0.7, 0.6],  // -> leaf 3.0
            [0.7, 0.8],  // -> leaf 4.0
            // Additional rows to test non-8-aligned blocks
            [0.2, 0.25], // -> leaf 1.0
            [0.8, 0.75], // -> leaf 4.0
        ];

        // Flatten into feature buffer
        let feature_buffer: Vec<f32> = test_rows.iter().flat_map(|r| r.iter().copied()).collect();

        // Compute expected outputs using scalar traversal
        let expected: Vec<f32> = test_rows
            .iter()
            .map(|row| StandardTraversal::traverse_tree(&tree, &std_state, row).0)
            .collect();

        // Test SIMD block traversal
        let mut output = vec![0.0f32; test_rows.len()];
        SimdTraversal4::traverse_block(&tree, &simd_state, &feature_buffer, num_features, &mut output, 1.0);

        for (i, (&actual, &exp)) in output.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - exp).abs() < 1e-6,
                "Row {i}: expected {exp}, got {actual}"
            );
        }
    }

    #[test]
    fn simd_block_with_weight() {
        let tree = simple_numeric_tree();
        let simd_state = SimdTraversal4::build_tree_state(&tree);

        let feature_buffer: Vec<f32> = vec![
            0.1, 0.1,  // -> 1.0
            0.9, 0.9,  // -> 4.0
            0.1, 0.1,  // -> 1.0
            0.9, 0.9,  // -> 4.0
            0.1, 0.1,  // -> 1.0
            0.9, 0.9,  // -> 4.0
            0.1, 0.1,  // -> 1.0
            0.9, 0.9,  // -> 4.0
        ];

        let mut output = vec![0.0f32; 8];
        let weight = 0.5;
        SimdTraversal4::traverse_block(&tree, &simd_state, &feature_buffer, 2, &mut output, weight);

        assert!((output[0] - 0.5).abs() < 1e-6); // 1.0 * 0.5
        assert!((output[1] - 2.0).abs() < 1e-6); // 4.0 * 0.5
    }
}

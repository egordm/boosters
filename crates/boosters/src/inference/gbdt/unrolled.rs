//! Unrolled tree layout for cache-friendly batch traversal.
//!
//! This module implements the UnrolledTreeLayout optimization (based on XGBoost's
//! `array_tree_layout`). The top K levels of a tree are unrolled into a flat array,
//! enabling level-by-level traversal with simple index arithmetic instead of
//! pointer-chasing.
//!
//! # Theory
//!
//! A complete binary tree of depth K has `2^K - 1` nodes. By unrolling
//! the top levels into a contiguous array, we get:
//!
//! - **Better cache locality**: All top-level nodes fit in L1/L2 cache
//! - **Simple index math**: `left_child = 2*i + 1`, `right_child = 2*i + 2`
//! - **SIMD-friendly**: Multiple rows can traverse the same level together
//!

// Allow many arguments for the builder function.
#![allow(clippy::too_many_arguments)]
//! # Array Layout
//!
//! ```text
//! Level 0:           [0]              <- root (array idx 0)
//! Level 1:        [1]   [2]           <- array idx 1, 2
//! Level 2:      [3][4] [5][6]         <- array idx 3, 4, 5, 6
//! ...
//! ```
//!
//! After traversing K levels, we have `2^K` possible exit points that
//! map back to nodes in the original tree for continued traversal.
//!
//! # Design
//!
//! The implementation uses a sealed trait pattern to provide const-generic-like
//! behavior with stable Rust, while also supporting nightly's `generic_const_exprs`.
//!
//! Type aliases provide convenient names for common depths:
//!
//! - `UnrolledTreeLayout4` — 15 nodes, 16 exits (small trees)
//! - `UnrolledTreeLayout6` — 63 nodes, 64 exits (default, matches XGBoost)
//! - `UnrolledTreeLayout8` — 255 nodes, 256 exits (deep trees)

use crate::data::SamplesView;
use crate::repr::gbdt::{LeafValue, SplitType, Tree, TreeView};
use ndarray::ArrayView1;

/// Maximum number of levels to unroll (matches XGBoost).
/// 6 levels = 63 nodes, 8 levels = 255 nodes.
pub const MAX_UNROLL_DEPTH: usize = 6;

/// Default unroll depth - 6 levels, matching XGBoost's default.
pub const DEFAULT_UNROLL_DEPTH: usize = 6;

/// Number of nodes in a complete binary tree of given depth.
/// For depth K: `2^K - 1` nodes.
#[inline]
pub const fn nodes_at_depth(depth: usize) -> usize {
    (1 << depth) - 1
}

// Number of exit points (leaves at bottom of unrolled section).
// For depth K: `2^K` exit points.

// =============================================================================
// Sealed Trait for Unroll Depth
// =============================================================================

/// Sealed trait that defines the array sizes for a given unroll depth.
///
/// This trait is sealed (cannot be implemented outside this crate) and provides
/// the array sizes needed for stack-allocated unrolled layouts.
pub trait UnrollDepth: private::Sealed + core::fmt::Debug + Clone + Copy {
    /// The unroll depth.
    const DEPTH: usize;
    /// Number of nodes: `2^DEPTH - 1`
    const N_NODES: usize;
    /// Number of exits: `2^DEPTH`
    const N_EXITS: usize;

    /// Type for the node array (sized to N_NODES).
    type NodeArray<T: Copy + core::fmt::Debug + Send + Sync>: AsRef<[T]>
        + AsMut<[T]>
        + Clone
        + core::fmt::Debug
        + Send
        + Sync;
    /// Type for the exit array (sized to N_EXITS).
    type ExitArray<T: Copy + core::fmt::Debug + Send + Sync>: AsRef<[T]>
        + AsMut<[T]>
        + Clone
        + core::fmt::Debug
        + Send
        + Sync;

    /// Create a node array filled with a value.
    fn node_array_fill<T: Copy + core::fmt::Debug + Send + Sync>(val: T) -> Self::NodeArray<T>;
    /// Create an exit array filled with a value.
    fn exit_array_fill<T: Copy + core::fmt::Debug + Send + Sync>(val: T) -> Self::ExitArray<T>;
}

mod private {
    pub trait Sealed {}
}

/// Macro to implement UnrollDepth for a given depth.
macro_rules! impl_unroll_depth {
    ($name:ident, $depth:expr, $n_nodes:expr, $n_exits:expr, $doc:expr) => {
        #[doc = $doc]
        #[derive(Debug, Clone, Copy)]
        pub struct $name;

        impl private::Sealed for $name {}
        impl UnrollDepth for $name {
            const DEPTH: usize = $depth;
            const N_NODES: usize = $n_nodes;
            const N_EXITS: usize = $n_exits;
            type NodeArray<T: Copy + core::fmt::Debug + Send + Sync> = [T; $n_nodes];
            type ExitArray<T: Copy + core::fmt::Debug + Send + Sync> = [T; $n_exits];

            #[inline]
            fn node_array_fill<T: Copy + core::fmt::Debug + Send + Sync>(
                val: T,
            ) -> Self::NodeArray<T> {
                [val; $n_nodes]
            }
            #[inline]
            fn exit_array_fill<T: Copy + core::fmt::Debug + Send + Sync>(
                val: T,
            ) -> Self::ExitArray<T> {
                [val; $n_exits]
            }
        }
    };
}

// Depth implementations: 2^DEPTH - 1 nodes, 2^DEPTH exits
impl_unroll_depth!(Depth4, 4, 15, 16, "Depth 4: 15 nodes, 16 exits.");
impl_unroll_depth!(
    Depth6,
    6,
    63,
    64,
    "Depth 6: 63 nodes, 64 exits (default, matches XGBoost)."
);
impl_unroll_depth!(Depth8, 8, 255, 256, "Depth 8: 255 nodes, 256 exits.");

// =============================================================================
// UnrolledTreeLayout
// =============================================================================

/// Unrolled layout for the top levels of a tree.
///
/// Stores the top `D::DEPTH` levels in flat stack-allocated arrays for
/// cache-friendly traversal. After traversing these levels, use
/// `exit_node_idx` to get the original tree node index for continued traversal.
///
/// # Type Parameters
///
/// - `D`: Depth marker type implementing [`UnrollDepth`] (e.g., [`Depth6`])
///
/// # Array Sizes
///
/// - Nodes: `2^DEPTH - 1` (e.g., 63 for depth 6)
/// - Exits: `2^DEPTH` (e.g., 64 for depth 6)
#[derive(Debug, Clone)]
pub struct UnrolledTreeLayout<D: UnrollDepth = Depth6> {
    /// Split feature index per array node.
    split_indices: D::NodeArray<u32>,

    /// Original tree node index per array node.
    tree_node_indices: D::NodeArray<u32>,

    /// Split threshold per array node.
    split_thresholds: D::NodeArray<f32>,

    /// Default direction for missing values (true = left).
    default_left: D::NodeArray<bool>,

    /// Whether each array node corresponds to a leaf in original tree.
    is_original_leaf: D::NodeArray<bool>,

    /// Split type per array node (Numeric or Categorical).
    split_types: D::NodeArray<SplitType>,

    /// Mapping from exit points to original tree node indices.
    exit_node_idx: D::ExitArray<u32>,

    /// Whether this tree has any categorical splits.
    has_categorical: bool,

    /// Categorical segments (heap allocated, variable size).
    cat_segments: Box<[(u32, u32)]>,

    /// Categorical bitsets (heap allocated, variable size).
    cat_bitsets: Box<[u32]>,
}

impl<D: UnrollDepth> UnrolledTreeLayout<D> {
    /// Create an unrolled layout from a Tree.
    pub fn from_tree<L: LeafValue>(tree: &Tree<L>) -> Self {
        // Initialize arrays with fill values
        let mut split_indices = D::node_array_fill(0u32);
        let mut tree_node_indices = D::node_array_fill(0u32);
        let mut split_thresholds = D::node_array_fill(f32::NAN);
        let mut default_left = D::node_array_fill(false);
        let mut is_original_leaf = D::node_array_fill(false);
        let mut split_types = D::node_array_fill(SplitType::Numeric);
        let mut exit_node_idx = D::exit_array_fill(0u32);

        // Copy categorical data
        let has_categorical = tree.has_categorical();
        let (cat_segments, cat_bitsets) = if has_categorical {
            let cats = tree.categories();
            (cats.segments().to_vec(), cats.bitsets().to_vec())
        } else {
            (vec![], vec![])
        };

        // Populate the array layout by traversing the original tree
        populate_recursive(
            tree,
            0,        // original tree node idx (root)
            0,        // array node idx
            0,        // current level
            D::DEPTH, // target depth
            split_indices.as_mut(),
            tree_node_indices.as_mut(),
            split_thresholds.as_mut(),
            default_left.as_mut(),
            is_original_leaf.as_mut(),
            split_types.as_mut(),
            exit_node_idx.as_mut(),
        );

        Self {
            split_indices,
            tree_node_indices,
            split_thresholds,
            default_left,
            is_original_leaf,
            split_types,
            exit_node_idx,
            has_categorical,
            cat_segments: cat_segments.into_boxed_slice(),
            cat_bitsets: cat_bitsets.into_boxed_slice(),
        }
    }

    /// Number of levels unrolled.
    #[inline]
    pub const fn depth(&self) -> usize {
        D::DEPTH
    }

    /// Number of nodes in the array layout.
    #[inline]
    pub const fn n_nodes(&self) -> usize {
        D::N_NODES
    }

    /// Number of exit points.
    #[inline]
    pub const fn n_exits(&self) -> usize {
        D::N_EXITS
    }

    /// Get split feature index for an array node.
    #[inline]
    pub fn split_index(&self, array_idx: usize) -> u32 {
        self.split_indices.as_ref()[array_idx]
    }

    /// Get split threshold for an array node.
    #[inline]
    pub fn split_threshold(&self, array_idx: usize) -> f32 {
        self.split_thresholds.as_ref()[array_idx]
    }

    /// Get default direction for missing values.
    #[inline]
    pub fn default_left(&self, array_idx: usize) -> bool {
        self.default_left.as_ref()[array_idx]
    }

    /// Check if the array node corresponds to a leaf in the original tree.
    #[inline]
    pub fn is_original_leaf(&self, array_idx: usize) -> bool {
        self.is_original_leaf.as_ref()[array_idx]
    }

    /// Get split type for an array node.
    #[inline]
    pub fn split_type(&self, array_idx: usize) -> SplitType {
        self.split_types.as_ref()[array_idx]
    }

    /// Get the original tree node index for an exit point.
    #[inline]
    pub fn exit_node_idx(&self, exit_idx: usize) -> u32 {
        self.exit_node_idx.as_ref()[exit_idx]
    }

    /// Get a slice of all split indices for SIMD traversal.
    #[inline]
    pub fn split_indices_slice(&self) -> &[u32] {
        self.split_indices.as_ref()
    }

    /// Get a slice of all split thresholds for SIMD traversal.
    #[inline]
    pub fn split_thresholds_slice(&self) -> &[f32] {
        self.split_thresholds.as_ref()
    }

    /// Get a slice of all default_left flags for SIMD traversal.
    #[inline]
    pub fn default_left_slice(&self) -> &[bool] {
        self.default_left.as_ref()
    }

    /// Whether this tree has categorical splits.
    #[inline]
    pub fn has_categorical(&self) -> bool {
        self.has_categorical
    }

    /// Check if a category goes right for a categorical split.
    pub fn category_goes_right(&self, tree_node_idx: u32, category: u32) -> bool {
        let (start, size) = self.cat_segments[tree_node_idx as usize];
        if size == 0 {
            return false;
        }
        let word_idx = (category >> 5) as usize;
        let bit_idx = category & 31;
        if word_idx >= size as usize {
            return false;
        }
        let word = self.cat_bitsets[start as usize + word_idx];
        (word >> bit_idx) & 1 == 1
    }

    /// Check if category goes right using array-based storage.
    #[inline]
    fn category_goes_right_array(&self, array_idx: usize, category: u32) -> bool {
        debug_assert!(self.has_categorical);
        let tree_node_idx = self.tree_node_indices.as_ref()[array_idx];
        self.category_goes_right(tree_node_idx, category)
    }

    /// Traverse the array layout for a single sample.
    #[inline]
    pub fn traverse_to_exit(&self, sample: ArrayView1<'_, f32>) -> usize {
        let mut idx = 0usize;
        let split_indices = self.split_indices.as_ref();
        let split_thresholds = self.split_thresholds.as_ref();
        let default_left = self.default_left.as_ref();
        let split_types = self.split_types.as_ref();

        for _ in 0..D::DEPTH {
            let feat_idx = split_indices[idx] as usize;
            debug_assert!(
                feat_idx < sample.len(),
                "sample has {} features but split needs feature {feat_idx}",
                sample.len()
            );
            let fvalue = sample[feat_idx];

            let go_left = if fvalue.is_nan() {
                default_left[idx]
            } else if self.has_categorical && split_types[idx] == SplitType::Categorical {
                let category = fvalue as u32;
                !self.category_goes_right_array(idx, category)
            } else {
                fvalue < split_thresholds[idx]
            };

            idx = if go_left { 2 * idx + 1 } else { 2 * idx + 2 };
        }

        // Convert final array index to exit index
        idx - nodes_at_depth(D::DEPTH)
    }

    /// Traverse a block of rows through the array layout.
    ///
    /// This uses `SamplesView` (from `Dataset::buffer_samples`) for cache-friendly
    /// sample-major access.
    pub fn traverse_block(&self, data: &SamplesView<'_>, exit_indices: &mut [usize]) {
        let split_indices = self.split_indices.as_ref();
        let split_thresholds = self.split_thresholds.as_ref();
        let default_left = self.default_left.as_ref();
        let split_types = self.split_types.as_ref();

        // Initialize all rows at position 0 within level
        for pos in exit_indices.iter_mut() {
            *pos = 0;
        }

        // Traverse level by level
        for level in 0..D::DEPTH {
            let level_start = nodes_at_depth(level);

            for (row_idx, pos) in exit_indices.iter_mut().enumerate() {
                let array_idx = level_start + *pos;

                let feat_idx = split_indices[array_idx] as usize;
                let sample = data.sample_view(row_idx);
                debug_assert!(
                    feat_idx < sample.len(),
                    "sample has {} features but split needs feature {feat_idx}",
                    sample.len()
                );
                let fvalue = sample[feat_idx];

                let go_left = if fvalue.is_nan() {
                    default_left[array_idx]
                } else if self.has_categorical && split_types[array_idx] == SplitType::Categorical {
                    let category = fvalue as u32;
                    !self.category_goes_right_array(array_idx, category)
                } else {
                    fvalue < split_thresholds[array_idx]
                };

                *pos = 2 * *pos + (!go_left as usize);
            }
        }
    }
}

// =============================================================================
// Type Aliases for Common Depths
// =============================================================================

/// Unrolled layout with depth 4 (15 nodes, 16 exits) - for small/shallow trees.
pub type UnrolledTreeLayout4 = UnrolledTreeLayout<Depth4>;

/// Unrolled layout with depth 6 (63 nodes, 64 exits) - default, matches XGBoost.
pub type UnrolledTreeLayout6 = UnrolledTreeLayout<Depth6>;

/// Unrolled layout with depth 8 (255 nodes, 256 exits) - for deep trees.
pub type UnrolledTreeLayout8 = UnrolledTreeLayout<Depth8>;

// =============================================================================
// Shared helper functions
// =============================================================================

/// Recursively populate the array layout from the original tree.
fn populate_recursive<L: LeafValue>(
    tree: &Tree<L>,
    tree_nidx: u32,
    array_nidx: usize,
    level: usize,
    target_depth: usize,
    split_indices: &mut [u32],
    tree_node_indices: &mut [u32],
    split_thresholds: &mut [f32],
    default_left: &mut [bool],
    is_original_leaf: &mut [bool],
    split_types: &mut [SplitType],
    exit_node_idx: &mut [u32],
) {
    // At target depth, record exit mapping
    if level == target_depth {
        let exit_idx = array_nidx - nodes_at_depth(target_depth);
        exit_node_idx[exit_idx] = tree_nidx;
        return;
    }

    // Internal node mapping (array_nidx is guaranteed < D::N_NODES here).
    tree_node_indices[array_nidx] = tree_nidx;

    // If this is a leaf in the original tree
    if tree.is_leaf(tree_nidx) {
        // Mark as leaf and use NaN threshold (comparison always false → goes right)
        split_indices[array_nidx] = 0;
        split_thresholds[array_nidx] = f32::NAN;
        default_left[array_nidx] = false; // NaN goes right
        is_original_leaf[array_nidx] = true;
        split_types[array_nidx] = SplitType::Numeric;

        // Both children at next level point to the same leaf
        let left_array_idx = 2 * array_nidx + 1;
        let right_array_idx = 2 * array_nidx + 2;

        populate_recursive(
            tree,
            tree_nidx, // Same node - it's a leaf
            left_array_idx,
            level + 1,
            target_depth,
            split_indices,
            tree_node_indices,
            split_thresholds,
            default_left,
            is_original_leaf,
            split_types,
            exit_node_idx,
        );
        populate_recursive(
            tree,
            tree_nidx, // Same node - it's a leaf
            right_array_idx,
            level + 1,
            target_depth,
            split_indices,
            tree_node_indices,
            split_thresholds,
            default_left,
            is_original_leaf,
            split_types,
            exit_node_idx,
        );
    } else {
        // Copy split info from original tree
        split_indices[array_nidx] = tree.split_index(tree_nidx);
        split_thresholds[array_nidx] = tree.split_threshold(tree_nidx);
        default_left[array_nidx] = tree.default_left(tree_nidx);
        is_original_leaf[array_nidx] = false;
        split_types[array_nidx] = tree.split_type(tree_nidx);

        let left_child = tree.left_child(tree_nidx);
        let right_child = tree.right_child(tree_nidx);

        let left_array_idx = 2 * array_nidx + 1;
        let right_array_idx = 2 * array_nidx + 2;

        populate_recursive(
            tree,
            left_child,
            left_array_idx,
            level + 1,
            target_depth,
            split_indices,
            tree_node_indices,
            split_thresholds,
            default_left,
            is_original_leaf,
            split_types,
            exit_node_idx,
        );
        populate_recursive(
            tree,
            right_child,
            right_array_idx,
            level + 1,
            target_depth,
            split_indices,
            tree_node_indices,
            split_thresholds,
            default_left,
            is_original_leaf,
            split_types,
            exit_node_idx,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repr::gbdt::{MutableTree, ScalarLeaf, Tree};

    fn build_complete_tree(depth: usize) -> Tree<ScalarLeaf> {
        let mut builder = MutableTree::with_capacity(nodes_at_depth(depth + 1));
        let root = builder.init_root();

        fn grow(
            builder: &mut MutableTree<ScalarLeaf>,
            node: u32,
            current_depth: usize,
            max_depth: usize,
            leaf_counter: &mut f32,
        ) {
            if current_depth == max_depth {
                let val = *leaf_counter;
                *leaf_counter += 1.0;
                builder.make_leaf(node, ScalarLeaf(val));
            } else {
                let (left, right) = builder.apply_numeric_split(node, 0, 0.5, true);
                grow(builder, left, current_depth + 1, max_depth, leaf_counter);
                grow(builder, right, current_depth + 1, max_depth, leaf_counter);
            }
        }

        let mut leaf_counter = 0.0;
        grow(&mut builder, root, 0, depth, &mut leaf_counter);
        builder.freeze()
    }

    #[test]
    fn unrolled_layout_basic() {
        let mut builder = MutableTree::with_capacity(3);
        let root = builder.init_root();
        let (l, r) = builder.apply_numeric_split(root, 0, 0.5, true);
        builder.make_leaf(l, ScalarLeaf(1.0));
        builder.make_leaf(r, ScalarLeaf(2.0));
        let tree = builder.freeze();

        let layout = UnrolledTreeLayout4::from_tree(&tree);

        assert_eq!(layout.depth(), 4);
        assert_eq!(layout.n_nodes(), 15);
        assert_eq!(layout.n_exits(), 16);
        assert_eq!(layout.split_index(0), 0);
        assert!((layout.split_threshold(0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn unrolled_layout_traverse() {
        let mut builder = MutableTree::with_capacity(3);
        let root = builder.init_root();
        let (l, r) = builder.apply_numeric_split(root, 0, 0.5, true);
        builder.make_leaf(l, ScalarLeaf(1.0));
        builder.make_leaf(r, ScalarLeaf(2.0));
        let tree = builder.freeze();

        let layout = UnrolledTreeLayout4::from_tree(&tree);

        let exit_left = layout.traverse_to_exit(ndarray::array![0.3f32].view());
        let exit_right = layout.traverse_to_exit(ndarray::array![0.7f32].view());

        let left_node = layout.exit_node_idx(exit_left);
        let right_node = layout.exit_node_idx(exit_right);

        assert!(tree.is_leaf(left_node));
        assert!(tree.is_leaf(right_node));
        assert_eq!(tree.leaf_value(left_node).0, 1.0);
        assert_eq!(tree.leaf_value(right_node).0, 2.0);
    }

    #[test]
    fn unrolled_layout_deeper_tree() {
        let tree = build_complete_tree(3);
        let layout = UnrolledTreeLayout4::from_tree(&tree);

        assert_eq!(layout.depth(), 4);
        assert_eq!(layout.n_nodes(), 15);
        assert_eq!(layout.n_exits(), 16);
    }

    #[test]
    fn unrolled_layout_block_processing() {
        use crate::data::SamplesView;

        let mut builder = MutableTree::with_capacity(3);
        let root = builder.init_root();
        let (l, r) = builder.apply_numeric_split(root, 0, 0.5, true);
        builder.make_leaf(l, ScalarLeaf(1.0));
        builder.make_leaf(r, ScalarLeaf(2.0));
        let tree = builder.freeze();

        let layout = UnrolledTreeLayout4::from_tree(&tree);

        // Create a SamplesView with 4 rows, 1 feature each
        let feature_values = [0.3f32, 0.7, 0.2, 0.9];
        let view = SamplesView::from_slice(&feature_values, 4, 1).unwrap();
        let mut exit_indices = vec![0usize; 4];

        layout.traverse_block(&view, &mut exit_indices);

        for (i, &exit_idx) in exit_indices.iter().enumerate() {
            let node_idx = layout.exit_node_idx(exit_idx);
            let expected_val = if feature_values[i] < 0.5 { 1.0 } else { 2.0 };
            assert_eq!(
                tree.leaf_value(node_idx).0,
                expected_val,
                "Row {} with feature {} should get leaf {}",
                i,
                feature_values[i],
                expected_val
            );
        }
    }

    #[test]
    fn unrolled_layout_missing_values() {
        let mut builder = MutableTree::with_capacity(3);
        let root = builder.init_root();
        let (l, r) = builder.apply_numeric_split(root, 0, 0.5, true);
        builder.make_leaf(l, ScalarLeaf(1.0));
        builder.make_leaf(r, ScalarLeaf(2.0));
        let tree = builder.freeze();

        let layout = UnrolledTreeLayout4::from_tree(&tree);

        let exit_nan = layout.traverse_to_exit(ndarray::array![f32::NAN].view());
        let node_idx = layout.exit_node_idx(exit_nan);
        assert_eq!(tree.leaf_value(node_idx).0, 1.0);
    }

    #[test]
    fn unrolled_layout_depth6_default() {
        let tree = build_complete_tree(4);
        let layout = UnrolledTreeLayout6::from_tree(&tree);

        assert_eq!(layout.depth(), 6);
        assert_eq!(layout.n_nodes(), 63);
        assert_eq!(layout.n_exits(), 64);
    }

    #[test]
    fn unrolled_layout_depth8() {
        let tree = build_complete_tree(4);
        let layout = UnrolledTreeLayout8::from_tree(&tree);

        assert_eq!(layout.depth(), 8);
        assert_eq!(layout.n_nodes(), 255);
        assert_eq!(layout.n_exits(), 256);
    }
}

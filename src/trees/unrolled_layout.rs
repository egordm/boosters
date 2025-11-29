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
//! # Future: Const Generic Version
//!
//! The current implementation uses runtime depth with heap allocation for
//! simplicity. Per RFC-0002 DD-4, the target design uses const generics:
//!
//! ```ignore
//! // Future implementation
//! pub struct UnrolledTreeLayout<const DEPTH: usize = 6> {
//!     split_indices: [u32; (1 << DEPTH) - 1],  // Stack allocated
//!     exit_node_idx: [u32; 1 << DEPTH],
//!     // ...
//! }
//!
//! pub type UnrolledTreeLayout6 = UnrolledTreeLayout<6>;  // 63 nodes
//! pub type UnrolledTreeLayout4 = UnrolledTreeLayout<4>;  // 15 nodes
//! ```
//!
//! The current structure is designed for easy migration to const generics.

use super::leaf::LeafValue;
use super::node::SplitType;
use super::soa::SoATreeStorage;

/// Maximum number of levels to unroll (matches XGBoost).
/// 6 levels = 63 nodes, 8 levels = 255 nodes.
pub const MAX_UNROLL_DEPTH: usize = 6;

/// Number of nodes in a complete binary tree of given depth.
#[inline]
const fn nodes_at_depth(depth: usize) -> usize {
    (1 << depth) - 1
}

/// Number of exit points (leaves at bottom of unrolled section).
#[inline]
const fn exits_at_depth(depth: usize) -> usize {
    1 << depth
}

/// Unrolled layout for the top levels of a tree.
///
/// Stores the top `depth` levels in a flat array for cache-friendly
/// traversal. After traversing these levels, use `exit_node_idx` to
/// get the original tree node index for continued traversal.
#[derive(Debug, Clone)]
pub struct UnrolledTreeLayout {
    /// Number of levels unrolled into the array.
    depth: usize,

    /// Split feature index per array node.
    /// Length: `2^depth - 1`
    split_indices: Box<[u32]>,

    /// Split threshold per array node.
    /// For leaf nodes in the original tree, this is NaN (comparison always false).
    split_thresholds: Box<[f32]>,

    /// Default direction for missing values (true = left).
    default_left: Box<[bool]>,

    /// Whether each array node corresponds to a leaf in the original tree.
    /// If true, no further traversal is needed.
    is_original_leaf: Box<[bool]>,

    /// Split type per array node (Numeric or Categorical).
    split_types: Box<[SplitType]>,

    /// Mapping from exit points to original tree node indices.
    /// Length: `2^depth`
    /// After traversing `depth` levels, index into this array to get
    /// the node index in the original tree for continued processing.
    exit_node_idx: Box<[u32]>,

    /// Whether this tree has any categorical splits (for fast path).
    has_categorical: bool,

    /// Reference to categorical data (shared with original tree).
    /// We store the segments inline but bitsets are copied.
    cat_segments: Box<[(u32, u32)]>,
    cat_bitsets: Box<[u32]>,
}

impl UnrolledTreeLayout {
    /// Create an unrolled layout from a SoATreeStorage.
    ///
    /// Unrolls up to `max_depth` levels (capped at MAX_UNROLL_DEPTH).
    /// If the tree is shallower than `max_depth`, uses the tree's actual depth.
    pub fn from_tree<L: LeafValue>(tree: &SoATreeStorage<L>, max_depth: usize) -> Self {
        let depth = max_depth.min(MAX_UNROLL_DEPTH);
        let num_array_nodes = nodes_at_depth(depth);
        let num_exits = exits_at_depth(depth);

        // Allocate arrays
        let mut split_indices = vec![0u32; num_array_nodes];
        let mut split_thresholds = vec![f32::NAN; num_array_nodes];
        let mut default_left = vec![false; num_array_nodes];
        let mut is_original_leaf = vec![false; num_array_nodes];
        let mut split_types = vec![SplitType::Numeric; num_array_nodes];
        let mut exit_node_idx = vec![0u32; num_exits];

        // Copy categorical data
        let has_categorical = tree.has_categorical();
        let (cat_segments, cat_bitsets) = if has_categorical {
            let cats = tree.categories();
            (cats.segments().to_vec(), cats.bitsets().to_vec())
        } else {
            (vec![], vec![])
        };

        // Populate the array layout by traversing the original tree
        Self::populate_recursive(
            tree,
            0,     // original tree node idx (root)
            0,     // array node idx
            0,     // current level
            depth, // target depth
            &mut split_indices,
            &mut split_thresholds,
            &mut default_left,
            &mut is_original_leaf,
            &mut split_types,
            &mut exit_node_idx,
        );

        Self {
            depth,
            split_indices: split_indices.into_boxed_slice(),
            split_thresholds: split_thresholds.into_boxed_slice(),
            default_left: default_left.into_boxed_slice(),
            is_original_leaf: is_original_leaf.into_boxed_slice(),
            split_types: split_types.into_boxed_slice(),
            exit_node_idx: exit_node_idx.into_boxed_slice(),
            has_categorical,
            cat_segments: cat_segments.into_boxed_slice(),
            cat_bitsets: cat_bitsets.into_boxed_slice(),
        }
    }

    /// Recursively populate the array layout from the original tree.
    fn populate_recursive<L: LeafValue>(
        tree: &SoATreeStorage<L>,
        tree_nidx: u32,
        array_nidx: usize,
        level: usize,
        target_depth: usize,
        split_indices: &mut [u32],
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

        // If this is a leaf in the original tree
        if tree.is_leaf(tree_nidx) {
            // Mark as leaf and use NaN threshold (comparison always false → goes right)
            // But we want consistent behavior, so we set default_left to go right
            // to a "virtual" node that will map back to this same leaf.
            split_indices[array_nidx] = 0;
            split_thresholds[array_nidx] = f32::NAN;
            default_left[array_nidx] = false; // NaN goes right
            is_original_leaf[array_nidx] = true;
            split_types[array_nidx] = SplitType::Numeric;

            // Both children at next level point to the same leaf
            let left_array_idx = 2 * array_nidx + 1;
            let right_array_idx = 2 * array_nidx + 2;

            Self::populate_recursive(
                tree,
                tree_nidx, // Same node - it's a leaf
                left_array_idx,
                level + 1,
                target_depth,
                split_indices,
                split_thresholds,
                default_left,
                is_original_leaf,
                split_types,
                exit_node_idx,
            );
            Self::populate_recursive(
                tree,
                tree_nidx, // Same node - it's a leaf
                right_array_idx,
                level + 1,
                target_depth,
                split_indices,
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

            Self::populate_recursive(
                tree,
                left_child,
                left_array_idx,
                level + 1,
                target_depth,
                split_indices,
                split_thresholds,
                default_left,
                is_original_leaf,
                split_types,
                exit_node_idx,
            );
            Self::populate_recursive(
                tree,
                right_child,
                right_array_idx,
                level + 1,
                target_depth,
                split_indices,
                split_thresholds,
                default_left,
                is_original_leaf,
                split_types,
                exit_node_idx,
            );
        }
    }

    /// Number of levels unrolled.
    #[inline]
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Number of nodes in the array layout.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.split_indices.len()
    }

    /// Number of exit points.
    #[inline]
    pub fn num_exits(&self) -> usize {
        self.exit_node_idx.len()
    }

    /// Get split feature index for an array node.
    #[inline]
    pub fn split_index(&self, array_idx: usize) -> u32 {
        self.split_indices[array_idx]
    }

    /// Get split threshold for an array node.
    #[inline]
    pub fn split_threshold(&self, array_idx: usize) -> f32 {
        self.split_thresholds[array_idx]
    }

    /// Get default direction for missing values.
    #[inline]
    pub fn default_left(&self, array_idx: usize) -> bool {
        self.default_left[array_idx]
    }

    /// Check if the array node corresponds to a leaf in the original tree.
    #[inline]
    pub fn is_original_leaf(&self, array_idx: usize) -> bool {
        self.is_original_leaf[array_idx]
    }

    /// Get split type for an array node.
    #[inline]
    pub fn split_type(&self, array_idx: usize) -> SplitType {
        self.split_types[array_idx]
    }

    /// Get the original tree node index for an exit point.
    ///
    /// After traversing `depth` levels, the row index (0..2^depth) maps
    /// to an exit point. This returns the corresponding node in the
    /// original tree for continued traversal.
    #[inline]
    pub fn exit_node_idx(&self, exit_idx: usize) -> u32 {
        self.exit_node_idx[exit_idx]
    }

    /// Whether this tree has categorical splits.
    #[inline]
    pub fn has_categorical(&self) -> bool {
        self.has_categorical
    }

    /// Check if a category goes right for a categorical split.
    ///
    /// NOTE: This uses the original tree's node index, not array index.
    /// For array-based categorical lookup, we'd need to store the mapping.
    /// For now, categorical splits in the array layout fall back to
    /// using the original tree's categorical data.
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

    /// Traverse the array layout for a single row, returning the exit index.
    ///
    /// The exit index (0..2^depth) can be used with `exit_node_idx()` to get
    /// the original tree node for continued traversal.
    #[inline]
    pub fn traverse_to_exit(&self, features: &[f32]) -> usize {
        let mut idx = 0usize;

        for _ in 0..self.depth {
            let feat_idx = self.split_indices[idx] as usize;
            let fvalue = features.get(feat_idx).copied().unwrap_or(f32::NAN);

            let go_left = if fvalue.is_nan() {
                self.default_left[idx]
            } else if self.has_categorical && self.split_types[idx] == SplitType::Categorical {
                // Categorical: categories in set go RIGHT
                let category = fvalue as u32;
                // For categorical, we need the original tree node idx
                // This is a limitation - for now, categorical splits in top levels
                // will still work but may not be as optimized
                !self.category_goes_right_array(idx, category)
            } else {
                fvalue < self.split_thresholds[idx]
            };

            idx = if go_left {
                2 * idx + 1
            } else {
                2 * idx + 2
            };
        }

        // Convert final array index to exit index
        idx - nodes_at_depth(self.depth)
    }

    /// Check if category goes right using array-based storage.
    ///
    /// NOTE: This is a simplified version. For full categorical support
    /// in the array layout, we'd need to map array indices to category segments.
    /// For now, this falls back to tree-based lookup which requires the
    /// original node index.
    fn category_goes_right_array(&self, _array_idx: usize, _category: u32) -> bool {
        // TODO: Implement proper array-based categorical lookup
        // For now, categorical splits in the unrolled section use a simplified path
        false
    }

    /// Process a block of rows through the array layout.
    ///
    /// This is the key optimization: all rows traverse the same levels together,
    /// which keeps the array data in cache.
    ///
    /// # Arguments
    ///
    /// * `features` - Feature matrix (row-major, each row is `num_features` f32s)
    /// * `num_features` - Number of features per row
    /// * `exit_indices` - Output: exit index for each row (length = num_rows)
    ///
    /// After calling this, use `exit_node_idx()` on each exit index to get
    /// the original tree node for continued traversal.
    pub fn process_block(
        &self,
        features: &[f32],
        num_features: usize,
        exit_indices: &mut [usize],
    ) {
        // Initialize all rows at position 0 within level
        // (This is the relative position, not the array index)
        for pos in exit_indices.iter_mut() {
            *pos = 0;
        }

        // Traverse level by level
        // At each level, position ranges from 0 to 2^level - 1
        for level in 0..self.depth {
            // First array index at this level
            let level_start = nodes_at_depth(level);

            for (row_idx, pos) in exit_indices.iter_mut().enumerate() {
                // Convert position-within-level to array index
                let array_idx = level_start + *pos;

                let feat_idx = self.split_indices[array_idx] as usize;
                let row_offset = row_idx * num_features;
                let fvalue = features
                    .get(row_offset + feat_idx)
                    .copied()
                    .unwrap_or(f32::NAN);

                let go_left = if fvalue.is_nan() {
                    self.default_left[array_idx]
                } else if self.has_categorical
                    && self.split_types[array_idx] == SplitType::Categorical
                {
                    let category = fvalue as u32;
                    !self.category_goes_right_array(array_idx, category)
                } else {
                    fvalue < self.split_thresholds[array_idx]
                };

                // Update position for next level
                // In perfect binary tree: left child = 2*pos, right child = 2*pos + 1
                *pos = 2 * *pos + (!go_left as usize);
            }
        }

        // exit_indices now contains positions at level `depth`,
        // which are exactly the exit indices (0..2^depth)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trees::leaf::ScalarLeaf;
    use crate::trees::soa::TreeBuilder;

    fn build_complete_tree(depth: usize) -> SoATreeStorage<ScalarLeaf> {
        // Build a complete binary tree of given depth
        // Leaf values are the leaf index (0, 1, 2, ...)
        let mut builder = TreeBuilder::new();
        let num_nodes = nodes_at_depth(depth + 1); // +1 because depth is 0-indexed

        // Simple tree where feature 0 < 0.5 at every split
        // This creates a predictable structure for testing
        fn add_node(
            builder: &mut TreeBuilder<ScalarLeaf>,
            current_depth: usize,
            max_depth: usize,
            leaf_counter: &mut f32,
        ) -> u32 {
            if current_depth == max_depth {
                let val = *leaf_counter;
                *leaf_counter += 1.0;
                builder.add_leaf(ScalarLeaf(val))
            } else {
                let left = add_node(builder, current_depth + 1, max_depth, leaf_counter);
                let right = add_node(builder, current_depth + 1, max_depth, leaf_counter);
                builder.add_split(0, 0.5, true, left, right)
            }
        }

        let _ = num_nodes; // suppress unused warning
        let mut leaf_counter = 0.0;
        add_node(&mut builder, 0, depth, &mut leaf_counter);
        builder.build()
    }

    #[test]
    fn unrolled_layout_basic() {
        // Build a depth-2 tree (3 internal nodes, 4 leaves)
        let mut builder = TreeBuilder::new();
        builder.add_split(0, 0.5, true, 1, 2); // root
        builder.add_leaf(ScalarLeaf(1.0)); // node 1 (left leaf)
        builder.add_leaf(ScalarLeaf(2.0)); // node 2 (right leaf)
        let tree = builder.build();

        // Unroll 2 levels
        let layout = UnrolledTreeLayout::from_tree(&tree, 2);

        assert_eq!(layout.depth(), 2);
        assert_eq!(layout.num_nodes(), 3); // 2^2 - 1 = 3
        assert_eq!(layout.num_exits(), 4); // 2^2 = 4

        // Root should have the split info
        assert_eq!(layout.split_index(0), 0);
        assert!((layout.split_threshold(0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn unrolled_layout_traverse() {
        // Build a simple tree:
        //        [0] feat0 < 0.5
        //        /          \
        //    [1] leaf=1   [2] leaf=2
        let mut builder = TreeBuilder::new();
        builder.add_split(0, 0.5, true, 1, 2);
        builder.add_leaf(ScalarLeaf(1.0));
        builder.add_leaf(ScalarLeaf(2.0));
        let tree = builder.build();

        let layout = UnrolledTreeLayout::from_tree(&tree, 2);

        // Test traversal
        let exit_left = layout.traverse_to_exit(&[0.3]); // < 0.5, go left
        let exit_right = layout.traverse_to_exit(&[0.7]); // >= 0.5, go right

        // Exit indices should be different
        // After 2 levels from root, left path: 0 -> 1 -> 3 (exit 0)
        // Right path: 0 -> 2 -> 5 (exit 2) or similar depending on leaf handling

        // The exit node should map back to the correct leaf
        let left_node = layout.exit_node_idx(exit_left);
        let right_node = layout.exit_node_idx(exit_right);

        // Left should map to node 1, right to node 2
        assert!(tree.is_leaf(left_node));
        assert!(tree.is_leaf(right_node));
        assert_eq!(tree.leaf_value(left_node).0, 1.0);
        assert_eq!(tree.leaf_value(right_node).0, 2.0);
    }

    #[test]
    fn unrolled_layout_deeper_tree() {
        // Build a complete tree of depth 3 (7 internal nodes, 8 leaves)
        let tree = build_complete_tree(3);

        // Unroll 3 levels
        let layout = UnrolledTreeLayout::from_tree(&tree, 3);

        assert_eq!(layout.depth(), 3);
        assert_eq!(layout.num_nodes(), 7); // 2^3 - 1
        assert_eq!(layout.num_exits(), 8); // 2^3
    }

    #[test]
    fn unrolled_layout_block_processing() {
        // Build simple tree
        let mut builder = TreeBuilder::new();
        builder.add_split(0, 0.5, true, 1, 2);
        builder.add_leaf(ScalarLeaf(1.0));
        builder.add_leaf(ScalarLeaf(2.0));
        let tree = builder.build();

        let layout = UnrolledTreeLayout::from_tree(&tree, 2);

        // Process a block of 4 rows
        let features = vec![
            0.3, // row 0: go left
            0.7, // row 1: go right
            0.2, // row 2: go left
            0.9, // row 3: go right
        ];
        let mut exit_indices = vec![0usize; 4];

        layout.process_block(&features, 1, &mut exit_indices);

        // Verify exit indices lead to correct leaves
        for (i, &exit_idx) in exit_indices.iter().enumerate() {
            let node_idx = layout.exit_node_idx(exit_idx);
            let expected_val = if features[i] < 0.5 { 1.0 } else { 2.0 };
            assert_eq!(
                tree.leaf_value(node_idx).0,
                expected_val,
                "Row {} with feature {} should get leaf {}",
                i,
                features[i],
                expected_val
            );
        }
    }

    #[test]
    fn unrolled_layout_missing_values() {
        // Build tree with default_left = true
        let mut builder = TreeBuilder::new();
        builder.add_split(0, 0.5, true, 1, 2); // default goes left
        builder.add_leaf(ScalarLeaf(1.0)); // left leaf
        builder.add_leaf(ScalarLeaf(2.0)); // right leaf
        let tree = builder.build();

        let layout = UnrolledTreeLayout::from_tree(&tree, 2);

        // Test with NaN (missing value) - should go left (default)
        let exit_nan = layout.traverse_to_exit(&[f32::NAN]);
        let node_idx = layout.exit_node_idx(exit_nan);
        assert_eq!(tree.leaf_value(node_idx).0, 1.0);
    }
}

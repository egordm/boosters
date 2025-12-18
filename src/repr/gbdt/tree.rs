//! Canonical tree representation (SoA) and mutable construction API.
//!
//! This module provides:
//! - [`Tree`]: Immutable SoA tree storage for efficient traversal
//! - [`MutableTree`]: Builder for constructing trees during training
//! - [`TreeView`]: Read-only trait for unified tree access
//!
//! # TreeView Trait
//!
//! The [`TreeView`] trait provides a uniform interface for tree traversal,
//! implemented by both `Tree` and `MutableTree`. This enables generic
//! traversal code that works with either representation.

// Allow many constructor arguments for creating trees with all their fields.
#![allow(clippy::too_many_arguments)]

use super::categories::{float_to_category, CategoriesStorage};
use super::leaf::LeafValue;
use super::node::SplitType;
use super::NodeId;

// ============================================================================
// TreeView Trait
// ============================================================================

/// Read-only view of a tree for traversal.
///
/// Provides the minimal interface needed to traverse a tree from root to leaf.
/// Implemented for both [`Tree`] and [`MutableTree`].
///
/// # Design
///
/// This trait abstracts tree structure access, enabling generic prediction code
/// that works with both immutable trees (inference) and mutable trees (training).
///
/// # Example
///
/// ```ignore
/// use booste_rs::repr::gbdt::TreeView;
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
}

// ============================================================================
// TreeValidationError
// ============================================================================

/// Structural validation errors for [`Tree`].
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
    CategoricalSegmentsLenMismatch {
        segments_len: usize,
        n_nodes: usize,
    },
}

/// Structure-of-Arrays tree storage for efficient traversal.
///
/// Stores tree nodes in flat arrays for cache-friendly traversal.
/// Child indices are local to this tree (0 = root).
#[derive(Debug, Clone)]
pub struct Tree<L: LeafValue> {
    split_indices: Box<[u32]>,
    split_thresholds: Box<[f32]>,
    left_children: Box<[u32]>,
    right_children: Box<[u32]>,
    default_left: Box<[bool]>,
    is_leaf: Box<[bool]>,
    leaf_values: Box<[L]>,
    split_types: Box<[SplitType]>,
    categories: CategoriesStorage,
}

impl<L: LeafValue> Tree<L> {
    /// Create a new tree from parallel arrays.
    ///
    /// All arrays must have the same length (number of nodes).
    /// Creates a tree with only numeric splits (no categorical).
    pub fn new(
        split_indices: Vec<u32>,
        split_thresholds: Vec<f32>,
        left_children: Vec<u32>,
        right_children: Vec<u32>,
        default_left: Vec<bool>,
        is_leaf: Vec<bool>,
        leaf_values: Vec<L>,
    ) -> Self {
        let num_nodes = split_indices.len();
        debug_assert_eq!(num_nodes, split_thresholds.len());
        debug_assert_eq!(num_nodes, left_children.len());
        debug_assert_eq!(num_nodes, right_children.len());
        debug_assert_eq!(num_nodes, default_left.len());
        debug_assert_eq!(num_nodes, is_leaf.len());
        debug_assert_eq!(num_nodes, leaf_values.len());

        Self {
            split_indices: split_indices.into_boxed_slice(),
            split_thresholds: split_thresholds.into_boxed_slice(),
            left_children: left_children.into_boxed_slice(),
            right_children: right_children.into_boxed_slice(),
            default_left: default_left.into_boxed_slice(),
            is_leaf: is_leaf.into_boxed_slice(),
            leaf_values: leaf_values.into_boxed_slice(),
            split_types: vec![SplitType::Numeric; num_nodes].into_boxed_slice(),
            categories: CategoriesStorage::empty(),
        }
    }

    /// Create a new tree with categorical split support.
    ///
    /// All arrays must have the same length (number of nodes).
    pub fn with_categories(
        split_indices: Vec<u32>,
        split_thresholds: Vec<f32>,
        left_children: Vec<u32>,
        right_children: Vec<u32>,
        default_left: Vec<bool>,
        is_leaf: Vec<bool>,
        leaf_values: Vec<L>,
        split_types: Vec<SplitType>,
        categories: CategoriesStorage,
    ) -> Self {
        let num_nodes = split_indices.len();
        debug_assert_eq!(num_nodes, split_thresholds.len());
        debug_assert_eq!(num_nodes, left_children.len());
        debug_assert_eq!(num_nodes, right_children.len());
        debug_assert_eq!(num_nodes, default_left.len());
        debug_assert_eq!(num_nodes, is_leaf.len());
        debug_assert_eq!(num_nodes, leaf_values.len());
        debug_assert_eq!(num_nodes, split_types.len());

        Self {
            split_indices: split_indices.into_boxed_slice(),
            split_thresholds: split_thresholds.into_boxed_slice(),
            left_children: left_children.into_boxed_slice(),
            right_children: right_children.into_boxed_slice(),
            default_left: default_left.into_boxed_slice(),
            is_leaf: is_leaf.into_boxed_slice(),
            leaf_values: leaf_values.into_boxed_slice(),
            split_types: split_types.into_boxed_slice(),
            categories,
        }
    }

    /// Number of nodes in this tree.
    #[inline]
    pub fn n_nodes(&self) -> usize {
        self.is_leaf.len()
    }

    /// Check if a node is a leaf.
    #[inline]
    pub fn is_leaf(&self, node_idx: NodeId) -> bool {
        self.is_leaf[node_idx as usize]
    }

    /// Get split feature index for a node.
    #[inline]
    pub fn split_index(&self, node_idx: NodeId) -> u32 {
        self.split_indices[node_idx as usize]
    }

    /// Get split threshold for a node.
    #[inline]
    pub fn split_threshold(&self, node_idx: NodeId) -> f32 {
        self.split_thresholds[node_idx as usize]
    }

    /// Get left child index.
    #[inline]
    pub fn left_child(&self, node_idx: NodeId) -> NodeId {
        self.left_children[node_idx as usize]
    }

    /// Get right child index.
    #[inline]
    pub fn right_child(&self, node_idx: NodeId) -> NodeId {
        self.right_children[node_idx as usize]
    }

    /// Get default direction for missing values.
    #[inline]
    pub fn default_left(&self, node_idx: NodeId) -> bool {
        self.default_left[node_idx as usize]
    }

    /// Get leaf value for a node.
    #[inline]
    pub fn leaf_value(&self, node_idx: NodeId) -> &L {
        &self.leaf_values[node_idx as usize]
    }

    /// Get split type for a node.
    #[inline]
    pub fn split_type(&self, node_idx: NodeId) -> SplitType {
        self.split_types[node_idx as usize]
    }

    /// Check if this tree has any categorical splits.
    #[inline]
    pub fn has_categorical(&self) -> bool {
        !self.categories.is_empty()
    }

    /// Get reference to categories storage.
    #[inline]
    pub fn categories(&self) -> &CategoriesStorage {
        &self.categories
    }

    /// Traverse the tree to find the leaf for given features.
    pub fn predict_row(&self, features: &[f32]) -> &L {
        let mut idx: NodeId = 0; // Start at root

        while !self.is_leaf(idx) {
            let feat_idx = self.split_index(idx) as usize;
            let fvalue = features.get(feat_idx).copied().unwrap_or(f32::NAN);

            idx = if fvalue.is_nan() {
                if self.default_left(idx) {
                    self.left_child(idx)
                } else {
                    self.right_child(idx)
                }
            } else {
                match self.split_type(idx) {
                    SplitType::Numeric => {
                        if fvalue < self.split_threshold(idx) {
                            self.left_child(idx)
                        } else {
                            self.right_child(idx)
                        }
                    }
                    SplitType::Categorical => {
                        let category = float_to_category(fvalue);
                        if self.categories.category_goes_right(idx, category) {
                            self.right_child(idx)
                        } else {
                            self.left_child(idx)
                        }
                    }
                }
            };
        }

        self.leaf_value(idx)
    }

    /// Validate basic structural invariants for this tree.
    ///
    /// Intended for debug checks and tests (e.g., model conversion invariants).
    pub fn validate(&self) -> Result<(), TreeValidationError> {
        let n_nodes = self.n_nodes();
        if n_nodes == 0 {
            return Err(TreeValidationError::EmptyTree);
        }

        // If categorical splits exist, segments must be indexed by node.
        let has_cat_split = self
            .split_types
            .iter()
            .any(|t| matches!(t, SplitType::Categorical));
        if has_cat_split {
            let segments_len = self.categories.segments().len();
            if segments_len != n_nodes {
                return Err(TreeValidationError::CategoricalSegmentsLenMismatch {
                    segments_len,
                    n_nodes,
                });
            }
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

                    if !self.is_leaf(node) {
                        let left = self.left_child(node);
                        let right = self.right_child(node);

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

    /// Traverse the tree to find the leaf for a binned row.
    ///
    /// Used during training when we have binned data. Numeric bins are converted
    /// to float values using the bin mappers before comparison. Categorical bins
    /// are treated as canonical category indices (0..K-1).
    pub fn predict_binned_row(
        &self,
        row: &crate::data::binned::RowView<'_>,
        dataset: &crate::data::BinnedDataset,
    ) -> &L {
        let mut idx: NodeId = 0;

        while !self.is_leaf(idx) {
            let feat_idx = self.split_index(idx) as usize;
            let bin_opt = row.get_bin(feat_idx);

            idx = match bin_opt {
                None => {
                    if self.default_left(idx) {
                        self.left_child(idx)
                    } else {
                        self.right_child(idx)
                    }
                }
                Some(bin) => match self.split_type(idx) {
                    SplitType::Numeric => {
                        let mapper = dataset.bin_mapper(feat_idx);
                        let fvalue = mapper.bin_to_value(bin) as f32;
                        if fvalue < self.split_threshold(idx) {
                            self.left_child(idx)
                        } else {
                            self.right_child(idx)
                        }
                    }
                    SplitType::Categorical => {
                        let category = bin;
                        if self.categories.category_goes_right(idx, category) {
                            self.right_child(idx)
                        } else {
                            self.left_child(idx)
                        }
                    }
                },
            };
        }

        self.leaf_value(idx)
    }

    /// Batch predict for multiple rows in a binned dataset.
    ///
    /// This is more efficient than calling `predict_binned_row` in a loop because:
    /// - Better memory access patterns (processes all rows through same tree path)
    /// - Reduces function call overhead
    /// - Can be parallelized with Rayon
    ///
    /// # Arguments
    /// * `dataset` - The binned dataset containing the rows
    /// * `predictions` - Slice to update with leaf values (one per row)
    ///
    /// The leaf values are **added** to the existing predictions.
    pub fn predict_binned_batch(
        &self,
        dataset: &crate::data::BinnedDataset,
        predictions: &mut [f32],
    ) where
        L: Into<f32> + Copy,
    {
        let n_rows = dataset.n_rows();
        debug_assert_eq!(predictions.len(), n_rows);

        for row_idx in 0..n_rows {
            if let Some(row) = dataset.row_view(row_idx) {
                let leaf = self.predict_binned_row(&row, dataset);
                predictions[row_idx] += (*leaf).into();
            }
        }
    }

    /// Parallel batch predict for multiple rows in a binned dataset.
    ///
    /// Uses Rayon for parallel execution across rows.
    pub fn par_predict_binned_batch(
        &self,
        dataset: &crate::data::BinnedDataset,
        predictions: &mut [f32],
    ) where
        L: Into<f32> + Copy + Send + Sync,
    {
        use rayon::prelude::*;

        let n_rows = dataset.n_rows();
        debug_assert_eq!(predictions.len(), n_rows);

        predictions.par_iter_mut().enumerate().for_each(|(row_idx, pred)| {
            if let Some(row) = dataset.row_view(row_idx) {
                let leaf = self.predict_binned_row(&row, dataset);
                *pred += (*leaf).into();
            }
        });
    }

    /// Batch predict for multiple rows in a row-major matrix.
    ///
    /// This is used for evaluation on non-binned data (e.g., eval sets).
    /// The leaf values are **added** to the existing predictions.
    ///
    /// # Arguments
    /// * `matrix` - Row-major feature matrix
    /// * `predictions` - Slice to update with leaf values (one per row)
    pub fn predict_batch(
        &self,
        matrix: &crate::data::RowMatrix<f32>,
        predictions: &mut [f32],
    ) where
        L: Into<f32> + Copy,
    {
        use crate::data::DataMatrix;
        
        let n_rows = matrix.num_rows();
        let n_features = matrix.num_features();
        debug_assert_eq!(predictions.len(), n_rows);

        let mut row_buf = vec![0.0f32; n_features];
        for row_idx in 0..n_rows {
            matrix.copy_row(row_idx, &mut row_buf);
            let leaf = self.predict_row(&row_buf);
            predictions[row_idx] += (*leaf).into();
        }
    }

    /// Generic batch predict using any feature accessor.
    ///
    /// This is the unified prediction method that works with any data source
    /// implementing [`FeatureAccessor`]: RowMatrix, ColMatrix, BinnedAccessor, etc.
    ///
    /// Leaf values are **added** to the existing predictions (accumulate pattern).
    ///
    /// # Arguments
    /// * `accessor` - Feature value source
    /// * `predictions` - Slice to update with leaf values (one per row)
    ///
    /// # Example
    ///
    /// ```ignore
    /// use booste_rs::inference::{Tree, FeatureAccessor, traverse_to_leaf};
    /// use booste_rs::data::RowMatrix;
    ///
    /// let tree: Tree<ScalarLeaf> = /* ... */;
    /// let data = RowMatrix::from_vec(vec![0.1, 0.2, 0.3, 0.4], 2, 2);
    /// let mut predictions = vec![0.0; 2];
    /// tree.predict_batch_accumulate(&data, &mut predictions);
    /// ```
    pub fn predict_batch_accumulate<A: crate::data::FeatureAccessor>(
        &self,
        accessor: &A,
        predictions: &mut [f32],
    ) where
        L: Into<f32> + Copy,
    {
        use crate::inference::gbdt::traverse_to_leaf;
        
        let n_rows = accessor.num_rows();
        debug_assert_eq!(predictions.len(), n_rows);

        for row_idx in 0..n_rows {
            let leaf_idx = traverse_to_leaf(self, accessor, row_idx);
            let leaf = self.leaf_value(leaf_idx);
            predictions[row_idx] += (*leaf).into();
        }
    }
}

// =============================================================================
// TreeView for Tree
// =============================================================================

impl<L: LeafValue> TreeView for Tree<L> {
    type LeafValue = L;

    #[inline]
    fn n_nodes(&self) -> usize {
        self.is_leaf.len()
    }

    #[inline]
    fn is_leaf(&self, node: NodeId) -> bool {
        self.is_leaf[node as usize]
    }

    #[inline]
    fn split_index(&self, node: NodeId) -> u32 {
        self.split_indices[node as usize]
    }

    #[inline]
    fn split_threshold(&self, node: NodeId) -> f32 {
        self.split_thresholds[node as usize]
    }

    #[inline]
    fn left_child(&self, node: NodeId) -> NodeId {
        self.left_children[node as usize]
    }

    #[inline]
    fn right_child(&self, node: NodeId) -> NodeId {
        self.right_children[node as usize]
    }

    #[inline]
    fn default_left(&self, node: NodeId) -> bool {
        self.default_left[node as usize]
    }

    #[inline]
    fn split_type(&self, node: NodeId) -> SplitType {
        self.split_types[node as usize]
    }

    #[inline]
    fn categories(&self) -> &CategoriesStorage {
        &self.categories
    }

    #[inline]
    fn leaf_value(&self, node: NodeId) -> &L {
        &self.leaf_values[node as usize]
    }
}

// =============================================================================
// MutableTree (training-time construction)
// =============================================================================

/// Mutable tree for use during training.
///
/// Supports the training pattern where nodes are allocated first (placeholders)
/// and filled in later when splits/leaves are determined.
#[derive(Debug, Clone)]
pub struct MutableTree<L: LeafValue> {
    split_indices: Vec<u32>,
    split_thresholds: Vec<f32>,
    left_children: Vec<u32>,
    right_children: Vec<u32>,
    default_left: Vec<bool>,
    is_leaf: Vec<bool>,
    leaf_values: Vec<L>,
    split_types: Vec<SplitType>,
    /// Categorical data: (node_idx, category_bitset)
    categorical_nodes: Vec<(NodeId, Vec<u32>)>,
    next_id: NodeId,
}

impl<L: LeafValue> Default for MutableTree<L> {
    fn default() -> Self {
        Self::new()
    }
}

impl<L: LeafValue> MutableTree<L> {
    /// Create a new mutable tree.
    pub fn new() -> Self {
        Self {
            split_indices: Vec::with_capacity(64),
            split_thresholds: Vec::with_capacity(64),
            left_children: Vec::with_capacity(64),
            right_children: Vec::with_capacity(64),
            default_left: Vec::with_capacity(64),
            is_leaf: Vec::with_capacity(64),
            leaf_values: Vec::with_capacity(64),
            split_types: Vec::with_capacity(64),
            categorical_nodes: Vec::new(),
            next_id: 0,
        }
    }

    /// Create a tree with capacity hint.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            split_indices: Vec::with_capacity(capacity),
            split_thresholds: Vec::with_capacity(capacity),
            left_children: Vec::with_capacity(capacity),
            right_children: Vec::with_capacity(capacity),
            default_left: Vec::with_capacity(capacity),
            is_leaf: Vec::with_capacity(capacity),
            leaf_values: Vec::with_capacity(capacity),
            split_types: Vec::with_capacity(capacity),
            categorical_nodes: Vec::new(),
            next_id: 0,
        }
    }

    /// Initialize the root node as a placeholder.
    ///
    /// Returns the root node ID (always 0).
    pub fn init_root(&mut self) -> NodeId {
        self.reset();
        self.allocate_node();
        0
    }

    /// Initialize the tree with a fixed number of placeholder nodes.
    ///
    /// This is useful for model loaders where node indices and child references
    /// are already known (e.g. XGBoost JSON). Returns the root node ID (0).
    pub fn init_root_with_num_nodes(&mut self, num_nodes: usize) -> NodeId {
        self.reset();
        for _ in 0..num_nodes {
            self.allocate_node();
        }
        0
    }

    /// Apply a numeric split to a node, allocating child nodes.
    ///
    /// Returns `(left_id, right_id)`.
    pub fn apply_numeric_split(
        &mut self,
        node: NodeId,
        feature: u32,
        threshold: f32,
        default_left: bool,
    ) -> (NodeId, NodeId) {
        let left_id = self.allocate_node();
        let right_id = self.allocate_node();

        let idx = node as usize;
        self.split_indices[idx] = feature;
        self.split_thresholds[idx] = threshold;
        self.left_children[idx] = left_id;
        self.right_children[idx] = right_id;
        self.default_left[idx] = default_left;
        self.is_leaf[idx] = false;
        self.split_types[idx] = SplitType::Numeric;

        (left_id, right_id)
    }

    /// Set a numeric split on an existing node, with explicit child indices.
    ///
    /// Intended for conversion code where the full node set is pre-allocated.
    pub fn set_numeric_split(
        &mut self,
        node: NodeId,
        feature: u32,
        threshold: f32,
        default_left: bool,
        left_child: NodeId,
        right_child: NodeId,
    ) {
        let idx = node as usize;
        self.split_indices[idx] = feature;
        self.split_thresholds[idx] = threshold;
        self.left_children[idx] = left_child;
        self.right_children[idx] = right_child;
        self.default_left[idx] = default_left;
        self.is_leaf[idx] = false;
        self.split_types[idx] = SplitType::Numeric;
    }

    /// Apply a categorical split to a node, allocating child nodes.
    ///
    /// The `category_bitset` contains the packed u32 words for the category bitset.
    /// Categories in this bitset go RIGHT, categories not in the set go LEFT.
    ///
    /// Returns `(left_id, right_id)`.
    pub fn apply_categorical_split(
        &mut self,
        node: NodeId,
        feature: u32,
        category_bitset: Vec<u32>,
        default_left: bool,
    ) -> (NodeId, NodeId) {
        let left_id = self.allocate_node();
        let right_id = self.allocate_node();

        let idx = node as usize;
        self.split_indices[idx] = feature;
        self.split_thresholds[idx] = 0.0;
        self.left_children[idx] = left_id;
        self.right_children[idx] = right_id;
        self.default_left[idx] = default_left;
        self.is_leaf[idx] = false;
        self.split_types[idx] = SplitType::Categorical;
        self.categorical_nodes.push((node, category_bitset));

        (left_id, right_id)
    }

    /// Set a categorical split on an existing node, with explicit child indices.
    ///
    /// Intended for conversion code where the full node set is pre-allocated.
    pub fn set_categorical_split(
        &mut self,
        node: NodeId,
        feature: u32,
        category_bitset: Vec<u32>,
        default_left: bool,
        left_child: NodeId,
        right_child: NodeId,
    ) {
        let idx = node as usize;
        self.split_indices[idx] = feature;
        self.split_thresholds[idx] = 0.0;
        self.left_children[idx] = left_child;
        self.right_children[idx] = right_child;
        self.default_left[idx] = default_left;
        self.is_leaf[idx] = false;
        self.split_types[idx] = SplitType::Categorical;

        if let Some(pos) = self.categorical_nodes.iter().position(|(n, _)| *n == node) {
            self.categorical_nodes.remove(pos);
        }
        self.categorical_nodes.push((node, category_bitset));
    }

    /// Set a node as a leaf with the given value.
    pub fn make_leaf(&mut self, node: NodeId, value: L) {
        let idx = node as usize;
        self.is_leaf[idx] = true;
        self.leaf_values[idx] = value;
    }

    /// Apply learning rate to all leaf values.
    pub fn apply_learning_rate(&mut self, learning_rate: f32) {
        for (is_leaf, value) in self.is_leaf.iter().zip(self.leaf_values.iter_mut()) {
            if *is_leaf {
                value.scale(learning_rate);
            }
        }
    }

    /// Current number of allocated nodes.
    #[inline]
    pub fn n_nodes(&self) -> usize {
        self.split_indices.len()
    }

    /// Reset the tree for reuse.
    pub fn reset(&mut self) {
        self.split_indices.clear();
        self.split_thresholds.clear();
        self.left_children.clear();
        self.right_children.clear();
        self.default_left.clear();
        self.is_leaf.clear();
        self.leaf_values.clear();
        self.split_types.clear();
        self.categorical_nodes.clear();
        self.next_id = 0;
    }

    /// Finalize the tree and return immutable storage.
    pub fn freeze(self) -> Tree<L> {
        let categories = if self.categorical_nodes.is_empty() {
            CategoriesStorage::empty()
        } else {
            let mut cat_nodes = self.categorical_nodes;
            cat_nodes.sort_by_key(|(idx, _)| *idx);

            let num_nodes = self.split_indices.len();
            let mut segments = vec![(0u32, 0u32); num_nodes];
            let mut bitsets = Vec::new();

            for (node_idx, bitset) in cat_nodes {
                let start = bitsets.len() as u32;
                let size = bitset.len() as u32;
                segments[node_idx as usize] = (start, size);
                bitsets.extend(bitset);
            }

            CategoriesStorage::new(bitsets, segments)
        };

        Tree::with_categories(
            self.split_indices,
            self.split_thresholds,
            self.left_children,
            self.right_children,
            self.default_left,
            self.is_leaf,
            self.leaf_values,
            self.split_types,
            categories,
        )
    }

    fn allocate_node(&mut self) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;

        self.split_indices.push(0);
        self.split_thresholds.push(0.0);
        self.left_children.push(0);
        self.right_children.push(0);
        self.default_left.push(false);
        self.is_leaf.push(false);
        self.leaf_values.push(L::default());
        self.split_types.push(SplitType::Numeric);

        id
    }
}

// =============================================================================
// TreeView for MutableTree
// =============================================================================

impl<L: LeafValue> TreeView for MutableTree<L> {
    type LeafValue = L;

    #[inline]
    fn n_nodes(&self) -> usize {
        self.split_indices.len()
    }

    #[inline]
    fn is_leaf(&self, node: NodeId) -> bool {
        self.is_leaf[node as usize]
    }

    #[inline]
    fn split_index(&self, node: NodeId) -> u32 {
        self.split_indices[node as usize]
    }

    #[inline]
    fn split_threshold(&self, node: NodeId) -> f32 {
        self.split_thresholds[node as usize]
    }

    #[inline]
    fn left_child(&self, node: NodeId) -> NodeId {
        self.left_children[node as usize]
    }

    #[inline]
    fn right_child(&self, node: NodeId) -> NodeId {
        self.right_children[node as usize]
    }

    #[inline]
    fn default_left(&self, node: NodeId) -> bool {
        self.default_left[node as usize]
    }

    #[inline]
    fn split_type(&self, node: NodeId) -> SplitType {
        self.split_types[node as usize]
    }

    #[inline]
    fn categories(&self) -> &CategoriesStorage {
        // During construction, categorical splits are stored per-node
        // and not yet packed. Returns a static empty storage.
        // Categorical traversal is only supported after freeze().
        CategoriesStorage::empty_ref()
    }

    #[inline]
    fn leaf_value(&self, node: NodeId) -> &L {
        &self.leaf_values[node as usize]
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn predict_simple_tree() {
        // Tree:
        //   root: feat0 < 0.5
        //     left: leaf 1.0
        //     right: leaf 2.0
        let tree = crate::scalar_tree! {
            0 => num(0, 0.5, L) -> 1, 2,
            1 => leaf(1.0),
            2 => leaf(2.0),
        };

        assert_eq!(tree.predict_row(&[0.3]).0, 1.0);
        assert_eq!(tree.predict_row(&[0.7]).0, 2.0);
    }

    #[test]
    fn predict_categorical_tree() {
        // Root: categorical, categories {1,3} go RIGHT.
        let tree = crate::scalar_tree! {
            0 => cat(0, [1, 3], L) -> 1, 2,
            1 => leaf(-1.0),
            2 => leaf(1.0),
        };

        assert_eq!(tree.predict_row(&[0.0]).0, -1.0);
        assert_eq!(tree.predict_row(&[1.0]).0, 1.0);
        assert_eq!(tree.predict_row(&[3.0]).0, 1.0);
        assert_eq!(tree.predict_row(&[2.0]).0, -1.0);
    }

    #[test]
    fn test_predict_batch_accumulate() {
        use crate::data::RowMatrix;

        let tree = crate::scalar_tree! {
            0 => num(0, 0.5, L) -> 1, 2,
            1 => leaf(1.0),
            2 => leaf(2.0),
        };

        // 3 rows, 1 feature
        let data = RowMatrix::from_vec(vec![0.3, 0.7, 0.5], 3, 1);
        
        // Test accumulate pattern (starts with existing values)
        let mut predictions = vec![10.0, 20.0, 30.0];
        tree.predict_batch_accumulate(&data, &mut predictions);

        // Row 0: 0.3 < 0.5 -> left (1.0), 10.0 + 1.0 = 11.0
        // Row 1: 0.7 >= 0.5 -> right (2.0), 20.0 + 2.0 = 22.0
        // Row 2: 0.5 >= 0.5 -> right (2.0), 30.0 + 2.0 = 32.0
        assert_eq!(predictions, vec![11.0, 22.0, 32.0]);
    }
}

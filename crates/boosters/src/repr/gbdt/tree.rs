//! Canonical tree representation (SoA) and read-only tree interface.
//!
//! This module provides:
//! - [`Tree`]: Immutable SoA tree storage for efficient traversal
//! - [`TreeView`]: Read-only trait for unified tree access
//! - [`TreeValidationError`]: Structural validation errors
//!
//! For mutable tree construction during training, see [`super::mutable_tree::MutableTree`].

// Allow many constructor arguments for creating trees with all their fields.
#![allow(clippy::too_many_arguments)]

use ndarray::ArrayViewMut1;

use crate::Parallelism;
use crate::data::{BinnedDataset, DataAccessor, SampleAccessor};

use super::categories::{float_to_category, CategoriesStorage};
use super::coefficients::LeafCoefficients;
use super::leaf::LeafValue;
use super::node::SplitType;
use super::NodeId;

/// Re-export for backward compatibility during migration.
pub use crate::data::binned::BinnedSampleSlice as BinnedRowView;

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
    /// This is the primary traversal method that works with any `SampleAccessor`.
    /// The traversal handles NaN values using the tree's default direction,
    /// and supports both numeric and categorical splits.
    ///
    /// # Arguments
    ///
    /// * `sample` - Feature values for the sample (implements [`SampleAccessor`])
    ///
    /// # Returns
    ///
    /// The `NodeId` of the reached leaf node.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use boosters::repr::gbdt::TreeView;
    /// use boosters::data::SampleAccessor;
    ///
    /// let features: &[f32] = &[0.5, 1.0, 2.3];
    /// let leaf_id = tree.traverse_to_leaf(features);
    /// ```
    #[inline]
    fn traverse_to_leaf<S: SampleAccessor>(&self, sample: &S) -> NodeId {
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
    /// * `sample` - Feature values for the sample (implements [`SampleAccessor`])
    ///
    /// # Returns
    ///
    /// The `NodeId` of the reached leaf node.
    #[inline]
    fn traverse_to_leaf_from<S: SampleAccessor>(&self, start_node: NodeId, sample: &S) -> NodeId {
        let mut node = start_node;

        while !self.is_leaf(node) {
            let feat_idx = self.split_index(node) as usize;
            let fvalue = sample.feature(feat_idx);

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
    leaf_coefficients: LeafCoefficients,
    /// Optional gain at each split node (for explainability).
    gains: Option<Box<[f32]>>,
    /// Optional cover (hessian sum) at each node (for explainability).
    covers: Option<Box<[f32]>>,
}

impl<L: LeafValue> Tree<L> {
    /// Create a new tree from parallel arrays.
    ///
    /// All arrays must have the same length (number of nodes).
    ///
    /// # Parameters
    ///
    /// - Core structure: `split_indices`, `split_thresholds`, `left_children`,
    ///   `right_children`, `default_left`, `is_leaf`, `leaf_values`
    /// - Categorical support: `split_types`, `categories`
    /// - Linear leaves: `leaf_coefficients`
    ///
    /// For trees without categorical splits, pass `SplitType::Numeric` for all nodes
    /// and `CategoriesStorage::empty()`.
    ///
    /// For trees without linear leaves, pass `LeafCoefficients::empty()`.
    pub fn new(
        split_indices: Vec<u32>,
        split_thresholds: Vec<f32>,
        left_children: Vec<u32>,
        right_children: Vec<u32>,
        default_left: Vec<bool>,
        is_leaf: Vec<bool>,
        leaf_values: Vec<L>,
        split_types: Vec<SplitType>,
        categories: CategoriesStorage,
        leaf_coefficients: LeafCoefficients,
    ) -> Self {
        let n_nodes = split_indices.len();
        debug_assert_eq!(n_nodes, split_thresholds.len());
        debug_assert_eq!(n_nodes, left_children.len());
        debug_assert_eq!(n_nodes, right_children.len());
        debug_assert_eq!(n_nodes, default_left.len());
        debug_assert_eq!(n_nodes, is_leaf.len());
        debug_assert_eq!(n_nodes, leaf_values.len());
        debug_assert_eq!(n_nodes, split_types.len());

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
            leaf_coefficients,
            gains: None,
            covers: None,
        }
    }

    // =========================================================================
    // Linear Leaf Accessors
    // =========================================================================

    /// Get the linear terms for a leaf node, if any.
    ///
    /// Returns `Some((feature_indices, coefficients))` for linear leaves,
    /// or `None` for constant leaves.
    #[inline]
    pub fn leaf_terms(&self, node: NodeId) -> Option<(&[u32], &[f32])> {
        self.leaf_coefficients.leaf_terms(node)
    }

    /// Get the intercept for a leaf node.
    ///
    /// Returns 0.0 for constant leaves or if linear terms are empty.
    #[inline]
    pub fn leaf_intercept(&self, node: NodeId) -> f32 {
        self.leaf_coefficients.intercept(node)
    }

    /// Check if this tree has any linear leaf nodes.
    #[inline]
    pub fn has_linear_leaves(&self) -> bool {
        !self.leaf_coefficients.is_empty()
    }

    // =========================================================================
    // Explainability: Gains and Covers
    // =========================================================================

    /// Check if this tree has gain statistics.
    #[inline]
    pub fn has_gains(&self) -> bool {
        self.gains.is_some()
    }

    /// Check if this tree has cover statistics.
    #[inline]
    pub fn has_covers(&self) -> bool {
        self.covers.is_some()
    }

    /// Set the gains for this tree (builder pattern).
    pub fn with_gains(mut self, gains: Vec<f32>) -> Self {
        debug_assert_eq!(gains.len(), self.n_nodes());
        self.gains = Some(gains.into_boxed_slice());
        self
    }

    /// Set the covers for this tree (builder pattern).
    pub fn with_covers(mut self, covers: Vec<f32>) -> Self {
        debug_assert_eq!(covers.len(), self.n_nodes());
        self.covers = Some(covers.into_boxed_slice());
        self
    }

    /// Set both gains and covers.
    pub fn with_stats(self, gains: Vec<f32>, covers: Vec<f32>) -> Self {
        self.with_gains(gains).with_covers(covers)
    }

    /// Get read-only access to gains slice.
    ///
    /// Leaf nodes have gain=0, split nodes have the information gain from that split.
    pub fn gains(&self) -> Option<&[f32]> {
        self.gains.as_deref()
    }

    /// Get read-only access to covers slice.
    ///
    /// Cover is the sum of hessians for samples reaching each node.
    pub fn covers(&self) -> Option<&[f32]> {
        self.covers.as_deref()
    }

    // =========================================================================
    // Validation
    // =========================================================================

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

    // =========================================================================
    // Prediction Methods
    // =========================================================================

    /// Traverse the tree to find the leaf for given features.
    ///
    /// This is a convenience method for single-row prediction. For batch
    /// prediction, use [`traverse_to_leaf`] or [`predict_into`] instead.
    ///
    /// # Arguments
    ///
    /// * `features` - Feature values for a single sample
    ///
    /// # Returns
    ///
    /// Reference to the leaf value for this sample.
    pub fn predict_row(&self, features: &[f32]) -> &L {
        let leaf_id = self.traverse_to_leaf(&features);
        self.leaf_value(leaf_id)
    }

    /// Traverse the tree to find the leaf for a binned row.
    ///
    /// Used during training when we have binned data. Numeric bins are converted
    /// to float values using the bin mappers before comparison. Categorical bins
    /// are treated as canonical category indices (0..K-1).
    pub fn predict_binned_row(
        &self,
        row: &BinnedRowView<'_>,
        dataset: &BinnedDataset,
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



    /// Unified batch prediction for binned datasets.
    ///
    /// Traverses the tree for each row and **adds** leaf values to the predictions
    /// buffer (accumulate pattern). Supports both sequential and parallel execution.
    ///
    /// # Arguments
    /// * `dataset` - The binned dataset containing the rows
    /// * `predictions` - Pre-allocated buffer to update (length = `dataset.n_rows()`)
    /// * `parallelism` - Whether to use parallel execution
    pub fn predict_binned_into(
        &self,
        dataset: &BinnedDataset,
        mut predictions: ArrayViewMut1<f32>,
        parallelism: Parallelism,
    ) where
        L: Into<f32> + Copy + Send + Sync,
    {
        let n_rows = dataset.n_rows();
        debug_assert_eq!(predictions.len(), n_rows);

        parallelism.maybe_par_for_each(
            predictions.iter_mut().enumerate().collect::<Vec<_>>(),
            |(row_idx, pred)| {
                if let Some(row) = dataset.row_view(row_idx) {
                    let leaf = self.predict_binned_row(&row, dataset);
                    *pred += (*leaf).into();
                }
            },
        );
    }

    /// Unified batch prediction using any data accessor.
    ///
    /// Traverses the tree for each row and **adds** leaf values to the predictions
    /// buffer (accumulate pattern). Supports both sequential and parallel execution.
    ///
    /// # Arguments
    /// * `data` - Data source (SamplesView, FeaturesView, BinnedAccessor, etc.)
    /// * `predictions` - Pre-allocated buffer to update (length = `data.n_samples()`)
    /// * `parallelism` - Whether to use parallel execution
    ///
    /// # Example
    ///
    /// ```ignore
    /// use boosters::repr::gbdt::{Tree, ScalarLeaf};
    /// use boosters::dataset::SamplesView;
    /// use boosters::Parallelism;
    ///
    /// let tree: Tree<ScalarLeaf> = /* ... */;
    /// let data = SamplesView::from_slice(&[0.1, 0.2, 0.3, 0.4], 2, 2).unwrap();
    /// let mut predictions = vec![0.0; 2];
    /// tree.predict_into(&data, &mut predictions, Parallelism::Parallel);
    /// ```
    pub fn predict_into<D: DataAccessor + Sync>(
        &self,
        data: &D,
        predictions: &mut [f32],
        parallelism: crate::utils::Parallelism,
    ) where
        L: Into<f32> + Copy + Send + Sync,
    {
        let n_samples = data.n_samples();
        debug_assert_eq!(predictions.len(), n_samples);

        // Create prediction closures that extract samples within the closure
        // to avoid HRTB lifetime issues with `for<'a> D::Sample<'a>: Send`
        let predict_linear = |row_idx: usize| {
            let sample = data.sample(row_idx);
            let leaf_idx = self.traverse_to_leaf(&sample);
            self.compute_leaf_value(leaf_idx, &sample)
        };

        let predict_constant = |row_idx: usize| {
            let sample = data.sample(row_idx);
            let leaf_idx = self.traverse_to_leaf(&sample);
            (*self.leaf_value(leaf_idx)).into()
        };

        if self.has_linear_leaves() {
            parallelism.maybe_par_for_each(0..n_samples, |row_idx| {
                // Safety: each row_idx is unique, so we can safely write to predictions[row_idx]
                let pred = unsafe { &mut *predictions.as_ptr().add(row_idx).cast_mut() };
                *pred += predict_linear(row_idx);
            });
        } else {
            parallelism.maybe_par_for_each(0..n_samples, |row_idx| {
                let pred = unsafe { &mut *predictions.as_ptr().add(row_idx).cast_mut() };
                *pred += predict_constant(row_idx);
            });
        }
    }

    /// Compute the prediction value for a leaf, including linear terms if present.
    ///
    /// For constant leaves, returns the leaf value.
    /// For linear leaves, returns `intercept + Σ(coef × feature)`.
    /// If any linear feature is NaN, falls back to the base leaf value.
    #[inline]
    pub(crate) fn compute_leaf_value<S: SampleAccessor>(
        &self,
        leaf_idx: NodeId,
        sample: &S,
    ) -> f32
    where
        L: Into<f32> + Copy,
    {
        let base = (*self.leaf_value(leaf_idx)).into();

        if let Some((feat_indices, coefs)) = self.leaf_terms(leaf_idx) {
            // Check for NaN in any linear feature
            for &feat_idx in feat_indices {
                let val = sample.feature(feat_idx as usize);
                if val.is_nan() {
                    return base; // Fall back to constant leaf
                }
            }

            // Compute linear prediction: intercept + Σ(coef × feature)
            let intercept = self.leaf_intercept(leaf_idx);
            let linear_sum: f32 = feat_indices
                .iter()
                .zip(coefs.iter())
                .map(|(&f, &c)| c * sample.feature(f as usize))
                .sum();

            intercept + linear_sum
        } else {
            base
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{FeaturesView, SamplesView};
    use crate::repr::gbdt::mutable_tree::MutableTree;
    use crate::repr::gbdt::ScalarLeaf;
    use ndarray::{array, Array2};

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
        let tree = crate::scalar_tree! {
            0 => num(0, 0.5, L) -> 1, 2,
            1 => leaf(1.0),
            2 => leaf(2.0),
        };

        // 3 rows, 1 feature: [[0.3], [0.7], [0.5]]
        let arr = Array2::from_shape_vec((3, 1), vec![0.3, 0.7, 0.5]).unwrap();
        let data = SamplesView::from_array(arr.view());
        
        // Test accumulate pattern (starts with existing values)
        let mut predictions = vec![10.0, 20.0, 30.0];
        tree.predict_into(&data, &mut predictions, crate::utils::Parallelism::Sequential);

        // Row 0: 0.3 < 0.5 -> left (1.0), 10.0 + 1.0 = 11.0
        // Row 1: 0.7 >= 0.5 -> right (2.0), 20.0 + 2.0 = 22.0
        // Row 2: 0.5 >= 0.5 -> right (2.0), 30.0 + 2.0 = 32.0
        assert_eq!(predictions, vec![11.0, 22.0, 32.0]);
    }

    #[test]
    fn test_linear_prediction_single() {
        let mut tree: MutableTree<ScalarLeaf> = MutableTree::new();

        // Create tree: root splits on feature 0 at 0.5
        // Left leaf: linear with intercept=0.5, coef[0]=2.0
        // Right leaf: constant 10.0
        let root = tree.init_root();
        let (left, right) = tree.apply_numeric_split(root, 0, 0.5, false);
        tree.make_leaf(left, ScalarLeaf(1.0)); // Base value (not used for linear)
        tree.make_leaf(right, ScalarLeaf(10.0));
        tree.set_linear_leaf(left, vec![0], 0.5, vec![2.0]);

        let frozen = tree.freeze();

        // Test data: 3 rows, 1 feature: [[0.3], [0.7], [0.1]]
        // Row 0: x=0.3 -> left leaf -> 0.5 + 2.0*0.3 = 1.1
        // Row 1: x=0.7 -> right leaf -> 10.0
        // Row 2: x=0.1 -> left leaf -> 0.5 + 2.0*0.1 = 0.7
        let arr = Array2::from_shape_vec((3, 1), vec![0.3, 0.7, 0.1]).unwrap();
        let data = SamplesView::from_array(arr.view());
        let mut predictions = vec![0.0; 3];
        frozen.predict_into(&data, &mut predictions, crate::utils::Parallelism::Sequential);

        assert!((predictions[0] - 1.1).abs() < 1e-5, "got {}", predictions[0]);
        assert!((predictions[1] - 10.0).abs() < 1e-5, "got {}", predictions[1]);
        assert!((predictions[2] - 0.7).abs() < 1e-5, "got {}", predictions[2]);
    }

    #[test]
    fn test_linear_prediction_nan_fallback() {
        let mut tree: MutableTree<ScalarLeaf> = MutableTree::new();

        // Create simple tree with linear leaf
        let root = tree.init_root();
        tree.make_leaf(root, ScalarLeaf(5.0)); // Base value
        tree.set_linear_leaf(root, vec![0], 1.0, vec![2.0]); // intercept=1.0, coef=2.0

        let frozen = tree.freeze();

        // Test data: 2 rows, 1 feature: [[0.5], [NaN]]
        // Row 0: x=0.5 -> 1.0 + 2.0*0.5 = 2.0
        // Row 1: x=NaN -> fall back to base=5.0
        let arr = Array2::from_shape_vec((2, 1), vec![0.5, f32::NAN]).unwrap();
        let data = SamplesView::from_array(arr.view());
        let mut predictions = vec![0.0; 2];
        frozen.predict_into(&data, &mut predictions, crate::utils::Parallelism::Sequential);

        assert!((predictions[0] - 2.0).abs() < 1e-5, "got {}", predictions[0]);
        assert!((predictions[1] - 5.0).abs() < 1e-5, "got {}", predictions[1]);
    }

    #[test]
    fn test_linear_prediction_multivariate() {
        let mut tree: MutableTree<ScalarLeaf> = MutableTree::new();

        // Single leaf with linear model: y = 1.0 + 2.0*x0 + 3.0*x1
        let root = tree.init_root();
        tree.make_leaf(root, ScalarLeaf(0.0));
        tree.set_linear_leaf(root, vec![0, 1], 1.0, vec![2.0, 3.0]);

        let frozen = tree.freeze();

        // Test data: 2 rows, 2 features: [[1.0, 1.0], [0.5, 2.0]]
        // Row 0: [1.0, 1.0] -> 1.0 + 2.0*1.0 + 3.0*1.0 = 6.0
        // Row 1: [0.5, 2.0] -> 1.0 + 2.0*0.5 + 3.0*2.0 = 8.0
        let arr = Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 0.5, 2.0]).unwrap();
        let data = SamplesView::from_array(arr.view());
        let mut predictions = vec![0.0; 2];
        frozen.predict_into(&data, &mut predictions, crate::utils::Parallelism::Sequential);

        assert!((predictions[0] - 6.0).abs() < 1e-5, "got {}", predictions[0]);
        assert!((predictions[1] - 8.0).abs() < 1e-5, "got {}", predictions[1]);
    }

    #[test]
    fn test_gains_and_covers() {
        // Test tree with gains/covers for explainability
        let tree = crate::scalar_tree! {
            0 => num(0, 0.5, L) -> 1, 2,
            1 => leaf(1.0),
            2 => leaf(2.0),
        };

        // Initially no gains/covers
        assert!(!tree.has_gains());
        assert!(!tree.has_covers());
        assert!(tree.gains().is_none());
        assert!(tree.covers().is_none());

        // Add gains and covers
        let gains = vec![10.0, 0.0, 0.0]; // Only root has gain
        let covers = vec![100.0, 40.0, 60.0]; // All nodes have cover
        let tree = tree.with_stats(gains, covers);

        assert!(tree.has_gains());
        assert!(tree.has_covers());
        assert_eq!(tree.gains().unwrap(), &[10.0, 0.0, 0.0]);
        assert_eq!(tree.covers().unwrap(), &[100.0, 40.0, 60.0]);
    }

    #[test]
    fn test_traverse_with_different_accessors() {
        use crate::data::DataAccessor;

        let tree = crate::scalar_tree! {
            0 => num(0, 0.5, L) -> 1, 2,
            1 => leaf(-1.0),
            2 => leaf(1.0),
        };

        // Sample-major data: 3 samples, 2 features
        let row_data = array![
            [0.3f32, 0.0],  // sample 0
            [0.7, 0.0],     // sample 1
            [f32::NAN, 0.0] // sample 2
        ];
        let samples_view = SamplesView::from_array(row_data.view());

        // Feature-major data: same logical values but in feature-major layout
        let col_data = array![
            [0.3f32, 0.7, f32::NAN],  // feature 0
            [0.0, 0.0, 0.0],          // feature 1
        ];
        let features_view = FeaturesView::from_array(col_data.view());

        // Test SamplesView accessor
        let leaf_row_0 = tree.traverse_to_leaf(&samples_view.sample(0));
        let leaf_row_1 = tree.traverse_to_leaf(&samples_view.sample(1));
        let leaf_row_2 = tree.traverse_to_leaf(&samples_view.sample(2));

        assert_eq!(leaf_row_0, 1, "0.3 < 0.5 should go left");
        assert_eq!(leaf_row_1, 2, "0.7 >= 0.5 should go right");
        assert_eq!(leaf_row_2, 1, "NaN with default_left=true should go left");

        // Test FeaturesView accessor - should reach same leaves
        let leaf_col_0 = tree.traverse_to_leaf(&features_view.sample(0));
        let leaf_col_1 = tree.traverse_to_leaf(&features_view.sample(1));
        let leaf_col_2 = tree.traverse_to_leaf(&features_view.sample(2));

        assert_eq!(leaf_col_0, leaf_row_0, "FeaturesView should match SamplesView");
        assert_eq!(leaf_col_1, leaf_row_1, "FeaturesView should match SamplesView");
        assert_eq!(leaf_col_2, leaf_row_2, "FeaturesView should match SamplesView");
    }

    #[test]
    fn test_traverse_on_mutable_tree() {
        use crate::data::DataAccessor;

        let mut tree = MutableTree::<ScalarLeaf>::new();
        let _root = tree.init_root();

        // Apply a split at root: feature 0, threshold 0.5
        let (left, right) = tree.apply_numeric_split(0, 0, 0.5, true);
        tree.make_leaf(left, ScalarLeaf(-1.0));
        tree.make_leaf(right, ScalarLeaf(1.0));

        // Verify TreeView works on MutableTree
        assert!(!TreeView::is_leaf(&tree, 0));
        assert!(TreeView::is_leaf(&tree, left));
        assert!(TreeView::is_leaf(&tree, right));
        assert_eq!(TreeView::split_index(&tree, 0), 0);
        assert_eq!(TreeView::split_threshold(&tree, 0), 0.5);

        // Test traversal with MutableTree
        // 2 samples, 1 feature each: [[0.3], [0.7]]
        let arr = Array2::from_shape_vec((2, 1), vec![0.3, 0.7]).unwrap();
        let row_data = SamplesView::from_array(arr.view());
        let leaf_0 = tree.traverse_to_leaf(&row_data.sample(0));
        let leaf_1 = tree.traverse_to_leaf(&row_data.sample(1));

        assert_eq!(leaf_0, left);
        assert_eq!(leaf_1, right);
    }
}


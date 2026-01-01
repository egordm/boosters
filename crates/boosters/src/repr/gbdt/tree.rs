//! Canonical tree representation (SoA) for efficient traversal.
//!
//! This module provides:
//! - [`Tree`]: Immutable SoA tree storage for efficient traversal
//!
//! For the read-only tree interface, see [`super::tree_view::TreeView`].
//! For mutable tree construction during training, see [`super::mutable_tree::MutableTree`].

// Allow many constructor arguments for creating trees with all their fields.
#![allow(clippy::too_many_arguments)]

use ndarray::{ArrayView1, ArrayViewMut1, Axis};

use crate::data::Dataset;

use super::NodeId;
use super::categories::CategoriesStorage;
use super::coefficients::LeafCoefficients;
use super::tree_view::{TreeValidationError, TreeView, validate_tree};
use super::types::LeafValue;
use super::types::SplitType;

// ============================================================================
// Tree Structure
// ============================================================================

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
    /// This delegates to [`validate_tree`] for generic validation, then
    /// performs Tree-specific checks like categorical segment validation.
    pub fn validate(&self) -> Result<(), TreeValidationError> {
        // Run generic TreeView validation first
        validate_tree(self)?;

        // Tree-specific: If categorical splits exist, segments must be indexed by node.
        let has_cat_split = self
            .split_types
            .iter()
            .any(|t| matches!(t, SplitType::Categorical));
        if has_cat_split {
            let segments_len = self.categories.segments().len();
            let n_nodes = self.n_nodes();
            if segments_len != n_nodes {
                return Err(TreeValidationError::CategoricalSegmentsLenMismatch {
                    segments_len,
                    n_nodes,
                });
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
        let leaf_id = self.traverse_to_leaf(ArrayView1::from(features));
        self.leaf_value(leaf_id)
    }

    /// Block-buffered batch prediction over [`Dataset`].
    ///
    /// This is crate-private because the public prediction API is provided by
    /// the inference predictor; training uses this for incremental prediction
    /// updates without going through `BinnedDataset`.
    pub(crate) fn predict_into(
        &self,
        dataset: &Dataset,
        mut predictions: ArrayViewMut1<'_, f32>,
        parallelism: crate::utils::Parallelism,
    ) where
        L: Into<f32> + Copy + Send + Sync,
    {
        debug_assert_eq!(predictions.len(), dataset.n_samples());

        // Match inference defaults (XGBoost) without adding a dependency edge.
        const BLOCK_SIZE: usize = 64;

        let n_features = dataset.n_features();

        parallelism.maybe_par_bridge_for_each_init(
            predictions
                .axis_chunks_iter_mut(Axis(0), BLOCK_SIZE)
                .enumerate(),
            || ndarray::Array2::<f32>::zeros((BLOCK_SIZE, n_features)),
            |buffer, (block_idx, mut pred_chunk)| {
                let start_sample = block_idx * BLOCK_SIZE;
                let samples = dataset.buffer_samples(buffer, start_sample);
                debug_assert_eq!(samples.n_samples(), pred_chunk.len());

                if self.has_linear_leaves() {
                    for (row_idx, pred) in pred_chunk.iter_mut().enumerate() {
                        let row = samples.sample_view(row_idx);
                        let leaf_idx = self.traverse_to_leaf(row);
                        *pred += self.compute_leaf_value(leaf_idx, row);
                    }
                } else {
                    for (row_idx, pred) in pred_chunk.iter_mut().enumerate() {
                        let row = samples.sample_view(row_idx);
                        let leaf_idx = self.traverse_to_leaf(row);
                        *pred += (*self.leaf_value(leaf_idx)).into();
                    }
                }
            },
        );
    }

    /// Compute the prediction value for a leaf, including linear terms if present.
    ///
    /// For constant leaves, returns the leaf value.
    /// For linear leaves, returns `intercept + Σ(coef × feature)`.
    /// If any linear feature is NaN, falls back to the base leaf value.
    #[inline]
    pub(crate) fn compute_leaf_value(&self, leaf_idx: NodeId, sample: ArrayView1<'_, f32>) -> f32
    where
        L: Into<f32> + Copy,
    {
        let base = (*self.leaf_value(leaf_idx)).into();

        if let Some((feat_indices, coefs)) = self.leaf_terms(leaf_idx) {
            // Check for NaN in any linear feature
            for &feat_idx in feat_indices {
                let val = sample[feat_idx as usize];
                if val.is_nan() {
                    return base; // Fall back to constant leaf
                }
            }

            // Compute linear prediction: intercept + Σ(coef × feature)
            let intercept = self.leaf_intercept(leaf_idx);
            let linear_sum: f32 = feat_indices
                .iter()
                .zip(coefs.iter())
                .map(|(&f, &c)| c * sample[f as usize])
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
    use crate::data::{Dataset, SamplesView};
    use crate::repr::gbdt::ScalarLeaf;
    use crate::repr::gbdt::mutable_tree::MutableTree;
    use ndarray::{Array2, array};

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

        // Feature-major [n_features, n_samples]
        let features = array![[0.3f32, 0.7, 0.5]];
        let dataset = Dataset::from_array(features.view(), None, None);

        // Test accumulate pattern (starts with existing values)
        let mut predictions = ndarray::Array1::from_vec(vec![10.0, 20.0, 30.0]);
        tree.predict_into(
            &dataset,
            predictions.view_mut(),
            crate::utils::Parallelism::Sequential,
        );

        // Row 0: 0.3 < 0.5 -> left (1.0), 10.0 + 1.0 = 11.0
        // Row 1: 0.7 >= 0.5 -> right (2.0), 20.0 + 2.0 = 22.0
        // Row 2: 0.5 >= 0.5 -> right (2.0), 30.0 + 2.0 = 32.0
        assert_eq!(predictions.to_vec(), vec![11.0, 22.0, 32.0]);
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

        // Test data (feature-major): 1 feature, 3 samples
        // Row 0: x=0.3 -> left leaf -> 0.5 + 2.0*0.3 = 1.1
        // Row 1: x=0.7 -> right leaf -> 10.0
        // Row 2: x=0.1 -> left leaf -> 0.5 + 2.0*0.1 = 0.7
        let features = array![[0.3f32, 0.7, 0.1]];
        let dataset = Dataset::from_array(features.view(), None, None);
        let mut predictions = ndarray::Array1::zeros(3);
        frozen.predict_into(
            &dataset,
            predictions.view_mut(),
            crate::utils::Parallelism::Sequential,
        );

        assert!(
            (predictions[0] - 1.1).abs() < 1e-5,
            "got {}",
            predictions[0]
        );
        assert!(
            (predictions[1] - 10.0).abs() < 1e-5,
            "got {}",
            predictions[1]
        );
        assert!(
            (predictions[2] - 0.7).abs() < 1e-5,
            "got {}",
            predictions[2]
        );
    }

    #[test]
    fn test_linear_prediction_nan_fallback() {
        let mut tree: MutableTree<ScalarLeaf> = MutableTree::new();

        // Create simple tree with linear leaf
        let root = tree.init_root();
        tree.make_leaf(root, ScalarLeaf(5.0)); // Base value
        tree.set_linear_leaf(root, vec![0], 1.0, vec![2.0]); // intercept=1.0, coef=2.0

        let frozen = tree.freeze();

        // Test data (feature-major): 1 feature, 2 samples
        // Row 0: x=0.5 -> 1.0 + 2.0*0.5 = 2.0
        // Row 1: x=NaN -> fall back to base=5.0
        let features = array![[0.5f32, f32::NAN]];
        let dataset = Dataset::from_array(features.view(), None, None);
        let mut predictions = ndarray::Array1::zeros(2);
        frozen.predict_into(
            &dataset,
            predictions.view_mut(),
            crate::utils::Parallelism::Sequential,
        );

        assert!(
            (predictions[0] - 2.0).abs() < 1e-5,
            "got {}",
            predictions[0]
        );
        assert!(
            (predictions[1] - 5.0).abs() < 1e-5,
            "got {}",
            predictions[1]
        );
    }

    #[test]
    fn test_linear_prediction_multivariate() {
        let mut tree: MutableTree<ScalarLeaf> = MutableTree::new();

        // Single leaf with linear model: y = 1.0 + 2.0*x0 + 3.0*x1
        let root = tree.init_root();
        tree.make_leaf(root, ScalarLeaf(0.0));
        tree.set_linear_leaf(root, vec![0, 1], 1.0, vec![2.0, 3.0]);

        let frozen = tree.freeze();

        // Test data (feature-major): 2 features, 2 samples
        // Row 0: [1.0, 1.0] -> 1.0 + 2.0*1.0 + 3.0*1.0 = 6.0
        // Row 1: [0.5, 2.0] -> 1.0 + 2.0*0.5 + 3.0*2.0 = 8.0
        let features = array![[1.0f32, 0.5], [1.0, 2.0]];
        let dataset = Dataset::from_array(features.view(), None, None);
        let mut predictions = ndarray::Array1::zeros(2);
        frozen.predict_into(
            &dataset,
            predictions.view_mut(),
            crate::utils::Parallelism::Sequential,
        );

        assert!(
            (predictions[0] - 6.0).abs() < 1e-5,
            "got {}",
            predictions[0]
        );
        assert!(
            (predictions[1] - 8.0).abs() < 1e-5,
            "got {}",
            predictions[1]
        );
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
        let tree = crate::scalar_tree! {
            0 => num(0, 0.5, L) -> 1, 2,
            1 => leaf(-1.0),
            2 => leaf(1.0),
        };

        // Sample-major data: 3 samples, 2 features
        let row_data = array![
            [0.3f32, 0.0],   // sample 0
            [0.7, 0.0],      // sample 1
            [f32::NAN, 0.0]  // sample 2
        ];
        let samples_view = SamplesView::from_array(row_data.view());

        // Test SamplesView: sample-major rows are contiguous
        let row0 = samples_view.sample_view(0);
        let row1 = samples_view.sample_view(1);
        let row2 = samples_view.sample_view(2);
        let leaf_row_0 = tree.traverse_to_leaf(row0);
        let leaf_row_1 = tree.traverse_to_leaf(row1);
        let leaf_row_2 = tree.traverse_to_leaf(row2);

        assert_eq!(leaf_row_0, 1, "0.3 < 0.5 should go left");
        assert_eq!(leaf_row_1, 2, "0.7 >= 0.5 should go right");
        assert_eq!(leaf_row_2, 1, "NaN with default_left=true should go left");
    }

    #[test]
    fn test_traverse_on_mutable_tree() {
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
        let row0 = row_data.sample_view(0);
        let row1 = row_data.sample_view(1);
        let leaf_0 = tree.traverse_to_leaf(row0);
        let leaf_1 = tree.traverse_to_leaf(row1);

        assert_eq!(leaf_0, left);
        assert_eq!(leaf_1, right);
    }
}

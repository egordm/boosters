//! Binned data accessor and generic tree traversal.
//!
//! This module provides:
//! - [`BinnedAccessor`]: Converts binned data to midpoint values for traversal
//! - [`SingleRowSlice`]: Wraps a `&[f32]` slice for single-row traversal
//! - [`traverse_to_leaf`]: Generic traversal function for any tree/accessor combination
//! - [`traverse_to_leaf_from`]: Same as above, but starting from a given node
//!
//! # Design
//!
//! The core traits [`TreeView`](crate::repr::gbdt::TreeView) and
//! [`FeatureAccessor`](crate::data::FeatureAccessor) are defined in their
//! respective modules (`repr::gbdt` and `data`). This module provides
//! the inference-specific utilities that use those traits.

use crate::data::binned::BinMapper;
use crate::data::{BinnedDataset, FeatureAccessor};
use crate::repr::gbdt::{float_to_category, NodeId, SplitType, TreeView};

// ============================================================================
// SingleRowSlice Accessor
// ============================================================================

/// Feature accessor wrapping a single row of features as a slice.
///
/// This adapter allows using `&[f32]` with the generic [`traverse_to_leaf`]
/// function. The slice represents features for exactly one row, so `row` 
/// parameter must be 0.
///
/// # Example
///
/// ```ignore
/// use boosters::inference::gbdt::{SingleRowSlice, traverse_to_leaf};
///
/// let features = [0.5f32, 1.2, -0.3];
/// let accessor = SingleRowSlice(&features);
/// let leaf = traverse_to_leaf(&tree, &accessor, 0);
/// ```
pub struct SingleRowSlice<'a>(pub &'a [f32]);

impl FeatureAccessor for SingleRowSlice<'_> {
    #[inline]
    fn get_feature(&self, row: usize, feature: usize) -> f32 {
        debug_assert_eq!(row, 0, "SingleRowSlice only supports row 0");
        self.0.get(feature).copied().unwrap_or(f32::NAN)
    }

    #[inline]
    fn num_rows(&self) -> usize {
        1
    }

    #[inline]
    fn num_features(&self) -> usize {
        self.0.len()
    }
}

// ============================================================================
// BinnedAccessor
// ============================================================================

/// Feature accessor that converts binned data to midpoint values.
///
/// For each bin, returns the midpoint between lower and upper bounds:
/// - Bin 0: `(min_val + upper_bound[0]) / 2`
/// - Bin n: `(upper_bound[n-1] + upper_bound[n]) / 2`
///
/// For categorical features, returns the category value as f32.
/// For missing bins, returns `f32::NAN`.
///
/// # Panics
///
/// The constructor panics if `bin_mappers.len() != dataset.n_features()`.
pub struct BinnedAccessor<'a> {
    dataset: &'a BinnedDataset,
    bin_mappers: &'a [BinMapper],
}

impl<'a> BinnedAccessor<'a> {
    /// Create a new binned accessor.
    ///
    /// # Panics
    ///
    /// Panics if the number of bin mappers doesn't match the dataset's feature count.
    pub fn new(dataset: &'a BinnedDataset, bin_mappers: &'a [BinMapper]) -> Self {
        assert_eq!(
            dataset.n_features(),
            bin_mappers.len(),
            "BinMapper count ({}) must match feature count ({})",
            bin_mappers.len(),
            dataset.n_features()
        );
        Self {
            dataset,
            bin_mappers,
        }
    }
}

impl FeatureAccessor for BinnedAccessor<'_> {
    #[inline]
    fn get_feature(&self, row: usize, feature: usize) -> f32 {
        let bin = self.dataset.get_bin(row, feature);
        match bin {
            Some(b) => self.bin_mappers[feature].bin_to_midpoint(b) as f32,
            None => f32::NAN,
        }
    }

    #[inline]
    fn num_rows(&self) -> usize {
        self.dataset.n_rows()
    }

    #[inline]
    fn num_features(&self) -> usize {
        self.dataset.n_features()
    }
}

// ============================================================================
// Generic Traversal Function
// ============================================================================

/// Traverse a tree to find the leaf node for a given row.
///
/// This is the unified traversal function that works with any `TreeView`
/// and `FeatureAccessor` combination. Use [`traverse_to_leaf_from`] if you
/// need to start from a specific node.
///
/// # Arguments
///
/// * `tree` - Tree to traverse (implements [`TreeView`])
/// * `accessor` - Feature value source (implements [`FeatureAccessor`])
/// * `row` - Row index in the accessor
///
/// # Returns
///
/// The `NodeId` of the reached leaf node.
///
/// # Example
///
/// ```ignore
/// use boosters::inference::gbdt::traverse_to_leaf;
/// use boosters::data::RowMatrix;
///
/// let features = RowMatrix::from_vec(vec![0.5, 1.0], 1, 2);
/// let leaf_id = traverse_to_leaf(&tree, &features, 0);
/// ```
#[inline]
pub fn traverse_to_leaf<T: TreeView, A: FeatureAccessor>(
    tree: &T,
    accessor: &A,
    row: usize,
) -> NodeId {
    traverse_to_leaf_from(tree, 0, accessor, row)
}

/// Traverse a tree to find the leaf node, starting from a specific node.
///
/// This is the core traversal function used by both [`traverse_to_leaf`] and
/// [`traverse_from_node`]. It supports starting from any node, useful for
/// resuming traversal after unrolled levels.
///
/// # Arguments
///
/// * `tree` - Tree to traverse (implements [`TreeView`])
/// * `start_node` - Node ID to start traversal from
/// * `accessor` - Feature value source (implements [`FeatureAccessor`])
/// * `row` - Row index in the accessor
///
/// # Returns
///
/// The `NodeId` of the reached leaf node.
#[inline]
pub fn traverse_to_leaf_from<T: TreeView, A: FeatureAccessor>(
    tree: &T,
    start_node: NodeId,
    accessor: &A,
    row: usize,
) -> NodeId {
    let mut node = start_node;

    while !tree.is_leaf(node) {
        let feat_idx = tree.split_index(node) as usize;
        let fvalue = accessor.get_feature(row, feat_idx);

        node = if fvalue.is_nan() {
            // Missing value: use default direction
            if tree.default_left(node) {
                tree.left_child(node)
            } else {
                tree.right_child(node)
            }
        } else {
            match tree.split_type(node) {
                SplitType::Numeric => {
                    if fvalue < tree.split_threshold(node) {
                        tree.left_child(node)
                    } else {
                        tree.right_child(node)
                    }
                }
                SplitType::Categorical => {
                    let category = float_to_category(fvalue);
                    if tree.categories().category_goes_right(node, category) {
                        tree.right_child(node)
                    } else {
                        tree.left_child(node)
                    }
                }
            }
        };
    }

    node
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{FeaturesView, SamplesView};
    use crate::repr::gbdt::{ScalarLeaf, Tree};

    /// Build a simple 3-node tree:
    ///     [0] feature=0, threshold=0.5
    ///    /   \
    ///  [1]   [2]
    /// leaf  leaf
    fn build_simple_tree() -> Tree<ScalarLeaf> {
        Tree::new(
            vec![0, 0, 0],             // split_indices
            vec![0.5, 0.0, 0.0],       // split_thresholds
            vec![1, 0, 0],             // left_children
            vec![2, 0, 0],             // right_children
            vec![true, false, false],  // default_left
            vec![false, true, true],   // is_leaf
            vec![ScalarLeaf(0.0), ScalarLeaf(-1.0), ScalarLeaf(1.0)], // leaf_values
        )
    }

    #[test]
    fn test_feature_accessors_all_types() {
        let tree = build_simple_tree();

        // Sample-major data: 3 rows, 2 features (row-major order)
        let row_data = [0.3f32, 0.0, 0.7, 0.0, f32::NAN, 0.0];
        let samples_view = SamplesView::from_slice(&row_data, 3, 2).unwrap();

        // Feature-major data: same logical values but in feature-major order
        // [f0_s0, f0_s1, f0_s2, f1_s0, f1_s1, f1_s2] = [0.3, 0.7, NaN, 0.0, 0.0, 0.0]
        let col_data = [0.3f32, 0.7, f32::NAN, 0.0, 0.0, 0.0];
        let features_view = FeaturesView::from_slice(&col_data, 3, 2).unwrap();

        // Test SamplesView accessor
        let leaf_row_0 = traverse_to_leaf(&tree, &samples_view, 0);
        let leaf_row_1 = traverse_to_leaf(&tree, &samples_view, 1);
        let leaf_row_2 = traverse_to_leaf(&tree, &samples_view, 2);

        assert_eq!(leaf_row_0, 1, "0.3 < 0.5 should go left");
        assert_eq!(leaf_row_1, 2, "0.7 >= 0.5 should go right");
        assert_eq!(leaf_row_2, 1, "NaN with default_left=true should go left");

        // Test FeaturesView accessor - should reach same leaves
        let leaf_col_0 = traverse_to_leaf(&tree, &features_view, 0);
        let leaf_col_1 = traverse_to_leaf(&tree, &features_view, 1);
        let leaf_col_2 = traverse_to_leaf(&tree, &features_view, 2);

        assert_eq!(leaf_col_0, leaf_row_0, "FeaturesView should match SamplesView");
        assert_eq!(leaf_col_1, leaf_row_1, "FeaturesView should match SamplesView");
        assert_eq!(leaf_col_2, leaf_row_2, "FeaturesView should match SamplesView");
    }

    #[test]
    fn test_binned_accessor_midpoint() {
        use crate::data::binned::{BinMapper, MissingType};

        // Create bin mapper: bins [0, 0.25), [0.25, 0.5), [0.5, 0.75), [0.75, 1.0]
        // Upper bounds: [0.25, 0.5, 0.75, 1.0], min=0.0
        let bin_mapper = BinMapper::numerical(
            vec![0.25, 0.5, 0.75, 1.0],
            MissingType::None,
            0, // default_bin
            0, // most_freq_bin
            0.0, // sparse_rate
            0.0, // min_val
            1.0, // max_val
        );

        // Verify midpoint calculation directly on BinMapper
        // Bin 0: (0.0 + 0.25) / 2 = 0.125
        // Bin 1: (0.25 + 0.5) / 2 = 0.375
        // Bin 2: (0.5 + 0.75) / 2 = 0.625
        // Bin 3: (0.75 + 1.0) / 2 = 0.875
        let eps = 1e-6;
        assert!((bin_mapper.bin_to_midpoint(0) - 0.125).abs() < eps);
        assert!((bin_mapper.bin_to_midpoint(1) - 0.375).abs() < eps);
        assert!((bin_mapper.bin_to_midpoint(2) - 0.625).abs() < eps);
        assert!((bin_mapper.bin_to_midpoint(3) - 0.875).abs() < eps);
    }

    #[test]
    fn test_treeview_mutable_tree() {
        use crate::repr::gbdt::MutableTree;
        use ndarray::Array2;
        use crate::data::SamplesView;

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
        let leaf_0 = traverse_to_leaf(&tree, &row_data, 0);
        let leaf_1 = traverse_to_leaf(&tree, &row_data, 1);

        assert_eq!(leaf_0, left);
        assert_eq!(leaf_1, right);
    }
}

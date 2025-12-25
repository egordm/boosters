//! BinnedDataset - the main dataset structure.

use super::bundling::{BundlePlan, FeatureLocation};
use super::group::{FeatureGroup, FeatureMeta};
use super::storage::{BinStorage, FeatureView, GroupLayout};

/// The main binned dataset for GBDT training.
///
/// Contains quantized feature values organized in feature groups.
/// Labels and weights are NOT included - pass them separately to training.
///
/// # Example
///
/// ```ignore
/// let dataset = BinnedDataset::builder()
///     .add_numeric(&values, 256, None)
///     .group_strategy(GroupStrategy::Auto)
///     .build()?;
///
/// // Access groups for histogram building
/// for group in dataset.iter_groups() {
///     match group.storage() {
///         BinStorage::DenseU8(data) => { ... }
///         BinStorage::DenseU16(data) => { ... }
///     }
/// }
/// ```
#[derive(Clone, Debug)]
pub struct BinnedDataset {
    /// Number of rows (samples).
    n_rows: usize,
    /// Feature metadata indexed by global feature ID.
    features: Box<[FeatureMeta]>,
    /// Feature groups.
    groups: Vec<FeatureGroup>,
    /// Cumulative bin offsets across ALL features (for histogram indexing).
    /// `global_bin_offsets[i]` = sum of n_bins for features 0..i
    /// Length is n_features + 1 (last element is total_bins).
    global_bin_offsets: Box<[u32]>,
    /// Optional bundle plan if bundling was applied.
    bundle_plan: Option<BundlePlan>,
}

impl BinnedDataset {
    /// Create a new binned dataset with an optional bundle plan.
    ///
    /// Typically constructed via `BinnedDatasetBuilder`.
    pub(crate) fn with_bundle_plan(
        n_rows: usize,
        features: Vec<FeatureMeta>,
        groups: Vec<FeatureGroup>,
        bundle_plan: Option<BundlePlan>,
    ) -> Self {
        // Compute global bin offsets
        let mut global_bin_offsets = Vec::with_capacity(features.len() + 1);
        let mut total = 0u32;
        for meta in &features {
            global_bin_offsets.push(total);
            total += meta.n_bins();
        }
        global_bin_offsets.push(total);

        Self {
            n_rows,
            features: features.into_boxed_slice(),
            groups,
            global_bin_offsets: global_bin_offsets.into_boxed_slice(),
            bundle_plan,
        }
    }

    /// Create an empty dataset.
    pub fn empty() -> Self {
        Self {
            n_rows: 0,
            features: Box::new([]),
            groups: Vec::new(),
            global_bin_offsets: vec![0].into_boxed_slice(),
            bundle_plan: None,
        }
    }

    /// Number of rows (samples).
    #[inline]
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Number of features.
    #[inline]
    pub fn n_features(&self) -> usize {
        self.features.len()
    }

    /// Number of feature groups.
    #[inline]
    pub fn n_groups(&self) -> usize {
        self.groups.len()
    }

    /// Total bins across all features (for histogram allocation).
    #[inline]
    pub fn total_bins(&self) -> u32 {
        *self.global_bin_offsets.last().unwrap_or(&0)
    }

    /// Get feature metadata by global feature index.
    #[inline]
    pub fn feature(&self, idx: usize) -> &FeatureMeta {
        &self.features[idx]
    }

    /// Iterate over feature metadata.
    #[inline]
    pub fn iter_features(&self) -> impl Iterator<Item = &FeatureMeta> {
        self.features.iter()
    }

    /// Get a feature group by index.
    #[inline]
    pub fn group(&self, idx: usize) -> &FeatureGroup {
        &self.groups[idx]
    }

    /// Iterate over feature groups.
    #[inline]
    pub fn iter_groups(&self) -> impl Iterator<Item = &FeatureGroup> {
        self.groups.iter()
    }

    /// Global bin offsets (length = n_features + 1).
    /// Use for indexing into flat histogram arrays.
    #[inline]
    pub fn global_bin_offsets(&self) -> &[u32] {
        &self.global_bin_offsets
    }

    /// Global bin offset for a specific feature.
    #[inline]
    pub fn global_bin_offset(&self, feature: usize) -> u32 {
        self.global_bin_offsets[feature]
    }

    /// Number of bins for a feature.
    #[inline]
    pub fn n_bins(&self, feature: usize) -> u32 {
        self.features[feature].n_bins()
    }

    /// Check if a feature is categorical.
    #[inline]
    pub fn is_categorical(&self, feature: usize) -> bool {
        self.features[feature].is_categorical()
    }

    /// Check if a feature has missing values.
    #[inline]
    pub fn has_missing(&self, feature: usize) -> bool {
        self.features[feature].bin_mapper.missing_type() != super::MissingType::None
    }

    /// Get the bin mapper for a feature.
    #[inline]
    pub fn bin_mapper(&self, feature: usize) -> &super::BinMapper {
        &self.features[feature].bin_mapper
    }

    /// Get all bin mappers as a vector.
    ///
    /// This is useful for creating a `BinnedAccessor`.
    pub fn bin_mappers(&self) -> Vec<super::BinMapper> {
        self.features.iter().map(|f| f.bin_mapper.clone()).collect()
    }

    /// Total memory size in bytes (data only, not metadata).
    pub fn data_size_bytes(&self) -> usize {
        self.groups.iter().map(|g| g.size_bytes()).sum()
    }

    /// Get the bundle plan if bundling was applied.
    pub fn bundle_plan(&self) -> Option<&BundlePlan> {
        self.bundle_plan.as_ref()
    }

    /// Check if bundling was applied and was effective.
    pub fn has_effective_bundling(&self) -> bool {
        self.bundle_plan.as_ref().is_some_and(|p| p.is_effective())
    }

    /// Get bundling statistics if bundling was applied.
    pub fn bundling_stats(&self) -> Option<BundlingStats> {
        let plan = self.bundle_plan.as_ref()?;
        Some(BundlingStats {
            original_sparse_features: plan.sparse_feature_count,
            bundles_created: plan.bundles.len(),
            standalone_features: plan.standalone_features.len(),
            skipped_features: plan.skipped_features.len(),
            total_conflicts: plan.total_conflicts,
            is_effective: plan.is_effective(),
            reduction_ratio: plan.reduction_ratio(),
            binned_columns: plan.binned_column_count(),
        })
    }

    /// Decode an encoded bin from a bundled column to (original_feature, original_bin).
    ///
    /// This is used when a split is found on a bundled column and we need to
    /// determine which original feature the split actually applies to.
    ///
    /// # Arguments
    /// * `bundle_idx` - Index of the bundle (0..n_bundles)
    /// * `encoded_bin` - The encoded bin value in the bundled column
    ///
    /// # Returns
    /// * `Some((original_feature_idx, original_bin))` if bundling is active and decode succeeds
    /// * `None` if no bundling or invalid indices
    ///
    /// # Example
    ///
    /// ```ignore
    /// // If a split is found on bundled column 2 at bin 7:
    /// if let Some((orig_feat, orig_bin)) = dataset.decode_bundle_split(2, 7) {
    ///     println!("Split is on original feature {} at bin {}", orig_feat, orig_bin);
    /// }
    /// ```
    pub fn decode_bundle_split(&self, bundle_idx: usize, encoded_bin: u32) -> Option<(usize, u32)> {
        self.bundle_plan.as_ref()?.decode_bundle_split(bundle_idx, encoded_bin)
    }

    /// Get the location of an original feature after bundling.
    ///
    /// # Arguments
    /// * `feature_idx` - Original feature index (0..n_features)
    ///
    /// # Returns
    /// * `Some(FeatureLocation)` if bundling is active
    /// * `None` if no bundling was applied
    pub fn original_to_location(&self, feature_idx: usize) -> Option<&FeatureLocation> {
        self.bundle_plan.as_ref()?.original_to_location(feature_idx)
    }

    /// Get all original feature indices that belong to a bundle.
    ///
    /// # Arguments
    /// * `bundle_idx` - Index of the bundle
    ///
    /// # Returns
    /// * `Some(&[usize])` - Slice of original feature indices
    /// * `None` if no bundling or invalid bundle index
    pub fn bundle_features(&self, bundle_idx: usize) -> Option<&[usize]> {
        self.bundle_plan.as_ref()?.bundle_features(bundle_idx)
    }
}

/// Statistics about bundling effectiveness.
#[derive(Clone, Debug)]
pub struct BundlingStats {
    /// Number of sparse features that were considered for bundling.
    pub original_sparse_features: usize,
    /// Number of bundles created.
    pub bundles_created: usize,
    /// Number of standalone (non-bundled) features.
    pub standalone_features: usize,
    /// Number of skipped (trivial) features.
    pub skipped_features: usize,
    /// Total conflicts in the bundle plan.
    pub total_conflicts: usize,
    /// Whether bundling was effective (reduced column count).
    pub is_effective: bool,
    /// Reduction ratio: bundles / original_sparse_features.
    pub reduction_ratio: f32,
    /// Total binned columns after bundling.
    pub binned_columns: usize,
}

impl BinnedDataset {
    /// Check if all groups use the same layout.
    pub fn is_uniform_layout(&self) -> Option<GroupLayout> {
        let first = self.groups.first()?.layout();
        if self.groups.iter().all(|g| g.layout() == first) {
            Some(first)
        } else {
            None
        }
    }

    /// Get groups by layout type (dense groups only).
    pub fn groups_by_layout(&self, layout: GroupLayout) -> impl Iterator<Item = (usize, &FeatureGroup)> {
        self.groups
            .iter()
            .enumerate()
            .filter(move |(_, g)| g.is_dense() && g.layout() == layout)
    }

    /// Get row-major groups (dense only).
    pub fn row_major_groups(&self) -> impl Iterator<Item = (usize, &FeatureGroup)> {
        self.groups_by_layout(GroupLayout::RowMajor)
    }

    /// Get column-major groups (dense only).
    pub fn column_major_groups(&self) -> impl Iterator<Item = (usize, &FeatureGroup)> {
        self.groups_by_layout(GroupLayout::ColumnMajor)
    }

    /// Get sparse groups.
    pub fn sparse_groups(&self) -> impl Iterator<Item = (usize, &FeatureGroup)> {
        self.groups
            .iter()
            .enumerate()
            .filter(|(_, g)| g.is_sparse())
    }

    /// Find the group and in-group index for a global feature index.
    ///
    /// Returns (group_ref, feature_index_in_group).
    #[inline]
    pub fn feature_location(&self, feature: usize) -> (&FeatureGroup, usize) {
        let meta = &self.features[feature];
        let group = &self.groups[meta.group_index as usize];
        (group, meta.index_in_group as usize)
    }

    /// Get a zero-cost slice view for a single feature's bins.
    ///
    /// Returns a `FeatureView` that provides direct typed access to the bins.
    /// For column-major groups, the slice is contiguous (stride=1).
    /// For row-major groups, the slice spans all features with stride > 1.
    /// For sparse groups, the slice contains only non-zero entries.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let slice = dataset.feature_view(0);
    /// match slice {
    ///     FeatureView::U8 { bins, stride } => {
    ///         if stride == 1 {
    ///             // Column-major: bins[row] is the bin for this row
    ///             for &bin in bins { ... }
    ///         } else {
    ///             // Row-major: bins[row * stride] is the bin
    ///             for row in 0..n_rows { let bin = bins[row * stride]; }
    ///         }
    ///     }
    ///     FeatureView::SparseU8 { row_indices, bin_values } => { ... }
    ///     ...
    /// }
    /// ```
    #[inline]
    pub fn feature_view(&self, feature: usize) -> FeatureView<'_> {
        let meta = &self.features[feature];
        let group = &self.groups[meta.group_index as usize];
        let idx_in_group = meta.index_in_group as usize;
        let n_features = group.n_features();

        match (group.storage(), group.layout()) {
            // Column-major dense: extract contiguous slice for this feature
            (BinStorage::DenseU8(data), GroupLayout::ColumnMajor) => {
                let start = idx_in_group * self.n_rows;
                let end = start + self.n_rows;
                FeatureView::U8 { bins: &data[start..end], stride: 1 }
            }
            (BinStorage::DenseU16(data), GroupLayout::ColumnMajor) => {
                let start = idx_in_group * self.n_rows;
                let end = start + self.n_rows;
                FeatureView::U16 { bins: &data[start..end], stride: 1 }
            }
            // Row-major dense: provide full data with stride
            (BinStorage::DenseU8(data), GroupLayout::RowMajor) => {
                // Slice from feature offset, stride = n_features
                FeatureView::U8 { bins: &data[idx_in_group..], stride: n_features }
            }
            (BinStorage::DenseU16(data), GroupLayout::RowMajor) => {
                FeatureView::U16 { bins: &data[idx_in_group..], stride: n_features }
            }
            // Sparse: always contiguous (one feature per group)
            (BinStorage::SparseU8 { row_indices, bin_values, .. }, _) => {
                FeatureView::SparseU8 { row_indices, bin_values }
            }
            (BinStorage::SparseU16 { row_indices, bin_values, .. }, _) => {
                FeatureView::SparseU16 { row_indices, bin_values }
            }
        }
    }

    /// Get zero-cost slice views for all features.
    ///
    /// This is the preferred method for histogram building as it pre-computes
    /// all feature slices in one pass, avoiding repeated index lookups.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let slices = dataset.feature_views();
    /// histogram_builder.build_with_bins(histogram, gradients, indices, &slices);
    /// ```
    #[inline]
    pub fn feature_views(&self) -> Vec<FeatureView<'_>> {
        (0..self.features.len())
            .map(|f| self.feature_view(f))
            .collect()
    }

    /// Get a single bin value for a specific row and feature.
    ///
    /// Returns `Some(bin)` for dense storage, `None` for sparse.
    /// For sparse features, use `feature_view()` and iterate over row_indices.
    ///
    /// This method is useful for partition splitting where you need single-row access.
    #[inline]
    pub fn get_bin(&self, row: usize, feature: usize) -> Option<u32> {
        self.feature_view(feature).get_bin(row)
    }

    /// Get a view for a specific row with bounds checking.
    ///
    /// Returns `None` if the row index is out of bounds.
    ///
    /// # Example
    ///
    /// ```ignore
    /// if let Some(row_view) = dataset.row_view(0) {
    ///     let bin = row_view.get_bin(feature_idx);
    /// }
    /// ```
    #[inline]
    pub fn row_view(&self, row: usize) -> Option<BinnedSampleSlice<'_>> {
        if row >= self.n_rows {
            return None;
        }
        Some(BinnedSampleSlice { dataset: self, row })
    }
}

// =============================================================================
// RowView
// =============================================================================

/// A validated view into a single row of a BinnedDataset.
///
/// Created via `BinnedDataset::row_view()` with bounds checking.
/// Provides convenient access to bin values for a specific row.
///
/// # Example
///
/// ```ignore
/// let row_view = dataset.row_view(0)?;
/// let bin = row_view.get_bin(feature_idx);
/// let is_missing = row_view.is_missing(feature_idx);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct BinnedSampleSlice<'a> {
    dataset: &'a BinnedDataset,
    row: usize,
}

impl<'a> BinnedSampleSlice<'a> {
    /// Get the bin value for a feature.
    ///
    /// Returns `Some(bin)` for dense features, `None` for sparse features
    /// (which indicates the value is the default/zero bin).
    #[inline]
    pub fn get_bin(&self, feature: usize) -> Option<u32> {
        self.dataset.get_bin(self.row, feature)
    }

    /// Check if a feature value is missing (None for sparse features).
    #[inline]
    pub fn is_missing(&self, feature: usize) -> bool {
        self.get_bin(feature).is_none()
    }

    /// Get the row index.
    #[inline]
    pub fn row(&self) -> usize {
        self.row
    }

    /// Get reference to the underlying dataset.
    #[inline]
    pub fn dataset(&self) -> &'a BinnedDataset {
        self.dataset
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::storage::BinStorage;
    use crate::data::binned::{BinMapper, MissingType};

    fn make_simple_mapper(n_bins: u32) -> BinMapper {
        let bounds: Vec<f64> = (0..n_bins).map(|i| i as f64 + 0.5).collect();
        BinMapper::numerical(bounds, MissingType::None, 0, 0, 0.0, 0.0, (n_bins - 1) as f64)
    }

    #[test]
    fn test_binned_dataset_single_group() {
        // 4 rows, 2 features in one group
        let storage = BinStorage::from_u8(vec![
            0, 1,  // row 0
            2, 3,  // row 1
            0, 2,  // row 2
            1, 3,  // row 3
        ]);

        let group = FeatureGroup::new(
            vec![0, 1],
            GroupLayout::RowMajor,
            4,
            storage,
            vec![4, 4],
        );

        let features = vec![
            FeatureMeta::new(make_simple_mapper(4), 0, 0),
            FeatureMeta::new(make_simple_mapper(4), 0, 1),
        ];

        let dataset = BinnedDataset::with_bundle_plan(4, features, vec![group], None);

        assert_eq!(dataset.n_rows(), 4);
        assert_eq!(dataset.n_features(), 2);
        assert_eq!(dataset.n_groups(), 1);
        assert_eq!(dataset.total_bins(), 8);
        assert_eq!(dataset.global_bin_offsets(), &[0, 4, 8]);

        // Access via feature_view
        let slice = dataset.feature_view(0);
        assert!(slice.is_dense());
        assert!(!slice.is_contiguous()); // row-major has stride > 1
        assert_eq!(slice.stride(), 2); // 2 features in group

        // Verify bin access
        assert_eq!(dataset.get_bin(0, 0), Some(0)); // row 0, feature 0
        assert_eq!(dataset.get_bin(0, 1), Some(1)); // row 0, feature 1
        assert_eq!(dataset.get_bin(1, 0), Some(2)); // row 1, feature 0
        assert_eq!(dataset.get_bin(3, 1), Some(3)); // row 3, feature 1

        assert!(dataset.is_uniform_layout().is_some());
        assert_eq!(dataset.is_uniform_layout(), Some(GroupLayout::RowMajor));
    }

    #[test]
    fn test_binned_dataset_multiple_groups() {
        // 3 rows, 3 features in 2 groups
        // Group 0: features 0,1 (row-major)
        // Group 1: feature 2 (column-major)

        let group0 = FeatureGroup::new(
            vec![0, 1],
            GroupLayout::RowMajor,
            3,
            BinStorage::from_u8(vec![0, 1, 2, 3, 4, 5]),  // 3 rows * 2 features
            vec![4, 4],
        );

        let group1 = FeatureGroup::new(
            vec![2],
            GroupLayout::ColumnMajor,
            3,
            BinStorage::from_u8(vec![10, 11, 12]),  // 3 rows * 1 feature
            vec![16],
        );

        let features = vec![
            FeatureMeta::new(make_simple_mapper(4), 0, 0),   // feature 0 -> group 0, idx 0
            FeatureMeta::new(make_simple_mapper(4), 0, 1),   // feature 1 -> group 0, idx 1
            FeatureMeta::new(make_simple_mapper(16), 1, 0),  // feature 2 -> group 1, idx 0
        ];

        let dataset = BinnedDataset::with_bundle_plan(3, features, vec![group0, group1], None);

        assert_eq!(dataset.n_rows(), 3);
        assert_eq!(dataset.n_features(), 3);
        assert_eq!(dataset.n_groups(), 2);
        assert_eq!(dataset.total_bins(), 24);  // 4 + 4 + 16
        assert_eq!(dataset.global_bin_offsets(), &[0, 4, 8, 24]);

        // Verify access patterns
        let slice0 = dataset.feature_view(0);
        assert!(slice0.is_dense());
        assert!(!slice0.is_contiguous()); // row-major

        let slice2 = dataset.feature_view(2);
        assert!(slice2.is_dense());
        assert!(slice2.is_contiguous()); // column-major

        // Verify bin values
        assert_eq!(dataset.get_bin(0, 2), Some(10));
        assert_eq!(dataset.get_bin(2, 2), Some(12));

        // Mixed layouts
        assert!(dataset.is_uniform_layout().is_none());

        // Group iteration by layout
        let row_major: Vec<_> = dataset.row_major_groups().collect();
        let col_major: Vec<_> = dataset.column_major_groups().collect();
        assert_eq!(row_major.len(), 1);
        assert_eq!(col_major.len(), 1);
        assert_eq!(row_major[0].0, 0);  // group 0
        assert_eq!(col_major[0].0, 1);  // group 1
    }

    #[test]
    fn test_feature_views() {
        // Column-major: 4 rows, 2 features
        let storage = BinStorage::from_u8(vec![
            0, 1, 2, 3,  // feature 0
            10, 11, 12, 13,  // feature 1
        ]);

        let group = FeatureGroup::new(
            vec![0, 1],
            GroupLayout::ColumnMajor,
            4,
            storage,
            vec![4, 16],
        );

        let features = vec![
            FeatureMeta::new(make_simple_mapper(4), 0, 0),
            FeatureMeta::new(make_simple_mapper(16), 0, 1),
        ];

        let dataset = BinnedDataset::with_bundle_plan(4, features, vec![group], None);
        let slices = dataset.feature_views();

        assert_eq!(slices.len(), 2);
        assert!(slices[0].is_contiguous());
        assert!(slices[1].is_contiguous());

        // Verify slice contents
        match slices[0] {
            FeatureView::U8 { bins, stride } => {
                assert_eq!(stride, 1);
                assert_eq!(bins, &[0, 1, 2, 3]);
            }
            _ => panic!("Expected U8"),
        }
        match slices[1] {
            FeatureView::U8 { bins, stride } => {
                assert_eq!(stride, 1);
                assert_eq!(bins, &[10, 11, 12, 13]);
            }
            _ => panic!("Expected U8"),
        }
    }

    #[test]
    fn test_empty_dataset() {
        let dataset = BinnedDataset::empty();
        assert_eq!(dataset.n_rows(), 0);
        assert_eq!(dataset.n_features(), 0);
        assert_eq!(dataset.n_groups(), 0);
        assert_eq!(dataset.total_bins(), 0);
        assert!(dataset.feature_views().is_empty());
    }
}

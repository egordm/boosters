//! BinnedDataset - the main dataset structure.

use super::bundling::{BundlePlan, FeatureLocation};
use super::group::{BinnedFeatureInfo, FeatureGroup};
use super::storage::{BinStorage, FeatureView, GroupLayout};
use crate::data::FeatureType;
use crate::data::{DataAccessor, SampleAccessor};

// =============================================================================
// BundledColumns - Pre-computed bundled bin data for EFB training
// =============================================================================

/// Pre-computed bundled column data for efficient histogram building.
///
/// When EFB is effective, this stores the encoded bins for each bundled column,
/// avoiding on-the-fly computation during histogram building.
#[derive(Clone, Debug)]
pub struct BundledColumns {
    /// Encoded bins for each bundled column (column-major: column × n_rows).
    /// Each bundle becomes one column with offset-encoded bins.
    bundle_bins: Vec<Vec<u16>>,
    /// Number of bins per bundled column (for histogram allocation).
    bundle_n_bins: Vec<u32>,
    /// Standalone feature indices (not bundled, passed through unchanged).
    standalone_features: Vec<usize>,
    /// Total number of bundled columns (bundles + standalone).
    n_columns: usize,
}

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
    features: Box<[BinnedFeatureInfo]>,
    /// Feature groups.
    groups: Vec<FeatureGroup>,
    /// Cumulative bin offsets across ALL features (for histogram indexing).
    /// `global_bin_offsets[i]` = sum of n_bins for features 0..i
    /// Length is n_features + 1 (last element is total_bins).
    global_bin_offsets: Box<[u32]>,
    /// Optional bundle plan if bundling was applied.
    bundle_plan: Option<BundlePlan>,
    /// Pre-computed bundled columns for efficient histogram building.
    /// Only populated when bundling is effective.
    bundled_columns: Option<BundledColumns>,
}

impl BinnedDataset {
    /// Create a new binned dataset with an optional bundle plan.
    ///
    /// Typically constructed via `BinnedDatasetBuilder`.
    pub(crate) fn with_bundle_plan(
        n_rows: usize,
        features: Vec<BinnedFeatureInfo>,
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
            bundled_columns: None, // Computed lazily via compute_bundled_columns()
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
            bundled_columns: None,
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
    pub fn feature(&self, idx: usize) -> &BinnedFeatureInfo {
        &self.features[idx]
    }

    /// Iterate over feature metadata.
    #[inline]
    pub fn iter_features(&self) -> impl Iterator<Item = &BinnedFeatureInfo> {
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

    /// Check if bundled columns are available for histogram building.
    ///
    /// Returns true if bundling is effective AND bundled columns have been computed.
    pub fn has_bundled_columns(&self) -> bool {
        self.bundled_columns.is_some()
    }

    /// Get the number of bundled columns (bundles + standalone features).
    ///
    /// Returns n_features if bundling is not effective.
    pub fn n_bundled_columns(&self) -> usize {
        if let Some(bc) = &self.bundled_columns {
            bc.n_columns
        } else {
            self.features.len()
        }
    }

    /// Get feature views for bundled columns (fewer than original features).
    ///
    /// If bundled columns are available, returns:
    /// - One view per bundle (encoded bins for mutually exclusive sparse features)
    /// - One view per standalone dense feature
    ///
    /// If bundling is not effective, returns same as `feature_views()`.
    ///
    /// # Note
    /// Call `compute_bundled_columns()` during dataset construction to enable this.
    pub fn bundled_feature_views(&self) -> Vec<FeatureView<'_>> {
        if let Some(bc) = &self.bundled_columns {
            let mut views = Vec::with_capacity(bc.n_columns);

            // Add bundle views (u16 encoded bins)
            for bundle_bins in &bc.bundle_bins {
                views.push(FeatureView::U16 {
                    bins: bundle_bins,
                    stride: 1,
                });
            }

            // Add standalone feature views (unchanged)
            for &feature_idx in &bc.standalone_features {
                views.push(self.feature_view(feature_idx));
            }

            views
        } else {
            self.feature_views()
        }
    }

    /// Get a single bundled column's view by bundled column index.
    ///
    /// If bundling is active:
    /// - `col_idx < n_bundles`: returns the bundle's u16 encoded bins
    /// - `col_idx >= n_bundles`: returns the standalone feature view
    ///
    /// If bundling is not active (no bundles computed):
    /// - `col_idx` is treated as original feature index
    ///
    /// # Panics
    /// - If col_idx >= n_bundled_columns()
    pub fn bundled_feature_view(&self, col_idx: usize) -> FeatureView<'_> {
        if let Some(bc) = &self.bundled_columns {
            let n_bundles = bc.bundle_bins.len();
            if col_idx < n_bundles {
                FeatureView::U16 {
                    bins: &bc.bundle_bins[col_idx],
                    stride: 1,
                }
            } else {
                let standalone_idx = col_idx - n_bundles;
                let orig_feat = bc.standalone_features[standalone_idx];
                self.feature_view(orig_feat)
            }
        } else {
            // No bundling: col_idx is original feature index
            self.feature_view(col_idx)
        }
    }

    /// Get bin counts for bundled columns (for histogram layout).
    ///
    /// Returns bin counts per bundled column if bundling is effective,
    /// otherwise returns bin counts for original features.
    pub fn bundled_bin_counts(&self) -> Vec<u32> {
        if let Some(bc) = &self.bundled_columns {
            let mut counts = bc.bundle_n_bins.clone();
            for &feature_idx in &bc.standalone_features {
                counts.push(self.features[feature_idx].n_bins());
            }
            counts
        } else {
            self.features.iter().map(|f| f.n_bins()).collect()
        }
    }

    /// Compute and store bundled columns for efficient histogram building.
    ///
    /// This pre-computes the offset-encoded bins for each bundle, enabling
    /// efficient histogram building during training. Should be called once
    /// after dataset construction when bundling is effective.
    ///
    /// # Returns
    /// - `true` if bundled columns were computed
    /// - `false` if bundling is not effective (no-op)
    pub fn compute_bundled_columns(&mut self) -> bool {
        let plan = match &self.bundle_plan {
            Some(p) if p.is_effective() => p,
            _ => return false,
        };

        let n_rows = self.n_rows;
        let n_bundles = plan.bundles.len();

        // Pre-compute bin counts for each bundle
        let bundle_n_bins: Vec<u32> = plan
            .bundles
            .iter()
            .map(|bundle| bundle.total_bins)
            .collect();

        // Pre-compute the default (most frequent) bin for each feature in each bundle.
        // A feature is "active" when its bin differs from its default bin.
        // This correctly handles features where the default bin is not 0.
        let bundle_default_bins: Vec<Vec<u32>> = plan
            .bundles
            .iter()
            .map(|bundle| {
                bundle
                    .feature_indices
                    .iter()
                    .map(|&feat_idx| self.bin_mapper(feat_idx).most_freq_bin())
                    .collect()
            })
            .collect();

        // Compute encoded bins for each bundle
        let bundle_bins: Vec<Vec<u16>> = plan
            .bundles
            .iter()
            .zip(bundle_default_bins.iter())
            .map(|(bundle, default_bins)| {
                let mut encoded = vec![0u16; n_rows];

                // For each row, find the first feature with a non-default bin value
                for row in 0..n_rows {
                    let mut found_active = false;

                    for (i, &feat_idx) in bundle.feature_indices.iter().enumerate() {
                        if let Some(bin) = self.get_bin(row, feat_idx) {
                            let default_bin = default_bins[i];
                            if bin != default_bin {
                                // Found active feature (non-default bin), encode it
                                let offset = bundle.bin_offsets[i];
                                encoded[row] = (offset + bin) as u16;
                                found_active = true;
                                break;
                            }
                        }
                    }

                    if !found_active {
                        encoded[row] = 0; // All features have default values
                    }
                }

                encoded
            })
            .collect();

        let standalone_features = plan.standalone_features.clone();
        let n_columns = n_bundles + standalone_features.len();

        self.bundled_columns = Some(BundledColumns {
            bundle_bins,
            bundle_n_bins,
            standalone_features,
            n_columns,
        });

        true
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
        self.bundle_plan
            .as_ref()?
            .decode_bundle_split(bundle_idx, encoded_bin)
    }

    /// Get the bin range [min_bin, max_bin] for a bundled split.
    ///
    /// Used during partitioning to determine if a row's encoded bin belongs to
    /// the same sub-feature as the split. Rows outside this range should go to
    /// the default direction.
    ///
    /// # Arguments
    /// * `bundle_idx` - Index of the bundle
    /// * `encoded_bin` - The split threshold (encoded bin value)
    ///
    /// # Returns
    /// * `Some((min_bin, max_bin))` - The range of encoded bins for this sub-feature
    /// * `None` if no bundling or encoded_bin is 0 (all defaults)
    pub fn bundle_split_bin_range(
        &self,
        bundle_idx: usize,
        encoded_bin: u32,
    ) -> Option<(u32, u32)> {
        self.bundle_plan
            .as_ref()?
            .bundle_split_bin_range(bundle_idx, encoded_bin)
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

    // =========================================================================
    // Bundled Column Helpers (for LightGBM-style histogram building)
    // =========================================================================

    /// Get the number of bundles (not including standalone features).
    ///
    /// Returns 0 if bundling is not active.
    pub fn n_bundles(&self) -> usize {
        self.bundle_plan
            .as_ref()
            .map(|p| p.bundles.len())
            .unwrap_or(0)
    }

    /// Get standalone feature indices (features not in bundles).
    ///
    /// Returns empty slice if bundling is not active.
    pub fn standalone_features(&self) -> &[usize] {
        self.bundled_columns
            .as_ref()
            .map(|bc| bc.standalone_features.as_slice())
            .unwrap_or(&[])
    }

    /// Check if a bundled column is categorical.
    ///
    /// Bundle columns are always numeric (categorical features are not bundled).
    /// Standalone columns inherit their original feature's type.
    /// When bundling is not active, col_idx is the original feature index.
    ///
    /// # Arguments
    /// * `col_idx` - Bundled column index (0..n_bundled_columns)
    pub fn bundled_column_is_categorical(&self, col_idx: usize) -> bool {
        let n_bundles = self.n_bundles();
        if col_idx < n_bundles {
            // Bundle columns are always numeric
            false
        } else if let Some(bc) = &self.bundled_columns {
            // Standalone column with bundling active: check original feature
            let standalone_idx = col_idx - n_bundles;
            if standalone_idx < bc.standalone_features.len() {
                let orig_feat = bc.standalone_features[standalone_idx];
                return self.is_categorical(orig_feat);
            }
            false
        } else {
            // No bundling: col_idx IS original feature index
            self.is_categorical(col_idx)
        }
    }

    /// Check if a bundled column has missing values.
    ///
    /// Bundle columns: bin 0 represents "all features at default", which
    /// acts like a missing indicator. We return true for bundles.
    ///
    /// Standalone columns inherit their original feature's missing status.
    /// When bundling is not active, col_idx is the original feature index.
    ///
    /// # Arguments
    /// * `col_idx` - Bundled column index (0..n_bundled_columns)
    pub fn bundled_column_has_missing(&self, col_idx: usize) -> bool {
        let n_bundles = self.n_bundles();
        if col_idx < n_bundles {
            // Bundle columns have "missing" semantics via bin 0
            true
        } else if let Some(bc) = &self.bundled_columns {
            // Standalone column with bundling active: check original feature
            let standalone_idx = col_idx - n_bundles;
            if standalone_idx < bc.standalone_features.len() {
                let orig_feat = bc.standalone_features[standalone_idx];
                return self.has_missing(orig_feat);
            }
            false
        } else {
            // No bundling: col_idx IS original feature index
            self.has_missing(col_idx)
        }
    }

    /// Convert a bundled column split to original feature split.
    ///
    /// This is used when storing a split in the tree. The tree must use
    /// original feature indices so prediction works without bundling.
    ///
    /// # Arguments
    /// * `col_idx` - Bundled column index (0..n_bundled_columns)
    /// * `bin` - Split threshold in the bundled column
    ///
    /// # Returns
    /// * `Some((original_feature, original_bin))` if the split is valid
    /// * `None` if the split is invalid (e.g., bin 0 on a bundle)
    pub fn bundled_column_to_split(&self, col_idx: usize, bin: u32) -> Option<(usize, u32)> {
        let n_bundles = self.n_bundles();
        if col_idx < n_bundles {
            // Bundle column: decode to original feature
            self.decode_bundle_split(col_idx, bin)
        } else {
            // Standalone column: direct mapping
            let standalone_idx = col_idx - n_bundles;
            if let Some(bc) = &self.bundled_columns {
                if standalone_idx < bc.standalone_features.len() {
                    let orig_feat = bc.standalone_features[standalone_idx];
                    return Some((orig_feat, bin));
                }
            }
            // Fallback: no bundling active, col_idx IS the original feature
            Some((col_idx, bin))
        }
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
    pub fn groups_by_layout(
        &self,
        layout: GroupLayout,
    ) -> impl Iterator<Item = (usize, &FeatureGroup)> {
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
                FeatureView::U8 {
                    bins: &data[start..end],
                    stride: 1,
                }
            }
            (BinStorage::DenseU16(data), GroupLayout::ColumnMajor) => {
                let start = idx_in_group * self.n_rows;
                let end = start + self.n_rows;
                FeatureView::U16 {
                    bins: &data[start..end],
                    stride: 1,
                }
            }
            // Row-major dense: provide full data with stride
            (BinStorage::DenseU8(data), GroupLayout::RowMajor) => {
                // Slice from feature offset, stride = n_features
                FeatureView::U8 {
                    bins: &data[idx_in_group..],
                    stride: n_features,
                }
            }
            (BinStorage::DenseU16(data), GroupLayout::RowMajor) => FeatureView::U16 {
                bins: &data[idx_in_group..],
                stride: n_features,
            },
            // Sparse: always contiguous (one feature per group)
            (
                BinStorage::SparseU8 {
                    row_indices,
                    bin_values,
                    ..
                },
                _,
            ) => FeatureView::SparseU8 {
                row_indices,
                bin_values,
            },
            (
                BinStorage::SparseU16 {
                    row_indices,
                    bin_values,
                    ..
                },
                _,
            ) => FeatureView::SparseU16 {
                row_indices,
                bin_values,
            },
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
    pub fn row_view(&self, row: usize) -> Option<BinnedSample<'_>> {
        if row >= self.n_rows {
            return None;
        }
        Some(BinnedSample { dataset: self, row })
    }
}

// =============================================================================
// BinnedSample
// =============================================================================

/// A validated view into a single row of a BinnedDataset.
///
/// Created via `BinnedDataset::row_view()` with bounds checking.
/// Provides convenient access to bin values for a specific row.
///
/// Implements [`SampleAccessor`] for tree traversal, converting bin indices
/// to midpoint values on-demand.
///
/// # Example
///
/// ```ignore
/// let sample = dataset.row_view(0)?;
/// let bin = sample.get_bin(feature_idx);
/// let is_missing = sample.is_missing(feature_idx);
///
/// // For tree traversal via SampleAccessor:
/// let value = sample.feature(feature_idx);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct BinnedSample<'a> {
    dataset: &'a BinnedDataset,
    row: usize,
}

impl<'a> BinnedSample<'a> {
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

// =============================================================================
// Trait Implementations
// =============================================================================

impl SampleAccessor for BinnedSample<'_> {
    #[inline]
    fn feature(&self, index: usize) -> f32 {
        let bin = self.dataset.get_bin(self.row, index);
        match bin {
            Some(b) => self.dataset.bin_mapper(index).bin_to_midpoint(b) as f32,
            None => f32::NAN,
        }
    }

    #[inline]
    fn n_features(&self) -> usize {
        self.dataset.n_features()
    }
}

impl DataAccessor for BinnedDataset {
    type Sample<'a>
        = BinnedSample<'a>
    where
        Self: 'a;

    #[inline]
    fn sample(&self, index: usize) -> Self::Sample<'_> {
        BinnedSample {
            dataset: self,
            row: index,
        }
    }

    #[inline]
    fn n_samples(&self) -> usize {
        self.n_rows()
    }

    #[inline]
    fn n_features(&self) -> usize {
        BinnedDataset::n_features(self)
    }

    #[inline]
    fn feature_type(&self, feature: usize) -> FeatureType {
        if self.features[feature].is_categorical() {
            FeatureType::Categorical
        } else {
            FeatureType::Numeric
        }
    }

    #[inline]
    fn has_categorical(&self) -> bool {
        self.features.iter().any(|f| f.is_categorical())
    }
}

#[cfg(test)]
mod tests {
    use super::super::storage::BinStorage;
    use super::*;
    use crate::data::binned::{BinMapper, MissingType};

    fn make_simple_mapper(n_bins: u32) -> BinMapper {
        let bounds: Vec<f64> = (0..n_bins).map(|i| i as f64 + 0.5).collect();
        BinMapper::numerical(
            bounds,
            MissingType::None,
            0,
            0,
            0.0,
            0.0,
            (n_bins - 1) as f64,
        )
    }

    #[test]
    fn test_binned_dataset_single_group() {
        // 4 rows, 2 features in one group
        let storage = BinStorage::from_u8(vec![
            0, 1, // row 0
            2, 3, // row 1
            0, 2, // row 2
            1, 3, // row 3
        ]);

        let group = FeatureGroup::new(vec![0, 1], GroupLayout::RowMajor, 4, storage, vec![4, 4]);

        let features = vec![
            BinnedFeatureInfo::new(make_simple_mapper(4), 0, 0),
            BinnedFeatureInfo::new(make_simple_mapper(4), 0, 1),
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
            BinStorage::from_u8(vec![0, 1, 2, 3, 4, 5]), // 3 rows * 2 features
            vec![4, 4],
        );

        let group1 = FeatureGroup::new(
            vec![2],
            GroupLayout::ColumnMajor,
            3,
            BinStorage::from_u8(vec![10, 11, 12]), // 3 rows * 1 feature
            vec![16],
        );

        let features = vec![
            BinnedFeatureInfo::new(make_simple_mapper(4), 0, 0), // feature 0 -> group 0, idx 0
            BinnedFeatureInfo::new(make_simple_mapper(4), 0, 1), // feature 1 -> group 0, idx 1
            BinnedFeatureInfo::new(make_simple_mapper(16), 1, 0), // feature 2 -> group 1, idx 0
        ];

        let dataset = BinnedDataset::with_bundle_plan(3, features, vec![group0, group1], None);

        assert_eq!(dataset.n_rows(), 3);
        assert_eq!(dataset.n_features(), 3);
        assert_eq!(dataset.n_groups(), 2);
        assert_eq!(dataset.total_bins(), 24); // 4 + 4 + 16
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
        assert_eq!(row_major[0].0, 0); // group 0
        assert_eq!(col_major[0].0, 1); // group 1
    }

    #[test]
    fn test_feature_views() {
        // Column-major: 4 rows, 2 features
        let storage = BinStorage::from_u8(vec![
            0, 1, 2, 3, // feature 0
            10, 11, 12, 13, // feature 1
        ]);

        let group = FeatureGroup::new(
            vec![0, 1],
            GroupLayout::ColumnMajor,
            4,
            storage,
            vec![4, 16],
        );

        let features = vec![
            BinnedFeatureInfo::new(make_simple_mapper(4), 0, 0),
            BinnedFeatureInfo::new(make_simple_mapper(16), 0, 1),
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

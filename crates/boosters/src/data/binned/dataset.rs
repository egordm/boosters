//! Unified BinnedDataset for training and inference.
//!
//! This module contains the main `BinnedDataset` type which replaces the
//! previous separate `Dataset` and `BinnedDataset` types. It contains both
//! binned data (for tree splits) and raw data (for linear regression).

// Allow dead code during migration - this will be used when we switch over in Epic 7
#![allow(dead_code)]

use super::bin_mapper::BinMapper;
use super::builder::BuiltGroups;
use super::group::FeatureGroup;
use super::sample_blocks::SampleBlocks;
use super::view::FeatureView;
use crate::data::types::accessor::{DataAccessor, SampleAccessor};
use crate::data::types::schema::FeatureType;

/// Where a feature's data lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureLocation {
    /// Feature in a regular (Dense or Sparse) group.
    Direct {
        group_idx: u32,
        idx_in_group: u32,
    },
    /// Feature bundled into a Bundle group (EFB).
    /// Not yet implemented - reserved for future use.
    Bundled {
        bundle_group_idx: u32,
        position_in_bundle: u32,
    },
    /// Feature was skipped (trivial, constant value).
    Skipped,
}

impl FeatureLocation {
    /// Returns true if the feature is directly stored (not bundled or skipped).
    #[inline]
    pub fn is_direct(&self) -> bool {
        matches!(self, FeatureLocation::Direct { .. })
    }

    /// Returns true if the feature is bundled (EFB).
    #[inline]
    pub fn is_bundled(&self) -> bool {
        matches!(self, FeatureLocation::Bundled { .. })
    }

    /// Returns true if the feature was skipped.
    #[inline]
    pub fn is_skipped(&self) -> bool {
        matches!(self, FeatureLocation::Skipped)
    }
}

/// Metadata for a single feature.
#[derive(Debug, Clone)]
pub struct BinnedFeatureInfo {
    /// Optional feature name.
    pub name: Option<String>,
    /// The bin mapper for this feature (contains thresholds/categories).
    pub bin_mapper: BinMapper,
    /// Where this feature's data lives.
    pub location: FeatureLocation,
}

impl BinnedFeatureInfo {
    /// Create a new feature info.
    pub fn new(name: Option<String>, bin_mapper: BinMapper, location: FeatureLocation) -> Self {
        Self {
            name,
            bin_mapper,
            location,
        }
    }

    /// Returns true if this feature is categorical.
    #[inline]
    pub fn is_categorical(&self) -> bool {
        self.bin_mapper.is_categorical()
    }

    /// Get the number of bins for this feature.
    #[inline]
    pub fn n_bins(&self) -> u32 {
        self.bin_mapper.n_bins()
    }
}

/// Result of `effective_feature_views()` - views in the effective layout.
///
/// The effective layout puts bundles first, then standalone features:
/// ```text
/// [bundle_0, bundle_1, ..., standalone_0, standalone_1, ...]
/// ```
///
/// This struct provides both the views and metadata needed by the grower.
#[derive(Debug)]
pub struct EffectiveViews<'a> {
    /// Feature views in effective order.
    pub views: Vec<FeatureView<'a>>,
    /// Bin counts for each effective column.
    pub bin_counts: Vec<u32>,
    /// Whether each effective column is a bundle.
    pub is_bundle: Vec<bool>,
    /// Number of bundle columns (first `n_bundles` entries are bundles).
    pub n_bundles: usize,
    /// Group indices for bundle columns (for decoding splits).
    pub bundle_groups: Vec<usize>,
    /// Original feature indices for standalone columns (effective_idx - n_bundles → original_feature).
    pub standalone_features: Vec<usize>,
    /// Whether each effective column is categorical.
    /// For bundles: always true (encoded bins are categorical-like).
    /// For standalone: matches the original feature type.
    pub is_categorical: Vec<bool>,
    /// Whether each effective column has missing values.
    /// For bundles: always true (bin 0 is "all defaults" / missing).
    /// For standalone: matches the original feature.
    pub has_missing: Vec<bool>,
}

impl<'a> EffectiveViews<'a> {
    /// Get the number of effective columns (bundles + standalone features).
    #[inline]
    pub fn n_columns(&self) -> usize {
        self.views.len()
    }

    /// Check if the dataset has any bundles.
    #[inline]
    pub fn has_bundles(&self) -> bool {
        self.n_bundles > 0
    }

    /// Map an effective column index to the original feature index.
    ///
    /// For bundle columns, returns `None` (bundles contain multiple features).
    /// For standalone columns, returns `Some(original_feature_idx)`.
    #[inline]
    pub fn effective_to_original(&self, effective_idx: usize) -> Option<usize> {
        if effective_idx < self.n_bundles {
            None // Bundle column - contains multiple features
        } else {
            Some(self.standalone_features[effective_idx - self.n_bundles])
        }
    }
}

/// The unified dataset type for training and inference.
///
/// Contains both binned data (for tree splits) and raw data (for linear regression).
/// This replaces the previous separate `Dataset` and `BinnedDataset` types.
#[derive(Debug, Clone)]
pub struct BinnedDataset {
    /// Number of samples.
    n_samples: usize,
    /// Per-feature metadata (name, bin mapper, location).
    features: Box<[BinnedFeatureInfo]>,
    /// Feature groups (actual storage).
    groups: Vec<FeatureGroup>,
    /// Global bin offsets for histogram allocation.
    /// `global_bin_offsets[i]` is the offset of feature i's bins in the global histogram.
    global_bin_offsets: Box<[u32]>,
    /// Optional labels (targets).
    labels: Option<Box<[f32]>>,
    /// Optional sample weights.
    weights: Option<Box<[f32]>>,
}

impl BinnedDataset {
    /// Create a new BinnedDataset from BuiltGroups.
    ///
    /// This is the main constructor used by the DatasetBuilder.
    pub fn from_built_groups(built: BuiltGroups) -> Self {
        let n_samples = built.n_samples;
        let n_features = built.analyses.len();

        // Build feature info and locations
        let mut features = Vec::with_capacity(n_features);
        let mut global_bin_offsets = Vec::with_capacity(n_features + 1);
        let mut current_offset = 0u32;

        // Map from global feature index to its location
        // First, build a reverse map from the groups
        let mut location_map: Vec<Option<FeatureLocation>> = vec![None; n_features];

        for (group_idx, group) in built.groups.iter().enumerate() {
            let is_bundle = group.is_bundle();
            for (idx_in_group, &global_idx) in group.feature_indices().iter().enumerate() {
                let location = if is_bundle {
                    FeatureLocation::Bundled {
                        bundle_group_idx: group_idx as u32,
                        position_in_bundle: idx_in_group as u32,
                    }
                } else {
                    FeatureLocation::Direct {
                        group_idx: group_idx as u32,
                        idx_in_group: idx_in_group as u32,
                    }
                };
                location_map[global_idx as usize] = Some(location);
            }
        }

        // Now build the feature info
        for feature_idx in 0..n_features {
            // Get bin mapper
            let bin_mapper = built.bin_mappers[feature_idx].clone();
            let n_bins = bin_mapper.n_bins();

            // Get location
            let location = if built.trivial_features.contains(&feature_idx) {
                FeatureLocation::Skipped
            } else if let Some(loc) = location_map[feature_idx] {
                loc
            } else {
                // Feature not in any group - should not happen
                FeatureLocation::Skipped
            };

            // Get feature name from analysis if available
            let name = None; // TODO: Add name to FeatureAnalysis if needed

            features.push(BinnedFeatureInfo::new(name, bin_mapper, location));

            // Track global bin offsets
            global_bin_offsets.push(current_offset);
            current_offset += n_bins;
        }
        global_bin_offsets.push(current_offset); // Final offset for total bins

        Self {
            n_samples,
            features: features.into_boxed_slice(),
            groups: built.groups,
            global_bin_offsets: global_bin_offsets.into_boxed_slice(),
            labels: built.labels.map(|v| v.into_boxed_slice()),
            weights: built.weights.map(|v| v.into_boxed_slice()),
        }
    }

    // =========================================================================
    // Basic accessors
    // =========================================================================

    /// Get the number of samples.
    #[inline]
    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    /// Get the number of features.
    #[inline]
    pub fn n_features(&self) -> usize {
        self.features.len()
    }

    /// Get the number of groups.
    #[inline]
    pub fn n_groups(&self) -> usize {
        self.groups.len()
    }

    /// Get the total number of bins across all features.
    #[inline]
    pub fn total_bins(&self) -> u32 {
        // Last element is the total
        self.global_bin_offsets
            .last()
            .copied()
            .unwrap_or(0)
    }

    /// Get feature info for a feature.
    #[inline]
    pub fn feature_info(&self, feature: usize) -> &BinnedFeatureInfo {
        &self.features[feature]
    }

    /// Get the location of a feature.
    #[inline]
    pub fn feature_location(&self, feature: usize) -> FeatureLocation {
        self.features[feature].location
    }

    /// Get the bin mapper for a feature.
    #[inline]
    pub fn bin_mapper(&self, feature: usize) -> &BinMapper {
        &self.features[feature].bin_mapper
    }

    /// Get the global bin offset for a feature.
    /// This is used for histogram indexing.
    #[inline]
    pub fn global_bin_offset(&self, feature: usize) -> u32 {
        self.global_bin_offsets[feature]
    }

    /// Check if a feature is categorical.
    #[inline]
    pub fn is_categorical(&self, feature: usize) -> bool {
        self.features[feature].is_categorical()
    }

    /// Get the number of bins for a feature.
    #[inline]
    pub fn n_bins(&self, feature: usize) -> u32 {
        self.features[feature].n_bins()
    }

    /// Check if the dataset has labels.
    #[inline]
    pub fn has_labels(&self) -> bool {
        self.labels.is_some()
    }

    /// Get the labels if present.
    #[inline]
    pub fn labels(&self) -> Option<&[f32]> {
        self.labels.as_deref()
    }

    /// Check if the dataset has weights.
    #[inline]
    pub fn has_weights(&self) -> bool {
        self.weights.is_some()
    }

    /// Get the weights if present.
    #[inline]
    pub fn weights(&self) -> Option<&[f32]> {
        self.weights.as_deref()
    }

    /// Get a reference to the groups.
    #[inline]
    pub fn groups(&self) -> &[FeatureGroup] {
        &self.groups
    }

    /// Get a reference to a specific group.
    #[inline]
    pub fn group(&self, group_idx: usize) -> &FeatureGroup {
        &self.groups[group_idx]
    }

    // =========================================================================
    // Bin/Raw Access Methods
    // =========================================================================

    /// Get the bin value for a sample and feature.
    ///
    /// # Parameters
    /// - `sample`: Sample index (0..n_samples)
    /// - `feature`: Global feature index (0..n_features)
    ///
    /// # Panics
    ///
    /// Panics if the feature is skipped (trivial) or indices are out of bounds.
    #[inline]
    pub fn bin(&self, sample: usize, feature: usize) -> u32 {
        let location = self.features[feature].location;
        match location {
            FeatureLocation::Direct {
                group_idx,
                idx_in_group,
            } => self.groups[group_idx as usize].bin(sample, idx_in_group as usize),
            FeatureLocation::Bundled { .. } => {
                // TODO: Implement bundled feature access
                panic!("Bundled feature access not yet implemented")
            }
            FeatureLocation::Skipped => {
                panic!("Cannot access bin for skipped feature {feature}")
            }
        }
    }

    /// Get the raw value for a sample and feature.
    ///
    /// Returns `None` for categorical features.
    ///
    /// # Parameters
    /// - `sample`: Sample index (0..n_samples)
    /// - `feature`: Global feature index (0..n_features)
    ///
    /// # Panics
    ///
    /// Panics if the feature is skipped (trivial) or indices are out of bounds.
    #[inline]
    pub fn raw_value(&self, sample: usize, feature: usize) -> Option<f32> {
        let location = self.features[feature].location;
        match location {
            FeatureLocation::Direct {
                group_idx,
                idx_in_group,
            } => self.groups[group_idx as usize].raw(sample, idx_in_group as usize),
            FeatureLocation::Bundled { .. } => {
                // Bundled features don't have raw values
                None
            }
            FeatureLocation::Skipped => {
                panic!("Cannot access raw value for skipped feature {feature}")
            }
        }
    }

    /// Get a contiguous slice of raw values for a feature.
    ///
    /// Returns `None` for categorical features or sparse storage.
    ///
    /// # Parameters
    /// - `feature`: Global feature index (0..n_features)
    ///
    /// # Panics
    ///
    /// Panics if the feature is skipped (trivial).
    #[inline]
    pub fn raw_feature_slice(&self, feature: usize) -> Option<&[f32]> {
        let location = self.features[feature].location;
        match location {
            FeatureLocation::Direct {
                group_idx,
                idx_in_group,
            } => self.groups[group_idx as usize].raw_slice(idx_in_group as usize),
            FeatureLocation::Bundled { .. } => {
                // Bundled features don't have raw slices
                None
            }
            FeatureLocation::Skipped => {
                panic!("Cannot access raw slice for skipped feature {feature}")
            }
        }
    }

    // =========================================================================
    // Histogram Building (Hot Path)
    // =========================================================================

    /// Get feature views for histogram building.
    ///
    /// Returns views for all non-trivial features, in global feature index order.
    /// This is the primary API for training - the hot path for histogram building.
    ///
    /// # Returns
    ///
    /// A vector of `FeatureView`s, one per non-trivial feature, in order of
    /// global feature index.
    pub fn feature_views(&self) -> Vec<FeatureView<'_>> {
        let mut views = Vec::with_capacity(self.features.len());

        for feature_idx in 0..self.features.len() {
            let location = self.features[feature_idx].location;
            match location {
                FeatureLocation::Direct {
                    group_idx,
                    idx_in_group,
                } => {
                    let view = self.groups[group_idx as usize].feature_view(idx_in_group as usize);
                    views.push(view);
                }
                FeatureLocation::Bundled { .. } => {
                    // TODO: Handle bundled features when EFB is implemented
                    // For now, skip bundled features
                }
                FeatureLocation::Skipped => {
                    // Trivial features are skipped - don't add a view
                }
            }
        }

        views
    }

    /// Get effective feature views for histogram building.
    ///
    /// This returns views in the "effective" feature layout used by the grower:
    /// - First: One view per bundle (the encoded bundle column)
    /// - Then: One view per standalone (non-bundled) feature
    ///
    /// The returned `EffectiveViews` struct contains both the views and metadata
    /// about which columns are bundles.
    ///
    /// # Layout
    ///
    /// ```text
    /// Effective columns: [bundle_0, bundle_1, ..., standalone_0, standalone_1, ...]
    /// ```
    ///
    /// For datasets without bundling, this is equivalent to `feature_views()`.
    pub fn effective_feature_views(&self) -> EffectiveViews<'_> {
        use crate::data::MissingType;

        let mut views = Vec::new();
        let mut bin_counts = Vec::new();
        let mut is_bundle = Vec::new();
        let mut is_categorical = Vec::new();
        let mut has_missing = Vec::new();
        let mut bundle_groups = Vec::new();
        let mut standalone_features = Vec::new();
        let mut n_bundles = 0;

        // Phase 1: Collect bundle views (bundle groups)
        for (group_idx, group) in self.groups.iter().enumerate() {
            if group.is_bundle() {
                // Bundle group: one view for the encoded bundle
                views.push(group.feature_view(0));
                if let Some(bundle) = group.as_bundle() {
                    bin_counts.push(bundle.total_bins());
                } else {
                    // Should not happen, but fallback to bin_counts[0]
                    bin_counts.push(group.bin_counts()[0]);
                }
                is_bundle.push(true);
                // Bundles are treated as categorical (encoded bin space)
                is_categorical.push(true);
                // Bundles always have "missing" (bin 0 = all defaults)
                has_missing.push(true);
                bundle_groups.push(group_idx);
                n_bundles += 1;
            }
        }

        // Phase 2: Collect standalone feature views (non-bundle features)
        for feature_idx in 0..self.features.len() {
            let location = self.features[feature_idx].location;
            match location {
                FeatureLocation::Direct {
                    group_idx,
                    idx_in_group,
                } => {
                    let group = &self.groups[group_idx as usize];
                    views.push(group.feature_view(idx_in_group as usize));
                    bin_counts.push(group.bin_counts()[idx_in_group as usize]);
                    is_bundle.push(false);
                    // Use original feature's categorical flag
                    is_categorical.push(self.features[feature_idx].is_categorical());
                    // Check if original feature has missing
                    has_missing.push(
                        self.features[feature_idx].bin_mapper.missing_type() != MissingType::None,
                    );
                    standalone_features.push(feature_idx);
                }
                FeatureLocation::Bundled { .. } | FeatureLocation::Skipped => {
                    // Bundled features are accessed via their bundle
                    // Skipped features are not included
                }
            }
        }

        EffectiveViews {
            views,
            bin_counts,
            is_bundle,
            n_bundles,
            bundle_groups,
            standalone_features,
            is_categorical,
            has_missing,
        }
    }

    /// Get view for a single original feature.
    ///
    /// Use this when you need access to a specific feature, not for bulk iteration.
    ///
    /// # Parameters
    /// - `feature`: Global feature index (0..n_features)
    ///
    /// # Panics
    ///
    /// Panics if the feature is skipped (trivial) or bundled.
    pub fn original_feature_view(&self, feature: usize) -> FeatureView<'_> {
        let location = self.features[feature].location;
        match location {
            FeatureLocation::Direct {
                group_idx,
                idx_in_group,
            } => self.groups[group_idx as usize].feature_view(idx_in_group as usize),
            FeatureLocation::Bundled { .. } => {
                panic!("Cannot get view for bundled feature {feature} - use feature_views() instead")
            }
            FeatureLocation::Skipped => {
                panic!("Cannot get view for skipped feature {feature}")
            }
        }
    }

    /// Decode a split on an effective column to the original feature and bin.
    ///
    /// When working with effective columns (from `effective_feature_views()`), splits
    /// need to be decoded back to original feature indices for tree storage.
    ///
    /// # Parameters
    /// - `effective_views`: The EffectiveViews from `effective_feature_views()`
    /// - `effective_col`: The effective column index (0..n_effective_columns)
    /// - `bin`: The bin value for the split
    ///
    /// # Returns
    /// `(original_feature_idx, original_bin)` for use in tree split nodes.
    ///
    /// # Bundle Handling
    /// For bundle columns, decodes the encoded bin to find which original feature
    /// owns that bin. If bin is 0 (all defaults), returns the first feature in the
    /// bundle with bin 0.
    pub fn decode_split_to_original(
        &self,
        effective_views: &EffectiveViews<'_>,
        effective_col: usize,
        bin: u32,
    ) -> (usize, u32) {
        if effective_col < effective_views.n_bundles {
            // Bundle column: decode using BundleStorage
            let group_idx = effective_views.bundle_groups[effective_col];
            let group = &self.groups[group_idx];
            if let Some(bundle) = group.as_bundle() {
                if let Some((pos_in_bundle, orig_bin)) = bundle.decode(bin as u16) {
                    // Map position in bundle to original feature
                    let orig_feature = bundle.feature_indices()[pos_in_bundle] as usize;
                    (orig_feature, orig_bin)
                } else {
                    // bin 0 = all defaults. Pick the first feature in the bundle with bin 0.
                    let first_feature = bundle.feature_indices()[0] as usize;
                    (first_feature, 0)
                }
            } else {
                panic!("Bundle group {} has no BundleStorage", group_idx);
            }
        } else {
            // Standalone column: direct mapping
            let standalone_idx = effective_col - effective_views.n_bundles;
            let orig_feature = effective_views.standalone_features[standalone_idx];
            (orig_feature, bin)
        }
    }

    // =========================================================================
    // Linear trees support / gblinear
    // =========================================================================

    /// Check if any feature has raw values (for linear trees).
    /// True if there's at least one numeric group.
    pub fn has_raw_values(&self) -> bool {
        self.groups.iter().any(|g| g.has_raw_values())
    }

    /// Get indices of numeric features (for linear tree feature selection).
    ///
    /// Linear trees use this to identify which features to include in regression.
    /// Features with `FeatureStorageType::Bundled` are excluded (splits only, no regression).
    pub fn numeric_feature_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.features.iter().enumerate().filter_map(|(idx, info)| {
            if info.location.is_direct() && !info.is_categorical() {
                Some(idx)
            } else {
                None
            }
        })
    }

    /// Iterator over (feature_index, raw_slice) for all numeric features.
    ///
    /// Zero-allocation access to raw values. Use this when you don't need a
    /// contiguous matrix.
    ///
    /// # Example
    ///
    /// ```ignore
    /// for (feature_idx, raw_values) in dataset.raw_feature_iter() {
    ///     // raw_values is &[f32] with n_samples elements
    /// }
    /// ```
    pub fn raw_feature_iter(&self) -> impl Iterator<Item = (usize, &[f32])> + '_ {
        self.features.iter().enumerate().filter_map(|(idx, info)| {
            match info.location {
                FeatureLocation::Direct {
                    group_idx,
                    idx_in_group,
                } => {
                    // Only numeric features have raw values
                    self.groups[group_idx as usize]
                        .raw_slice(idx_in_group as usize)
                        .map(|slice| (idx, slice))
                }
                FeatureLocation::Bundled { .. } | FeatureLocation::Skipped => None,
            }
        })
    }

    // =========================================================================
    // Feature Value Iteration (RFC-0019)
    // =========================================================================

    /// Apply a function to each (sample_idx, raw_value) pair for a feature.
    ///
    /// This is the recommended pattern for iterating over feature values in
    /// GBLinear training/prediction and Linear SHAP. The storage type is
    /// matched **once** at the start, then we iterate directly on the
    /// underlying data—no per-iteration branching.
    ///
    /// # Performance
    ///
    /// - **Dense numeric**: Equivalent to `for (i, &v) in slice.iter().enumerate()`
    /// - **Sparse numeric**: Iterates only stored (non-zero) values
    ///
    /// # Panics
    ///
    /// - Panics if the feature is categorical (linear models don't use them)
    /// - Panics if the feature is bundled (EFB bundles are categorical)
    /// - Panics if the feature is skipped (trivial/constant)
    ///
    /// # Example
    ///
    /// ```ignore
    /// // GBLinear weight update pattern
    /// let mut sum_grad = 0.0;
    /// let mut sum_hess = 0.0;
    /// dataset.for_each_feature_value(feature_idx, |sample_idx, value| {
    ///     sum_grad += grad_hess[sample_idx].grad * value;
    ///     sum_hess += grad_hess[sample_idx].hess * value * value;
    /// });
    /// ```
    #[inline]
    pub fn for_each_feature_value<F>(&self, feature: usize, mut f: F)
    where
        F: FnMut(usize, f32),
    {
        let info = &self.features[feature];

        match info.location {
            FeatureLocation::Direct {
                group_idx,
                idx_in_group,
            } => {
                let group = &self.groups[group_idx as usize];

                // Check if this is a categorical feature (no raw values)
                if group.is_categorical() {
                    panic!(
                        "for_each_feature_value: feature {} is categorical, \
                        linear models don't use categorical features",
                        feature
                    );
                }

                // Match on storage type ONCE, then iterate directly
                match group.storage() {
                    super::FeatureStorage::Numeric(storage) => {
                        // Dense numeric: direct slice iteration (zero-cost)
                        let slice = storage.raw_slice(idx_in_group as usize, self.n_samples);
                        for (idx, &val) in slice.iter().enumerate() {
                            f(idx, val);
                        }
                    }
                    super::FeatureStorage::SparseNumeric(storage) => {
                        // Sparse numeric: iterate only stored (non-zero) values
                        for (sample_idx, _bin, raw_val) in storage.iter() {
                            f(sample_idx as usize, raw_val);
                        }
                    }
                    super::FeatureStorage::Categorical(_)
                    | super::FeatureStorage::SparseCategorical(_) => {
                        panic!(
                            "for_each_feature_value: feature {} is categorical, \
                            linear models don't use categorical features",
                            feature
                        );
                    }
                    super::FeatureStorage::Bundle(_) => {
                        panic!(
                            "for_each_feature_value: feature {} is in a bundle, \
                            bundles don't have raw values",
                            feature
                        );
                    }
                }
            }
            FeatureLocation::Bundled { .. } => {
                panic!(
                    "for_each_feature_value: feature {} is bundled, \
                    bundles don't have raw values (they're categorical)",
                    feature
                );
            }
            FeatureLocation::Skipped => {
                panic!(
                    "for_each_feature_value: feature {} was skipped (trivial/constant)",
                    feature
                );
            }
        }
    }

    /// Gather raw values for a feature at specified sample indices into a buffer.
    ///
    /// This is the recommended pattern for linear tree fitting where we need
    /// values for a subset of samples (e.g., samples that landed in a leaf).
    ///
    /// # Arguments
    ///
    /// * `feature` - The feature index
    /// * `sample_indices` - Slice of sample indices to gather (should be sorted for sparse efficiency)
    /// * `buffer` - Output buffer, must have length >= sample_indices.len()
    ///
    /// # Performance
    ///
    /// - **Dense numeric**: Simple indexed gather, O(k) where k = sample_indices.len()
    /// - **Sparse numeric**: Merge-join leveraging sorted indices, O(k + nnz)
    ///
    /// # Panics
    ///
    /// - Panics if the feature is categorical (linear trees don't use them)
    /// - Panics if the feature is bundled (EFB bundles are categorical)
    /// - Panics if buffer.len() < sample_indices.len()
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Linear tree fitting: gather values for leaf samples
    /// let leaf_samples: &[u32] = &[10, 25, 42, 100];
    /// let mut buffer = vec![0.0f32; leaf_samples.len()];
    /// dataset.gather_feature_values(feature_idx, leaf_samples, &mut buffer);
    /// // buffer now contains values at indices 10, 25, 42, 100
    /// ```
    #[inline]
    pub fn gather_feature_values(
        &self,
        feature: usize,
        sample_indices: &[u32],
        buffer: &mut [f32],
    ) {
        debug_assert!(
            buffer.len() >= sample_indices.len(),
            "buffer too small: {} < {}",
            buffer.len(),
            sample_indices.len()
        );

        let info = &self.features[feature];

        match info.location {
            FeatureLocation::Direct {
                group_idx,
                idx_in_group,
            } => {
                let group = &self.groups[group_idx as usize];

                // Check if this is a categorical feature (no raw values)
                if group.is_categorical() {
                    panic!(
                        "gather_feature_values: feature {} is categorical, \
                        linear trees don't use categorical features",
                        feature
                    );
                }

                // Match on storage type ONCE, then gather efficiently
                match group.storage() {
                    super::FeatureStorage::Numeric(storage) => {
                        // Dense numeric: simple indexed gather
                        let slice = storage.raw_slice(idx_in_group as usize, self.n_samples);
                        for (out_idx, &sample_idx) in sample_indices.iter().enumerate() {
                            buffer[out_idx] = slice[sample_idx as usize];
                        }
                    }
                    super::FeatureStorage::SparseNumeric(storage) => {
                        // Sparse numeric: merge-join using sorted indices
                        // Both sample_indices (from stable partitioning) and
                        // storage.sample_indices() are sorted
                        Self::gather_sparse_merge_join(
                            storage.sample_indices(),
                            storage.raw_values(),
                            sample_indices,
                            buffer,
                        );
                    }
                    super::FeatureStorage::Categorical(_)
                    | super::FeatureStorage::SparseCategorical(_) => {
                        panic!(
                            "gather_feature_values: feature {} is categorical, \
                            linear trees don't use categorical features",
                            feature
                        );
                    }
                    super::FeatureStorage::Bundle(_) => {
                        panic!(
                            "gather_feature_values: feature {} is in a bundle, \
                            bundles don't have raw values",
                            feature
                        );
                    }
                }
            }
            FeatureLocation::Bundled { .. } => {
                panic!(
                    "gather_feature_values: feature {} is bundled, \
                    bundles don't have raw values (they're categorical)",
                    feature
                );
            }
            FeatureLocation::Skipped => {
                // Skipped features have constant value (0.0)
                buffer[..sample_indices.len()].fill(0.0);
            }
        }
    }

    /// Merge-join for gathering values from sparse storage.
    ///
    /// Both `sparse_indices` and `sample_indices` must be sorted.
    /// For samples not in sparse storage, writes 0.0.
    #[inline]
    fn gather_sparse_merge_join(
        sparse_indices: &[u32],
        sparse_values: &[f32],
        sample_indices: &[u32],
        buffer: &mut [f32],
    ) {
        let mut sparse_pos = 0;
        let sparse_len = sparse_indices.len();

        for (out_idx, &sample_idx) in sample_indices.iter().enumerate() {
            // Advance sparse pointer until we reach or pass sample_idx
            while sparse_pos < sparse_len && sparse_indices[sparse_pos] < sample_idx {
                sparse_pos += 1;
            }

            // Check if we have a match
            if sparse_pos < sparse_len && sparse_indices[sparse_pos] == sample_idx {
                buffer[out_idx] = sparse_values[sparse_pos];
            } else {
                // Sample not in sparse storage = zero value
                buffer[out_idx] = 0.0;
            }
        }
    }

    /// Check if the dataset contains any categorical features.
    ///
    /// GBLinear requires all features to be numeric (have raw values).
    /// Use this to validate before calling GBLinear training.
    pub fn has_categorical(&self) -> bool {
        self.features.iter().any(|f| {
            matches!(f.location, FeatureLocation::Direct { group_idx, .. } if {
                self.groups[group_idx as usize].is_categorical()
            })
        })
    }

    // =========================================================================
    // Sample block iteration (for prediction)
    // =========================================================================

    /// Create a sample block iterator for efficient prediction.
    ///
    /// Buffers samples into contiguous blocks in sample-major layout,
    /// providing ~2x speedup for prediction vs random column access.
    ///
    /// # Arguments
    ///
    /// * `block_size` - Number of samples per block (default: 64)
    ///
    /// # Example
    ///
    /// ```ignore
    /// use boosters::utils::Parallelism;
    /// use boosters::data::SamplesView;
    ///
    /// let blocks = dataset.sample_blocks(64);
    /// blocks.for_each_with(Parallelism::Parallel, |start_idx, block| {
    ///     // block is ArrayView2<f32> with shape [block_size, n_features]
    ///     let samples = SamplesView::from_array(block);
    ///     // Use samples for prediction...
    /// });
    /// ```
    #[inline]
    pub fn sample_blocks(&self, block_size: usize) -> SampleBlocks<'_> {
        SampleBlocks::new(self, block_size)
    }
}

// ============================================================================
// DataAccessor Implementation
// ============================================================================

/// A view into a single sample (row) of a BinnedDataset.
///
/// Implements `SampleAccessor` to provide feature values for tree traversal
/// and linear model fitting. Returns actual raw values for numeric features.
#[derive(Clone, Copy)]
pub struct BinnedSampleView<'a> {
    dataset: &'a BinnedDataset,
    sample: usize,
}

impl<'a> BinnedSampleView<'a> {
    /// Create a new sample view.
    #[inline]
    fn new(dataset: &'a BinnedDataset, sample: usize) -> Self {
        Self { dataset, sample }
    }

    /// Get the sample index.
    #[inline]
    pub fn index(&self) -> usize {
        self.sample
    }
}

impl SampleAccessor for BinnedSampleView<'_> {
    /// Get the feature value at the given index.
    ///
    /// For numeric features, returns the actual raw value.
    /// For categorical features, returns the bin index as f32.
    /// For skipped features, returns NaN.
    #[inline]
    fn feature(&self, feature: usize) -> f32 {
        // Try to get raw value first (for numeric features)
        if let Some(raw) = self.dataset.raw_value(self.sample, feature) {
            return raw;
        }

        // Fall back to bin value for categorical features
        let location = self.dataset.features[feature].location;
        match location {
            FeatureLocation::Direct {
                group_idx,
                idx_in_group,
            } => {
                // Categorical feature - return bin as f32
                self.dataset.groups[group_idx as usize].bin(self.sample, idx_in_group as usize)
                    as f32
            }
            FeatureLocation::Bundled { .. } => {
                // Bundled features don't have raw values - return NaN for now
                // (bundled features shouldn't be used for linear regression)
                f32::NAN
            }
            FeatureLocation::Skipped => {
                // Skipped features are constant/trivial - return NaN
                f32::NAN
            }
        }
    }

    #[inline]
    fn n_features(&self) -> usize {
        self.dataset.n_features()
    }
}

impl DataAccessor for BinnedDataset {
    type Sample<'a>
        = BinnedSampleView<'a>
    where
        Self: 'a;

    #[inline]
    fn sample(&self, index: usize) -> Self::Sample<'_> {
        BinnedSampleView::new(self, index)
    }

    #[inline]
    fn n_samples(&self) -> usize {
        self.n_samples
    }

    #[inline]
    fn n_features(&self) -> usize {
        self.features.len()
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

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::binned::builder::DatasetBuilder;
    use crate::data::binned::feature_analysis::BinningConfig;
    use ndarray::{array, Array2};

    fn make_array(values: &[f32], rows: usize, cols: usize) -> Array2<f32> {
        Array2::from_shape_vec((rows, cols), values.to_vec()).unwrap()
    }

    #[test]
    fn test_create_from_built_groups() {
        // Use floats to ensure numeric detection
        let data = make_array(&[1.5, 2.5, 3.5, 4.5, 5.5], 5, 1);
        let config = BinningConfig::default();

        let built = DatasetBuilder::from_array(data.view(), &config)
            .unwrap()
            .build_groups()
            .unwrap();

        let dataset = BinnedDataset::from_built_groups(built);

        assert_eq!(dataset.n_samples(), 5);
        assert_eq!(dataset.n_features(), 1);
        assert_eq!(dataset.n_groups(), 1);
        assert!(dataset.has_raw_values());
    }

    #[test]
    fn test_feature_location() {
        let data = make_array(
            &[1.1, 2.2, 3.3, 10.1, 20.2, 30.3],
            3,
            2,
        );
        let config = BinningConfig::default();

        let built = DatasetBuilder::from_array(data.view(), &config)
            .unwrap()
            .build_groups()
            .unwrap();

        let dataset = BinnedDataset::from_built_groups(built);

        // Both features should be direct (not skipped or bundled)
        assert!(dataset.feature_location(0).is_direct());
        assert!(dataset.feature_location(1).is_direct());

        // Both should be in the same group (numeric dense)
        if let FeatureLocation::Direct { group_idx: g0, .. } = dataset.feature_location(0) {
            if let FeatureLocation::Direct { group_idx: g1, .. } = dataset.feature_location(1) {
                assert_eq!(g0, g1);
            }
        }
    }

    #[test]
    fn test_global_bin_offsets() {
        let data = make_array(&[1.1, 2.2, 3.3, 4.4, 5.5], 5, 1);
        let config = BinningConfig::builder().max_bins(10).build();

        let built = DatasetBuilder::from_array(data.view(), &config)
            .unwrap()
            .build_groups()
            .unwrap();

        let dataset = BinnedDataset::from_built_groups(built);

        // Feature 0 should start at offset 0
        assert_eq!(dataset.global_bin_offset(0), 0);
        // Total bins should equal the number of bins for feature 0
        assert_eq!(dataset.total_bins(), dataset.n_bins(0));
    }

    #[test]
    fn test_labels_and_weights() {
        let data = make_array(&[1.1, 2.2, 3.3, 4.4, 5.5], 5, 1);
        let labels = array![0.0, 1.0, 0.0, 1.0, 0.0];
        let weights = array![1.0, 2.0, 1.0, 2.0, 1.0];
        let config = BinningConfig::default();

        let built = DatasetBuilder::from_array(data.view(), &config)
            .unwrap()
            .set_labels(labels.view())
            .set_weights(weights.view())
            .build_groups()
            .unwrap();

        let dataset = BinnedDataset::from_built_groups(built);

        assert!(dataset.has_labels());
        assert!(dataset.has_weights());
        assert_eq!(dataset.labels().unwrap(), &[0.0, 1.0, 0.0, 1.0, 0.0]);
        assert_eq!(dataset.weights().unwrap(), &[1.0, 2.0, 1.0, 2.0, 1.0]);
    }

    #[test]
    fn test_feature_is_categorical() {
        // Mix of numeric and categorical
        let built = DatasetBuilder::new()
            .add_numeric("x", array![1.1, 2.2, 3.3, 4.4, 5.5].view())
            .add_categorical("y", array![0.0, 1.0, 2.0, 1.0, 0.0].view())
            .build_groups()
            .unwrap();

        let dataset = BinnedDataset::from_built_groups(built);

        assert!(!dataset.is_categorical(0)); // Numeric
        assert!(dataset.is_categorical(1)); // Categorical
    }

    #[test]
    fn test_bin_access() {
        // Create a simple dataset with known values
        let data = make_array(&[1.1, 2.2, 3.3, 4.4, 5.5], 5, 1);
        let config = BinningConfig::builder().max_bins(5).build();

        let built = DatasetBuilder::from_array(data.view(), &config)
            .unwrap()
            .build_groups()
            .unwrap();

        let dataset = BinnedDataset::from_built_groups(built);

        // Bin values should be 0..n_bins-1 for evenly spaced values
        // With 5 unique values and max_bins=5, each value should map to its own bin
        for sample in 0..5 {
            let bin = dataset.bin(sample, 0);
            assert!(bin < dataset.n_bins(0));
        }
    }

    #[test]
    fn test_raw_value_access() {
        let data = make_array(&[1.5, 2.5, 3.5, 4.5, 5.5], 5, 1);
        let config = BinningConfig::default();

        let built = DatasetBuilder::from_array(data.view(), &config)
            .unwrap()
            .build_groups()
            .unwrap();

        let dataset = BinnedDataset::from_built_groups(built);

        // Raw values should be preserved exactly
        assert_eq!(dataset.raw_value(0, 0), Some(1.5));
        assert_eq!(dataset.raw_value(1, 0), Some(2.5));
        assert_eq!(dataset.raw_value(2, 0), Some(3.5));
        assert_eq!(dataset.raw_value(3, 0), Some(4.5));
        assert_eq!(dataset.raw_value(4, 0), Some(5.5));
    }

    #[test]
    fn test_raw_feature_slice() {
        let data = make_array(&[1.5, 2.5, 3.5, 4.5, 5.5], 5, 1);
        let config = BinningConfig::default();

        let built = DatasetBuilder::from_array(data.view(), &config)
            .unwrap()
            .build_groups()
            .unwrap();

        let dataset = BinnedDataset::from_built_groups(built);

        // Get the raw slice for the feature
        let slice = dataset.raw_feature_slice(0);
        assert!(slice.is_some());
        assert_eq!(slice.unwrap(), &[1.5, 2.5, 3.5, 4.5, 5.5]);
    }

    #[test]
    fn test_categorical_no_raw_values() {
        let built = DatasetBuilder::new()
            .add_categorical("cat", array![0.0, 1.0, 2.0, 1.0, 0.0].view())
            .build_groups()
            .unwrap();

        let dataset = BinnedDataset::from_built_groups(built);

        // Categorical features don't have raw values
        assert_eq!(dataset.raw_value(0, 0), None);
        assert_eq!(dataset.raw_feature_slice(0), None);
    }

    #[test]
    fn test_feature_views_count() {
        let data = make_array(
            &[1.1, 2.2, 3.3, 10.1, 20.2, 30.3],
            3,
            2,
        );
        let config = BinningConfig::default();

        let built = DatasetBuilder::from_array(data.view(), &config)
            .unwrap()
            .build_groups()
            .unwrap();

        let dataset = BinnedDataset::from_built_groups(built);
        let views = dataset.feature_views();

        // Should have exactly 2 views (one per non-trivial feature)
        assert_eq!(views.len(), 2);
    }

    #[test]
    fn test_feature_views_dense() {
        let data = make_array(&[1.1, 2.2, 3.3, 4.4, 5.5], 5, 1);
        let config = BinningConfig::default();

        let built = DatasetBuilder::from_array(data.view(), &config)
            .unwrap()
            .build_groups()
            .unwrap();

        let dataset = BinnedDataset::from_built_groups(built);
        let views = dataset.feature_views();

        assert_eq!(views.len(), 1);
        assert!(views[0].is_dense());
        assert_eq!(views[0].len(), 5); // 5 samples
    }

    #[test]
    fn test_original_feature_view() {
        let built = DatasetBuilder::new()
            .add_numeric("x", array![1.1, 2.2, 3.3, 4.4, 5.5].view())
            .add_numeric("y", array![10.1, 20.2, 30.3, 40.4, 50.5].view())
            .build_groups()
            .unwrap();

        let dataset = BinnedDataset::from_built_groups(built);

        // Get views for individual features
        let view0 = dataset.original_feature_view(0);
        let view1 = dataset.original_feature_view(1);

        assert!(view0.is_dense());
        assert!(view1.is_dense());
        assert_eq!(view0.len(), 5);
        assert_eq!(view1.len(), 5);
    }

    #[test]
    fn test_mixed_feature_views() {
        let built = DatasetBuilder::new()
            .add_numeric("num", array![1.1, 2.2, 3.3, 4.4, 5.5].view())
            .add_categorical("cat", array![0.0, 1.0, 2.0, 1.0, 0.0].view())
            .build_groups()
            .unwrap();

        let dataset = BinnedDataset::from_built_groups(built);
        let views = dataset.feature_views();

        // Should have 2 views (one numeric, one categorical)
        assert_eq!(views.len(), 2);

        // Both should be dense (not sparse)
        for view in &views {
            assert!(view.is_dense());
            assert_eq!(view.len(), 5);
        }
    }

    #[test]
    fn test_numeric_feature_indices() {
        let built = DatasetBuilder::new()
            .add_numeric("num1", array![1.1, 2.2, 3.3, 4.4, 5.5].view())
            .add_categorical("cat", array![0.0, 1.0, 2.0, 1.0, 0.0].view())
            .add_numeric("num2", array![10.1, 20.2, 30.3, 40.4, 50.5].view())
            .build_groups()
            .unwrap();

        let dataset = BinnedDataset::from_built_groups(built);
        let indices: Vec<_> = dataset.numeric_feature_indices().collect();

        // Features 0 and 2 are numeric
        assert_eq!(indices, vec![0, 2]);
    }

    #[test]
    fn test_raw_feature_iter() {
        let built = DatasetBuilder::new()
            .add_numeric("x", array![1.5, 2.5, 3.5, 4.5, 5.5].view())
            .add_categorical("cat", array![0.0, 1.0, 2.0, 1.0, 0.0].view())
            .add_numeric("y", array![10.5, 20.5, 30.5, 40.5, 50.5].view())
            .build_groups()
            .unwrap();

        let dataset = BinnedDataset::from_built_groups(built);

        // Collect raw feature iterator results
        let raw_features: Vec<_> = dataset.raw_feature_iter().collect();

        // Should have 2 numeric features with raw values
        assert_eq!(raw_features.len(), 2);

        // Feature 0 (numeric)
        assert_eq!(raw_features[0].0, 0);
        assert_eq!(raw_features[0].1, &[1.5, 2.5, 3.5, 4.5, 5.5]);

        // Feature 2 (numeric)
        assert_eq!(raw_features[1].0, 2);
        assert_eq!(raw_features[1].1, &[10.5, 20.5, 30.5, 40.5, 50.5]);
    }

    #[test]
    fn test_raw_feature_iter_all_categorical() {
        let built = DatasetBuilder::new()
            .add_categorical("cat1", array![0.0, 1.0, 2.0, 1.0, 0.0].view())
            .add_categorical("cat2", array![1.0, 0.0, 1.0, 0.0, 1.0].view())
            .build_groups()
            .unwrap();

        let dataset = BinnedDataset::from_built_groups(built);

        // No numeric features, so raw_feature_iter should be empty
        let raw_features: Vec<_> = dataset.raw_feature_iter().collect();
        assert!(raw_features.is_empty());
    }

    // ========================================================================
    // DataAccessor Tests
    // ========================================================================

    #[test]
    fn test_data_accessor_numeric() {
        use crate::data::DataAccessor;
        use crate::data::SampleAccessor;

        // Create dataset with numeric features
        // Use non-integer values to avoid auto-categorical detection
        let built = DatasetBuilder::new()
            .add_numeric("x", array![1.5, 2.5, 3.5, 4.5, 5.5].view())
            .add_numeric("y", array![10.1, 20.2, 30.3, 40.4, 50.5].view())
            .build_groups()
            .unwrap();

        let dataset = BinnedDataset::from_built_groups(built);

        // Use DataAccessor trait
        assert_eq!(dataset.n_samples(), 5);
        assert_eq!(dataset.n_features(), 2);

        // Both features should be numeric (not auto-detected as categorical)
        assert!(!dataset.is_categorical(0));
        assert!(!dataset.is_categorical(1));

        // Check raw values via SampleAccessor
        let sample = dataset.sample(0);
        assert_eq!(sample.n_features(), 2);
        // Raw values should be preserved (not bin midpoints)
        assert!((sample.feature(0) - 1.5).abs() < 0.01, "feature(0) = {}", sample.feature(0));
        assert!((sample.feature(1) - 10.1).abs() < 0.01, "feature(1) = {}", sample.feature(1));

        // Check other samples
        let sample2 = dataset.sample(2);
        assert!((sample2.feature(0) - 3.5).abs() < 0.01);
        assert!((sample2.feature(1) - 30.3).abs() < 0.01);

        let sample4 = dataset.sample(4);
        assert!((sample4.feature(0) - 5.5).abs() < 0.01);
        assert!((sample4.feature(1) - 50.5).abs() < 0.01);
    }

    #[test]
    fn test_data_accessor_categorical() {
        use crate::data::DataAccessor;
        use crate::data::SampleAccessor;

        // Create dataset with categorical features
        let built = DatasetBuilder::new()
            .add_categorical("cat", array![0.0, 1.0, 2.0, 1.0, 0.0].view())
            .build_groups()
            .unwrap();

        let dataset = BinnedDataset::from_built_groups(built);

        // Categorical returns bin as f32
        let sample0 = dataset.sample(0);
        let sample1 = dataset.sample(1);
        let sample2 = dataset.sample(2);

        // Categories should be preserved as bin indices
        // The exact bin depends on how categoricals are encoded, but
        // they should be small integers
        assert!(sample0.feature(0) >= 0.0);
        assert!(sample1.feature(0) >= 0.0);
        assert!(sample2.feature(0) >= 0.0);

        // Feature type should be categorical
        assert!(dataset.has_categorical());
        assert_eq!(dataset.feature_type(0), FeatureType::Categorical);
    }

    #[test]
    fn test_data_accessor_mixed() {
        use crate::data::DataAccessor;
        use crate::data::SampleAccessor;

        // Create dataset with mixed numeric and categorical
        // Use non-integer values for numeric to avoid auto-categorical detection
        let built = DatasetBuilder::new()
            .add_numeric("num", array![1.1, 2.2, 3.3, 4.4, 5.5].view())
            .add_categorical("cat", array![0.0, 1.0, 0.0, 1.0, 0.0].view())
            .build_groups()
            .unwrap();

        let dataset = BinnedDataset::from_built_groups(built);

        assert_eq!(dataset.n_features(), 2);
        assert!(dataset.has_categorical());

        // Numeric feature should return raw value
        let sample = dataset.sample(2);
        assert!((sample.feature(0) - 3.3).abs() < 0.01, "feature(0) = {}", sample.feature(0));

        // Feature types
        assert_eq!(dataset.feature_type(0), FeatureType::Numeric);
        assert_eq!(dataset.feature_type(1), FeatureType::Categorical);
    }

    #[test]
    fn test_decode_split_to_original_with_bundle() {
        use crate::data::{BinnedDatasetBuilder, BinningConfig, FeatureMetadata};

        // Create two sparse categorical features that will be bundled
        // Feature 0: non-zero only for rows 0-4
        // Feature 1: non-zero only for rows 50-54
        let n_samples = 100;

        let mut data_vec = Vec::with_capacity(n_samples * 2);
        for sample in 0..n_samples {
            let f0 = if sample < 5 {
                (sample % 3 + 1) as f32
            } else {
                0.0
            };
            let f1 = if (50..55).contains(&sample) {
                ((sample - 50) % 3 + 1) as f32
            } else {
                0.0
            };
            data_vec.push(f0);
            data_vec.push(f1);
        }

        let data = ndarray::Array2::from_shape_vec((n_samples, 2), data_vec).unwrap();
        let metadata = FeatureMetadata::default().categorical(vec![0, 1]);
        let config = BinningConfig::builder()
            .enable_bundling(true)
            .sparsity_threshold(0.9)
            .build();

        let dataset = BinnedDatasetBuilder::from_array_with_metadata(data.view(), Some(&metadata), &config)
            .unwrap()
            .build()
            .unwrap();

        let effective = dataset.effective_feature_views();

        // Verify bundling occurred
        assert_eq!(effective.n_bundles, 1, "Should have 1 bundle");
        assert_eq!(effective.n_columns(), 1, "Should have 1 effective column");

        // Test decode_split_to_original for the bundle column
        // Bin 0 should return the first feature with bin 0
        let (feature, bin) = dataset.decode_split_to_original(&effective, 0, 0);
        assert_eq!(feature, 0, "Bin 0 should map to first feature");
        assert_eq!(bin, 0, "Bin 0 should decode to original bin 0");

        // A non-zero bin should decode to the original feature that owns it
        // The exact bin depends on how the bundle encodes values
        // For a bin from the first feature (bins 1-4 typically):
        let (feature1, _bin1) = dataset.decode_split_to_original(&effective, 0, 1);
        assert!(
            feature1 == 0 || feature1 == 1,
            "Bin 1 should map to feature 0 or 1, got {}",
            feature1
        );

        // For a higher bin (likely from second feature):
        // Feature 0 has ~4 bins (categories 1,2,3 + default), so bins 5+ should be feature 1
        let (feature5, _bin5) = dataset.decode_split_to_original(&effective, 0, 5);
        // bin 5 is in feature 1's range (offset after feature 0's bins)
        assert!(
            feature5 == 0 || feature5 == 1,
            "Bin 5 should map to a feature in the bundle"
        );
    }

    // ========================================================================
    // for_each_feature_value() Tests
    // ========================================================================

    #[test]
    fn test_for_each_feature_value_dense_numeric() {
        // Create a dataset with known numeric values
        let built = DatasetBuilder::new()
            .add_numeric("x", array![1.5, 2.5, 3.5, 4.5, 5.5].view())
            .add_numeric("y", array![10.0, 20.0, 30.0, 40.0, 50.0].view())
            .build_groups()
            .unwrap();

        let dataset = BinnedDataset::from_built_groups(built);

        // Collect values from feature 0
        let mut values_0 = Vec::new();
        dataset.for_each_feature_value(0, |idx, val| {
            values_0.push((idx, val));
        });

        // Should iterate all 5 samples in order
        assert_eq!(values_0.len(), 5);
        assert_eq!(values_0[0], (0, 1.5));
        assert_eq!(values_0[1], (1, 2.5));
        assert_eq!(values_0[2], (2, 3.5));
        assert_eq!(values_0[3], (3, 4.5));
        assert_eq!(values_0[4], (4, 5.5));

        // Collect values from feature 1
        let mut values_1 = Vec::new();
        dataset.for_each_feature_value(1, |idx, val| {
            values_1.push((idx, val));
        });

        assert_eq!(values_1.len(), 5);
        assert_eq!(values_1[0], (0, 10.0));
        assert_eq!(values_1[4], (4, 50.0));
    }

    #[test]
    fn test_for_each_feature_value_accumulation() {
        // Simulate GBLinear weight update pattern
        let built = DatasetBuilder::new()
            .add_numeric("x", array![1.0, 2.0, 3.0, 4.0, 5.0].view())
            .build_groups()
            .unwrap();

        let dataset = BinnedDataset::from_built_groups(built);

        // Compute sum of values (like sum_grad for unit gradients)
        let mut sum = 0.0f64;
        dataset.for_each_feature_value(0, |_idx, val| {
            sum += val as f64;
        });

        assert!((sum - 15.0).abs() < 1e-6, "sum = {}", sum);

        // Compute sum of squared values (like sum_hess for unit hessians)
        let mut sum_sq = 0.0f64;
        dataset.for_each_feature_value(0, |_idx, val| {
            sum_sq += (val * val) as f64;
        });

        assert!((sum_sq - 55.0).abs() < 1e-6, "sum_sq = {}", sum_sq);
    }

    #[test]
    fn test_for_each_feature_value_mixed_dataset() {
        // Dataset with numeric and categorical - only iterate numeric
        let built = DatasetBuilder::new()
            .add_numeric("num", array![1.1, 2.2, 3.3].view())
            .add_categorical("cat", array![0.0, 1.0, 2.0].view())
            .build_groups()
            .unwrap();

        let dataset = BinnedDataset::from_built_groups(built);

        // Numeric feature should work
        let mut values = Vec::new();
        dataset.for_each_feature_value(0, |idx, val| {
            values.push((idx, val));
        });
        assert_eq!(values.len(), 3);
        assert!((values[0].1 - 1.1).abs() < 0.01);
    }

    #[test]
    #[should_panic(expected = "categorical")]
    fn test_for_each_feature_value_panics_on_categorical() {
        let built = DatasetBuilder::new()
            .add_categorical("cat", array![0.0, 1.0, 2.0].view())
            .build_groups()
            .unwrap();

        let dataset = BinnedDataset::from_built_groups(built);

        // Should panic when trying to iterate a categorical feature
        dataset.for_each_feature_value(0, |_idx, _val| {});
    }

    #[test]
    fn test_for_each_feature_value_nan_handling() {
        // NaN values should be passed through
        let built = DatasetBuilder::new()
            .add_numeric("x", array![1.0, f32::NAN, 3.0].view())
            .build_groups()
            .unwrap();

        let dataset = BinnedDataset::from_built_groups(built);

        let mut values = Vec::new();
        dataset.for_each_feature_value(0, |idx, val| {
            values.push((idx, val));
        });

        assert_eq!(values.len(), 3);
        assert_eq!(values[0], (0, 1.0));
        assert!(values[1].1.is_nan(), "Expected NaN at index 1");
        assert_eq!(values[2], (2, 3.0));
    }

    #[test]
    fn test_for_each_feature_value_single_sample() {
        let built = DatasetBuilder::new()
            .add_numeric("x", array![42.0].view())
            .build_groups()
            .unwrap();

        let dataset = BinnedDataset::from_built_groups(built);

        let mut values = Vec::new();
        dataset.for_each_feature_value(0, |idx, val| {
            values.push((idx, val));
        });

        assert_eq!(values, vec![(0, 42.0)]);
    }

    #[test]
    fn test_for_each_feature_value_sparse_numeric() {
        // Create sparse data: mostly zeros with a few non-zero values
        // Need >90% zeros with sparsity_threshold(0.9)
        let mut values = vec![0.0f32; 100];
        values[5] = 1.5;
        values[25] = 2.7;
        values[75] = 3.9;
        // 97% zeros = sparse

        let data = ndarray::Array2::from_shape_vec((100, 1), values).unwrap();
        let config = BinningConfig::builder().sparsity_threshold(0.9).build();

        let dataset = DatasetBuilder::from_array(data.view(), &config)
            .unwrap()
            .build()
            .unwrap();

        // Verify it's actually sparse storage
        let groups = &dataset.groups;
        assert!(
            matches!(
                groups[0].storage(),
                crate::data::binned::FeatureStorage::SparseNumeric(_)
            ),
            "Expected sparse numeric storage"
        );

        // Collect values using for_each_feature_value
        let mut collected = Vec::new();
        dataset.for_each_feature_value(0, |idx, val| {
            collected.push((idx, val));
        });

        // Should only yield the 3 non-zero values (sparse iteration skips zeros)
        assert_eq!(collected.len(), 3, "Should have 3 non-zero entries");

        // Verify correct values at correct indices
        assert!(collected.iter().any(|&(i, v)| i == 5 && (v - 1.5).abs() < 0.01));
        assert!(collected.iter().any(|&(i, v)| i == 25 && (v - 2.7).abs() < 0.01));
        assert!(collected.iter().any(|&(i, v)| i == 75 && (v - 3.9).abs() < 0.01));
    }

    #[test]
    #[should_panic(expected = "bundled")]
    fn test_for_each_feature_value_panics_on_bundled() {
        // Create two sparse categorical features that will be bundled
        let n_samples = 100;
        let mut data_vec = Vec::with_capacity(n_samples * 2);

        for i in 0..n_samples {
            let f0 = if i < 10 { (i % 3 + 1) as f32 } else { 0.0 };
            let f1 = if i >= 90 { ((i - 90) % 3 + 1) as f32 } else { 0.0 };
            data_vec.push(f0);
            data_vec.push(f1);
        }

        let data = ndarray::Array2::from_shape_vec((n_samples, 2), data_vec).unwrap();
        let metadata = crate::data::FeatureMetadata::default().categorical(vec![0, 1]);
        let config = BinningConfig::builder()
            .enable_bundling(true)
            .sparsity_threshold(0.9)
            .build();

        let dataset = crate::data::BinnedDatasetBuilder::from_array_with_metadata(
            data.view(),
            Some(&metadata),
            &config,
        )
        .unwrap()
        .build()
        .unwrap();

        // Should panic when trying to iterate a bundled feature
        dataset.for_each_feature_value(0, |_idx, _val| {});
    }

    // ========================================================================
    // gather_feature_values() Tests
    // ========================================================================

    #[test]
    fn test_gather_feature_values_dense_numeric() {
        let built = DatasetBuilder::new()
            .add_numeric("x", array![10.0, 20.0, 30.0, 40.0, 50.0].view())
            .build_groups()
            .unwrap();

        let dataset = BinnedDataset::from_built_groups(built);

        // Gather subset of samples
        let indices = [1u32, 3, 4];
        let mut buffer = vec![0.0f32; indices.len()];
        dataset.gather_feature_values(0, &indices, &mut buffer);

        assert_eq!(buffer, vec![20.0, 40.0, 50.0]);
    }

    #[test]
    fn test_gather_feature_values_dense_all_samples() {
        let built = DatasetBuilder::new()
            .add_numeric("x", array![1.0, 2.0, 3.0, 4.0].view())
            .build_groups()
            .unwrap();

        let dataset = BinnedDataset::from_built_groups(built);

        let indices = [0u32, 1, 2, 3];
        let mut buffer = vec![0.0f32; 4];
        dataset.gather_feature_values(0, &indices, &mut buffer);

        assert_eq!(buffer, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_gather_feature_values_sparse_numeric() {
        // Create sparse data: mostly zeros with a few non-zero values
        let mut values = vec![0.0f32; 100];
        values[10] = 1.5;
        values[30] = 2.5;
        values[50] = 3.5;
        values[70] = 4.5;
        values[90] = 5.5;

        let data = ndarray::Array2::from_shape_vec((100, 1), values).unwrap();
        let config = BinningConfig::builder().sparsity_threshold(0.9).build();

        let dataset = DatasetBuilder::from_array(data.view(), &config)
            .unwrap()
            .build()
            .unwrap();

        // Verify sparse storage
        assert!(matches!(
            dataset.groups[0].storage(),
            crate::data::binned::FeatureStorage::SparseNumeric(_)
        ));

        // Gather mix of sparse and zero samples
        let indices = [5u32, 10, 30, 45, 70]; // 5,45 are zeros; 10,30,70 are non-zero
        let mut buffer = vec![-1.0f32; indices.len()]; // Pre-fill to verify zeros are written
        dataset.gather_feature_values(0, &indices, &mut buffer);

        // Check: 5->0.0, 10->1.5, 30->2.5, 45->0.0, 70->4.5
        assert!((buffer[0] - 0.0).abs() < 0.01, "idx 5 should be 0.0");
        assert!((buffer[1] - 1.5).abs() < 0.01, "idx 10 should be 1.5");
        assert!((buffer[2] - 2.5).abs() < 0.01, "idx 30 should be 2.5");
        assert!((buffer[3] - 0.0).abs() < 0.01, "idx 45 should be 0.0");
        assert!((buffer[4] - 4.5).abs() < 0.01, "idx 70 should be 4.5");
    }

    #[test]
    fn test_gather_feature_values_empty_indices() {
        let built = DatasetBuilder::new()
            .add_numeric("x", array![1.0, 2.0, 3.0].view())
            .build_groups()
            .unwrap();

        let dataset = BinnedDataset::from_built_groups(built);

        let indices: [u32; 0] = [];
        let mut buffer: Vec<f32> = vec![];
        dataset.gather_feature_values(0, &indices, &mut buffer);
        // Should not panic, just do nothing
    }

    #[test]
    #[should_panic(expected = "categorical")]
    fn test_gather_feature_values_panics_on_categorical() {
        let built = DatasetBuilder::new()
            .add_categorical("cat", array![0.0, 1.0, 2.0].view())
            .build_groups()
            .unwrap();

        let dataset = BinnedDataset::from_built_groups(built);

        let indices = [0u32, 1];
        let mut buffer = vec![0.0f32; 2];
        dataset.gather_feature_values(0, &indices, &mut buffer);
    }

    #[test]
    fn test_gather_feature_values_single_sample() {
        let built = DatasetBuilder::new()
            .add_numeric("x", array![42.0, 99.0, 17.0].view())
            .build_groups()
            .unwrap();

        let dataset = BinnedDataset::from_built_groups(built);

        let indices = [1u32];
        let mut buffer = vec![0.0f32; 1];
        dataset.gather_feature_values(0, &indices, &mut buffer);

        assert_eq!(buffer, vec![99.0]);
    }

    #[test]
    fn test_gather_sparse_merge_join_algorithm() {
        // Direct test of the merge-join helper function
        let sparse_indices = [10u32, 20, 30, 50, 70];
        let sparse_values = [1.0f32, 2.0, 3.0, 5.0, 7.0];

        // Query indices that partially overlap with sparse
        let sample_indices = [5u32, 10, 25, 30, 40, 70, 80];
        let mut buffer = vec![-1.0f32; sample_indices.len()];

        BinnedDataset::gather_sparse_merge_join(
            &sparse_indices,
            &sparse_values,
            &sample_indices,
            &mut buffer,
        );

        // Expected: 5->0.0, 10->1.0, 25->0.0, 30->3.0, 40->0.0, 70->7.0, 80->0.0
        assert_eq!(buffer[0], 0.0); // 5 not in sparse
        assert_eq!(buffer[1], 1.0); // 10 in sparse
        assert_eq!(buffer[2], 0.0); // 25 not in sparse
        assert_eq!(buffer[3], 3.0); // 30 in sparse
        assert_eq!(buffer[4], 0.0); // 40 not in sparse
        assert_eq!(buffer[5], 7.0); // 70 in sparse
        assert_eq!(buffer[6], 0.0); // 80 not in sparse
    }
}

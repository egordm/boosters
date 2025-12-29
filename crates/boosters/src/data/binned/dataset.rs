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
use super::view::FeatureView;

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

        // Map from global feature index to (group_idx, idx_in_group)
        // First, build a reverse map from the groups
        let mut location_map: Vec<Option<(u32, u32)>> = vec![None; n_features];

        for (group_idx, group) in built.groups.iter().enumerate() {
            for (idx_in_group, &global_idx) in group.feature_indices().iter().enumerate() {
                location_map[global_idx as usize] = Some((group_idx as u32, idx_in_group as u32));
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
            } else if let Some((group_idx, idx_in_group)) = location_map[feature_idx] {
                FeatureLocation::Direct {
                    group_idx,
                    idx_in_group,
                }
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
}

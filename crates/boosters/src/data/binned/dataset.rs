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
    // Linear trees support
    // =========================================================================

    /// Check if any feature has raw values (for linear trees).
    /// True if there's at least one numeric group.
    pub fn has_raw_values(&self) -> bool {
        self.groups.iter().any(|g| g.has_raw_values())
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
}

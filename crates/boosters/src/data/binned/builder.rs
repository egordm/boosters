//! BinnedDatasetBuilder - constructs BinnedDataset from raw data.
//!
//! This module provides the `DatasetBuilder` which is the primary way to create
//! a `BinnedDataset` from raw feature data. It supports two usage patterns:
//!
//! 1. **Batch ingestion** (recommended): Provide a full matrix, auto-detect types
//! 2. **Single-feature**: Add features one-by-one for fine-grained control
//!
//! # Example: Batch Ingestion
//!
//! ```ignore
//! use boosters::data::{DatasetBuilder, BinningConfig};
//! use ndarray::Array2;
//!
//! let data: Array2<f32> = load_data();
//! let dataset = DatasetBuilder::from_array(data.view(), &BinningConfig::default())?
//!     .set_labels(labels.view())
//!     .build()?;
//! ```

#![allow(dead_code)] // During migration

use ndarray::{ArrayView1, ArrayView2};
use std::collections::HashMap;

use super::bin_mapper::BinMapper;
use super::feature_analysis::{
    analyze_features, compute_groups, BinningConfig, FeatureAnalysis, FeatureMetadata, GroupSpec,
    GroupType, GroupingResult,
};
use super::group::FeatureGroup;
use super::storage::{CategoricalStorage, NumericStorage, SparseCategoricalStorage, SparseNumericStorage};
use super::{BinData, FeatureStorage};
use crate::data::FeaturesView;

/// Errors that can occur during dataset construction.
#[derive(Debug, Clone)]
pub enum DatasetError {
    /// Feature count mismatch
    FeatureCountMismatch { expected: usize, got: usize },
    /// Sample count mismatch
    SampleCountMismatch { expected: usize, got: usize },
    /// Invalid feature index
    InvalidFeatureIndex(usize),
    /// Empty dataset
    EmptyDataset,
    /// Binning error
    BinningError(String),
}

impl std::fmt::Display for DatasetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DatasetError::FeatureCountMismatch { expected, got } => {
                write!(f, "Feature count mismatch: expected {expected}, got {got}")
            }
            DatasetError::SampleCountMismatch { expected, got } => {
                write!(f, "Sample count mismatch: expected {expected}, got {got}")
            }
            DatasetError::InvalidFeatureIndex(idx) => {
                write!(f, "Invalid feature index: {idx}")
            }
            DatasetError::EmptyDataset => write!(f, "Cannot build empty dataset"),
            DatasetError::BinningError(msg) => write!(f, "Binning error: {msg}"),
        }
    }
}

impl std::error::Error for DatasetError {}

/// Per-feature info stored in the builder.
#[derive(Clone)]
struct PendingFeature {
    /// Feature name (may be empty)
    name: String,
    /// Raw values for this feature
    values: Vec<f32>,
    /// Is categorical (user-specified or auto-detected)
    is_categorical: bool,
    /// Optional max_bins override
    max_bins: Option<u32>,
}

/// Builder for constructing `BinnedDataset` from raw data.
///
/// Two usage patterns:
/// 1. Batch: `DatasetBuilder::from_array(data, config)?.set_labels(labels).build()?`
/// 2. Single: `DatasetBuilder::new().add_numeric("x", values).build()?`
#[derive(Clone)]
pub struct DatasetBuilder {
    /// Features added so far (single-feature API)
    pending_features: Vec<PendingFeature>,
    /// Pre-computed analyses (batch API)
    analyses: Option<Vec<FeatureAnalysis>>,
    /// Pre-computed grouping (batch API)
    grouping: Option<GroupingResult>,
    /// Pre-computed bin mappers (batch API)
    bin_mappers: Option<Vec<BinMapper>>,
    /// Raw feature data (batch API) - shape (n_samples, n_features)
    batch_data: Option<Vec<Vec<f32>>>, // Column-major: [feature][sample]
    /// Labels for the dataset
    labels: Option<Vec<f32>>,
    /// Sample weights
    weights: Option<Vec<f32>>,
    /// Configuration used for building
    config: BinningConfig,
    /// Number of samples (if known)
    n_samples: Option<usize>,
}

impl Default for DatasetBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl DatasetBuilder {
    /// Create a new empty builder.
    pub fn new() -> Self {
        Self {
            pending_features: Vec::new(),
            analyses: None,
            grouping: None,
            bin_mappers: None,
            batch_data: None,
            labels: None,
            weights: None,
            config: BinningConfig::default(),
            n_samples: None,
        }
    }

    /// Create builder from a 2D array with auto-detection.
    ///
    /// Analyzes all features to detect:
    /// - Numeric vs categorical features
    /// - Optimal bin widths (U8 vs U16)
    /// - Sparsity patterns
    ///
    /// Groups features homogeneously for optimal storage.
    pub fn from_array(
        data: ArrayView2<f32>,
        config: &BinningConfig,
    ) -> Result<Self, DatasetError> {
        Self::from_array_with_metadata(data, None, config)
    }

    /// Create builder from array with optional feature metadata.
    ///
    /// User-provided metadata takes precedence over auto-detection:
    /// - Feature names
    /// - Which columns are categorical vs numeric
    /// - Per-feature max_bins overrides
    pub fn from_array_with_metadata(
        data: ArrayView2<f32>,
        metadata: Option<&FeatureMetadata>,
        config: &BinningConfig,
    ) -> Result<Self, DatasetError> {
        let (n_samples, n_features) = data.dim();
        if n_samples == 0 || n_features == 0 {
            return Err(DatasetError::EmptyDataset);
        }

        // Store data in column-major format (feature-major): [feature][sample]
        // This matches the expected layout for FeaturesView
        let batch_data: Vec<Vec<f32>> = (0..n_features)
            .map(|f| data.column(f).iter().copied().collect())
            .collect();

        // Create flat feature-major slice for FeaturesView
        let flat_feature_major: Vec<f32> = batch_data.iter().flatten().copied().collect();
        let features_view = FeaturesView::from_slice(&flat_feature_major, n_samples, n_features)
            .ok_or_else(|| DatasetError::BinningError("Failed to create features view".into()))?;

        // Analyze features using parallel analysis
        let analyses = analyze_features(features_view, config, metadata);

        // Compute grouping strategy
        let grouping = compute_groups(&analyses, config);

        // Create bin mappers for each feature
        let bin_mappers = create_bin_mappers(&analyses, &batch_data, config, metadata);

        Ok(Self {
            pending_features: Vec::new(),
            analyses: Some(analyses),
            grouping: Some(grouping),
            bin_mappers: Some(bin_mappers),
            batch_data: Some(batch_data),
            labels: None,
            weights: None,
            config: config.clone(),
            n_samples: Some(n_samples),
        })
    }

    /// Add a numeric feature.
    pub fn add_numeric(mut self, name: &str, values: ArrayView1<f32>) -> Self {
        self.add_feature_internal(name, values, false, None);
        self
    }

    /// Add a categorical feature.
    pub fn add_categorical(mut self, name: &str, values: ArrayView1<f32>) -> Self {
        self.add_feature_internal(name, values, true, None);
        self
    }

    /// Add a feature with explicit type and optional max_bins.
    pub fn add_feature_with_options(
        mut self,
        name: &str,
        values: ArrayView1<f32>,
        is_categorical: bool,
        max_bins: Option<u32>,
    ) -> Self {
        self.add_feature_internal(name, values, is_categorical, max_bins);
        self
    }

    fn add_feature_internal(
        &mut self,
        name: &str,
        values: ArrayView1<f32>,
        is_categorical: bool,
        max_bins: Option<u32>,
    ) {
        // Validate sample count consistency
        if let Some(n) = self.n_samples {
            assert_eq!(
                n,
                values.len(),
                "Sample count mismatch: expected {n}, got {}",
                values.len()
            );
        } else {
            self.n_samples = Some(values.len());
        }

        self.pending_features.push(PendingFeature {
            name: name.to_string(),
            values: values.iter().copied().collect(),
            is_categorical,
            max_bins,
        });
    }

    /// Set labels for the dataset.
    pub fn set_labels(mut self, labels: ArrayView1<f32>) -> Self {
        if let Some(n) = self.n_samples {
            assert_eq!(
                n,
                labels.len(),
                "Label count mismatch: expected {n}, got {}",
                labels.len()
            );
        } else {
            self.n_samples = Some(labels.len());
        }
        self.labels = Some(labels.iter().copied().collect());
        self
    }

    /// Set sample weights.
    pub fn set_weights(mut self, weights: ArrayView1<f32>) -> Self {
        if let Some(n) = self.n_samples {
            assert_eq!(
                n,
                weights.len(),
                "Weight count mismatch: expected {n}, got {}",
                weights.len()
            );
        }
        self.weights = Some(weights.iter().copied().collect());
        self
    }

    /// Build the groups from collected data.
    ///
    /// Returns the feature groups and associated metadata.
    /// This is an intermediate step before full BinnedDataset construction.
    pub fn build_groups(self) -> Result<BuiltGroups, DatasetError> {
        if self.batch_data.is_some() {
            self.build_groups_batch()
        } else if !self.pending_features.is_empty() {
            self.build_groups_single_feature()
        } else {
            Err(DatasetError::EmptyDataset)
        }
    }

    fn build_groups_batch(self) -> Result<BuiltGroups, DatasetError> {
        let analyses = self.analyses.unwrap();
        let grouping = self.grouping.unwrap();
        let bin_mappers = self.bin_mappers.unwrap();
        let batch_data = self.batch_data.unwrap();
        let n_samples = self.n_samples.unwrap();

        let mut groups = Vec::with_capacity(grouping.groups.len());

        for group_spec in &grouping.groups {
            let group = build_feature_group(
                group_spec,
                &analyses,
                &bin_mappers,
                &batch_data,
                n_samples,
            );
            groups.push(group);
        }

        Ok(BuiltGroups {
            groups,
            bin_mappers,
            analyses,
            trivial_features: grouping.trivial_features,
            labels: self.labels,
            weights: self.weights,
            n_samples,
        })
    }

    fn build_groups_single_feature(self) -> Result<BuiltGroups, DatasetError> {
        let n_samples = self.n_samples.ok_or(DatasetError::EmptyDataset)?;
        let n_features = self.pending_features.len();

        // Convert pending features to column-major data
        let batch_data: Vec<Vec<f32>> = self
            .pending_features
            .iter()
            .map(|f| f.values.clone())
            .collect();

        // Build metadata from pending features
        let mut categorical_features = Vec::new();
        let mut max_bins_map = HashMap::new();
        let names: Vec<String> = self
            .pending_features
            .iter()
            .enumerate()
            .map(|(i, f)| {
                if f.is_categorical {
                    categorical_features.push(i);
                }
                if let Some(bins) = f.max_bins {
                    max_bins_map.insert(i, bins);
                }
                f.name.clone()
            })
            .collect();

        // Build metadata - set max_bins one at a time
        let mut metadata = FeatureMetadata::default()
            .names(names)
            .categorical(categorical_features);
        for (feature, bins) in max_bins_map {
            metadata = metadata.max_bins_for(feature, bins);
        }

        // Create flat feature-major slice for FeaturesView
        let flat_feature_major: Vec<f32> = batch_data.iter().flatten().copied().collect();
        let features_view = FeaturesView::from_slice(&flat_feature_major, n_samples, n_features)
            .ok_or_else(|| DatasetError::BinningError("Failed to create features view".into()))?;

        // Analyze features
        let analyses = analyze_features(features_view, &self.config, Some(&metadata));

        // Compute grouping
        let grouping = compute_groups(&analyses, &self.config);

        // Create bin mappers
        let bin_mappers = create_bin_mappers(&analyses, &batch_data, &self.config, Some(&metadata));

        // Build groups
        let mut groups = Vec::with_capacity(grouping.groups.len());

        for group_spec in &grouping.groups {
            let group = build_feature_group(
                group_spec,
                &analyses,
                &bin_mappers,
                &batch_data,
                n_samples,
            );
            groups.push(group);
        }

        Ok(BuiltGroups {
            groups,
            bin_mappers,
            analyses,
            trivial_features: grouping.trivial_features,
            labels: self.labels,
            weights: self.weights,
            n_samples,
        })
    }
}

/// Result of building groups from the builder.
///
/// This is an intermediate representation before full BinnedDataset construction.
pub struct BuiltGroups {
    /// Feature groups with populated storage
    pub groups: Vec<FeatureGroup>,
    /// Bin mappers for each feature (in original feature order)
    pub bin_mappers: Vec<BinMapper>,
    /// Analysis results for each feature
    pub analyses: Vec<FeatureAnalysis>,
    /// Indices of trivial features that were skipped
    pub trivial_features: Vec<usize>,
    /// Labels (if set)
    pub labels: Option<Vec<f32>>,
    /// Weights (if set)
    pub weights: Option<Vec<f32>>,
    /// Number of samples
    pub n_samples: usize,
}

impl BuiltGroups {
    /// Get the total number of features (excluding trivial).
    pub fn n_features(&self) -> usize {
        self.analyses.len() - self.trivial_features.len()
    }

    /// Get the total number of groups.
    pub fn n_groups(&self) -> usize {
        self.groups.len()
    }
}

// =============================================================================
// Helper functions - BinMapper creation
// =============================================================================

use super::bin_mapper::MissingType;

/// Create bin mappers for all features based on analysis results.
/// Data is in column-major format: [feature][sample]
fn create_bin_mappers(
    analyses: &[FeatureAnalysis],
    data: &[Vec<f32>], // Column-major: [feature][sample]
    config: &BinningConfig,
    metadata: Option<&FeatureMetadata>,
) -> Vec<BinMapper> {
    analyses
        .iter()
        .map(|analysis| {
            let feature_idx = analysis.feature_idx;
            let max_bins = metadata
                .and_then(|m| m.max_bins.get(&feature_idx).copied())
                .unwrap_or(config.max_bins);

            let values = &data[feature_idx];
            if analysis.is_numeric {
                bin_numeric(values, max_bins)
            } else {
                bin_categorical(values)
            }
        })
        .collect()
}

/// Create a BinMapper for a numeric feature from raw values.
fn bin_numeric(data: &[f32], max_bins: u32) -> BinMapper {
    // Collect non-NaN values and compute min/max
    let mut min_val = f32::MAX;
    let mut max_val = f32::MIN;
    let mut n_valid = 0usize;

    for &val in data.iter() {
        if val.is_finite() {
            min_val = min_val.min(val);
            max_val = max_val.max(val);
            n_valid += 1;
        }
    }

    // Handle degenerate cases
    if n_valid == 0 || min_val >= max_val {
        return BinMapper::numerical(vec![f64::MAX], MissingType::None, 0, 0, 0.0, 0.0, 0.0);
    }

    let n_bins = max_bins.min(n_valid as u32);

    // Compute equal-width bin boundaries
    let width = (max_val - min_val) / n_bins as f32;
    let bounds: Vec<f64> = (1..=n_bins)
        .map(|i| {
            if i == n_bins {
                f64::MAX
            } else {
                (min_val + i as f32 * width) as f64
            }
        })
        .collect();

    BinMapper::numerical(bounds, MissingType::None, 0, 0, 0.0, min_val as f64, max_val as f64)
}

/// Create a BinMapper for a categorical feature from raw values.
fn bin_categorical(data: &[f32]) -> BinMapper {
    // Collect unique category values (as integers)
    let mut seen = std::collections::HashSet::new();
    for &val in data.iter() {
        if val.is_finite() {
            seen.insert(val as i32);
        }
    }

    let mut categories: Vec<i32> = seen.into_iter().collect();
    categories.sort();

    if categories.is_empty() {
        categories.push(0);
    }

    BinMapper::categorical(categories, MissingType::None, 0, 0, 0.0)
}

// =============================================================================
// Helper functions - FeatureGroup building
// =============================================================================

/// Build a FeatureGroup from a GroupSpec.
fn build_feature_group(
    spec: &GroupSpec,
    _analyses: &[FeatureAnalysis],
    bin_mappers: &[BinMapper],
    data: &[Vec<f32>], // Column-major: [feature][sample]
    n_samples: usize,
) -> FeatureGroup {
    // Build storage based on group type
    let storage = match spec.group_type {
        GroupType::NumericDense => build_numeric_dense(spec, bin_mappers, data, n_samples),
        GroupType::CategoricalDense => build_categorical_dense(spec, bin_mappers, data, n_samples),
        GroupType::SparseNumeric => build_sparse_numeric(spec, bin_mappers, data, n_samples),
        GroupType::SparseCategorical => build_sparse_categorical(spec, bin_mappers, data, n_samples),
        GroupType::Bundle => {
            // EFB bundling - for now just treat as sparse categorical
            // Full EFB support is deferred to Story 1.6
            build_sparse_categorical(spec, bin_mappers, data, n_samples)
        }
    };

    // Get bin counts from bin mappers
    let bin_counts: Box<[u32]> = spec
        .feature_indices
        .iter()
        .map(|&idx| bin_mappers[idx].n_bins())
        .collect();

    // Convert feature indices to u32
    let feature_indices: Box<[u32]> = spec
        .feature_indices
        .iter()
        .map(|&idx| idx as u32)
        .collect();

    FeatureGroup::new(feature_indices, n_samples, storage, bin_counts)
}

/// Build NumericStorage for a group.
fn build_numeric_dense(
    spec: &GroupSpec,
    bin_mappers: &[BinMapper],
    data: &[Vec<f32>],
    n_samples: usize,
) -> FeatureStorage {
    let n_features = spec.n_features();

    // Collect raw values in column-major order
    let mut raw_values = Vec::with_capacity(n_features * n_samples);
    for &feature_idx in &spec.feature_indices {
        raw_values.extend_from_slice(&data[feature_idx]);
    }

    // Bin the values
    let bins = if spec.needs_u16 {
        let mut bin_data = Vec::with_capacity(n_features * n_samples);
        for &feature_idx in &spec.feature_indices {
            let mapper = &bin_mappers[feature_idx];
            for &value in &data[feature_idx] {
                bin_data.push(mapper.value_to_bin(value as f64) as u16);
            }
        }
        BinData::U16(bin_data.into_boxed_slice())
    } else {
        let mut bin_data = Vec::with_capacity(n_features * n_samples);
        for &feature_idx in &spec.feature_indices {
            let mapper = &bin_mappers[feature_idx];
            for &value in &data[feature_idx] {
                bin_data.push(mapper.value_to_bin(value as f64) as u8);
            }
        }
        BinData::U8(bin_data.into_boxed_slice())
    };

    FeatureStorage::Numeric(NumericStorage::new(bins, raw_values.into_boxed_slice()))
}

/// Build CategoricalStorage for a group.
fn build_categorical_dense(
    spec: &GroupSpec,
    bin_mappers: &[BinMapper],
    data: &[Vec<f32>],
    n_samples: usize,
) -> FeatureStorage {
    let n_features = spec.n_features();

    // Bin the values (no raw storage for categorical)
    let bins = if spec.needs_u16 {
        let mut bin_data = Vec::with_capacity(n_features * n_samples);
        for &feature_idx in &spec.feature_indices {
            let mapper = &bin_mappers[feature_idx];
            for &value in &data[feature_idx] {
                bin_data.push(mapper.value_to_bin(value as f64) as u16);
            }
        }
        BinData::U16(bin_data.into_boxed_slice())
    } else {
        let mut bin_data = Vec::with_capacity(n_features * n_samples);
        for &feature_idx in &spec.feature_indices {
            let mapper = &bin_mappers[feature_idx];
            for &value in &data[feature_idx] {
                bin_data.push(mapper.value_to_bin(value as f64) as u8);
            }
        }
        BinData::U8(bin_data.into_boxed_slice())
    };

    FeatureStorage::Categorical(CategoricalStorage::new(bins))
}

/// Build SparseNumericStorage for a group (single feature).
fn build_sparse_numeric(
    spec: &GroupSpec,
    bin_mappers: &[BinMapper],
    data: &[Vec<f32>],
    n_samples: usize,
) -> FeatureStorage {
    // Sparse groups have exactly one feature
    assert_eq!(spec.n_features(), 1, "Sparse groups must have exactly one feature");
    let feature_idx = spec.feature_indices[0];
    let mapper = &bin_mappers[feature_idx];
    let values = &data[feature_idx];

    // Collect non-zero entries
    let mut indices = Vec::new();
    let mut bins = Vec::new();
    let mut raw_values = Vec::new();

    for (i, &value) in values.iter().enumerate() {
        if value != 0.0 && !value.is_nan() {
            indices.push(i as u32);
            bins.push(mapper.value_to_bin(value as f64) as u8); // Sparse always U8
            raw_values.push(value);
        }
    }

    let bins_data = BinData::U8(bins.into_boxed_slice());
    FeatureStorage::SparseNumeric(SparseNumericStorage::new(
        indices.into_boxed_slice(),
        bins_data,
        raw_values.into_boxed_slice(),
        n_samples,
    ))
}

/// Build SparseCategoricalStorage for a group (single feature).
fn build_sparse_categorical(
    spec: &GroupSpec,
    bin_mappers: &[BinMapper],
    data: &[Vec<f32>],
    n_samples: usize,
) -> FeatureStorage {
    // Sparse groups have exactly one feature
    assert_eq!(spec.n_features(), 1, "Sparse groups must have exactly one feature");
    let feature_idx = spec.feature_indices[0];
    let mapper = &bin_mappers[feature_idx];
    let values = &data[feature_idx];

    // Collect non-zero entries
    let mut indices = Vec::new();
    let mut bins = Vec::new();

    for (i, &value) in values.iter().enumerate() {
        if value != 0.0 && !value.is_nan() {
            indices.push(i as u32);
            bins.push(mapper.value_to_bin(value as f64) as u8); // Sparse always U8
        }
    }

    let bins_data = BinData::U8(bins.into_boxed_slice());
    FeatureStorage::SparseCategorical(SparseCategoricalStorage::new(
        indices.into_boxed_slice(),
        bins_data,
        n_samples,
    ))
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    fn make_array(data: &[f32], n_samples: usize, n_features: usize) -> Array2<f32> {
        Array2::from_shape_vec((n_samples, n_features), data.to_vec()).unwrap()
    }

    #[test]
    fn test_builder_from_array_simple() {
        let data = make_array(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
        let config = BinningConfig::default();

        let builder = DatasetBuilder::from_array(data.view(), &config).unwrap();
        assert_eq!(builder.n_samples, Some(3));
    }

    #[test]
    fn test_builder_from_array_empty() {
        let data: Array2<f32> = Array2::zeros((0, 5));
        let config = BinningConfig::default();

        let result = DatasetBuilder::from_array(data.view(), &config);
        assert!(matches!(result, Err(DatasetError::EmptyDataset)));
    }

    #[test]
    fn test_builder_single_feature_api() {
        let builder = DatasetBuilder::new()
            .add_numeric("age", array![25.0, 30.0, 35.0, 40.0, 45.0].view())
            .add_numeric("income", array![50000.0, 75000.0, 100000.0, 125000.0, 150000.0].view())
            .add_categorical("gender", array![0.0, 1.0, 0.0, 1.0, 0.0].view());

        assert_eq!(builder.pending_features.len(), 3);
        assert_eq!(builder.n_samples, Some(5));
    }

    #[test]
    fn test_builder_set_labels() {
        let data = make_array(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
        let config = BinningConfig::default();

        let builder = DatasetBuilder::from_array(data.view(), &config)
            .unwrap()
            .set_labels(array![0.0, 1.0, 0.0].view());

        assert_eq!(builder.labels, Some(vec![0.0, 1.0, 0.0]));
    }

    #[test]
    fn test_build_groups_numeric() {
        // Two numeric features - use floats to avoid auto-categorical detection
        let data = make_array(
            &[
                1.1, 2.2, 3.3, 4.4, 5.5, // feature 0 - floats
                10.1, 20.2, 30.3, 40.4, 50.5, // feature 1 - floats
            ],
            5,
            2,
        );
        let config = BinningConfig::default();

        let groups = DatasetBuilder::from_array(data.view(), &config)
            .unwrap()
            .build_groups()
            .unwrap();

        // Should have one numeric dense group
        assert_eq!(groups.n_groups(), 1);
        assert_eq!(groups.groups[0].n_features(), 2);
        assert!(matches!(
            groups.groups[0].storage(),
            FeatureStorage::Numeric(_)
        ));
    }

    #[test]
    fn test_build_groups_mixed_types() {
        // Feature 0: numeric, Feature 1: categorical (integer values 0,1,2)
        let data = make_array(
            &[
                1.5, 2.5, 3.5, 4.5, 5.5, // feature 0 - floats
                0.0, 1.0, 2.0, 0.0, 1.0, // feature 1 - integers
            ],
            5,
            2,
        );

        let metadata = FeatureMetadata::default().categorical(vec![1]);
        let config = BinningConfig::default();

        let groups = DatasetBuilder::from_array_with_metadata(
            data.view(),
            Some(&metadata),
            &config,
        )
        .unwrap()
        .build_groups()
        .unwrap();

        // Should have two groups: one numeric, one categorical
        assert_eq!(groups.n_groups(), 2);

        let has_numeric = groups.groups.iter().any(|g| {
            matches!(g.storage(), FeatureStorage::Numeric(_))
        });
        let has_categorical = groups.groups.iter().any(|g| {
            matches!(g.storage(), FeatureStorage::Categorical(_))
        });

        assert!(has_numeric);
        assert!(has_categorical);
    }

    #[test]
    fn test_build_groups_sparse() {
        // Sparse feature (mostly zeros) - use floats to avoid categorical detection
        let mut values = vec![0.0; 100];
        values[0] = 1.5; // Non-integer to avoid categorical detection
        values[50] = 2.7; // Non-integer

        let data = Array2::from_shape_vec((100, 1), values).unwrap();
        let config = BinningConfig::builder().sparsity_threshold(0.9).build();

        let groups = DatasetBuilder::from_array(data.view(), &config)
            .unwrap()
            .build_groups()
            .unwrap();

        // Should have one sparse numeric group
        assert_eq!(groups.n_groups(), 1);
        assert!(matches!(
            groups.groups[0].storage(),
            FeatureStorage::SparseNumeric(_)
        ));
    }

    #[test]
    fn test_build_groups_preserves_raw_values() {
        // Use floats to ensure numeric detection
        let data = make_array(&[1.5, 2.5, 3.5, 4.5, 5.5], 5, 1);
        let config = BinningConfig::default();

        let groups = DatasetBuilder::from_array(data.view(), &config)
            .unwrap()
            .build_groups()
            .unwrap();

        // Check raw values are preserved
        if let FeatureStorage::Numeric(storage) = groups.groups[0].storage() {
            let raw = storage.raw_slice(0, 5);
            assert_eq!(raw, &[1.5, 2.5, 3.5, 4.5, 5.5]);
        } else {
            panic!("Expected Numeric storage");
        }
    }

    #[test]
    fn test_build_groups_column_major_layout() {
        // Two features, verify column-major layout - use floats for numeric detection
        let data = make_array(
            &[
                // Row 0: 1.1, 10.1
                // Row 1: 2.2, 20.2
                // Row 2: 3.3, 30.3
                1.1, 10.1, 2.2, 20.2, 3.3, 30.3,
            ],
            3,
            2,
        );
        let config = BinningConfig::default();

        let groups = DatasetBuilder::from_array(data.view(), &config)
            .unwrap()
            .build_groups()
            .unwrap();

        // In column-major: feature 0 values are [1.1, 2.2, 3.3], feature 1 values are [10.1, 20.2, 30.3]
        if let FeatureStorage::Numeric(storage) = groups.groups[0].storage() {
            let raw0 = storage.raw_slice(0, 3);
            let raw1 = storage.raw_slice(1, 3);
            assert_eq!(raw0, &[1.1, 2.2, 3.3]);
            assert_eq!(raw1, &[10.1, 20.2, 30.3]);
        } else {
            panic!("Expected Numeric storage");
        }
    }

    #[test]
    fn test_single_feature_api_builds_correctly() {
        // Use floats for numeric detection
        let built = DatasetBuilder::new()
            .add_numeric("x", array![1.1, 2.2, 3.3, 4.4, 5.5].view())
            .add_numeric("y", array![10.1, 20.2, 30.3, 40.4, 50.5].view())
            .build_groups()
            .unwrap();

        assert_eq!(built.n_samples, 5);
        assert_eq!(built.n_groups(), 1);
        assert!(matches!(
            built.groups[0].storage(),
            FeatureStorage::Numeric(_)
        ));
    }
}

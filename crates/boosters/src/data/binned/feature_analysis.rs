//! Feature analysis for detecting feature properties.
//!
//! This module provides single-pass feature analysis to detect:
//! - Binary features (exactly 2 unique values)
//! - Trivial features (all zeros/missing, can be skipped)
//! - Sparse features (high fraction of zeros)
//! - Integer-valued features (candidates for categorical)
//! - Required bin width (U8 vs U16)
//!
//! The analysis uses O(1) memory per feature and is fully parallelizable.

// Allow dead_code during migration - will be used when builder is implemented
#![allow(dead_code)]

use std::collections::HashMap;

use rayon::prelude::*;

use crate::data::{Dataset, Feature};

use super::BinData;

// ============================================================================
// BinningConfig
// ============================================================================

/// Configuration for binning features.
#[derive(Clone, Debug)]
pub struct BinningConfig {
    /// Maximum bins per feature (global default).
    /// Can be overridden per-feature via `FeatureMetadata::max_bins`.
    pub max_bins: u32,

    /// Sparsity threshold (fraction of zeros to use sparse storage).
    /// Features with density ≤ (1 - threshold) are considered sparse.
    pub sparsity_threshold: f32,

    /// Enable EFB bundling.
    pub enable_bundling: bool,

    /// Max cardinality to auto-detect as categorical.
    /// Features with ≤ this many unique integer values may be treated as categorical.
    pub max_categorical_cardinality: u32,

    /// Number of samples for computing bin boundaries (for large datasets).
    pub sample_cnt: usize,
}

impl Default for BinningConfig {
    fn default() -> Self {
        Self {
            max_bins: 256,
            sparsity_threshold: 0.9,
            enable_bundling: true,
            // Disable auto-detection of categorical features by default.
            // Integer features should be treated as ordinal/numeric unless
            // explicitly marked as categorical via FeatureMetadata.
            // This matches XGBoost behavior where categorical support is opt-in.
            max_categorical_cardinality: 0,
            sample_cnt: 200_000,
        }
    }
}

impl BinningConfig {
    /// Create a new builder for BinningConfig.
    pub fn builder() -> BinningConfigBuilder {
        BinningConfigBuilder::default()
    }
}

/// Builder for BinningConfig.
#[derive(Clone, Debug, Default)]
pub struct BinningConfigBuilder {
    max_bins: Option<u32>,
    sparsity_threshold: Option<f32>,
    enable_bundling: Option<bool>,
    max_categorical_cardinality: Option<u32>,
    sample_cnt: Option<usize>,
}

impl BinningConfigBuilder {
    /// Set maximum bins per feature.
    pub fn max_bins(mut self, max_bins: u32) -> Self {
        self.max_bins = Some(max_bins);
        self
    }

    /// Set sparsity threshold.
    pub fn sparsity_threshold(mut self, threshold: f32) -> Self {
        self.sparsity_threshold = Some(threshold);
        self
    }

    /// Enable or disable bundling.
    pub fn enable_bundling(mut self, enable: bool) -> Self {
        self.enable_bundling = Some(enable);
        self
    }

    /// Set max categorical cardinality for auto-detection.
    pub fn max_categorical_cardinality(mut self, cardinality: u32) -> Self {
        self.max_categorical_cardinality = Some(cardinality);
        self
    }

    /// Set sample count for bin boundary computation.
    pub fn sample_cnt(mut self, cnt: usize) -> Self {
        self.sample_cnt = Some(cnt);
        self
    }

    /// Build the config.
    pub fn build(self) -> BinningConfig {
        let default = BinningConfig::default();
        BinningConfig {
            max_bins: self.max_bins.unwrap_or(default.max_bins),
            sparsity_threshold: self
                .sparsity_threshold
                .unwrap_or(default.sparsity_threshold),
            enable_bundling: self.enable_bundling.unwrap_or(default.enable_bundling),
            max_categorical_cardinality: self
                .max_categorical_cardinality
                .unwrap_or(default.max_categorical_cardinality),
            sample_cnt: self.sample_cnt.unwrap_or(default.sample_cnt),
        }
    }
}

// ============================================================================
// FeatureMetadata
// ============================================================================

/// Metadata for features in a matrix.
/// All fields are optional - unspecified features use auto-detection.
#[derive(Clone, Debug, Default)]
pub struct FeatureMetadata {
    /// Feature names (length must match n_features or be empty).
    pub names: Vec<String>,

    /// Indices of categorical features. All others are numeric.
    /// If empty, auto-detect based on cardinality.
    pub categorical_features: Vec<usize>,

    /// Per-feature max_bins overrides. Key = feature index.
    pub max_bins: HashMap<usize, u32>,
}

impl FeatureMetadata {
    /// Create with just feature names.
    pub fn with_names(names: Vec<String>) -> Self {
        Self {
            names,
            ..Default::default()
        }
    }

    /// Create with categorical feature indices.
    pub fn with_categorical(categorical_features: Vec<usize>) -> Self {
        Self {
            categorical_features,
            ..Default::default()
        }
    }

    /// Builder pattern: set names.
    pub fn names(mut self, names: Vec<String>) -> Self {
        self.names = names;
        self
    }

    /// Builder pattern: set categorical features.
    pub fn categorical(mut self, indices: Vec<usize>) -> Self {
        self.categorical_features = indices;
        self
    }

    /// Builder pattern: set max_bins for a feature.
    pub fn max_bins_for(mut self, feature: usize, bins: u32) -> Self {
        self.max_bins.insert(feature, bins);
        self
    }

    /// Check if a feature is explicitly marked as categorical.
    pub fn is_categorical(&self, feature: usize) -> bool {
        self.categorical_features.contains(&feature)
    }

    /// Get max_bins for a feature, or None if not specified.
    pub fn get_max_bins(&self, feature: usize) -> Option<u32> {
        self.max_bins.get(&feature).copied()
    }

    /// Get name for a feature, or None if not specified.
    pub fn get_name(&self, feature: usize) -> Option<&str> {
        self.names.get(feature).map(|s| s.as_str())
    }
}

// ============================================================================
// FeatureAnalysis
// ============================================================================

/// Analysis results for a single feature.
#[derive(Clone, Debug, PartialEq)]
pub struct FeatureAnalysis {
    /// Original feature index in the input matrix.
    pub feature_idx: usize,

    /// Whether this feature should be treated as numeric (vs categorical).
    pub is_numeric: bool,

    /// Whether this feature is sparse (high fraction of zeros).
    pub is_sparse: bool,

    /// Whether this feature requires U16 bins (more than 256 bins).
    pub needs_u16: bool,

    /// Whether this feature is trivial (all zeros, single value, or all NaN).
    /// Trivial features can be skipped during training.
    pub is_trivial: bool,

    /// Whether this feature is binary (exactly 2 unique values).
    pub is_binary: bool,

    /// Number of unique non-NaN values seen.
    pub n_unique: usize,

    /// Number of non-zero values.
    pub n_nonzero: usize,

    /// Total number of valid (non-NaN) values.
    pub n_valid: usize,

    /// Total number of NaN values.
    pub n_nan: usize,

    /// Total number of samples.
    pub n_samples: usize,

    /// Minimum non-NaN value.
    pub min_val: f32,

    /// Maximum non-NaN value.
    pub max_val: f32,
}

impl FeatureAnalysis {
    /// Get the density (fraction of non-zero values).
    pub fn density(&self) -> f32 {
        if self.n_samples == 0 {
            0.0
        } else {
            self.n_nonzero as f32 / self.n_samples as f32
        }
    }

    /// Check if sparse based on a threshold.
    pub fn is_sparse_with_threshold(&self, threshold: f32) -> bool {
        self.density() <= (1.0 - threshold)
    }

    /// Get the number of bins this feature will need.
    /// For numeric: min(n_unique, max_bins) + possible NaN bin
    /// For categorical: n_unique + possible missing bin
    pub fn required_bins(&self, max_bins: u32, has_nan: bool) -> u32 {
        let base_bins = (self.n_unique as u32).min(max_bins);
        if has_nan { base_bins + 1 } else { base_bins }
    }
}

/// Efficient feature statistics (O(1) memory per feature, single-pass).
#[derive(Clone, Debug)]
struct FeatureStats {
    /// Minimum non-NaN value seen.
    min: f32,
    /// Maximum non-NaN value seen.
    max: f32,
    /// Count of non-zero, non-NaN values.
    non_zero_count: usize,
    /// Total count of non-NaN values.
    valid_count: usize,
    /// Count of NaN values.
    nan_count: usize,
    /// All values seen are integers.
    all_integers: bool,
    /// Set true when we see more than max_unique distinct values.
    exceeded_unique_limit: bool,
    /// Unique values (up to limit).
    unique_values: Vec<f32>,
    /// Max unique values to track.
    max_unique: usize,
}

impl FeatureStats {
    /// Create new stats with a limit on unique values to track.
    fn new(max_unique: usize) -> Self {
        Self {
            min: f32::MAX,
            max: f32::MIN,
            non_zero_count: 0,
            valid_count: 0,
            nan_count: 0,
            all_integers: true,
            exceeded_unique_limit: false,
            unique_values: Vec::new(),
            max_unique,
        }
    }

    /// Update stats with a new value.
    #[inline]
    fn update(&mut self, val: f32) {
        // Skip NaN values
        if val.is_nan() {
            self.nan_count += 1;
            return;
        }

        self.valid_count += 1;

        if val != 0.0 {
            self.non_zero_count += 1;
        }

        self.min = self.min.min(val);
        self.max = self.max.max(val);

        // Check if value is an integer
        if self.all_integers && val.fract() != 0.0 {
            self.all_integers = false;
        }

        // Track unique values (up to limit)
        if !self.exceeded_unique_limit && !self.unique_values.contains(&val) {
            if self.unique_values.len() >= self.max_unique {
                self.exceeded_unique_limit = true;
            } else {
                self.unique_values.push(val);
            }
        }
    }

    /// Convert stats to FeatureAnalysis.
    fn into_analysis(
        self,
        feature_idx: usize,
        n_samples: usize,
        config: &BinningConfig,
        metadata: Option<&FeatureMetadata>,
    ) -> FeatureAnalysis {
        let n_unique = self.unique_values.len();

        // Trivial if: all NaN, all zeros, or single value
        let is_trivial = self.valid_count == 0
            || (n_unique == 1 && self.unique_values.first() == Some(&0.0))
            || (n_unique == 0);

        // Binary if exactly 2 distinct values
        let is_binary = n_unique == 2;

        // Determine if categorical:
        // 1. User-specified always wins
        // 2. Auto-detect: integer values with low cardinality
        let user_specified_categorical = metadata
            .map(|m| m.is_categorical(feature_idx))
            .unwrap_or(false);

        let auto_categorical = !user_specified_categorical
            && self.all_integers
            && n_unique <= config.max_categorical_cardinality as usize;

        // Per RFC: Binary features default to numeric unless user specifies categorical
        let is_numeric = if user_specified_categorical {
            false
        } else if is_binary {
            true // Binary defaults to numeric
        } else {
            !auto_categorical
        };

        // Determine max_bins for this feature
        let max_bins = metadata
            .and_then(|m| m.get_max_bins(feature_idx))
            .unwrap_or(config.max_bins);

        // Calculate required bins
        let has_nan = self.nan_count > 0;
        let required_bins = if is_trivial {
            1
        } else if !is_numeric {
            // Categorical: one bin per unique value
            (n_unique as u32).min(max_bins) + if has_nan { 1 } else { 0 }
        } else {
            // Numeric: quantile-based, limited by max_bins
            max_bins.min(n_unique as u32) + if has_nan { 1 } else { 0 }
        };

        let needs_u16 = BinData::needs_u16(required_bins);

        // Sparse detection
        let density = if n_samples == 0 {
            0.0
        } else {
            self.non_zero_count as f32 / n_samples as f32
        };
        let is_sparse = density <= (1.0 - config.sparsity_threshold);

        FeatureAnalysis {
            feature_idx,
            is_numeric,
            is_sparse,
            needs_u16,
            is_trivial,
            is_binary,
            n_unique,
            n_nonzero: self.non_zero_count,
            n_valid: self.valid_count,
            n_nan: self.nan_count,
            n_samples,
            min_val: if self.valid_count > 0 {
                self.min
            } else {
                f32::NAN
            },
            max_val: if self.valid_count > 0 {
                self.max
            } else {
                f32::NAN
            },
        }
    }

    /// Include an implicit repeated value in min/max and uniqueness checks.
    ///
    /// This is used for sparse features where the default value is present for
    /// many rows but is not explicitly stored.
    #[inline]
    fn include_implicit_value(&mut self, val: f32) {
        if val.is_nan() {
            return;
        }

        self.min = self.min.min(val);
        self.max = self.max.max(val);

        if self.all_integers && val.fract() != 0.0 {
            self.all_integers = false;
        }

        if !self.exceeded_unique_limit && !self.unique_values.contains(&val) {
            if self.unique_values.len() >= self.max_unique {
                self.exceeded_unique_limit = true;
            } else {
                self.unique_values.push(val);
            }
        }
    }
}

// NOTE: In RFC-0021, `Dataset` owns raw values (dense/sparse). We intentionally
// keep only the Dataset-based entrypoint to avoid duplicated analysis pipelines.

/// Analyze all features in a [`Dataset`] without materializing a dense matrix.
///
/// This is the preferred analysis entrypoint for binning in RFC-0021:
/// `Dataset` owns raw values (dense or sparse), and `BinnedDataset` should not
/// force-densify sparse features during analysis.
pub fn analyze_features_dataset(
    dataset: &Dataset,
    config: &BinningConfig,
    metadata: Option<&FeatureMetadata>,
) -> Vec<FeatureAnalysis> {
    let n_features = dataset.n_features();
    let n_samples = dataset.n_samples();

    if n_features == 0 || n_samples == 0 {
        return (0..n_features)
            .map(|idx| FeatureAnalysis {
                feature_idx: idx,
                is_numeric: true,
                is_sparse: false,
                needs_u16: false,
                is_trivial: true,
                is_binary: false,
                n_unique: 0,
                n_nonzero: 0,
                n_valid: 0,
                n_nan: 0,
                n_samples,
                min_val: f32::NAN,
                max_val: f32::NAN,
            })
            .collect();
    }

    let max_unique = (config.max_categorical_cardinality as usize)
        .max(config.max_bins as usize)
        .saturating_add(10);

    (0..n_features)
        .into_par_iter()
        .map(|feature_idx| {
            let mut stats = FeatureStats::new(max_unique);
            let feature = dataset.feature(feature_idx);

            match feature {
                Feature::Dense(values) => {
                    if let Some(slice) = values.as_slice() {
                        for &val in slice {
                            stats.update(val);
                        }
                    } else {
                        for &val in values.iter() {
                            stats.update(val);
                        }
                    }
                }
                Feature::Sparse {
                    indices,
                    values,
                    n_samples: _,
                    default,
                } => {
                    // For sparse features, stored values represent non-default entries.
                    // We still need to include the implicit default value in analysis
                    // (usually 0.0) without materializing a dense column.

                    // Update stats for stored values.
                    for &val in values.iter() {
                        stats.update(val);
                    }

                    // Account for implicit defaults.
                    // This matches existing semantics that treat missing entries
                    // as `default` (typically 0.0).
                    let n_defaults = n_samples.saturating_sub(indices.len());
                    if n_defaults > 0 {
                        // Fold the default value into min/max and uniqueness.
                        // We do NOT iterate `n_defaults` times.
                        stats.include_implicit_value(*default);

                        // Non-zero count: existing binning treats "sparse" as many zeros,
                        // so we only increment if default is non-zero.
                        if *default != 0.0 && !default.is_nan() {
                            stats.non_zero_count = stats.non_zero_count.saturating_add(n_defaults);
                        }

                        // Valid count: default entries are valid unless default is NaN.
                        if default.is_nan() {
                            stats.nan_count = stats.nan_count.saturating_add(n_defaults);
                        } else {
                            stats.valid_count = stats.valid_count.saturating_add(n_defaults);
                        }
                    }
                }
            }

            let mut analysis = stats.into_analysis(feature_idx, n_samples, config, metadata);

            // If the feature is stored as sparse, treat it as sparse for grouping.
            if feature.is_sparse() {
                analysis.is_sparse = true;
            }

            analysis
        })
        .collect()
}

// NOTE: A sequential implementation was removed to keep a single source of
// truth. If we ever need it again, we can implement it by running the same
// Dataset-based loop without rayon.

// ============================================================================
// GroupSpec - Feature Grouping Strategy
// ============================================================================

/// Type of feature group storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GroupType {
    /// Dense numeric storage (bins).
    NumericDense,
    /// Dense categorical storage (bins only).
    CategoricalDense,
    /// Sparse numeric storage (single feature, bins).
    SparseNumeric,
    /// Sparse categorical storage (single feature, bins only).
    SparseCategorical,
    /// Bundled sparse categorical features (EFB).
    Bundle,
}

/// Specification for a feature group.
#[derive(Clone, Debug, PartialEq)]
pub struct GroupSpec {
    /// Feature indices in this group.
    pub feature_indices: Vec<usize>,
    /// Type of storage for this group.
    pub group_type: GroupType,
    /// Whether bins need U16 (vs U8).
    pub needs_u16: bool,
}

impl GroupSpec {
    /// Create a new group specification.
    pub fn new(feature_indices: Vec<usize>, group_type: GroupType, needs_u16: bool) -> Self {
        Self {
            feature_indices,
            group_type,
            needs_u16,
        }
    }

    /// Number of features in this group.
    pub fn n_features(&self) -> usize {
        self.feature_indices.len()
    }
}

/// Result of feature grouping computation.
#[derive(Clone, Debug)]
pub struct GroupingResult {
    /// Feature groups.
    pub groups: Vec<GroupSpec>,
    /// Indices of trivial features (skipped).
    pub trivial_features: Vec<usize>,
}

/// Compute homogeneous feature groups from analysis results.
///
/// Groups features by:
/// 1. Type (numeric vs categorical)
/// 2. Density (dense vs sparse)
/// 3. Bin width (U8 vs U16)
///
/// # Arguments
///
/// * `analyses` - Feature analysis results from `analyze_features_dataset()`
/// * `config` - Binning configuration
///
/// # Returns
///
/// `GroupingResult` containing feature groups and trivial feature indices.
///
/// # Grouping Rules
///
/// Per RFC-0018:
/// - Numeric dense U8: all numeric, non-sparse, ≤256 bins → one group
/// - Numeric dense U16: all numeric, non-sparse, >256 bins → one group
/// - Categorical dense U8: all categorical, non-sparse, ≤256 bins → one group
/// - Categorical dense U16: all categorical, non-sparse, >256 bins → one group
/// - Sparse features: each sparse feature → its own group
/// - Trivial features: skipped (not grouped)
pub fn compute_groups(analyses: &[FeatureAnalysis], _config: &BinningConfig) -> GroupingResult {
    let mut numeric_dense_u8: Vec<usize> = Vec::new();
    let mut numeric_dense_u16: Vec<usize> = Vec::new();
    let mut categorical_dense_u8: Vec<usize> = Vec::new();
    let mut categorical_dense_u16: Vec<usize> = Vec::new();
    let mut sparse_numeric: Vec<usize> = Vec::new();
    let mut sparse_categorical: Vec<usize> = Vec::new();
    let mut trivial_features: Vec<usize> = Vec::new();

    for analysis in analyses {
        if analysis.is_trivial {
            trivial_features.push(analysis.feature_idx);
            continue;
        }

        if analysis.is_sparse {
            if analysis.is_numeric {
                sparse_numeric.push(analysis.feature_idx);
            } else {
                sparse_categorical.push(analysis.feature_idx);
            }
        } else if analysis.is_numeric {
            if analysis.needs_u16 {
                numeric_dense_u16.push(analysis.feature_idx);
            } else {
                numeric_dense_u8.push(analysis.feature_idx);
            }
        } else {
            // Categorical
            if analysis.needs_u16 {
                categorical_dense_u16.push(analysis.feature_idx);
            } else {
                categorical_dense_u8.push(analysis.feature_idx);
            }
        }
    }

    let mut groups = Vec::new();

    // Create dense numeric groups
    if !numeric_dense_u8.is_empty() {
        groups.push(GroupSpec::new(
            numeric_dense_u8,
            GroupType::NumericDense,
            false,
        ));
    }
    if !numeric_dense_u16.is_empty() {
        groups.push(GroupSpec::new(
            numeric_dense_u16,
            GroupType::NumericDense,
            true,
        ));
    }

    // Create dense categorical groups
    if !categorical_dense_u8.is_empty() {
        groups.push(GroupSpec::new(
            categorical_dense_u8,
            GroupType::CategoricalDense,
            false,
        ));
    }
    if !categorical_dense_u16.is_empty() {
        groups.push(GroupSpec::new(
            categorical_dense_u16,
            GroupType::CategoricalDense,
            true,
        ));
    }

    // Create individual sparse groups
    for idx in sparse_numeric {
        let needs_u16 = analyses[idx].needs_u16;
        groups.push(GroupSpec::new(
            vec![idx],
            GroupType::SparseNumeric,
            needs_u16,
        ));
    }
    for idx in sparse_categorical {
        let needs_u16 = analyses[idx].needs_u16;
        groups.push(GroupSpec::new(
            vec![idx],
            GroupType::SparseCategorical,
            needs_u16,
        ));
    }

    // TODO: Story 1.6 - Bundle sparse categorical features together when
    // config.enable_bundling is true and conflict graph allows.

    GroupingResult {
        groups,
        trivial_features,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    use ndarray::Array2;

    fn make_dense_dataset(data: &[f32], n_samples: usize, n_features: usize) -> Dataset {
        assert_eq!(data.len(), n_samples * n_features);
        let features =
            Array2::from_shape_vec((n_features, n_samples), data.to_vec()).expect("shape mismatch");
        Dataset::from_array(features.view(), None, None)
    }

    fn default_config() -> BinningConfig {
        BinningConfig::default()
    }

    // -------------------------------------------------------------------------
    // BinningConfig tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_binning_config_default() {
        let config = BinningConfig::default();
        assert_eq!(config.max_bins, 256);
        assert!((config.sparsity_threshold - 0.9).abs() < 0.001);
        assert!(config.enable_bundling);
        // Auto-detection of categorical features is disabled by default
        assert_eq!(config.max_categorical_cardinality, 0);
    }

    #[test]
    fn test_binning_config_builder() {
        let config = BinningConfig::builder()
            .max_bins(128)
            .sparsity_threshold(0.8)
            .enable_bundling(false)
            .max_categorical_cardinality(64)
            .build();

        assert_eq!(config.max_bins, 128);
        assert!((config.sparsity_threshold - 0.8).abs() < 0.001);
        assert!(!config.enable_bundling);
        assert_eq!(config.max_categorical_cardinality, 64);
    }

    // -------------------------------------------------------------------------
    // FeatureMetadata tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_feature_metadata_categorical() {
        let metadata = FeatureMetadata::with_categorical(vec![1, 3, 5]);

        assert!(!metadata.is_categorical(0));
        assert!(metadata.is_categorical(1));
        assert!(!metadata.is_categorical(2));
        assert!(metadata.is_categorical(3));
    }

    #[test]
    fn test_feature_metadata_max_bins() {
        let metadata = FeatureMetadata::default()
            .max_bins_for(0, 64)
            .max_bins_for(5, 1024);

        assert_eq!(metadata.get_max_bins(0), Some(64));
        assert_eq!(metadata.get_max_bins(5), Some(1024));
        assert_eq!(metadata.get_max_bins(1), None);
    }

    // -------------------------------------------------------------------------
    // FeatureAnalysis tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_binary_feature() {
        let data = [0.0f32, 1.0, 0.0, 1.0];
        let ds = make_dense_dataset(&data, 4, 1);
        let analysis = analyze_features_dataset(&ds, &default_config(), None);

        assert!(analysis[0].is_binary);
        assert!(analysis[0].is_numeric); // Binary defaults to numeric
        assert!(!analysis[0].is_trivial);
        assert_eq!(analysis[0].n_unique, 2);
    }

    #[test]
    fn test_categorical_detection_integers() {
        // Low cardinality integers can be auto-detected as categorical when enabled
        let data = [0.0f32, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0];
        let ds = make_dense_dataset(&data, 8, 1);

        // With auto-detection disabled (default), integers remain numeric
        let analysis = analyze_features_dataset(&ds, &default_config(), None);
        assert!(analysis[0].is_numeric); // Should be numeric when auto-detection is disabled
        assert_eq!(analysis[0].n_unique, 3);

        // With auto-detection enabled, low-cardinality integers become categorical
        let config_with_auto = BinningConfig::builder()
            .max_categorical_cardinality(256)
            .build();
        let ds2 = make_dense_dataset(&data, 8, 1);
        let analysis_auto = analyze_features_dataset(&ds2, &config_with_auto, None);
        assert!(!analysis_auto[0].is_numeric); // Should be categorical
        assert!(!analysis_auto[0].is_binary);
        assert_eq!(analysis_auto[0].n_unique, 3);
    }

    #[test]
    fn test_user_specified_categorical() {
        // User specifies categorical, should override auto-detection
        let data = [0.0f32, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5];
        let ds = make_dense_dataset(&data, 8, 1);
        let metadata = FeatureMetadata::with_categorical(vec![0]);
        let analysis = analyze_features_dataset(&ds, &default_config(), Some(&metadata));

        assert!(!analysis[0].is_numeric); // User said categorical
    }

    #[test]
    fn test_numeric_floats() {
        // Non-integer values should be numeric
        let data = [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let ds = make_dense_dataset(&data, 8, 1);
        let analysis = analyze_features_dataset(&ds, &default_config(), None);

        assert!(analysis[0].is_numeric);
    }

    #[test]
    fn test_trivial_all_zeros() {
        let data = [0.0f32, 0.0, 0.0, 0.0];
        let ds = make_dense_dataset(&data, 4, 1);
        let analysis = analyze_features_dataset(&ds, &default_config(), None);

        assert!(analysis[0].is_trivial);
        assert_eq!(analysis[0].n_unique, 1);
        assert_eq!(analysis[0].n_nonzero, 0);
    }

    #[test]
    fn test_trivial_all_nan() {
        let data = [f32::NAN, f32::NAN, f32::NAN, f32::NAN];
        let ds = make_dense_dataset(&data, 4, 1);
        let analysis = analyze_features_dataset(&ds, &default_config(), None);

        assert!(analysis[0].is_trivial);
        assert_eq!(analysis[0].n_unique, 0);
        assert_eq!(analysis[0].n_valid, 0);
    }

    #[test]
    fn test_sparse_detection() {
        // 1 non-zero out of 10 = 10% density = 90% sparse
        let data = [0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let ds = make_dense_dataset(&data, 10, 1);
        let analysis = analyze_features_dataset(&ds, &default_config(), None);

        assert!(analysis[0].is_sparse); // Default threshold is 0.9
        assert!((analysis[0].density() - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_not_sparse() {
        // 5 non-zero out of 10 = 50% density
        let data = [0.0f32, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let ds = make_dense_dataset(&data, 10, 1);
        let analysis = analyze_features_dataset(&ds, &default_config(), None);

        assert!(!analysis[0].is_sparse);
    }

    #[test]
    fn test_needs_u16() {
        // If we have many unique values and high max_bins, we might need U16
        let config = BinningConfig::builder().max_bins(500).build();
        let data: Vec<f32> = (0..300).map(|i| i as f32).collect();
        let ds = make_dense_dataset(&data, 300, 1);
        let analysis = analyze_features_dataset(&ds, &config, None);

        assert!(analysis[0].needs_u16); // 300 unique values > 256
    }

    #[test]
    fn test_no_u16_needed() {
        // Default max_bins is 256, so even with many values we stay U8
        let data: Vec<f32> = (0..300).map(|i| i as f32).collect();
        let ds = make_dense_dataset(&data, 300, 1);
        let analysis = analyze_features_dataset(&ds, &default_config(), None);

        // Default max_bins is 256, but 300 unique values get capped
        // Actually, numeric features use quantile binning limited by max_bins
        assert!(!analysis[0].needs_u16);
    }

    #[test]
    fn test_min_max() {
        let data = [1.0f32, 5.0, 3.0, 2.0, 4.0];
        let ds = make_dense_dataset(&data, 5, 1);
        let analysis = analyze_features_dataset(&ds, &default_config(), None);

        assert!((analysis[0].min_val - 1.0).abs() < 0.001);
        assert!((analysis[0].max_val - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_multiple_features() {
        // Feature 0: binary {0, 1}
        // Feature 1: categorical integers {0, 1, 2}
        // Feature 2: numeric floats
        let data = [
            // Feature 0 (4 samples)
            0.0f32, 1.0, 0.0, 1.0, // Feature 1 (4 samples)
            0.0, 1.0, 2.0, 0.0, // Feature 2 (4 samples)
            0.1, 0.2, 0.3, 0.4,
        ];
        let ds = make_dense_dataset(&data, 4, 3);

        // With auto-detection disabled (default), all features are numeric
        let analysis = analyze_features_dataset(&ds, &default_config(), None);

        // Feature 0: binary → numeric
        assert!(analysis[0].is_binary);
        assert!(analysis[0].is_numeric);

        // Feature 1: integer but auto-detection disabled → still numeric
        assert!(!analysis[1].is_binary);
        assert!(analysis[1].is_numeric);

        // Feature 2: floats → numeric
        assert!(!analysis[2].is_binary);
        assert!(analysis[2].is_numeric);

        // With auto-detection enabled, integers become categorical
        let config_with_auto = BinningConfig::builder()
            .max_categorical_cardinality(256)
            .build();
        let ds2 = make_dense_dataset(&data, 4, 3);
        let analysis_auto = analyze_features_dataset(&ds2, &config_with_auto, None);

        // Feature 0: binary → still numeric (binary defaults to numeric)
        assert!(analysis_auto[0].is_binary);
        assert!(analysis_auto[0].is_numeric);

        // Feature 1: integer → categorical with auto-detection
        assert!(!analysis_auto[1].is_binary);
        assert!(!analysis_auto[1].is_numeric); // Auto-detected categorical

        // Feature 2: floats → numeric
        assert!(!analysis_auto[2].is_binary);
        assert!(analysis_auto[2].is_numeric);
    }

    #[test]
    fn test_parallel_matches_sequential() {
        let data = [
            0.0f32, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0, -1.0, 1.0,
        ];
        let config = default_config();
        let ds = make_dense_dataset(&data, 4, 4);

        // Determinism check: same input should yield identical analyses.
        let a1 = analyze_features_dataset(&ds, &config, None);
        let a2 = analyze_features_dataset(&ds, &config, None);
        assert_eq!(a1, a2);
    }

    #[test]
    fn test_empty_features() {
        let data: [f32; 0] = [];
        let ds = make_dense_dataset(&data, 0, 3);
        let analysis = analyze_features_dataset(&ds, &default_config(), None);

        assert_eq!(analysis.len(), 3);
        for a in &analysis {
            assert!(a.is_trivial);
        }
    }

    #[test]
    fn test_high_cardinality_not_categorical() {
        // More unique values than max_categorical_cardinality
        let config = BinningConfig::builder()
            .max_categorical_cardinality(5)
            .build();
        let data: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let ds = make_dense_dataset(&data, 20, 1);
        let analysis = analyze_features_dataset(&ds, &config, None);

        assert!(analysis[0].is_numeric); // Too many categories → numeric
    }

    // -------------------------------------------------------------------------
    // GroupSpec and compute_groups tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_group_spec_properties() {
        let group = GroupSpec::new(vec![0, 1, 2], GroupType::NumericDense, false);
        assert_eq!(group.n_features(), 3);
        assert_eq!(group.group_type, GroupType::NumericDense);

        let cat_group = GroupSpec::new(vec![3], GroupType::CategoricalDense, false);
        assert_eq!(cat_group.group_type, GroupType::CategoricalDense);

        let sparse_num = GroupSpec::new(vec![4], GroupType::SparseNumeric, false);
        assert_eq!(sparse_num.group_type, GroupType::SparseNumeric);

        let sparse_cat = GroupSpec::new(vec![5], GroupType::SparseCategorical, false);
        assert_eq!(sparse_cat.group_type, GroupType::SparseCategorical);
    }

    #[test]
    fn test_compute_groups_mixed() {
        // Create analyses manually for testing
        let analyses = vec![
            // Feature 0: numeric dense U8
            FeatureAnalysis {
                feature_idx: 0,
                is_numeric: true,
                is_sparse: false,
                needs_u16: false,
                is_trivial: false,
                is_binary: true,
                n_unique: 2,
                n_nonzero: 50,
                n_valid: 100,
                n_nan: 0,
                n_samples: 100,
                min_val: 0.0,
                max_val: 1.0,
            },
            // Feature 1: numeric dense U8
            FeatureAnalysis {
                feature_idx: 1,
                is_numeric: true,
                is_sparse: false,
                needs_u16: false,
                is_trivial: false,
                is_binary: false,
                n_unique: 100,
                n_nonzero: 100,
                n_valid: 100,
                n_nan: 0,
                n_samples: 100,
                min_val: 0.0,
                max_val: 99.0,
            },
            // Feature 2: categorical dense U8
            FeatureAnalysis {
                feature_idx: 2,
                is_numeric: false,
                is_sparse: false,
                needs_u16: false,
                is_trivial: false,
                is_binary: false,
                n_unique: 5,
                n_nonzero: 100,
                n_valid: 100,
                n_nan: 0,
                n_samples: 100,
                min_val: 0.0,
                max_val: 4.0,
            },
            // Feature 3: sparse numeric
            FeatureAnalysis {
                feature_idx: 3,
                is_numeric: true,
                is_sparse: true,
                needs_u16: false,
                is_trivial: false,
                is_binary: false,
                n_unique: 10,
                n_nonzero: 5,
                n_valid: 100,
                n_nan: 0,
                n_samples: 100,
                min_val: 0.0,
                max_val: 9.0,
            },
            // Feature 4: trivial
            FeatureAnalysis {
                feature_idx: 4,
                is_numeric: true,
                is_sparse: false,
                needs_u16: false,
                is_trivial: true,
                is_binary: false,
                n_unique: 1,
                n_nonzero: 0,
                n_valid: 100,
                n_nan: 0,
                n_samples: 100,
                min_val: 0.0,
                max_val: 0.0,
            },
        ];

        let result = compute_groups(&analyses, &default_config());

        // Should have:
        // - 1 numeric dense U8 group (features 0, 1)
        // - 1 categorical dense U8 group (feature 2)
        // - 1 sparse numeric group (feature 3)
        // - 1 trivial feature (feature 4)

        assert_eq!(result.groups.len(), 3);
        assert_eq!(result.trivial_features, vec![4]);

        // Find numeric dense group
        let num_dense = result
            .groups
            .iter()
            .find(|g| g.group_type == GroupType::NumericDense)
            .unwrap();
        assert_eq!(num_dense.feature_indices, vec![0, 1]);
        assert!(!num_dense.needs_u16);

        // Find categorical dense group
        let cat_dense = result
            .groups
            .iter()
            .find(|g| g.group_type == GroupType::CategoricalDense)
            .unwrap();
        assert_eq!(cat_dense.feature_indices, vec![2]);

        // Find sparse numeric group
        let sparse_num = result
            .groups
            .iter()
            .find(|g| g.group_type == GroupType::SparseNumeric)
            .unwrap();
        assert_eq!(sparse_num.feature_indices, vec![3]);
    }

    #[test]
    fn test_compute_groups_u16() {
        let analyses = vec![
            FeatureAnalysis {
                feature_idx: 0,
                is_numeric: true,
                is_sparse: false,
                needs_u16: true, // U16
                is_trivial: false,
                is_binary: false,
                n_unique: 500,
                n_nonzero: 100,
                n_valid: 100,
                n_nan: 0,
                n_samples: 100,
                min_val: 0.0,
                max_val: 499.0,
            },
            FeatureAnalysis {
                feature_idx: 1,
                is_numeric: true,
                is_sparse: false,
                needs_u16: false, // U8
                is_trivial: false,
                is_binary: false,
                n_unique: 100,
                n_nonzero: 100,
                n_valid: 100,
                n_nan: 0,
                n_samples: 100,
                min_val: 0.0,
                max_val: 99.0,
            },
        ];

        let result = compute_groups(&analyses, &default_config());

        // Should have 2 groups: one U8, one U16
        assert_eq!(result.groups.len(), 2);

        let u8_group = result.groups.iter().find(|g| !g.needs_u16).unwrap();
        let u16_group = result.groups.iter().find(|g| g.needs_u16).unwrap();

        assert_eq!(u8_group.feature_indices, vec![1]);
        assert_eq!(u16_group.feature_indices, vec![0]);
    }

    #[test]
    fn test_compute_groups_all_sparse() {
        let analyses = vec![
            FeatureAnalysis {
                feature_idx: 0,
                is_numeric: true,
                is_sparse: true,
                needs_u16: false,
                is_trivial: false,
                is_binary: false,
                n_unique: 10,
                n_nonzero: 5,
                n_valid: 100,
                n_nan: 0,
                n_samples: 100,
                min_val: 0.0,
                max_val: 9.0,
            },
            FeatureAnalysis {
                feature_idx: 1,
                is_numeric: false,
                is_sparse: true,
                needs_u16: false,
                is_trivial: false,
                is_binary: false,
                n_unique: 3,
                n_nonzero: 5,
                n_valid: 100,
                n_nan: 0,
                n_samples: 100,
                min_val: 0.0,
                max_val: 2.0,
            },
        ];

        let result = compute_groups(&analyses, &default_config());

        // Each sparse feature gets its own group
        assert_eq!(result.groups.len(), 2);
        assert!(
            result
                .groups
                .iter()
                .any(|g| g.group_type == GroupType::SparseNumeric)
        );
        assert!(
            result
                .groups
                .iter()
                .any(|g| g.group_type == GroupType::SparseCategorical)
        );
    }

    #[test]
    fn test_compute_groups_all_trivial() {
        let analyses = vec![
            FeatureAnalysis {
                feature_idx: 0,
                is_numeric: true,
                is_sparse: false,
                needs_u16: false,
                is_trivial: true,
                is_binary: false,
                n_unique: 0,
                n_nonzero: 0,
                n_valid: 0,
                n_nan: 0,
                n_samples: 100,
                min_val: f32::NAN,
                max_val: f32::NAN,
            },
            FeatureAnalysis {
                feature_idx: 1,
                is_numeric: true,
                is_sparse: false,
                needs_u16: false,
                is_trivial: true,
                is_binary: false,
                n_unique: 1,
                n_nonzero: 0,
                n_valid: 100,
                n_nan: 0,
                n_samples: 100,
                min_val: 0.0,
                max_val: 0.0,
            },
        ];

        let result = compute_groups(&analyses, &default_config());

        assert_eq!(result.groups.len(), 0);
        assert_eq!(result.trivial_features, vec![0, 1]);
    }

    #[test]
    fn test_compute_groups_empty() {
        let analyses: Vec<FeatureAnalysis> = vec![];
        let result = compute_groups(&analyses, &default_config());

        assert_eq!(result.groups.len(), 0);
        assert_eq!(result.trivial_features.len(), 0);
    }
}

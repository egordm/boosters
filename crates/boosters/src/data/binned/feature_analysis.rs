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

use super::BinData;
use crate::data::FeaturesView;

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
            max_categorical_cardinality: 256,
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
            sparsity_threshold: self.sparsity_threshold.unwrap_or(default.sparsity_threshold),
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
        if has_nan {
            base_bins + 1
        } else {
            base_bins
        }
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
}

/// Analyze all features in a data matrix.
///
/// Performs single-pass analysis to detect feature properties.
/// Analysis is parallelized across features using rayon.
///
/// # Arguments
///
/// * `features` - Feature-major view of the data matrix `[n_features, n_samples]`
/// * `config` - Binning configuration
/// * `metadata` - Optional user-provided metadata for categorical specification
///
/// # Returns
///
/// Vector of `FeatureAnalysis`, one per feature.
pub fn analyze_features(
    features: FeaturesView<'_>,
    config: &BinningConfig,
    metadata: Option<&FeatureMetadata>,
) -> Vec<FeatureAnalysis> {
    let n_samples = features.n_samples();
    let n_features = features.n_features();

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
                n_samples,
                min_val: f32::NAN,
                max_val: f32::NAN,
            })
            .collect();
    }

    // Track enough unique values for categorical detection + some margin
    let max_unique = (config.max_categorical_cardinality as usize) + 10;

    (0..n_features)
        .into_par_iter()
        .map(|col| {
            let mut stats = FeatureStats::new(max_unique);
            let feature_values = features.feature(col);

            for &val in feature_values.iter() {
                stats.update(val);
            }

            stats.into_analysis(col, n_samples, config, metadata)
        })
        .collect()
}

/// Analyze features sequentially (for small datasets or testing).
pub fn analyze_features_sequential(
    features: FeaturesView<'_>,
    config: &BinningConfig,
    metadata: Option<&FeatureMetadata>,
) -> Vec<FeatureAnalysis> {
    let n_samples = features.n_samples();
    let n_features = features.n_features();

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
                n_samples,
                min_val: f32::NAN,
                max_val: f32::NAN,
            })
            .collect();
    }

    let max_unique = (config.max_categorical_cardinality as usize) + 10;

    (0..n_features)
        .map(|col| {
            let mut stats = FeatureStats::new(max_unique);
            let feature_values = features.feature(col);

            for &val in feature_values.iter() {
                stats.update(val);
            }

            stats.into_analysis(col, n_samples, config, metadata)
        })
        .collect()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::FeaturesView;

    fn make_features(data: &[f32], n_samples: usize, n_features: usize) -> FeaturesView<'_> {
        FeaturesView::from_slice(data, n_samples, n_features).unwrap()
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
        assert_eq!(config.max_categorical_cardinality, 256);
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
        let features = make_features(&data, 4, 1);
        let analysis = analyze_features_sequential(features, &default_config(), None);

        assert!(analysis[0].is_binary);
        assert!(analysis[0].is_numeric); // Binary defaults to numeric
        assert!(!analysis[0].is_trivial);
        assert_eq!(analysis[0].n_unique, 2);
    }

    #[test]
    fn test_categorical_detection_integers() {
        // Low cardinality integers should be auto-detected as categorical
        let data = [0.0f32, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0];
        let features = make_features(&data, 8, 1);
        let analysis = analyze_features_sequential(features, &default_config(), None);

        assert!(!analysis[0].is_numeric); // Should be categorical
        assert!(!analysis[0].is_binary);
        assert_eq!(analysis[0].n_unique, 3);
    }

    #[test]
    fn test_user_specified_categorical() {
        // User specifies categorical, should override auto-detection
        let data = [0.0f32, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5];
        let features = make_features(&data, 8, 1);
        let metadata = FeatureMetadata::with_categorical(vec![0]);
        let analysis =
            analyze_features_sequential(features, &default_config(), Some(&metadata));

        assert!(!analysis[0].is_numeric); // User said categorical
    }

    #[test]
    fn test_numeric_floats() {
        // Non-integer values should be numeric
        let data = [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let features = make_features(&data, 8, 1);
        let analysis = analyze_features_sequential(features, &default_config(), None);

        assert!(analysis[0].is_numeric);
    }

    #[test]
    fn test_trivial_all_zeros() {
        let data = [0.0f32, 0.0, 0.0, 0.0];
        let features = make_features(&data, 4, 1);
        let analysis = analyze_features_sequential(features, &default_config(), None);

        assert!(analysis[0].is_trivial);
        assert_eq!(analysis[0].n_unique, 1);
        assert_eq!(analysis[0].n_nonzero, 0);
    }

    #[test]
    fn test_trivial_all_nan() {
        let data = [f32::NAN, f32::NAN, f32::NAN, f32::NAN];
        let features = make_features(&data, 4, 1);
        let analysis = analyze_features_sequential(features, &default_config(), None);

        assert!(analysis[0].is_trivial);
        assert_eq!(analysis[0].n_unique, 0);
        assert_eq!(analysis[0].n_valid, 0);
    }

    #[test]
    fn test_sparse_detection() {
        // 1 non-zero out of 10 = 10% density = 90% sparse
        let data = [0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let features = make_features(&data, 10, 1);
        let analysis = analyze_features_sequential(features, &default_config(), None);

        assert!(analysis[0].is_sparse); // Default threshold is 0.9
        assert!((analysis[0].density() - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_not_sparse() {
        // 5 non-zero out of 10 = 50% density
        let data = [0.0f32, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let features = make_features(&data, 10, 1);
        let analysis = analyze_features_sequential(features, &default_config(), None);

        assert!(!analysis[0].is_sparse);
    }

    #[test]
    fn test_needs_u16() {
        // If we have many unique values and high max_bins, we might need U16
        let config = BinningConfig::builder().max_bins(500).build();
        let data: Vec<f32> = (0..300).map(|i| i as f32).collect();
        let features = make_features(&data, 300, 1);
        let analysis = analyze_features_sequential(features, &config, None);

        assert!(analysis[0].needs_u16); // 300 unique values > 256
    }

    #[test]
    fn test_no_u16_needed() {
        // Default max_bins is 256, so even with many values we stay U8
        let data: Vec<f32> = (0..300).map(|i| i as f32).collect();
        let features = make_features(&data, 300, 1);
        let analysis = analyze_features_sequential(features, &default_config(), None);

        // Default max_bins is 256, but 300 unique values get capped
        // Actually, numeric features use quantile binning limited by max_bins
        assert!(!analysis[0].needs_u16);
    }

    #[test]
    fn test_min_max() {
        let data = [1.0f32, 5.0, 3.0, 2.0, 4.0];
        let features = make_features(&data, 5, 1);
        let analysis = analyze_features_sequential(features, &default_config(), None);

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
        let features = make_features(&data, 4, 3);
        let analysis = analyze_features_sequential(features, &default_config(), None);

        // Feature 0: binary → numeric
        assert!(analysis[0].is_binary);
        assert!(analysis[0].is_numeric);

        // Feature 1: integer → categorical
        assert!(!analysis[1].is_binary);
        assert!(!analysis[1].is_numeric); // Auto-detected categorical

        // Feature 2: floats → numeric
        assert!(!analysis[2].is_binary);
        assert!(analysis[2].is_numeric);
    }

    #[test]
    fn test_parallel_matches_sequential() {
        let data = [
            0.0f32, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0, -1.0, 1.0,
        ];
        let features = make_features(&data, 4, 4);
        let config = default_config();

        let seq = analyze_features_sequential(features, &config, None);
        let par = analyze_features(features, &config, None);

        assert_eq!(seq, par);
    }

    #[test]
    fn test_empty_features() {
        let data: [f32; 0] = [];
        let features = make_features(&data, 0, 3);
        let analysis = analyze_features_sequential(features, &default_config(), None);

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
        let features = make_features(&data, 20, 1);
        let analysis = analyze_features_sequential(features, &config, None);

        assert!(analysis[0].is_numeric); // Too many categories → numeric
    }
}

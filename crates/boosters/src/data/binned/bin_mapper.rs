//! Bin mapper types for mapping values to bin indices.
//!
//! This module contains the core types for describing how feature values
//! are mapped to discrete bins:
//! - [`BinMapper`] - Maps values to bins and back
//! - [`MissingType`] - How missing values are handled
//! - [`FeatureType`] - Numerical vs categorical features

// Allow dead_code during migration - will be used when builder is implemented in Epic 4
#![allow(dead_code)]

use std::collections::HashMap;

// ============================================================================
// MissingType
// ============================================================================

/// How missing values are handled for a feature.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MissingType {
    /// No missing values in this feature.
    #[default]
    None,
    /// Zeros are treated as missing (useful for sparse data).
    Zero,
    /// NaN values get a dedicated bin (typically the last bin).
    NaN,
}

// ============================================================================
// FeatureType
// ============================================================================

/// Whether a feature is numerical or categorical.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FeatureType {
    /// Continuous numerical feature with ordered bins.
    #[default]
    Numerical,
    /// Categorical feature where bins represent unordered categories.
    Categorical,
}

// ============================================================================
// BinMapper
// ============================================================================

/// Mapping from continuous values to bin indices.
///
/// Stores the bin boundaries and metadata needed to:
/// 1. Map new values to bins during prediction
/// 2. Convert bin indices back to split thresholds for tree output
/// 3. Handle missing values and categorical features
#[derive(Clone, Debug)]
pub struct BinMapper {
    /// Upper bounds for each bin (numerical features only).
    /// Value v maps to first bin where v <= bound[bin].
    bin_upper_bounds: Box<[f64]>,

    /// Categorical: maps category value → bin index.
    /// Only populated for categorical features.
    cat_to_bin: Option<HashMap<i32, u32>>,

    /// Categorical: maps bin index → category value.
    /// Only populated for categorical features.
    bin_to_cat: Option<Box<[i32]>>,

    /// Number of bins (including missing bin if applicable).
    n_bins: u32,

    /// How missing values are handled.
    missing_type: MissingType,

    /// Bin index for missing/default values.
    default_bin: u32,

    /// Most frequent bin (enables histogram subtraction optimization).
    most_freq_bin: u32,

    /// Feature type (numerical vs categorical).
    feature_type: FeatureType,

    /// Fraction of values that are zero or missing (0.0 to 1.0).
    /// Used to decide sparse vs dense representation.
    sparse_rate: f64,

    /// Min/max values seen during binning (numerical only).
    min_val: f64,
    max_val: f64,
}

impl BinMapper {
    /// Create a bin mapper for a numerical feature.
    pub fn numerical(
        bin_upper_bounds: Vec<f64>,
        missing_type: MissingType,
        default_bin: u32,
        most_freq_bin: u32,
        sparse_rate: f64,
        min_val: f64,
        max_val: f64,
    ) -> Self {
        let n_bins = bin_upper_bounds.len() as u32;
        Self {
            bin_upper_bounds: bin_upper_bounds.into_boxed_slice(),
            cat_to_bin: None,
            bin_to_cat: None,
            n_bins,
            missing_type,
            default_bin,
            most_freq_bin,
            feature_type: FeatureType::Numerical,
            sparse_rate,
            min_val,
            max_val,
        }
    }

    /// Create a bin mapper for a categorical feature.
    pub fn categorical(
        categories: Vec<i32>,
        missing_type: MissingType,
        default_bin: u32,
        most_freq_bin: u32,
        sparse_rate: f64,
    ) -> Self {
        let n_bins = categories.len() as u32;
        let cat_to_bin: HashMap<i32, u32> = categories
            .iter()
            .enumerate()
            .map(|(bin, &cat)| (cat, bin as u32))
            .collect();
        let bin_to_cat = categories.into_boxed_slice();

        Self {
            bin_upper_bounds: Box::new([]),
            cat_to_bin: Some(cat_to_bin),
            bin_to_cat: Some(bin_to_cat),
            n_bins,
            missing_type,
            default_bin,
            most_freq_bin,
            feature_type: FeatureType::Categorical,
            sparse_rate,
            min_val: 0.0,
            max_val: 0.0,
        }
    }

    /// Number of bins.
    #[inline]
    pub fn n_bins(&self) -> u32 {
        self.n_bins
    }

    /// Whether this feature requires U16 bins (more than 256 bins).
    ///
    /// This is a convenience method for deciding storage type:
    /// - `false` → can use `BinData::U8`
    /// - `true` → must use `BinData::U16`
    #[inline]
    pub fn needs_u16(&self) -> bool {
        self.n_bins > 256
    }

    /// Feature type.
    #[inline]
    pub fn feature_type(&self) -> FeatureType {
        self.feature_type
    }

    /// Check if this is a categorical feature.
    #[inline]
    pub fn is_categorical(&self) -> bool {
        self.feature_type == FeatureType::Categorical
    }

    /// Missing value handling type.
    #[inline]
    pub fn missing_type(&self) -> MissingType {
        self.missing_type
    }

    /// Bin for missing/default values.
    #[inline]
    pub fn default_bin(&self) -> u32 {
        self.default_bin
    }

    /// Most frequent bin (for subtraction trick optimization).
    #[inline]
    pub fn most_freq_bin(&self) -> u32 {
        self.most_freq_bin
    }

    /// Sparsity rate (fraction of zeros/missing).
    #[inline]
    pub fn sparse_rate(&self) -> f64 {
        self.sparse_rate
    }

    /// Check if this feature is trivial (only one bin, no splits possible).
    #[inline]
    pub fn is_trivial(&self) -> bool {
        self.n_bins <= 1
    }

    /// Map a continuous value to a bin index (numerical features).
    #[inline]
    pub fn value_to_bin(&self, value: f64) -> u32 {
        // Handle missing values
        if value.is_nan() {
            return match self.missing_type {
                MissingType::NaN => self.n_bins - 1, // Last bin for NaN
                _ => self.default_bin,
            };
        }

        if value == 0.0 && self.missing_type == MissingType::Zero {
            return self.default_bin;
        }

        if self.is_categorical() {
            // Categorical: look up in map
            let cat = value as i32;
            self.cat_to_bin
                .as_ref()
                .and_then(|m| m.get(&cat).copied())
                .unwrap_or(0) // Unknown category → bin 0
        } else {
            // Numerical: binary search
            self.search_bin(value)
        }
    }

    /// Binary search for bin index (numerical features).
    #[inline]
    fn search_bin(&self, value: f64) -> u32 {
        let bounds = &self.bin_upper_bounds;
        let mut lo = 0usize;
        let mut hi = bounds.len().saturating_sub(1);

        // Handle NaN bin if present
        if self.missing_type == MissingType::NaN {
            hi = hi.saturating_sub(1);
        }

        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if value <= bounds[mid] {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }

        lo as u32
    }

    /// Get the upper bound for a bin (for converting splits back to thresholds).
    #[inline]
    pub fn bin_to_value(&self, bin: u32) -> f64 {
        if self.is_categorical() {
            self.bin_to_cat
                .as_ref()
                .map(|cats| cats[bin as usize] as f64)
                .unwrap_or(0.0)
        } else {
            self.bin_upper_bounds[bin as usize]
        }
    }

    /// Get the midpoint of a bin (for feature value approximation).
    ///
    /// For numerical features:
    /// - Bin 0: `(min_val + upper_bound[0]) / 2`
    /// - Bin n: `(upper_bound[n-1] + upper_bound[n]) / 2`
    /// - Last bin: Uses `max_val` instead of the sentinel `f64::MAX` upper bound
    ///
    /// For categorical features, returns the bin index (which is the category index
    /// used during tree traversal). This differs from the original category value.
    /// For missing/NaN bins, returns `f64::NAN`.
    ///
    /// **Note**: For linear trees, prefer using raw feature values stored in
    /// `NumericStorage` rather than midpoint approximations.
    #[inline]
    pub fn bin_to_midpoint(&self, bin: u32) -> f64 {
        if self.is_categorical() {
            // Categorical: return bin index directly (used as category index in trees)
            return bin as f64;
        }

        // Handle NaN bin (typically last bin when missing_type == NaN)
        if self.missing_type == MissingType::NaN && bin == self.n_bins - 1 {
            return f64::NAN;
        }

        let bin_idx = bin as usize;
        let upper = self.bin_upper_bounds[bin_idx];

        // Use max_val for the last bin to avoid f64::MAX sentinel values
        let effective_upper = if upper.is_infinite() || upper > self.max_val {
            self.max_val
        } else {
            upper
        };

        let lower = if bin_idx == 0 {
            self.min_val
        } else {
            self.bin_upper_bounds[bin_idx - 1]
        };

        (lower + effective_upper) / 2.0
    }

    /// Get max category value (categorical only).
    pub fn max_cat_value(&self) -> Option<i32> {
        self.bin_to_cat
            .as_ref()
            .and_then(|cats| cats.iter().max().copied())
    }

    /// Min value seen during binning (numerical only).
    #[inline]
    pub fn min_val(&self) -> f64 {
        self.min_val
    }

    /// Max value seen during binning (numerical only).
    #[inline]
    pub fn max_val(&self) -> f64 {
        self.max_val
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------------
    // Numerical BinMapper Tests
    // ------------------------------------------------------------------------

    fn create_numeric_mapper() -> BinMapper {
        // 4 bins with boundaries: [1.0, 2.0, 3.0, 4.0]
        // bin 0: (-∞, 1.0]
        // bin 1: (1.0, 2.0]
        // bin 2: (2.0, 3.0]
        // bin 3: (3.0, 4.0]
        BinMapper::numerical(
            vec![1.0, 2.0, 3.0, 4.0],
            MissingType::None,
            0,
            0,
            0.0,
            0.5, // min_val
            3.5, // max_val
        )
    }

    #[test]
    fn test_numerical_value_to_bin() {
        let mapper = create_numeric_mapper();

        // Values within bins
        assert_eq!(mapper.value_to_bin(0.5), 0);
        assert_eq!(mapper.value_to_bin(1.0), 0);
        assert_eq!(mapper.value_to_bin(1.5), 1);
        assert_eq!(mapper.value_to_bin(2.0), 1);
        assert_eq!(mapper.value_to_bin(2.5), 2);
        assert_eq!(mapper.value_to_bin(3.0), 2);
        assert_eq!(mapper.value_to_bin(3.5), 3);
    }

    #[test]
    fn test_numerical_bin_to_value() {
        let mapper = create_numeric_mapper();

        assert_eq!(mapper.bin_to_value(0), 1.0);
        assert_eq!(mapper.bin_to_value(1), 2.0);
        assert_eq!(mapper.bin_to_value(2), 3.0);
        assert_eq!(mapper.bin_to_value(3), 4.0);
    }

    #[test]
    fn test_numerical_bin_to_midpoint() {
        let mapper = create_numeric_mapper();

        // bin 0: (min_val + upper[0]) / 2 = (0.5 + 1.0) / 2 = 0.75
        assert!((mapper.bin_to_midpoint(0) - 0.75).abs() < 1e-10);

        // bin 1: (upper[0] + upper[1]) / 2 = (1.0 + 2.0) / 2 = 1.5
        assert!((mapper.bin_to_midpoint(1) - 1.5).abs() < 1e-10);

        // bin 2: (upper[1] + upper[2]) / 2 = (2.0 + 3.0) / 2 = 2.5
        assert!((mapper.bin_to_midpoint(2) - 2.5).abs() < 1e-10);

        // bin 3: (upper[2] + max_val) / 2 = (3.0 + 3.5) / 2 = 3.25
        // (uses max_val instead of 4.0 since 4.0 > max_val)
        assert!((mapper.bin_to_midpoint(3) - 3.25).abs() < 1e-10);
    }

    #[test]
    fn test_numerical_with_nan_missing() {
        // 3 bins + NaN bin
        let mapper = BinMapper::numerical(
            vec![1.0, 2.0, 3.0, f64::INFINITY], // Last is NaN bin
            MissingType::NaN,
            3, // default_bin
            0,
            0.0,
            0.0,
            3.0,
        );

        assert_eq!(mapper.n_bins(), 4);
        assert_eq!(mapper.value_to_bin(f64::NAN), 3); // NaN goes to last bin
        assert!(mapper.bin_to_midpoint(3).is_nan()); // NaN bin returns NaN
    }

    #[test]
    fn test_numerical_with_zero_missing() {
        let mapper = BinMapper::numerical(
            vec![1.0, 2.0, 3.0],
            MissingType::Zero,
            0, // default_bin for zeros
            0,
            0.5, // 50% sparse
            0.0,
            3.0,
        );

        assert_eq!(mapper.value_to_bin(0.0), 0); // Zero goes to default bin
        assert_eq!(mapper.value_to_bin(1.5), 1); // Non-zero works normally
    }

    // ------------------------------------------------------------------------
    // Categorical BinMapper Tests
    // ------------------------------------------------------------------------

    fn create_categorical_mapper() -> BinMapper {
        // Categories: 10, 20, 30 map to bins 0, 1, 2
        BinMapper::categorical(vec![10, 20, 30], MissingType::None, 0, 0, 0.0)
    }

    #[test]
    fn test_categorical_value_to_bin() {
        let mapper = create_categorical_mapper();

        assert!(mapper.is_categorical());
        assert_eq!(mapper.value_to_bin(10.0), 0);
        assert_eq!(mapper.value_to_bin(20.0), 1);
        assert_eq!(mapper.value_to_bin(30.0), 2);

        // Unknown category maps to bin 0
        assert_eq!(mapper.value_to_bin(99.0), 0);
    }

    #[test]
    fn test_categorical_bin_to_value() {
        let mapper = create_categorical_mapper();

        assert_eq!(mapper.bin_to_value(0), 10.0);
        assert_eq!(mapper.bin_to_value(1), 20.0);
        assert_eq!(mapper.bin_to_value(2), 30.0);
    }

    #[test]
    fn test_categorical_bin_to_midpoint() {
        let mapper = create_categorical_mapper();

        // Categorical returns bin index as midpoint
        assert_eq!(mapper.bin_to_midpoint(0), 0.0);
        assert_eq!(mapper.bin_to_midpoint(1), 1.0);
        assert_eq!(mapper.bin_to_midpoint(2), 2.0);
    }

    #[test]
    fn test_categorical_max_cat_value() {
        let mapper = create_categorical_mapper();
        assert_eq!(mapper.max_cat_value(), Some(30));
    }

    // ------------------------------------------------------------------------
    // Common Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_n_bins() {
        let numerical = create_numeric_mapper();
        assert_eq!(numerical.n_bins(), 4);

        let categorical = create_categorical_mapper();
        assert_eq!(categorical.n_bins(), 3);
    }

    #[test]
    fn test_needs_u16() {
        // Small bins - U8 is sufficient
        let small = BinMapper::numerical(
            (0..100).map(|i| i as f64).collect(),
            MissingType::None,
            0,
            0,
            0.0,
            0.0,
            100.0,
        );
        assert!(!small.needs_u16());
        assert_eq!(small.n_bins(), 100);

        // Exactly 256 bins - U8 is sufficient
        let boundary = BinMapper::numerical(
            (0..256).map(|i| i as f64).collect(),
            MissingType::None,
            0,
            0,
            0.0,
            0.0,
            256.0,
        );
        assert!(!boundary.needs_u16());

        // More than 256 bins - needs U16
        let large = BinMapper::numerical(
            (0..500).map(|i| i as f64).collect(),
            MissingType::None,
            0,
            0,
            0.0,
            0.0,
            500.0,
        );
        assert!(large.needs_u16());
        assert_eq!(large.n_bins(), 500);
    }

    #[test]
    fn test_is_trivial() {
        // 1 bin is trivial (no splits possible)
        let trivial = BinMapper::numerical(vec![1.0], MissingType::None, 0, 0, 0.0, 0.0, 1.0);
        assert!(trivial.is_trivial());

        // 0 bins is trivial
        let empty = BinMapper::numerical(vec![], MissingType::None, 0, 0, 0.0, 0.0, 0.0);
        assert!(empty.is_trivial());

        // 2+ bins is not trivial
        let normal = create_numeric_mapper();
        assert!(!normal.is_trivial());
    }

    #[test]
    fn test_feature_type() {
        let numerical = create_numeric_mapper();
        assert_eq!(numerical.feature_type(), FeatureType::Numerical);
        assert!(!numerical.is_categorical());

        let categorical = create_categorical_mapper();
        assert_eq!(categorical.feature_type(), FeatureType::Categorical);
        assert!(categorical.is_categorical());
    }

    #[test]
    fn test_sparse_rate() {
        let sparse = BinMapper::numerical(
            vec![1.0, 2.0],
            MissingType::Zero,
            0,
            0,
            0.8, // 80% sparse
            0.0,
            2.0,
        );
        assert!((sparse.sparse_rate() - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_min_max_val() {
        let mapper = create_numeric_mapper();
        assert_eq!(mapper.min_val(), 0.5);
        assert_eq!(mapper.max_val(), 3.5);
    }
}

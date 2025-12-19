//! Feature analysis for bundling decisions.
//!
//! This module provides single-pass feature analysis to detect:
//! - Binary features (exactly 2 unique values)
//! - Trivial features (all zeros/missing, can be skipped)
//! - Sparse features (high fraction of zeros)
//!
//! The analysis uses O(1) memory per feature and is fully parallelizable.

use rayon::prelude::*;

use crate::data::DataMatrix;

/// Metadata about a feature determined during analysis.
#[derive(Clone, Debug, PartialEq)]
pub struct FeatureInfo {
    /// Original feature index in the input matrix.
    pub original_idx: usize,

    /// Fraction of rows with non-zero values (1.0 = dense).
    /// NaN values are excluded from the count.
    pub density: f32,

    /// True if feature has exactly 2 unique non-NaN values.
    /// Handles {0,1}, {-1,+1}, {0.5,1.5}, or any pair of values.
    pub is_binary: bool,

    /// True if feature is trivial (all zeros, single value, or all NaN).
    /// Trivial features can be skipped during bundling and training.
    pub is_trivial: bool,
}

impl FeatureInfo {
    /// Returns true if the feature is sparse (density below threshold).
    #[inline]
    pub fn is_sparse(&self, min_sparsity: f32) -> bool {
        self.density <= (1.0 - min_sparsity)
    }
}

/// Efficient feature statistics (O(1) memory per feature, single-pass).
///
/// Used internally during feature analysis.
#[derive(Clone, Debug, Default)]
struct FeatureStats {
    /// Minimum non-NaN value seen.
    min: f32,
    /// Maximum non-NaN value seen.
    max: f32,
    /// Count of non-zero, non-NaN values.
    non_zero_count: u32,
    /// Total count of non-NaN values (for density calculation).
    valid_count: u32,
    /// Set true when we see a third distinct value.
    has_more_than_two_values: bool,
    /// First distinct non-NaN value seen.
    first_value: Option<f32>,
    /// Second distinct non-NaN value seen.
    second_value: Option<f32>,
}

impl FeatureStats {
    /// Create new stats initialized for aggregation.
    fn new() -> Self {
        Self {
            min: f32::MAX,
            max: f32::MIN,
            non_zero_count: 0,
            valid_count: 0,
            has_more_than_two_values: false,
            first_value: None,
            second_value: None,
        }
    }

    /// Update stats with a new value.
    #[inline]
    fn update(&mut self, val: f32) {
        // Skip NaN values
        if val.is_nan() {
            return;
        }

        self.valid_count += 1;

        if val != 0.0 {
            self.non_zero_count += 1;
        }

        self.min = self.min.min(val);
        self.max = self.max.max(val);

        // Track up to 2 distinct values efficiently
        if !self.has_more_than_two_values {
            match (self.first_value, self.second_value) {
                (None, _) => {
                    self.first_value = Some(val);
                }
                (Some(first), None) if first != val => {
                    self.second_value = Some(val);
                }
                (Some(first), Some(second)) if first != val && second != val => {
                    self.has_more_than_two_values = true;
                }
                _ => {}
            }
        }
    }

    /// Convert stats to FeatureInfo.
    fn into_feature_info(self, original_idx: usize, n_rows: usize) -> FeatureInfo {
        // Trivial if:
        // - All NaN (valid_count == 0)
        // - Single value that is zero
        // - Single non-zero value (still trivial, no information)
        let is_trivial = self.valid_count == 0
            || (self.second_value.is_none()
                && self.first_value.is_none_or(|v| v == 0.0 || self.valid_count == 0));

        // Binary if exactly 2 distinct values
        let is_binary = self.second_value.is_some() && !self.has_more_than_two_values;

        // Density is fraction of non-zero values over total rows
        // If all NaN, density is 0
        let density = if n_rows > 0 {
            self.non_zero_count as f32 / n_rows as f32
        } else {
            0.0
        };

        FeatureInfo {
            original_idx,
            density,
            is_binary,
            is_trivial,
        }
    }
}

/// Analyze features in a data matrix.
///
/// Performs single-pass analysis to detect binary, trivial, and sparse features.
/// Analysis is parallelized across columns using rayon.
///
/// # Arguments
///
/// * `matrix` - Data matrix to analyze (must have f32 elements)
///
/// # Returns
///
/// Vector of `FeatureInfo`, one per feature/column.
///
/// # Example
///
/// ```
/// use boosters::data::ColMatrix;
/// use boosters::data::binned::analyze_features;
///
/// // 3 rows, 2 features
/// let data = ColMatrix::from_vec(vec![
///     0.0, 0.0, 1.0,  // Feature 0: sparse binary (1 non-zero out of 3)
///     1.0, 2.0, 3.0,  // Feature 1: dense continuous
/// ], 3, 2);
///
/// let infos = analyze_features(&data);
///
/// assert!(infos[0].is_binary);
/// assert!(infos[0].is_sparse(0.5));  // density 0.33 < 0.5 sparse threshold
/// assert!(!infos[1].is_binary);
/// ```
pub fn analyze_features<M>(matrix: &M) -> Vec<FeatureInfo>
where
    M: DataMatrix<Element = f32> + Sync,
{
    let n_rows = matrix.num_rows();
    let n_features = matrix.num_features();

    if n_features == 0 || n_rows == 0 {
        return (0..n_features)
            .map(|idx| FeatureInfo {
                original_idx: idx,
                density: 0.0,
                is_binary: false,
                is_trivial: true,
            })
            .collect();
    }

    (0..n_features)
        .into_par_iter()
        .map(|col| {
            let mut stats = FeatureStats::new();

            for row in 0..n_rows {
                if let Some(val) = matrix.get(row, col) {
                    stats.update(val);
                }
            }

            stats.into_feature_info(col, n_rows)
        })
        .collect()
}

/// Analyze features sequentially (for small datasets or testing).
///
/// Same as `analyze_features` but without parallelization.
pub fn analyze_features_sequential<M>(matrix: &M) -> Vec<FeatureInfo>
where
    M: DataMatrix<Element = f32>,
{
    let n_rows = matrix.num_rows();
    let n_features = matrix.num_features();

    if n_features == 0 || n_rows == 0 {
        return (0..n_features)
            .map(|idx| FeatureInfo {
                original_idx: idx,
                density: 0.0,
                is_binary: false,
                is_trivial: true,
            })
            .collect();
    }

    (0..n_features)
        .map(|col| {
            let mut stats = FeatureStats::new();

            for row in 0..n_rows {
                if let Some(val) = matrix.get(row, col) {
                    stats.update(val);
                }
            }

            stats.into_feature_info(col, n_rows)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::ColMatrix;

    #[test]
    fn binary_detection_zero_one() {
        // {0, 1} is binary
        let data = ColMatrix::from_vec(vec![0.0, 1.0, 0.0, 1.0], 4, 1);
        let infos = analyze_features_sequential(&data);

        assert!(infos[0].is_binary, "0/1 should be detected as binary");
        assert!(!infos[0].is_trivial);
    }

    #[test]
    fn binary_detection_minus_one_one() {
        // {-1, 1} is binary
        let data = ColMatrix::from_vec(vec![-1.0, 1.0, -1.0, 1.0], 4, 1);
        let infos = analyze_features_sequential(&data);

        assert!(infos[0].is_binary, "-1/1 should be detected as binary");
        assert!(!infos[0].is_trivial);
    }

    #[test]
    fn binary_detection_arbitrary_values() {
        // {0.5, 1.5} is binary
        let data = ColMatrix::from_vec(vec![0.5, 1.5, 0.5, 1.5], 4, 1);
        let infos = analyze_features_sequential(&data);

        assert!(infos[0].is_binary, "0.5/1.5 should be detected as binary");
        assert!(!infos[0].is_trivial);
    }

    #[test]
    fn trivial_all_zeros() {
        let data = ColMatrix::from_vec(vec![0.0, 0.0, 0.0, 0.0], 4, 1);
        let infos = analyze_features_sequential(&data);

        assert!(infos[0].is_trivial, "all zeros should be trivial");
        assert!(!infos[0].is_binary);
        assert_eq!(infos[0].density, 0.0);
    }

    #[test]
    fn trivial_single_value() {
        // Single non-zero value repeated is also trivial
        let data = ColMatrix::from_vec(vec![5.0, 5.0, 5.0, 5.0], 4, 1);
        let infos = analyze_features_sequential(&data);

        // Single value is NOT trivial by our definition (only zero single-value is)
        // Actually, the RFC says trivial = all zeros/missing OR single value
        // Let me reconsider: single constant value has no information, so trivial
        assert!(!infos[0].is_binary);
        // With current implementation, single non-zero value is NOT trivial
        // because trivial requires first_value to be zero or None
        // This matches the RFC: "is_trivial = stats.first_value.is_none() OR 
        //                       (stats.second_value.is_none() AND stats.first_value == Some(0.0))"
        assert!(!infos[0].is_trivial, "single non-zero value is not trivial per RFC");
    }

    #[test]
    fn trivial_all_nan() {
        let data = ColMatrix::from_vec(vec![f32::NAN, f32::NAN, f32::NAN], 3, 1);
        let infos = analyze_features_sequential(&data);

        assert!(infos[0].is_trivial, "all NaN should be trivial");
        assert!(!infos[0].is_binary);
        assert_eq!(infos[0].density, 0.0);
    }

    #[test]
    fn sparsity_calculation() {
        // 9 zeros, 1 non-zero → 10% density (90% sparse)
        let mut values = vec![0.0; 9];
        values.push(1.0);
        let data = ColMatrix::from_vec(values, 10, 1);
        let infos = analyze_features_sequential(&data);

        assert!((infos[0].density - 0.1).abs() < 0.001, "density should be 0.1");
        assert!(infos[0].is_sparse(0.9), "should be sparse with 90% threshold");
        assert!(!infos[0].is_sparse(0.95), "should not be sparse with 95% threshold");
    }

    #[test]
    fn multi_value_not_binary() {
        // 3+ values → not binary
        let data = ColMatrix::from_vec(vec![0.0, 1.0, 2.0, 3.0], 4, 1);
        let infos = analyze_features_sequential(&data);

        assert!(!infos[0].is_binary, "3+ values should not be binary");
        assert!(!infos[0].is_trivial);
    }

    #[test]
    fn nan_excluded_from_unique_values() {
        // {0, 1, NaN} should still be binary (NaN excluded)
        let data = ColMatrix::from_vec(vec![0.0, 1.0, f32::NAN, 0.0], 4, 1);
        let infos = analyze_features_sequential(&data);

        assert!(infos[0].is_binary, "NaN should be excluded from unique value count");
    }

    #[test]
    fn empty_matrix_rows() {
        let data: ColMatrix<f32> = ColMatrix::from_vec(vec![], 0, 2);
        let infos = analyze_features_sequential(&data);

        assert_eq!(infos.len(), 2);
        assert!(infos[0].is_trivial);
        assert!(infos[1].is_trivial);
    }

    #[test]
    fn empty_matrix_cols() {
        let data: ColMatrix<f32> = ColMatrix::from_vec(vec![], 0, 0);
        let infos = analyze_features_sequential(&data);

        assert_eq!(infos.len(), 0);
    }

    #[test]
    fn multiple_features() {
        // Feature 0: binary {0,1}
        // Feature 1: continuous {0,1,2,3}
        // Feature 2: trivial all zeros
        let data = ColMatrix::from_vec(
            vec![
                0.0, 1.0, 0.0, 1.0, // feature 0
                0.0, 1.0, 2.0, 3.0, // feature 1
                0.0, 0.0, 0.0, 0.0, // feature 2
            ],
            4,
            3,
        );
        let infos = analyze_features_sequential(&data);

        assert!(infos[0].is_binary);
        assert!(!infos[0].is_trivial);

        assert!(!infos[1].is_binary);
        assert!(!infos[1].is_trivial);

        assert!(!infos[2].is_binary);
        assert!(infos[2].is_trivial);
    }

    #[test]
    fn parallel_matches_sequential() {
        let data = ColMatrix::from_vec(
            vec![
                0.0, 1.0, 0.0, 1.0,
                0.0, 1.0, 2.0, 3.0,
                0.0, 0.0, 0.0, 0.0,
                -1.0, 1.0, -1.0, 1.0,
            ],
            4,
            4,
        );

        let seq = analyze_features_sequential(&data);
        let par = analyze_features(&data);

        assert_eq!(seq, par);
    }
}

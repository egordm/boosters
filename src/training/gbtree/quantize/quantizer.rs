//! Quantization strategies and the Quantizer.

use std::sync::Arc;

use rayon::prelude::*;

use super::cuts::{BinCuts, BinIndex, CategoricalInfo};
use super::matrix::QuantizedMatrix;
use crate::data::ColumnAccess;

// ============================================================================
// CutFinder trait and implementations
// ============================================================================

/// Strategy for computing bin boundaries.
pub trait CutFinder {
    /// Compute bin cuts from feature data.
    ///
    /// # Arguments
    ///
    /// * `data` - Feature matrix with column access
    /// * `max_bins` - Maximum number of bins per feature (including missing bin)
    fn find_cuts<D>(&self, data: &D, max_bins: usize) -> BinCuts
    where
        D: ColumnAccess<Element = f32> + Sync;

    /// Compute bin cuts with categorical feature information.
    ///
    /// For categorical features, bins represent categories directly.
    /// For numerical features, standard quantile-based cuts are used.
    fn find_cuts_with_categorical<D>(
        &self,
        data: &D,
        max_bins: usize,
        cat_info: &CategoricalInfo,
    ) -> BinCuts
    where
        D: ColumnAccess<Element = f32> + Sync,
    {
        // Default implementation: ignore categorical info
        let _ = cat_info;
        self.find_cuts(data, max_bins)
    }
}

/// Exact quantile computation for bin boundaries.
///
/// Sorts each feature and picks evenly-spaced quantile values as cut points.
/// Best for small to medium datasets (< 1M rows).
///
/// For larger datasets, consider streaming quantile sketches (future work).
#[derive(Debug, Clone)]
pub struct ExactQuantileCuts {
    /// Minimum samples per bin (to avoid very sparse bins).
    pub min_samples_per_bin: usize,
}

impl Default for ExactQuantileCuts {
    fn default() -> Self {
        Self {
            min_samples_per_bin: 1,
        }
    }
}

impl ExactQuantileCuts {
    /// Create with custom minimum samples per bin.
    pub fn new(min_samples_per_bin: usize) -> Self {
        Self {
            min_samples_per_bin: min_samples_per_bin.max(1),
        }
    }
}

impl CutFinder for ExactQuantileCuts {
    fn find_cuts<D>(&self, data: &D, max_bins: usize) -> BinCuts
    where
        D: ColumnAccess<Element = f32> + Sync,
    {
        let num_features = data.num_columns();
        let num_rows = data.num_rows();

        // Collect cuts for all features in parallel
        let feature_cuts: Vec<Vec<f32>> = (0..num_features)
            .into_par_iter()
            .map(|feat| {
                // Collect non-missing values
                let mut values: Vec<f32> = data
                    .column(feat)
                    .filter_map(|(_, v)| if v.is_nan() { None } else { Some(v) })
                    .collect();

                if values.is_empty() {
                    return vec![];
                }

                // Sort values
                values.sort_by(|a, b| a.partial_cmp(b).unwrap());

                // Deduplicate
                values.dedup();

                if values.is_empty() {
                    return vec![];
                }

                // Compute number of bins (max_bins - 1 cuts, since bin 0 is for missing)
                // We need at least min_samples_per_bin samples per bin
                let max_cuts = max_bins.saturating_sub(1); // -1 for missing bin
                let min_bins_by_samples = num_rows / self.min_samples_per_bin.max(1);
                let num_cuts = max_cuts.min(values.len()).min(min_bins_by_samples);

                if num_cuts == 0 {
                    return vec![];
                }

                // Pick evenly-spaced quantile points
                let mut cuts = Vec::with_capacity(num_cuts);
                for i in 1..=num_cuts {
                    let idx = (i * values.len() / (num_cuts + 1)).min(values.len() - 1);
                    cuts.push(values[idx]);
                }

                // Ensure cuts are unique and sorted
                cuts.dedup();
                cuts
            })
            .collect();

        // Build CSR-style storage
        let mut cut_values = Vec::new();
        let mut cut_ptrs = vec![0u32];

        for cuts in feature_cuts {
            cut_values.extend(cuts);
            cut_ptrs.push(cut_values.len() as u32);
        }

        BinCuts::new(cut_values, cut_ptrs)
    }

    fn find_cuts_with_categorical<D>(
        &self,
        data: &D,
        max_bins: usize,
        cat_info: &CategoricalInfo,
    ) -> BinCuts
    where
        D: ColumnAccess<Element = f32> + Sync,
    {
        let num_features = data.num_columns();
        let num_rows = data.num_rows();

        // Track which features are categorical and their cardinality
        let mut is_categorical = vec![false; num_features];
        let mut num_categories = vec![0u32; num_features];

        // Collect cuts for all features in parallel
        let feature_cuts: Vec<Vec<f32>> = (0..num_features)
            .into_par_iter()
            .map(|feat| {
                // Check if this feature is categorical
                if cat_info.is_categorical(feat) {
                    // Categorical: no cuts needed, bins are direct category indices
                    // Return empty cuts; num_bins will be num_categories + 1 (for missing)
                    return vec![];
                }

                // Numerical: standard quantile cuts
                let mut values: Vec<f32> = data
                    .column(feat)
                    .filter_map(|(_, v)| if v.is_nan() { None } else { Some(v) })
                    .collect();

                if values.is_empty() {
                    return vec![];
                }

                values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                values.dedup();

                if values.is_empty() {
                    return vec![];
                }

                let max_cuts = max_bins.saturating_sub(1);
                let min_bins_by_samples = num_rows / self.min_samples_per_bin.max(1);
                let num_cuts = max_cuts.min(values.len()).min(min_bins_by_samples);

                if num_cuts == 0 {
                    return vec![];
                }

                let mut cuts = Vec::with_capacity(num_cuts);
                for i in 1..=num_cuts {
                    let idx = (i * values.len() / (num_cuts + 1)).min(values.len() - 1);
                    cuts.push(values[idx]);
                }
                cuts.dedup();
                cuts
            })
            .collect();

        // Set categorical flags (sequential, after parallel work)
        for feat in 0..num_features {
            if let Some(n_cats) = cat_info.num_categories(feat) {
                is_categorical[feat] = true;
                num_categories[feat] = n_cats;
            }
        }

        // Build CSR-style storage
        let mut cut_values = Vec::new();
        let mut cut_ptrs = vec![0u32];

        for cuts in feature_cuts {
            cut_values.extend(cuts);
            cut_ptrs.push(cut_values.len() as u32);
        }

        BinCuts::with_categorical(cut_values, cut_ptrs, is_categorical, num_categories)
    }
}

// ============================================================================
// Quantizer
// ============================================================================

/// Transforms raw features into quantized form.
///
/// The quantizer holds bin cuts and can transform any compatible data matrix
/// into a [`QuantizedMatrix`].
#[derive(Debug, Clone)]
pub struct Quantizer {
    cuts: Arc<BinCuts>,
}

impl Quantizer {
    /// Create from pre-computed cuts.
    pub fn new(cuts: BinCuts) -> Self {
        Self {
            cuts: Arc::new(cuts),
        }
    }

    /// Create by computing cuts from data.
    pub fn from_data<D, C>(data: &D, cut_finder: &C, max_bins: usize) -> Self
    where
        D: ColumnAccess<Element = f32> + Sync,
        C: CutFinder,
    {
        let cuts = cut_finder.find_cuts(data, max_bins);
        Self::new(cuts)
    }

    /// Create by computing cuts from data with categorical feature support.
    pub fn from_data_with_categorical<D, C>(
        data: &D,
        cut_finder: &C,
        max_bins: usize,
        cat_info: &CategoricalInfo,
    ) -> Self
    where
        D: ColumnAccess<Element = f32> + Sync,
        C: CutFinder,
    {
        let cuts = cut_finder.find_cuts_with_categorical(data, max_bins, cat_info);
        Self::new(cuts)
    }

    /// Get the bin cuts.
    pub fn cuts(&self) -> &BinCuts {
        &self.cuts
    }

    /// Quantize a feature matrix.
    ///
    /// Transforms each feature value to its bin index.
    /// Missing values (NaN) are mapped to bin 0.
    pub fn quantize<D, B>(&self, data: &D) -> QuantizedMatrix<B>
    where
        D: ColumnAccess<Element = f32> + Sync,
        B: BinIndex,
    {
        let num_rows = data.num_rows() as u32;
        let num_features = data.num_columns() as u32;

        // Allocate column-major storage
        let total_size = (num_rows as usize) * (num_features as usize);
        let mut index = vec![B::default(); total_size];

        // Parallel quantization per feature
        index
            .par_chunks_mut(num_rows as usize)
            .enumerate()
            .for_each(|(feat, col)| {
                for (row, value) in data.column(feat) {
                    col[row] = self.cuts.bin_value_as(feat as u32, value);
                }
            });

        QuantizedMatrix::new(index, num_rows, num_features, Arc::clone(&self.cuts))
    }

    /// Quantize using default bin index type (u8).
    pub fn quantize_u8<D>(&self, data: &D) -> QuantizedMatrix<u8>
    where
        D: ColumnAccess<Element = f32> + Sync,
    {
        self.quantize(data)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::ColMatrix;

    #[test]
    fn test_exact_quantile_cuts() {
        // Create a simple column-major matrix
        // 10 rows, 2 features
        let data: Vec<f32> = vec![
            // Feature 0: values 0-9
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
            // Feature 1: values 10-19
            10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0,
        ];
        let matrix = ColMatrix::from_vec(data, 10, 2);

        let cut_finder = ExactQuantileCuts::default();
        let cuts = cut_finder.find_cuts(&matrix, 5);

        // Should have 2 features
        assert_eq!(cuts.num_features(), 2);

        // Each feature should have some cuts (exact number depends on quantile selection)
        assert!(cuts.num_bins(0) >= 2);
        assert!(cuts.num_bins(1) >= 2);

        // Feature 0 cuts should be in range [0, 9]
        for &cut in cuts.feature_cuts(0) {
            assert!(cut >= 0.0 && cut <= 9.0);
        }

        // Feature 1 cuts should be in range [10, 19]
        for &cut in cuts.feature_cuts(1) {
            assert!(cut >= 10.0 && cut <= 19.0);
        }
    }

    #[test]
    fn test_quantizer_roundtrip() {
        // Create data with known distribution
        let data: Vec<f32> = vec![
            // Feature 0: 0.0, 0.5, 1.0, 1.5, 2.0 (5 rows)
            0.0, 0.5, 1.0, 1.5, 2.0,
            // Feature 1: all same value
            5.0, 5.0, 5.0, 5.0, 5.0,
        ];
        let matrix = ColMatrix::from_vec(data, 5, 2);

        let quantizer = Quantizer::from_data(&matrix, &ExactQuantileCuts::default(), 256);
        let quantized: QuantizedMatrix<u8> = quantizer.quantize(&matrix);

        assert_eq!(quantized.num_rows(), 5);
        assert_eq!(quantized.num_features(), 2);

        // Feature 0 should have different bins for different values
        let col0 = quantized.feature_column(0);
        // Values are sorted and distinct, so bins should be non-decreasing
        for i in 1..col0.len() {
            assert!(
                col0[i] >= col0[i - 1],
                "Bins should be non-decreasing for sorted values"
            );
        }

        // Feature 1 (all same value) should have all same bins
        let col1 = quantized.feature_column(1);
        let first_bin = col1[0];
        for &bin in col1 {
            assert_eq!(bin, first_bin, "All same values should have same bin");
        }
    }

    #[test]
    fn test_quantizer_with_missing() {
        // Data with missing values
        let data: Vec<f32> = vec![
            // Feature 0: some NaN
            f32::NAN, 1.0, 2.0, f32::NAN, 4.0,
        ];
        let matrix = ColMatrix::from_vec(data, 5, 1);

        let quantizer = Quantizer::from_data(&matrix, &ExactQuantileCuts::default(), 256);
        let quantized: QuantizedMatrix<u8> = quantizer.quantize(&matrix);

        let col = quantized.feature_column(0);

        // NaN values should be bin 0
        assert_eq!(col[0], 0, "NaN should map to bin 0");
        assert_eq!(col[3], 0, "NaN should map to bin 0");

        // Non-NaN values should be > 0
        assert!(col[1] > 0, "Non-NaN should have bin > 0");
        assert!(col[2] > 0, "Non-NaN should have bin > 0");
        assert!(col[4] > 0, "Non-NaN should have bin > 0");
    }

    #[test]
    fn test_quantizer_preserves_ordering() {
        // Test that quantization preserves value ordering
        let data: Vec<f32> = vec![
            // Feature 0: random-ish values
            5.0, 2.0, 8.0, 1.0, 9.0, 3.0, 7.0, 4.0, 6.0, 0.0,
        ];
        let matrix = ColMatrix::from_vec(data.clone(), 10, 1);

        let quantizer = Quantizer::from_data(&matrix, &ExactQuantileCuts::default(), 256);
        let quantized: QuantizedMatrix<u8> = quantizer.quantize(&matrix);

        // For any two values, if value[i] < value[j], then bin[i] <= bin[j]
        let col = quantized.feature_column(0);
        for i in 0..10 {
            for j in 0..10 {
                if data[i] < data[j] {
                    assert!(
                        col[i] <= col[j],
                        "Ordering violated: data[{}]={} < data[{}]={}, but bin[{}]={} > bin[{}]={}",
                        i,
                        data[i],
                        j,
                        data[j],
                        i,
                        col[i],
                        j,
                        col[j]
                    );
                }
            }
        }
    }

    #[test]
    fn test_single_row_data() {
        // Edge case: single row
        let data: Vec<f32> = vec![1.0, 2.0, 3.0]; // 1 row, 3 features
        let matrix = ColMatrix::from_vec(data, 1, 3);

        let quantizer = Quantizer::from_data(&matrix, &ExactQuantileCuts::default(), 256);
        let quantized: QuantizedMatrix<u8> = quantizer.quantize(&matrix);

        assert_eq!(quantized.num_rows(), 1);
        assert_eq!(quantized.num_features(), 3);

        // All bins should be > 0 (not missing)
        for feat in 0..3 {
            assert!(
                quantized.get(0, feat) > 0,
                "Non-NaN value should not be in missing bin"
            );
        }
    }

    #[test]
    fn test_all_missing_column() {
        // Edge case: all values are NaN
        let data: Vec<f32> = vec![
            // Feature 0: all NaN
            f32::NAN, f32::NAN, f32::NAN, f32::NAN, f32::NAN,
        ];
        let matrix = ColMatrix::from_vec(data, 5, 1);

        let quantizer = Quantizer::from_data(&matrix, &ExactQuantileCuts::default(), 256);
        let quantized: QuantizedMatrix<u8> = quantizer.quantize(&matrix);

        // All should be bin 0
        let col = quantized.feature_column(0);
        for &bin in col {
            assert_eq!(bin, 0, "All NaN column should all be bin 0");
        }

        // With 0 cuts: bin 0 (missing) + bin 1 (single region) = 2 bins
        assert_eq!(quantizer.cuts().num_bins(0), 2);
    }

    #[test]
    fn test_single_unique_value() {
        // Edge case: all same non-NaN value
        let data: Vec<f32> = vec![42.0, 42.0, 42.0, 42.0, 42.0];
        let matrix = ColMatrix::from_vec(data, 5, 1);

        let quantizer = Quantizer::from_data(&matrix, &ExactQuantileCuts::default(), 256);
        let quantized: QuantizedMatrix<u8> = quantizer.quantize(&matrix);

        // All should be same bin (and not bin 0 since not missing)
        let col = quantized.feature_column(0);
        let first_bin = col[0];
        assert!(first_bin > 0, "Non-NaN should not be in missing bin");
        for &bin in col {
            assert_eq!(bin, first_bin, "All same values should have same bin");
        }
    }

    #[test]
    fn test_max_bins_respected() {
        // Create data that would naturally want many bins
        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let matrix = ColMatrix::from_vec(data, 100, 1);

        // Request only 4 bins (3 cuts + missing bin)
        let quantizer = Quantizer::from_data(&matrix, &ExactQuantileCuts::default(), 4);
        let quantized: QuantizedMatrix<u8> = quantizer.quantize(&matrix);

        // Count unique bins
        let col = quantized.feature_column(0);
        let mut unique_bins: Vec<u8> = col.to_vec();
        unique_bins.sort();
        unique_bins.dedup();

        // Should have at most 4 unique bins
        assert!(
            unique_bins.len() <= 4,
            "Should have at most 4 bins, got {}",
            unique_bins.len()
        );
    }

    #[test]
    fn test_quantize_u16() {
        // Test with u16 bin indices (for > 256 bins)
        let data: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let matrix = ColMatrix::from_vec(data, 1000, 1);

        let quantizer = Quantizer::from_data(&matrix, &ExactQuantileCuts::default(), 500);
        let quantized: QuantizedMatrix<u16> = quantizer.quantize(&matrix);

        assert_eq!(quantized.num_rows(), 1000);
        assert_eq!(quantized.num_features(), 1);

        // Values should span a good range of bins
        let col = quantized.feature_column(0);
        let max_bin = *col.iter().max().unwrap();
        assert!(max_bin > 1, "Should have multiple bins for varied data");
    }

    #[test]
    fn test_multi_feature_independence() {
        // Test that features are quantized independently
        let data: Vec<f32> = vec![
            // Feature 0: small range [0, 1]
            0.0, 0.25, 0.5, 0.75, 1.0,
            // Feature 1: large range [0, 1000]
            0.0, 250.0, 500.0, 750.0, 1000.0,
        ];
        let matrix = ColMatrix::from_vec(data, 5, 2);

        let quantizer = Quantizer::from_data(&matrix, &ExactQuantileCuts::default(), 256);

        // Feature 0 cuts should be in [0, 1]
        for &cut in quantizer.cuts().feature_cuts(0) {
            assert!(
                cut >= 0.0 && cut <= 1.0,
                "Feature 0 cut {} out of range [0, 1]",
                cut
            );
        }

        // Feature 1 cuts should be in [0, 1000]
        for &cut in quantizer.cuts().feature_cuts(1) {
            assert!(
                cut >= 0.0 && cut <= 1000.0,
                "Feature 1 cut {} out of range [0, 1000]",
                cut
            );
        }
    }

    #[test]
    fn test_categorical_feature_binning() {
        // 6 rows, 1 categorical feature with 3 categories
        let data: Vec<f32> = vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0];
        let matrix = ColMatrix::from_vec(data.clone(), 6, 1);

        let cat_info = CategoricalInfo::with_categorical(1, &[(0, 3)]);
        let cut_finder = ExactQuantileCuts::default();
        let quantizer =
            Quantizer::from_data_with_categorical(&matrix, &cut_finder, 256, &cat_info);
        let quantized: QuantizedMatrix<u8> = quantizer.quantize(&matrix);

        let cuts = quantizer.cuts();

        // Verify categorical metadata
        assert!(cuts.is_categorical(0));
        assert_eq!(cuts.num_categories(0), 3);
        assert_eq!(cuts.num_bins(0), 4); // 3 categories + missing bin

        // Verify binning: category i -> bin i+1
        let col = quantized.feature_column(0);
        assert_eq!(col[0], 1); // Category 0 -> bin 1
        assert_eq!(col[1], 2); // Category 1 -> bin 2
        assert_eq!(col[2], 3); // Category 2 -> bin 3
        assert_eq!(col[3], 1); // Category 0 -> bin 1
        assert_eq!(col[4], 2); // Category 1 -> bin 2
        assert_eq!(col[5], 3); // Category 2 -> bin 3
    }

    #[test]
    fn test_categorical_with_missing() {
        // Categorical feature with some missing values
        let data: Vec<f32> = vec![0.0, f32::NAN, 1.0, f32::NAN, 2.0];
        let matrix = ColMatrix::from_vec(data, 5, 1);

        let cat_info = CategoricalInfo::with_categorical(1, &[(0, 3)]);
        let quantizer = Quantizer::from_data_with_categorical(
            &matrix,
            &ExactQuantileCuts::default(),
            256,
            &cat_info,
        );
        let quantized: QuantizedMatrix<u8> = quantizer.quantize(&matrix);

        let col = quantized.feature_column(0);
        assert_eq!(col[0], 1); // Category 0 -> bin 1
        assert_eq!(col[1], 0); // Missing -> bin 0
        assert_eq!(col[2], 2); // Category 1 -> bin 2
        assert_eq!(col[3], 0); // Missing -> bin 0
        assert_eq!(col[4], 3); // Category 2 -> bin 3
    }

    #[test]
    fn test_categorical_unknown_category() {
        // Categorical feature with value outside declared range
        let data: Vec<f32> = vec![0.0, 1.0, 5.0]; // 5 is > num_categories (3)
        let matrix = ColMatrix::from_vec(data, 3, 1);

        let cat_info = CategoricalInfo::with_categorical(1, &[(0, 3)]);
        let quantizer = Quantizer::from_data_with_categorical(
            &matrix,
            &ExactQuantileCuts::default(),
            256,
            &cat_info,
        );
        let quantized: QuantizedMatrix<u8> = quantizer.quantize(&matrix);

        let col = quantized.feature_column(0);
        assert_eq!(col[0], 1); // Category 0 -> bin 1
        assert_eq!(col[1], 2); // Category 1 -> bin 2
        assert_eq!(col[2], 0); // Unknown category 5 -> bin 0 (treated as missing)
    }

    #[test]
    fn test_mixed_numerical_categorical() {
        // 2 features: numerical and categorical
        let data: Vec<f32> = vec![
            // Feature 0 (numerical)
            0.0, 1.0, 2.0, 3.0, 4.0,
            // Feature 1 (categorical, 3 categories)
            0.0, 1.0, 2.0, 0.0, 1.0,
        ];
        let matrix = ColMatrix::from_vec(data, 5, 2);

        let cat_info = CategoricalInfo::with_categorical(2, &[(1, 3)]);
        let quantizer = Quantizer::from_data_with_categorical(
            &matrix,
            &ExactQuantileCuts::default(),
            256,
            &cat_info,
        );
        let cuts = quantizer.cuts();

        // Feature 0 is numerical
        assert!(!cuts.is_categorical(0));
        assert_eq!(cuts.num_categories(0), 0);
        assert!(!cuts.feature_cuts(0).is_empty()); // Should have cuts

        // Feature 1 is categorical
        assert!(cuts.is_categorical(1));
        assert_eq!(cuts.num_categories(1), 3);
        assert!(cuts.feature_cuts(1).is_empty()); // Categorical has no cuts
        assert_eq!(cuts.num_bins(1), 4); // 3 categories + missing
    }
}

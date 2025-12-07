//! Quantization and binning for histogram-based training.
//!
//! This module provides the infrastructure to discretize continuous feature values
//! into bins for histogram-based gradient boosting (RFC-0011).
//!
//! # Overview
//!
//! Histogram-based training requires converting continuous features into a small
//! number of discrete bins (typically 256). This enables:
//!
//! - O(n_bins) split search instead of O(n_samples)
//! - Efficient histogram aggregation
//! - Better cache locality (u8 bin indices fit more data in cache)
//! - Histogram subtraction optimization
//!
//! # Key Types
//!
//! - [`BinCuts`]: Bin boundaries for all features (thresholds)
//! - [`QuantizedMatrix`]: Quantized feature matrix storing bin indices
//! - [`BinIndex`]: Trait for bin index types (u8, u16, u32)
//! - [`Quantizer`]: Transforms raw features into quantized form
//!
//! # Example
//!
//! ```ignore
//! use booste_rs::training::quantize::{ExactQuantileCuts, Quantizer, CutFinder};
//! use booste_rs::data::ColMatrix;
//!
//! // Create feature matrix
//! let data: ColMatrix<f32> = /* ... */;
//!
//! // Find bin boundaries using exact quantiles
//! let cut_finder = ExactQuantileCuts::default();
//! let cuts = cut_finder.find_cuts(&data, 256);
//!
//! // Quantize the data
//! let quantizer = Quantizer::new(cuts);
//! let quantized = quantizer.quantize(&data);
//!
//! // Access bin indices
//! let bin = quantized.get(row, feature);
//! let column = quantized.feature_column(feature);
//! ```
//!
//! # Missing Values
//!
//! Missing values (NaN) are mapped to bin 0 by convention. This allows the
//! split finder to handle missing values by checking if `bin == 0`.
//!
//! See RFC-0011 for design rationale.

use std::sync::Arc;

use rayon::prelude::*;

use crate::data::ColumnAccess;

// ============================================================================
// BinIndex trait
// ============================================================================

/// Trait for bin index types.
///
/// Bin indices can be u8 (256 bins), u16 (65536 bins), or u32 (4B bins).
/// Most use cases need only u8, but the generic parameter allows flexibility.
pub trait BinIndex: Copy + Send + Sync + Default + 'static {
    /// Maximum number of bins this type can represent.
    const MAX_BINS: usize;

    /// Convert from usize, saturating at MAX_BINS - 1.
    fn from_usize(v: usize) -> Self;

    /// Convert to usize.
    fn to_usize(self) -> usize;
}

impl BinIndex for u8 {
    const MAX_BINS: usize = 256;

    #[inline]
    fn from_usize(v: usize) -> Self {
        v.min(255) as u8
    }

    #[inline]
    fn to_usize(self) -> usize {
        self as usize
    }
}

impl BinIndex for u16 {
    const MAX_BINS: usize = 65536;

    #[inline]
    fn from_usize(v: usize) -> Self {
        v.min(65535) as u16
    }

    #[inline]
    fn to_usize(self) -> usize {
        self as usize
    }
}

impl BinIndex for u32 {
    const MAX_BINS: usize = u32::MAX as usize;

    #[inline]
    fn from_usize(v: usize) -> Self {
        v.min(u32::MAX as usize) as u32
    }

    #[inline]
    fn to_usize(self) -> usize {
        self as usize
    }
}

// ============================================================================
// BinCuts
// ============================================================================

/// Bin boundaries for all features.
///
/// Stores cut points (thresholds) for each feature in a CSR-like format:
/// - `cut_values`: All cut values concatenated
/// - `cut_ptrs`: Offsets into `cut_values` for each feature
///
/// A value `v` for feature `f` maps to bin `b` where `cuts[b] <= v < cuts[b+1]`.
/// Bin 0 is reserved for missing values (NaN).
///
/// # Categorical Features
///
/// For categorical features, bins represent category indices directly:
/// - Bin 0: missing values (NaN)
/// - Bin 1: category 0
/// - Bin 2: category 1
/// - ...
///
/// Categorical features have no cut values (cuts are empty), and `is_categorical[f]`
/// is set to true. The number of bins equals the cardinality + 1 (for missing).
///
/// # Memory Layout
///
/// ```text
/// cut_ptrs:    [0, 5, 8, 12]  (offsets)
/// cut_values:  [0.1, 0.3, 0.5, 0.7, 0.9,   ← Feature 0: 5 cuts (6 bins)
///               1.0, 2.0, 3.0,              ← Feature 1: 3 cuts (4 bins)
///               0.0, 0.25, 0.5, 0.75]       ← Feature 2: 4 cuts (5 bins)
/// ```
#[derive(Debug, Clone)]
pub struct BinCuts {
    /// All cut values concatenated, sorted per feature.
    /// These are the upper bounds of each bin (exclusive).
    /// For categorical features, this is empty (no cuts needed).
    cut_values: Box<[f32]>,

    /// Offsets into cut_values: cut_ptrs[f] is start of feature f's cuts.
    /// Length: num_features + 1
    cut_ptrs: Box<[u32]>,

    /// Number of features.
    num_features: u32,

    /// Per-feature categorical flag.
    /// If `is_categorical[f]` is true, feature f is treated as categorical.
    is_categorical: Box<[bool]>,

    /// Per-feature number of categories (only meaningful for categorical features).
    /// For numerical features, this is 0.
    num_categories: Box<[u32]>,
}

impl BinCuts {
    /// Create new bin cuts from pre-computed values.
    ///
    /// # Arguments
    ///
    /// * `cut_values` - All cut values concatenated
    /// * `cut_ptrs` - Offsets into cut_values for each feature (length: num_features + 1)
    ///
    /// # Panics
    ///
    /// Panics if `cut_ptrs` is empty or last element doesn't match `cut_values.len()`.
    pub fn new(cut_values: Vec<f32>, cut_ptrs: Vec<u32>) -> Self {
        assert!(!cut_ptrs.is_empty(), "cut_ptrs must not be empty");
        assert_eq!(
            *cut_ptrs.last().unwrap() as usize,
            cut_values.len(),
            "Last cut_ptr must equal cut_values.len()"
        );

        let num_features = (cut_ptrs.len() - 1) as u32;

        Self {
            cut_values: cut_values.into_boxed_slice(),
            cut_ptrs: cut_ptrs.into_boxed_slice(),
            num_features,
            is_categorical: vec![false; num_features as usize].into_boxed_slice(),
            num_categories: vec![0; num_features as usize].into_boxed_slice(),
        }
    }

    /// Create bin cuts with categorical feature support.
    ///
    /// # Arguments
    ///
    /// * `cut_values` - All cut values concatenated (for numerical features)
    /// * `cut_ptrs` - Offsets into cut_values for each feature
    /// * `is_categorical` - Per-feature categorical flag
    /// * `num_categories` - Per-feature category count (0 for numerical)
    pub fn with_categorical(
        cut_values: Vec<f32>,
        cut_ptrs: Vec<u32>,
        is_categorical: Vec<bool>,
        num_categories: Vec<u32>,
    ) -> Self {
        assert!(!cut_ptrs.is_empty(), "cut_ptrs must not be empty");
        assert_eq!(
            *cut_ptrs.last().unwrap() as usize,
            cut_values.len(),
            "Last cut_ptr must equal cut_values.len()"
        );

        let num_features = (cut_ptrs.len() - 1) as u32;
        assert_eq!(
            is_categorical.len(),
            num_features as usize,
            "is_categorical length must match num_features"
        );
        assert_eq!(
            num_categories.len(),
            num_features as usize,
            "num_categories length must match num_features"
        );

        Self {
            cut_values: cut_values.into_boxed_slice(),
            cut_ptrs: cut_ptrs.into_boxed_slice(),
            num_features,
            is_categorical: is_categorical.into_boxed_slice(),
            num_categories: num_categories.into_boxed_slice(),
        }
    }

    /// Number of features.
    #[inline]
    pub fn num_features(&self) -> u32 {
        self.num_features
    }

    /// Check if a feature is categorical.
    #[inline]
    pub fn is_categorical(&self, feature: u32) -> bool {
        self.is_categorical[feature as usize]
    }

    /// Get the number of categories for a categorical feature.
    ///
    /// Returns 0 for numerical features.
    #[inline]
    pub fn num_categories(&self, feature: u32) -> u32 {
        self.num_categories[feature as usize]
    }

    /// Get bin boundaries for a specific feature.
    ///
    /// Returns a slice of cut values (bin upper bounds).
    /// For categorical features, this returns an empty slice.
    #[inline]
    pub fn feature_cuts(&self, feature: u32) -> &[f32] {
        let start = self.cut_ptrs[feature as usize] as usize;
        let end = self.cut_ptrs[feature as usize + 1] as usize;
        &self.cut_values[start..end]
    }

    /// Number of bins for a feature.
    ///
    /// For numerical features: number of cuts + 1 (for bin 0 which handles missing/below-min).
    /// For categorical features: num_categories + 1 (for bin 0 which handles missing).
    ///
    /// For numerical features with N cuts, there are N+2 bins:
    /// - Bin 0: missing values (NaN)
    /// - Bins 1..=N: regions (-∞, cut[0]], (cut[0], cut[1]], ..., (cut[N-2], cut[N-1]]
    /// - Bin N+1: region (cut[N-1], +∞)
    #[inline]
    pub fn num_bins(&self, feature: u32) -> usize {
        if self.is_categorical[feature as usize] {
            // Categorical: bins are 0 (missing) + categories
            self.num_categories[feature as usize] as usize + 1
        } else {
            // Numerical: bins are 0 (missing) + (N+1 regions for N cuts)
            let start = self.cut_ptrs[feature as usize];
            let end = self.cut_ptrs[feature as usize + 1];
            let num_cuts = (end - start) as usize;
            // N cuts create N+1 regions, plus bin 0 for missing = N+2 bins total
            num_cuts + 2
        }
    }

    /// Total bins across all features.
    ///
    /// Useful for pre-allocating histogram storage.
    pub fn total_bins(&self) -> usize {
        (0..self.num_features).map(|f| self.num_bins(f)).sum()
    }

    /// Map a single value to its bin index.
    ///
    /// For numerical features:
    /// - NaN values map to bin 0
    /// - Values below min cut map to bin 1
    /// - Values >= max cut map to max bin
    ///
    /// For categorical features:
    /// - NaN values map to bin 0
    /// - Category i maps to bin i + 1 (0-indexed categories)
    ///
    /// Uses binary search for numerical: O(log num_bins).
    /// Direct mapping for categorical: O(1).
    #[inline]
    pub fn bin_value(&self, feature: u32, value: f32) -> usize {
        // Missing values go to bin 0
        if value.is_nan() {
            return 0;
        }

        if self.is_categorical[feature as usize] {
            // Categorical: direct mapping, category i -> bin i+1
            // Value should be a non-negative integer
            let cat = value.round() as usize;
            let max_cat = self.num_categories[feature as usize] as usize;
            if cat >= max_cat {
                // Unknown category treated as missing (bin 0)
                0
            } else {
                cat + 1
            }
        } else {
            // Numerical: binary search for correct bin
            let cuts = self.feature_cuts(feature);
            if cuts.is_empty() {
                return 1; // Single bin for all non-missing values
            }

            // Binary search for the bin
            // We want the first cut that is > value, then bin = that index + 1
            // (because bin 0 is reserved for missing)
            //
            // Bin layout for cuts [c0, c1, c2]:
            // - bin 0: missing (NaN)
            // - bin 1: value <= c0
            // - bin 2: c0 < value <= c1
            // - bin 3: c1 < value <= c2
            // - bin 4: value > c2
            match cuts.binary_search_by(|c| c.partial_cmp(&value).unwrap()) {
                Ok(idx) => idx + 1,  // Exact match: value == cuts[idx], goes to bin idx+1
                Err(idx) => idx + 1, // Insert position: cuts[idx-1] < value < cuts[idx]
            }
        }
    }

    /// Map a value to bin index, returning a BinIndex type.
    #[inline]
    pub fn bin_value_as<B: BinIndex>(&self, feature: u32, value: f32) -> B {
        B::from_usize(self.bin_value(feature, value))
    }
}

// ============================================================================
// QuantizedMatrix
// ============================================================================

/// Quantized feature matrix storing bin indices.
///
/// Stored in **column-major** order for efficient histogram building:
/// iterating rows for a single feature is contiguous memory access.
///
/// # Type Parameter
///
/// `B` controls bin index width:
/// - `u8`: Up to 256 bins per feature (default, 1 byte per cell)
/// - `u16`: Up to 65536 bins (2 bytes per cell)
/// - `u32`: Unlimited bins (4 bytes per cell)
///
/// # Memory Layout
///
/// ```text
/// For 4 rows × 3 features:
///
/// index: [r0f0, r1f0, r2f0, r3f0,   ← Feature 0 column (contiguous)
///         r0f1, r1f1, r2f1, r3f1,   ← Feature 1 column
///         r0f2, r1f2, r2f2, r3f2]   ← Feature 2 column
/// ```
#[derive(Debug, Clone)]
pub struct QuantizedMatrix<B: BinIndex = u8> {
    /// Bin indices in column-major layout: index[col * num_rows + row]
    index: Box<[B]>,

    /// Number of rows.
    num_rows: u32,

    /// Number of features.
    num_features: u32,

    /// Reference to the bin cuts used for quantization.
    cuts: Arc<BinCuts>,
}

impl<B: BinIndex> QuantizedMatrix<B> {
    /// Create a new quantized matrix.
    ///
    /// # Arguments
    ///
    /// * `index` - Bin indices in column-major layout
    /// * `num_rows` - Number of rows
    /// * `num_features` - Number of features
    /// * `cuts` - Bin cuts used for quantization
    pub fn new(index: Vec<B>, num_rows: u32, num_features: u32, cuts: Arc<BinCuts>) -> Self {
        assert_eq!(
            index.len(),
            (num_rows as usize) * (num_features as usize),
            "Index length must equal num_rows * num_features"
        );

        Self {
            index: index.into_boxed_slice(),
            num_rows,
            num_features,
            cuts,
        }
    }

    /// Number of rows.
    #[inline]
    pub fn num_rows(&self) -> u32 {
        self.num_rows
    }

    /// Number of features.
    #[inline]
    pub fn num_features(&self) -> u32 {
        self.num_features
    }

    /// Get the bin cuts used for quantization.
    #[inline]
    pub fn cuts(&self) -> &BinCuts {
        &self.cuts
    }

    /// Get bin index for a specific cell.
    #[inline]
    pub fn get(&self, row: u32, feature: u32) -> B {
        let idx = (feature as usize) * (self.num_rows as usize) + (row as usize);
        self.index[idx]
    }

    /// Get all bin indices for a feature (contiguous slice).
    ///
    /// This is the primary access pattern for histogram building.
    #[inline]
    pub fn feature_column(&self, feature: u32) -> &[B] {
        let start = (feature as usize) * (self.num_rows as usize);
        let end = start + self.num_rows as usize;
        &self.index[start..end]
    }

    /// Iterate over bin indices for specific rows of a feature.
    ///
    /// Used for histogram building on a subset of rows (e.g., rows in a tree node).
    #[inline]
    pub fn iter_rows_for_feature<'a>(
        &'a self,
        feature: u32,
        rows: &'a [u32],
    ) -> impl Iterator<Item = B> + 'a {
        let col = self.feature_column(feature);
        rows.iter().map(move |&row| col[row as usize])
    }
}

// ============================================================================
// CutFinder trait
// ============================================================================

/// Information about which features are categorical and their cardinality.
#[derive(Debug, Clone, Default)]
pub struct CategoricalInfo {
    /// For each feature: Some(num_categories) if categorical, None if numerical.
    pub feature_types: Vec<Option<u32>>,
}

impl CategoricalInfo {
    /// Create with no categorical features.
    pub fn all_numerical(num_features: usize) -> Self {
        Self {
            feature_types: vec![None; num_features],
        }
    }

    /// Create from a list specifying categorical features.
    ///
    /// # Arguments
    /// * `num_features` - Total number of features
    /// * `categorical` - List of (feature_index, num_categories) pairs
    pub fn with_categorical(num_features: usize, categorical: &[(usize, u32)]) -> Self {
        let mut feature_types = vec![None; num_features];
        for &(feat, num_cats) in categorical {
            if feat < num_features {
                feature_types[feat] = Some(num_cats);
            }
        }
        Self { feature_types }
    }

    /// Check if a feature is categorical.
    pub fn is_categorical(&self, feature: usize) -> bool {
        self.feature_types
            .get(feature)
            .map(|t| t.is_some())
            .unwrap_or(false)
    }

    /// Get the number of categories for a feature, or None if numerical.
    pub fn num_categories(&self, feature: usize) -> Option<u32> {
        self.feature_types.get(feature).copied().flatten()
    }
}

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

// ============================================================================
// ExactQuantileCuts
// ============================================================================

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

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to create BinCuts for testing
    fn make_cuts(features: &[&[f32]]) -> BinCuts {
        let mut cut_values = Vec::new();
        let mut cut_ptrs = vec![0u32];

        for cuts in features {
            cut_values.extend(*cuts);
            cut_ptrs.push(cut_values.len() as u32);
        }

        BinCuts::new(cut_values, cut_ptrs)
    }

    #[test]
    fn test_bin_cuts_basic() {
        // Feature 0: cuts at [0.5, 1.5, 2.5] -> bins 0 (missing), 1, 2, 3, 4
        // Feature 1: cuts at [10.0] -> bins 0 (missing), 1, 2
        let cuts = make_cuts(&[&[0.5, 1.5, 2.5], &[10.0]]);

        assert_eq!(cuts.num_features(), 2);
        assert_eq!(cuts.feature_cuts(0), &[0.5, 1.5, 2.5]);
        assert_eq!(cuts.feature_cuts(1), &[10.0]);
        assert_eq!(cuts.num_bins(0), 5); // 3 cuts -> 4 regions + 1 missing = 5 bins
        assert_eq!(cuts.num_bins(1), 3); // 1 cut -> 2 regions + 1 missing = 3 bins
    }

    #[test]
    fn test_bin_value_mapping() {
        let cuts = make_cuts(&[&[0.5, 1.5, 2.5]]);

        // NaN -> bin 0
        assert_eq!(cuts.bin_value(0, f32::NAN), 0);

        // Values map to correct bins
        assert_eq!(cuts.bin_value(0, 0.0), 1); // < 0.5 -> bin 1
        assert_eq!(cuts.bin_value(0, 0.5), 1); // == 0.5 -> bin 1
        assert_eq!(cuts.bin_value(0, 0.7), 2); // 0.5 < v < 1.5 -> bin 2
        assert_eq!(cuts.bin_value(0, 1.5), 2); // == 1.5 -> bin 2
        assert_eq!(cuts.bin_value(0, 2.0), 3); // 1.5 < v < 2.5 -> bin 3
        assert_eq!(cuts.bin_value(0, 2.5), 3); // == 2.5 -> bin 3
        assert_eq!(cuts.bin_value(0, 3.0), 4); // > 2.5 -> bin 4
        assert_eq!(cuts.bin_value(0, 100.0), 4); // way above -> bin 4
    }

    #[test]
    fn test_bin_value_edge_cases() {
        // Empty cuts (bin 0 for missing, bin 1 for all values)
        let cuts = make_cuts(&[&[]]);
        assert_eq!(cuts.num_bins(0), 2); // 0 cuts -> 1 region + 1 missing = 2 bins
        assert_eq!(cuts.bin_value(0, f32::NAN), 0);
        assert_eq!(cuts.bin_value(0, 0.0), 1);
        assert_eq!(cuts.bin_value(0, 100.0), 1);

        // Single cut
        let cuts = make_cuts(&[&[5.0]]);
        assert_eq!(cuts.num_bins(0), 3); // 1 cut -> 2 regions + 1 missing = 3 bins
        assert_eq!(cuts.bin_value(0, f32::NAN), 0);
        assert_eq!(cuts.bin_value(0, 4.0), 1);
        assert_eq!(cuts.bin_value(0, 5.0), 1);
        assert_eq!(cuts.bin_value(0, 6.0), 2);
    }

    #[test]
    fn test_quantized_matrix_layout() {
        let cuts = Arc::new(make_cuts(&[&[0.5], &[10.0]]));

        // 3 rows, 2 features, column-major
        // Feature 0: [1, 2, 1] (bins for values < 0.5, > 0.5, < 0.5)
        // Feature 1: [1, 2, 1] (bins for values < 10, > 10, < 10)
        let index = vec![
            1u8, 2, 1, // Feature 0 column
            1, 2, 1, // Feature 1 column
        ];

        let qm = QuantizedMatrix::new(index, 3, 2, cuts);

        assert_eq!(qm.num_rows(), 3);
        assert_eq!(qm.num_features(), 2);

        // Test get()
        assert_eq!(qm.get(0, 0), 1);
        assert_eq!(qm.get(1, 0), 2);
        assert_eq!(qm.get(2, 0), 1);
        assert_eq!(qm.get(0, 1), 1);
        assert_eq!(qm.get(1, 1), 2);
        assert_eq!(qm.get(2, 1), 1);

        // Test feature_column()
        assert_eq!(qm.feature_column(0), &[1, 2, 1]);
        assert_eq!(qm.feature_column(1), &[1, 2, 1]);
    }

    #[test]
    fn test_iter_rows_for_feature() {
        let cuts = Arc::new(make_cuts(&[&[0.5]]));
        let index = vec![1u8, 2, 1, 2, 1]; // 5 rows, 1 feature
        let qm = QuantizedMatrix::new(index, 5, 1, cuts);

        // Iterate over subset of rows
        let rows = vec![0, 2, 4];
        let bins: Vec<u8> = qm.iter_rows_for_feature(0, &rows).collect();
        assert_eq!(bins, vec![1, 1, 1]);

        let rows = vec![1, 3];
        let bins: Vec<u8> = qm.iter_rows_for_feature(0, &rows).collect();
        assert_eq!(bins, vec![2, 2]);
    }

    #[test]
    fn test_bin_index_trait() {
        // u8
        assert_eq!(u8::MAX_BINS, 256);
        assert_eq!(u8::from_usize(0), 0u8);
        assert_eq!(u8::from_usize(255), 255u8);
        assert_eq!(u8::from_usize(256), 255u8); // saturates
        assert_eq!(100u8.to_usize(), 100usize);

        // u16
        assert_eq!(u16::MAX_BINS, 65536);
        assert_eq!(u16::from_usize(0), 0u16);
        assert_eq!(u16::from_usize(65535), 65535u16);
        assert_eq!(u16::from_usize(65536), 65535u16); // saturates
    }

    // Integration test with actual data quantization
    mod integration {
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
                assert!(col0[i] >= col0[i - 1], "Bins should be non-decreasing for sorted values");
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
                            i, data[i], j, data[j], i, col[i], j, col[j]
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
                assert!(quantized.get(0, feat) > 0, "Non-NaN value should not be in missing bin");
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
            assert!(unique_bins.len() <= 4, "Should have at most 4 bins, got {}", unique_bins.len());
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
                assert!(cut >= 0.0 && cut <= 1.0, "Feature 0 cut {} out of range [0, 1]", cut);
            }

            // Feature 1 cuts should be in [0, 1000]
            for &cut in quantizer.cuts().feature_cuts(1) {
                assert!(cut >= 0.0 && cut <= 1000.0, "Feature 1 cut {} out of range [0, 1000]", cut);
            }
        }

        #[test]
        fn test_categorical_feature_binning() {
            use super::CategoricalInfo;

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
            use super::CategoricalInfo;

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
            use super::CategoricalInfo;

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
            use super::CategoricalInfo;

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

        #[test]
        fn test_categorical_info() {
            use super::CategoricalInfo;

            let info = CategoricalInfo::all_numerical(5);
            assert!(!info.is_categorical(0));
            assert!(!info.is_categorical(4));
            assert_eq!(info.num_categories(0), None);

            let info = CategoricalInfo::with_categorical(5, &[(1, 10), (3, 5)]);
            assert!(!info.is_categorical(0));
            assert!(info.is_categorical(1));
            assert!(!info.is_categorical(2));
            assert!(info.is_categorical(3));
            assert!(!info.is_categorical(4));
            assert_eq!(info.num_categories(1), Some(10));
            assert_eq!(info.num_categories(3), Some(5));
        }
    }
}

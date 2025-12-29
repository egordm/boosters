//! Storage types for binned feature data.
//!
//! This module provides typed storage for different kinds of features:
//! - `NumericStorage`: Dense numeric features with bins + raw values
//! - `CategoricalStorage`: Dense categorical features with bins only (lossless)
//! - `SparseNumericStorage`: Sparse numeric features (CSC-like)
//! - `SparseCategoricalStorage`: Sparse categorical features (CSC-like)
//!
//! All storage is column-major: feature values are contiguous per feature.

use super::BinData;

/// Dense numeric storage: [n_features × n_samples], column-major.
///
/// For feature `f` at sample `s` with `n_samples` total:
/// - Index: `f * n_samples + s`
///
/// Raw values store actual f32 values including NaN for missing.
/// Missing handling semantics are defined by BinMapper.
///
/// # Examples
///
/// ```
/// use boosters::data::binned::{BinData, NumericStorage};
///
/// // 2 features, 3 samples each
/// let bins = BinData::from(vec![
///     0u8, 1, 2,   // Feature 0: samples 0,1,2
///     3, 4, 5,     // Feature 1: samples 0,1,2
/// ]);
/// let raw = vec![
///     1.0, 2.0, 3.0,   // Feature 0
///     4.0, 5.0, 6.0,   // Feature 1
/// ].into_boxed_slice();
///
/// let storage = NumericStorage::new(bins, raw);
/// let n_samples = 3;
///
/// // Access bin values
/// assert_eq!(storage.bin(0, 0, n_samples), 0); // Feature 0, sample 0
/// assert_eq!(storage.bin(1, 1, n_samples), 4); // Feature 1, sample 1
///
/// // Access raw values
/// assert_eq!(storage.raw(0, 0, n_samples), 1.0);
/// assert_eq!(storage.raw(2, 1, n_samples), 6.0);
///
/// // Get contiguous slice for linear trees
/// assert_eq!(storage.raw_slice(0, n_samples), &[1.0, 2.0, 3.0]);
/// ```
#[derive(Debug, Clone)]
pub struct NumericStorage {
    /// Bin values: [n_features × n_samples], column-major.
    bins: BinData,
    /// Raw values: [n_features × n_samples], column-major.
    /// Always present for numeric features.
    raw_values: Box<[f32]>,
}

impl NumericStorage {
    /// Creates a new NumericStorage.
    ///
    /// # Panics
    ///
    /// Panics if `bins.len() != raw_values.len()`.
    #[inline]
    pub fn new(bins: BinData, raw_values: Box<[f32]>) -> Self {
        assert_eq!(
            bins.len(),
            raw_values.len(),
            "bins and raw_values must have the same length"
        );
        Self { bins, raw_values }
    }

    /// Returns the bin value for the given sample and feature.
    ///
    /// # Parameters
    /// - `sample`: Sample index (0..n_samples)
    /// - `feature_in_group`: Feature index within this storage group
    /// - `n_samples`: Total number of samples
    #[inline]
    pub fn bin(&self, sample: usize, feature_in_group: usize, n_samples: usize) -> u32 {
        let idx = feature_in_group * n_samples + sample;
        self.bins.get(idx).expect("index out of bounds")
    }

    /// Returns the bin value without bounds checking.
    ///
    /// # Safety
    ///
    /// The index `feature_in_group * n_samples + sample` must be within bounds.
    #[inline]
    pub unsafe fn bin_unchecked(
        &self,
        sample: usize,
        feature_in_group: usize,
        n_samples: usize,
    ) -> u32 {
        let idx = feature_in_group * n_samples + sample;
        // SAFETY: Caller guarantees index is within bounds
        unsafe { self.bins.get_unchecked(idx) }
    }

    /// Returns the raw value for the given sample and feature.
    ///
    /// # Parameters
    /// - `sample`: Sample index (0..n_samples)
    /// - `feature_in_group`: Feature index within this storage group
    /// - `n_samples`: Total number of samples
    #[inline]
    pub fn raw(&self, sample: usize, feature_in_group: usize, n_samples: usize) -> f32 {
        let idx = feature_in_group * n_samples + sample;
        self.raw_values[idx]
    }

    /// Returns the raw value without bounds checking.
    ///
    /// # Safety
    ///
    /// The index `feature_in_group * n_samples + sample` must be within bounds.
    #[inline]
    pub unsafe fn raw_unchecked(
        &self,
        sample: usize,
        feature_in_group: usize,
        n_samples: usize,
    ) -> f32 {
        let idx = feature_in_group * n_samples + sample;
        // SAFETY: Caller guarantees index is within bounds
        unsafe { *self.raw_values.get_unchecked(idx) }
    }

    /// Returns a contiguous slice of raw values for the given feature.
    ///
    /// This is the efficient access pattern for linear tree regression,
    /// as all values for a feature are contiguous in memory.
    ///
    /// # Parameters
    /// - `feature_in_group`: Feature index within this storage group
    /// - `n_samples`: Total number of samples
    #[inline]
    pub fn raw_slice(&self, feature_in_group: usize, n_samples: usize) -> &[f32] {
        let start = feature_in_group * n_samples;
        &self.raw_values[start..start + n_samples]
    }

    /// Returns the number of features in this storage.
    ///
    /// Computed from `total_elements / n_samples`.
    #[inline]
    pub fn n_features(&self, n_samples: usize) -> usize {
        if n_samples == 0 {
            0
        } else {
            self.bins.len() / n_samples
        }
    }

    /// Returns the total size in bytes used by this storage.
    #[inline]
    pub fn size_bytes(&self) -> usize {
        self.bins.size_bytes() + self.raw_values.len() * std::mem::size_of::<f32>()
    }

    /// Returns a reference to the underlying bin data.
    #[inline]
    pub fn bins(&self) -> &BinData {
        &self.bins
    }

    /// Returns a reference to the underlying raw values.
    #[inline]
    pub fn raw_values(&self) -> &[f32] {
        &self.raw_values
    }
}

/// Dense categorical storage: [n_features × n_samples], column-major.
///
/// For feature `f` at sample `s` with `n_samples` total:
/// - Index: `f * n_samples + s`
///
/// No raw values - bin = category ID (lossless). This is because categorical
/// features are encoded directly as their category index.
///
/// # Examples
///
/// ```
/// use boosters::data::binned::{BinData, CategoricalStorage};
///
/// // 2 features, 3 samples each
/// let bins = BinData::from(vec![
///     0u8, 1, 0,   // Feature 0: categories 0, 1, 0
///     2, 0, 1,     // Feature 1: categories 2, 0, 1
/// ]);
///
/// let storage = CategoricalStorage::new(bins);
/// let n_samples = 3;
///
/// // Access category values (bins)
/// assert_eq!(storage.bin(0, 0, n_samples), 0);
/// assert_eq!(storage.bin(1, 0, n_samples), 1);
/// assert_eq!(storage.bin(0, 1, n_samples), 2);
/// ```
#[derive(Debug, Clone)]
pub struct CategoricalStorage {
    /// Bin values (bin index = category ID): [n_features × n_samples], column-major.
    bins: BinData,
}

impl CategoricalStorage {
    /// Creates a new CategoricalStorage.
    #[inline]
    pub fn new(bins: BinData) -> Self {
        Self { bins }
    }

    /// Returns the bin (category) value for the given sample and feature.
    ///
    /// # Parameters
    /// - `sample`: Sample index (0..n_samples)
    /// - `feature_in_group`: Feature index within this storage group
    /// - `n_samples`: Total number of samples
    #[inline]
    pub fn bin(&self, sample: usize, feature_in_group: usize, n_samples: usize) -> u32 {
        let idx = feature_in_group * n_samples + sample;
        self.bins.get(idx).expect("index out of bounds")
    }

    /// Returns the bin (category) value without bounds checking.
    ///
    /// # Safety
    ///
    /// The index `feature_in_group * n_samples + sample` must be within bounds.
    #[inline]
    pub unsafe fn bin_unchecked(
        &self,
        sample: usize,
        feature_in_group: usize,
        n_samples: usize,
    ) -> u32 {
        let idx = feature_in_group * n_samples + sample;
        // SAFETY: Caller guarantees index is within bounds
        unsafe { self.bins.get_unchecked(idx) }
    }

    /// Returns the number of features in this storage.
    ///
    /// Computed from `total_elements / n_samples`.
    #[inline]
    pub fn n_features(&self, n_samples: usize) -> usize {
        if n_samples == 0 {
            0
        } else {
            self.bins.len() / n_samples
        }
    }

    /// Returns the total size in bytes used by this storage.
    #[inline]
    pub fn size_bytes(&self) -> usize {
        self.bins.size_bytes()
    }

    /// Returns a reference to the underlying bin data.
    #[inline]
    pub fn bins(&self) -> &BinData {
        &self.bins
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numeric_storage_new() {
        let bins = BinData::from(vec![0u8, 1, 2, 3, 4, 5]);
        let raw = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0].into_boxed_slice();
        let storage = NumericStorage::new(bins, raw);
        assert_eq!(storage.bins().len(), 6);
        assert_eq!(storage.raw_values().len(), 6);
    }

    #[test]
    #[should_panic(expected = "bins and raw_values must have the same length")]
    fn test_numeric_storage_new_mismatched_lengths() {
        let bins = BinData::from(vec![0u8, 1, 2]);
        let raw = vec![1.0, 2.0].into_boxed_slice();
        NumericStorage::new(bins, raw);
    }

    #[test]
    fn test_bin_access() {
        // 2 features, 3 samples
        let bins = BinData::from(vec![
            0u8, 1, 2, // Feature 0
            10, 11, 12, // Feature 1
        ]);
        let raw = vec![0.0; 6].into_boxed_slice();
        let storage = NumericStorage::new(bins, raw);
        let n_samples = 3;

        // Feature 0
        assert_eq!(storage.bin(0, 0, n_samples), 0);
        assert_eq!(storage.bin(1, 0, n_samples), 1);
        assert_eq!(storage.bin(2, 0, n_samples), 2);

        // Feature 1
        assert_eq!(storage.bin(0, 1, n_samples), 10);
        assert_eq!(storage.bin(1, 1, n_samples), 11);
        assert_eq!(storage.bin(2, 1, n_samples), 12);
    }

    #[test]
    fn test_raw_access() {
        // 2 features, 3 samples
        let bins = BinData::from(vec![0u8; 6]);
        let raw = vec![
            1.0, 2.0, 3.0, // Feature 0
            4.0, 5.0, 6.0, // Feature 1
        ]
        .into_boxed_slice();
        let storage = NumericStorage::new(bins, raw);
        let n_samples = 3;

        // Feature 0
        assert_eq!(storage.raw(0, 0, n_samples), 1.0);
        assert_eq!(storage.raw(1, 0, n_samples), 2.0);
        assert_eq!(storage.raw(2, 0, n_samples), 3.0);

        // Feature 1
        assert_eq!(storage.raw(0, 1, n_samples), 4.0);
        assert_eq!(storage.raw(1, 1, n_samples), 5.0);
        assert_eq!(storage.raw(2, 1, n_samples), 6.0);
    }

    #[test]
    fn test_raw_slice() {
        // 2 features, 3 samples
        let bins = BinData::from(vec![0u8; 6]);
        let raw = vec![
            1.0, 2.0, 3.0, // Feature 0
            4.0, 5.0, 6.0, // Feature 1
        ]
        .into_boxed_slice();
        let storage = NumericStorage::new(bins, raw);
        let n_samples = 3;

        assert_eq!(storage.raw_slice(0, n_samples), &[1.0, 2.0, 3.0]);
        assert_eq!(storage.raw_slice(1, n_samples), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_n_features() {
        // 3 features, 4 samples
        let bins = BinData::from(vec![0u8; 12]);
        let raw = vec![0.0; 12].into_boxed_slice();
        let storage = NumericStorage::new(bins, raw);

        assert_eq!(storage.n_features(4), 3);
        assert_eq!(storage.n_features(12), 1);
        assert_eq!(storage.n_features(1), 12);
        assert_eq!(storage.n_features(0), 0);
    }

    #[test]
    fn test_size_bytes() {
        // U8 bins: 6 bytes, raw values: 6 * 4 = 24 bytes
        let bins = BinData::from(vec![0u8; 6]);
        let raw = vec![0.0f32; 6].into_boxed_slice();
        let storage = NumericStorage::new(bins, raw);
        assert_eq!(storage.size_bytes(), 6 + 24);

        // U16 bins: 6 * 2 = 12 bytes, raw values: 6 * 4 = 24 bytes
        let bins = BinData::from(vec![0u16; 6]);
        let raw = vec![0.0f32; 6].into_boxed_slice();
        let storage = NumericStorage::new(bins, raw);
        assert_eq!(storage.size_bytes(), 12 + 24);
    }

    #[test]
    fn test_unchecked_access() {
        let bins = BinData::from(vec![5u8, 10, 15]);
        let raw = vec![1.5, 2.5, 3.5].into_boxed_slice();
        let storage = NumericStorage::new(bins, raw);
        let n_samples = 3;

        unsafe {
            assert_eq!(storage.bin_unchecked(0, 0, n_samples), 5);
            assert_eq!(storage.bin_unchecked(2, 0, n_samples), 15);
            assert_eq!(storage.raw_unchecked(1, 0, n_samples), 2.5);
        }
    }

    #[test]
    fn test_single_feature_single_sample() {
        let bins = BinData::from(vec![42u8]);
        let raw = vec![3.14].into_boxed_slice();
        let storage = NumericStorage::new(bins, raw);

        assert_eq!(storage.n_features(1), 1);
        assert_eq!(storage.bin(0, 0, 1), 42);
        assert_eq!(storage.raw(0, 0, 1), 3.14);
        assert_eq!(storage.raw_slice(0, 1), &[3.14]);
    }

    #[test]
    fn test_with_nan_values() {
        let bins = BinData::from(vec![0u8, 1, 0]); // 0 might represent missing
        let raw = vec![1.0, 2.0, f32::NAN].into_boxed_slice();
        let storage = NumericStorage::new(bins, raw);
        let n_samples = 3;

        assert_eq!(storage.raw(0, 0, n_samples), 1.0);
        assert_eq!(storage.raw(1, 0, n_samples), 2.0);
        assert!(storage.raw(2, 0, n_samples).is_nan());
    }

    #[test]
    fn test_u16_bins() {
        let bins = BinData::from(vec![256u16, 512, 1000]);
        let raw = vec![1.0, 2.0, 3.0].into_boxed_slice();
        let storage = NumericStorage::new(bins, raw);
        let n_samples = 3;

        assert_eq!(storage.bin(0, 0, n_samples), 256);
        assert_eq!(storage.bin(1, 0, n_samples), 512);
        assert_eq!(storage.bin(2, 0, n_samples), 1000);
    }

    // =========================================================================
    // CategoricalStorage tests
    // =========================================================================

    #[test]
    fn test_categorical_storage_new() {
        let bins = BinData::from(vec![0u8, 1, 2, 3, 4, 5]);
        let storage = CategoricalStorage::new(bins);
        assert_eq!(storage.bins().len(), 6);
    }

    #[test]
    fn test_categorical_bin_access() {
        // 2 features, 3 samples
        let bins = BinData::from(vec![
            0u8, 1, 2, // Feature 0: categories
            5, 6, 7,   // Feature 1: categories
        ]);
        let storage = CategoricalStorage::new(bins);
        let n_samples = 3;

        // Feature 0
        assert_eq!(storage.bin(0, 0, n_samples), 0);
        assert_eq!(storage.bin(1, 0, n_samples), 1);
        assert_eq!(storage.bin(2, 0, n_samples), 2);

        // Feature 1
        assert_eq!(storage.bin(0, 1, n_samples), 5);
        assert_eq!(storage.bin(1, 1, n_samples), 6);
        assert_eq!(storage.bin(2, 1, n_samples), 7);
    }

    #[test]
    fn test_categorical_n_features() {
        // 3 features, 4 samples
        let bins = BinData::from(vec![0u8; 12]);
        let storage = CategoricalStorage::new(bins);

        assert_eq!(storage.n_features(4), 3);
        assert_eq!(storage.n_features(12), 1);
        assert_eq!(storage.n_features(1), 12);
        assert_eq!(storage.n_features(0), 0);
    }

    #[test]
    fn test_categorical_size_bytes() {
        // U8 bins: 6 bytes (no raw values)
        let bins = BinData::from(vec![0u8; 6]);
        let storage = CategoricalStorage::new(bins);
        assert_eq!(storage.size_bytes(), 6);

        // U16 bins: 6 * 2 = 12 bytes (no raw values)
        let bins = BinData::from(vec![0u16; 6]);
        let storage = CategoricalStorage::new(bins);
        assert_eq!(storage.size_bytes(), 12);
    }

    #[test]
    fn test_categorical_unchecked_access() {
        let bins = BinData::from(vec![5u8, 10, 15]);
        let storage = CategoricalStorage::new(bins);
        let n_samples = 3;

        unsafe {
            assert_eq!(storage.bin_unchecked(0, 0, n_samples), 5);
            assert_eq!(storage.bin_unchecked(1, 0, n_samples), 10);
            assert_eq!(storage.bin_unchecked(2, 0, n_samples), 15);
        }
    }

    #[test]
    fn test_categorical_single_feature_single_sample() {
        let bins = BinData::from(vec![42u8]);
        let storage = CategoricalStorage::new(bins);

        assert_eq!(storage.n_features(1), 1);
        assert_eq!(storage.bin(0, 0, 1), 42);
    }

    #[test]
    fn test_categorical_u16_bins() {
        let bins = BinData::from(vec![256u16, 512, 1000]);
        let storage = CategoricalStorage::new(bins);
        let n_samples = 3;

        assert_eq!(storage.bin(0, 0, n_samples), 256);
        assert_eq!(storage.bin(1, 0, n_samples), 512);
        assert_eq!(storage.bin(2, 0, n_samples), 1000);
    }
}

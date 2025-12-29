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

/// Sparse numeric storage: CSC-like, single feature.
///
/// Only stores non-zero/non-default sample entries. Samples not in
/// `sample_indices` have implicit bin=0, raw=0.0.
///
/// **Important**: Sparse storage assumes zeros are meaningful values, not missing.
/// For features where missing should be NaN, use dense storage instead.
///
/// # Examples
///
/// ```
/// use boosters::data::binned::{BinData, SparseNumericStorage};
///
/// // Feature with 100 samples, only 3 non-zero entries
/// let sample_indices = vec![5u32, 20, 50].into_boxed_slice();
/// let bins = BinData::from(vec![1u8, 2, 3]);
/// let raw_values = vec![10.0, 20.0, 30.0].into_boxed_slice();
///
/// let storage = SparseNumericStorage::new(sample_indices, bins, raw_values, 100);
///
/// // Access non-zero entries
/// assert_eq!(storage.get(5), (1, 10.0));
/// assert_eq!(storage.get(20), (2, 20.0));
///
/// // Access zero entries (not in sample_indices)
/// assert_eq!(storage.get(0), (0, 0.0));
/// assert_eq!(storage.get(99), (0, 0.0));
/// ```
#[derive(Debug, Clone)]
pub struct SparseNumericStorage {
    /// Sample indices of non-zero entries (sorted).
    sample_indices: Box<[u32]>,
    /// Bin values for non-zero entries (parallel to sample_indices).
    bins: BinData,
    /// Raw values for non-zero entries (parallel to bins).
    raw_values: Box<[f32]>,
    /// Total number of samples.
    n_samples: usize,
}

impl SparseNumericStorage {
    /// Creates a new SparseNumericStorage.
    ///
    /// # Arguments
    /// - `sample_indices`: Sorted indices of non-zero samples
    /// - `bins`: Bin values for non-zero samples
    /// - `raw_values`: Raw values for non-zero samples
    /// - `n_samples`: Total number of samples
    ///
    /// # Panics
    ///
    /// Panics if lengths don't match or sample_indices is not sorted.
    pub fn new(
        sample_indices: Box<[u32]>,
        bins: BinData,
        raw_values: Box<[f32]>,
        n_samples: usize,
    ) -> Self {
        assert_eq!(
            sample_indices.len(),
            bins.len(),
            "sample_indices and bins must have the same length"
        );
        assert_eq!(
            sample_indices.len(),
            raw_values.len(),
            "sample_indices and raw_values must have the same length"
        );
        debug_assert!(
            sample_indices.windows(2).all(|w| w[0] < w[1]),
            "sample_indices must be sorted and unique"
        );
        Self {
            sample_indices,
            bins,
            raw_values,
            n_samples,
        }
    }

    /// Returns (bin, raw) for the given sample.
    /// Returns (0, 0.0) if sample is not in the sparse indices.
    #[inline]
    pub fn get(&self, sample: usize) -> (u32, f32) {
        match self.sample_indices.binary_search(&(sample as u32)) {
            Ok(pos) => {
                // SAFETY: pos is within bounds from binary_search
                let bin = unsafe { self.bins.get_unchecked(pos) };
                let raw = self.raw_values[pos];
                (bin, raw)
            }
            Err(_) => (0, 0.0),
        }
    }

    /// Returns the bin value for the given sample.
    /// Returns 0 if sample is not in the sparse indices.
    #[inline]
    pub fn bin(&self, sample: usize) -> u32 {
        self.get(sample).0
    }

    /// Returns the raw value for the given sample.
    /// Returns 0.0 if sample is not in the sparse indices.
    #[inline]
    pub fn raw(&self, sample: usize) -> f32 {
        self.get(sample).1
    }

    /// Returns the number of non-zero entries.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.sample_indices.len()
    }

    /// Returns the total number of samples.
    #[inline]
    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    /// Returns the sparsity ratio (fraction of zeros).
    #[inline]
    pub fn sparsity(&self) -> f64 {
        if self.n_samples == 0 {
            1.0
        } else {
            1.0 - (self.nnz() as f64 / self.n_samples as f64)
        }
    }

    /// Returns the total size in bytes used by this storage.
    #[inline]
    pub fn size_bytes(&self) -> usize {
        self.sample_indices.len() * std::mem::size_of::<u32>()
            + self.bins.size_bytes()
            + self.raw_values.len() * std::mem::size_of::<f32>()
    }

    /// Returns a reference to the sample indices.
    #[inline]
    pub fn sample_indices(&self) -> &[u32] {
        &self.sample_indices
    }

    /// Returns a reference to the bin data.
    #[inline]
    pub fn bins(&self) -> &BinData {
        &self.bins
    }

    /// Returns a reference to the raw values.
    #[inline]
    pub fn raw_values(&self) -> &[f32] {
        &self.raw_values
    }

    /// Iterates over (sample_index, bin, raw) for non-zero entries.
    pub fn iter(&self) -> impl Iterator<Item = (u32, u32, f32)> + '_ {
        self.sample_indices
            .iter()
            .enumerate()
            .map(|(pos, &sample_idx)| {
                // SAFETY: pos is within bounds
                let bin = unsafe { self.bins.get_unchecked(pos) };
                let raw = self.raw_values[pos];
                (sample_idx, bin, raw)
            })
    }
}

/// Sparse categorical storage: CSC-like, single feature.
///
/// Only stores non-zero/non-default sample entries. Samples not in
/// `sample_indices` have implicit bin=0 (default category).
///
/// No raw values - bin = category ID (lossless).
///
/// # Examples
///
/// ```
/// use boosters::data::binned::{BinData, SparseCategoricalStorage};
///
/// // Feature with 100 samples, only 3 non-default entries
/// let sample_indices = vec![5u32, 20, 50].into_boxed_slice();
/// let bins = BinData::from(vec![1u8, 2, 3]);
///
/// let storage = SparseCategoricalStorage::new(sample_indices, bins, 100);
///
/// // Access non-default entries
/// assert_eq!(storage.bin(5), 1);
/// assert_eq!(storage.bin(20), 2);
///
/// // Access default entries (not in sample_indices)
/// assert_eq!(storage.bin(0), 0);
/// assert_eq!(storage.bin(99), 0);
/// ```
#[derive(Debug, Clone)]
pub struct SparseCategoricalStorage {
    /// Sample indices of non-zero entries (sorted).
    sample_indices: Box<[u32]>,
    /// Bin values for non-zero entries (parallel to sample_indices).
    bins: BinData,
    /// Total number of samples.
    n_samples: usize,
}

impl SparseCategoricalStorage {
    /// Creates a new SparseCategoricalStorage.
    ///
    /// # Arguments
    /// - `sample_indices`: Sorted indices of non-zero samples
    /// - `bins`: Bin values for non-zero samples
    /// - `n_samples`: Total number of samples
    ///
    /// # Panics
    ///
    /// Panics if lengths don't match or sample_indices is not sorted.
    pub fn new(sample_indices: Box<[u32]>, bins: BinData, n_samples: usize) -> Self {
        assert_eq!(
            sample_indices.len(),
            bins.len(),
            "sample_indices and bins must have the same length"
        );
        debug_assert!(
            sample_indices.windows(2).all(|w| w[0] < w[1]),
            "sample_indices must be sorted and unique"
        );
        Self {
            sample_indices,
            bins,
            n_samples,
        }
    }

    /// Returns the bin (category) value for the given sample.
    /// Returns 0 if sample is not in the sparse indices.
    #[inline]
    pub fn bin(&self, sample: usize) -> u32 {
        match self.sample_indices.binary_search(&(sample as u32)) {
            Ok(pos) => {
                // SAFETY: pos is within bounds from binary_search
                unsafe { self.bins.get_unchecked(pos) }
            }
            Err(_) => 0,
        }
    }

    /// Returns the number of non-zero entries.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.sample_indices.len()
    }

    /// Returns the total number of samples.
    #[inline]
    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    /// Returns the sparsity ratio (fraction of zeros).
    #[inline]
    pub fn sparsity(&self) -> f64 {
        if self.n_samples == 0 {
            1.0
        } else {
            1.0 - (self.nnz() as f64 / self.n_samples as f64)
        }
    }

    /// Returns the total size in bytes used by this storage.
    #[inline]
    pub fn size_bytes(&self) -> usize {
        self.sample_indices.len() * std::mem::size_of::<u32>() + self.bins.size_bytes()
    }

    /// Returns a reference to the sample indices.
    #[inline]
    pub fn sample_indices(&self) -> &[u32] {
        &self.sample_indices
    }

    /// Returns a reference to the bin data.
    #[inline]
    pub fn bins(&self) -> &BinData {
        &self.bins
    }

    /// Iterates over (sample_index, bin) for non-zero entries.
    pub fn iter(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        self.sample_indices
            .iter()
            .enumerate()
            .map(|(pos, &sample_idx)| {
                // SAFETY: pos is within bounds
                let bin = unsafe { self.bins.get_unchecked(pos) };
                (sample_idx, bin)
            })
    }
}

/// Unified feature storage with bins and optional raw values.
///
/// This enum wraps all storage types for homogeneous groups.
/// The variant determines whether raw values are available.
///
/// # Storage Type Properties
///
/// | Variant          | Raw Values | Sparse |
/// |------------------|------------|--------|
/// | Numeric          | ✓          | ✗      |
/// | Categorical      | ✗          | ✗      |
/// | SparseNumeric    | ✓          | ✓      |
/// | SparseCategorical| ✗          | ✓      |
///
/// # Examples
///
/// ```
/// use boosters::data::binned::{BinData, NumericStorage, CategoricalStorage, FeatureStorage};
///
/// // Numeric storage has raw values
/// let numeric = FeatureStorage::Numeric(NumericStorage::new(
///     BinData::from(vec![0u8, 1, 2]),
///     vec![1.0, 2.0, 3.0].into_boxed_slice(),
/// ));
/// assert!(numeric.has_raw_values());
/// assert!(!numeric.is_categorical());
/// assert!(!numeric.is_sparse());
///
/// // Categorical storage does not have raw values
/// let categorical = FeatureStorage::Categorical(CategoricalStorage::new(
///     BinData::from(vec![0u8, 1, 2]),
/// ));
/// assert!(!categorical.has_raw_values());
/// assert!(categorical.is_categorical());
/// ```
#[derive(Debug, Clone)]
pub enum FeatureStorage {
    /// Dense numeric features with bins + raw values.
    Numeric(NumericStorage),
    /// Dense categorical features with bins only (lossless).
    Categorical(CategoricalStorage),
    /// Sparse numeric features with bins + raw values.
    SparseNumeric(SparseNumericStorage),
    /// Sparse categorical features with bins only (lossless).
    SparseCategorical(SparseCategoricalStorage),
    // Note: Bundle variant will be added in a future story if needed.
}

impl FeatureStorage {
    /// Returns `true` if this storage has raw values.
    ///
    /// Numeric and SparseNumeric storage have raw values.
    /// Categorical and SparseCategorical storage do not (bin = category ID).
    #[inline]
    pub fn has_raw_values(&self) -> bool {
        matches!(
            self,
            FeatureStorage::Numeric(_) | FeatureStorage::SparseNumeric(_)
        )
    }

    /// Returns `true` if this storage is for categorical features.
    ///
    /// Categorical and SparseCategorical storage are for categorical features.
    #[inline]
    pub fn is_categorical(&self) -> bool {
        matches!(
            self,
            FeatureStorage::Categorical(_) | FeatureStorage::SparseCategorical(_)
        )
    }

    /// Returns `true` if this storage is sparse.
    #[inline]
    pub fn is_sparse(&self) -> bool {
        matches!(
            self,
            FeatureStorage::SparseNumeric(_) | FeatureStorage::SparseCategorical(_)
        )
    }

    /// Returns `true` if this storage is dense.
    #[inline]
    pub fn is_dense(&self) -> bool {
        !self.is_sparse()
    }

    /// Returns the total size in bytes used by this storage.
    #[inline]
    pub fn size_bytes(&self) -> usize {
        match self {
            FeatureStorage::Numeric(s) => s.size_bytes(),
            FeatureStorage::Categorical(s) => s.size_bytes(),
            FeatureStorage::SparseNumeric(s) => s.size_bytes(),
            FeatureStorage::SparseCategorical(s) => s.size_bytes(),
        }
    }

    // Note: as_*() accessor methods intentionally omitted.
    // Use exhaustive match to access the underlying storage types.
    // This encourages correct handling of all variants and prevents bugs
    // from non-exhaustive matching patterns.
}

impl From<NumericStorage> for FeatureStorage {
    fn from(storage: NumericStorage) -> Self {
        FeatureStorage::Numeric(storage)
    }
}

impl From<CategoricalStorage> for FeatureStorage {
    fn from(storage: CategoricalStorage) -> Self {
        FeatureStorage::Categorical(storage)
    }
}

impl From<SparseNumericStorage> for FeatureStorage {
    fn from(storage: SparseNumericStorage) -> Self {
        FeatureStorage::SparseNumeric(storage)
    }
}

impl From<SparseCategoricalStorage> for FeatureStorage {
    fn from(storage: SparseCategoricalStorage) -> Self {
        FeatureStorage::SparseCategorical(storage)
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

    // =========================================================================
    // SparseNumericStorage tests
    // =========================================================================

    #[test]
    fn test_sparse_numeric_new() {
        let indices = vec![1u32, 5, 10].into_boxed_slice();
        let bins = BinData::from(vec![1u8, 2, 3]);
        let raw = vec![10.0, 50.0, 100.0].into_boxed_slice();
        let storage = SparseNumericStorage::new(indices, bins, raw, 100);

        assert_eq!(storage.nnz(), 3);
        assert_eq!(storage.n_samples(), 100);
    }

    #[test]
    #[should_panic(expected = "sample_indices and bins must have the same length")]
    fn test_sparse_numeric_mismatched_bins() {
        let indices = vec![1u32, 5, 10].into_boxed_slice();
        let bins = BinData::from(vec![1u8, 2]);
        let raw = vec![10.0, 50.0, 100.0].into_boxed_slice();
        SparseNumericStorage::new(indices, bins, raw, 100);
    }

    #[test]
    #[should_panic(expected = "sample_indices and raw_values must have the same length")]
    fn test_sparse_numeric_mismatched_raw() {
        let indices = vec![1u32, 5, 10].into_boxed_slice();
        let bins = BinData::from(vec![1u8, 2, 3]);
        let raw = vec![10.0, 50.0].into_boxed_slice();
        SparseNumericStorage::new(indices, bins, raw, 100);
    }

    #[test]
    fn test_sparse_numeric_get() {
        let indices = vec![5u32, 20, 50].into_boxed_slice();
        let bins = BinData::from(vec![1u8, 2, 3]);
        let raw = vec![10.0, 20.0, 30.0].into_boxed_slice();
        let storage = SparseNumericStorage::new(indices, bins, raw, 100);

        // Non-zero entries
        assert_eq!(storage.get(5), (1, 10.0));
        assert_eq!(storage.get(20), (2, 20.0));
        assert_eq!(storage.get(50), (3, 30.0));

        // Zero entries (not in indices)
        assert_eq!(storage.get(0), (0, 0.0));
        assert_eq!(storage.get(10), (0, 0.0));
        assert_eq!(storage.get(99), (0, 0.0));
    }

    #[test]
    fn test_sparse_numeric_bin_and_raw() {
        let indices = vec![5u32].into_boxed_slice();
        let bins = BinData::from(vec![42u8]);
        let raw = vec![3.14].into_boxed_slice();
        let storage = SparseNumericStorage::new(indices, bins, raw, 10);

        assert_eq!(storage.bin(5), 42);
        assert_eq!(storage.raw(5), 3.14);
        assert_eq!(storage.bin(0), 0);
        assert_eq!(storage.raw(0), 0.0);
    }

    #[test]
    fn test_sparse_numeric_sparsity() {
        let indices = vec![1u32, 5].into_boxed_slice();
        let bins = BinData::from(vec![1u8, 2]);
        let raw = vec![1.0, 2.0].into_boxed_slice();
        let storage = SparseNumericStorage::new(indices, bins, raw, 10);

        assert!((storage.sparsity() - 0.8).abs() < 1e-6); // 8/10 zeros
    }

    #[test]
    fn test_sparse_numeric_size_bytes() {
        let indices = vec![1u32, 5, 10].into_boxed_slice(); // 12 bytes
        let bins = BinData::from(vec![1u8, 2, 3]); // 3 bytes
        let raw = vec![1.0, 2.0, 3.0].into_boxed_slice(); // 12 bytes
        let storage = SparseNumericStorage::new(indices, bins, raw, 100);

        assert_eq!(storage.size_bytes(), 12 + 3 + 12);
    }

    #[test]
    fn test_sparse_numeric_iter() {
        let indices = vec![5u32, 20, 50].into_boxed_slice();
        let bins = BinData::from(vec![1u8, 2, 3]);
        let raw = vec![10.0, 20.0, 30.0].into_boxed_slice();
        let storage = SparseNumericStorage::new(indices, bins, raw, 100);

        let items: Vec<_> = storage.iter().collect();
        assert_eq!(items, vec![(5, 1, 10.0), (20, 2, 20.0), (50, 3, 30.0)]);
    }

    #[test]
    fn test_sparse_numeric_empty() {
        let indices = vec![].into_boxed_slice();
        let bins = BinData::from(vec![0u8; 0]);
        let raw = vec![].into_boxed_slice();
        let storage = SparseNumericStorage::new(indices, bins, raw, 100);

        assert_eq!(storage.nnz(), 0);
        assert_eq!(storage.get(50), (0, 0.0));
        assert!((storage.sparsity() - 1.0).abs() < 1e-6);
    }

    // =========================================================================
    // SparseCategoricalStorage tests
    // =========================================================================

    #[test]
    fn test_sparse_categorical_new() {
        let indices = vec![1u32, 5, 10].into_boxed_slice();
        let bins = BinData::from(vec![1u8, 2, 3]);
        let storage = SparseCategoricalStorage::new(indices, bins, 100);

        assert_eq!(storage.nnz(), 3);
        assert_eq!(storage.n_samples(), 100);
    }

    #[test]
    #[should_panic(expected = "sample_indices and bins must have the same length")]
    fn test_sparse_categorical_mismatched() {
        let indices = vec![1u32, 5, 10].into_boxed_slice();
        let bins = BinData::from(vec![1u8, 2]);
        SparseCategoricalStorage::new(indices, bins, 100);
    }

    #[test]
    fn test_sparse_categorical_bin() {
        let indices = vec![5u32, 20, 50].into_boxed_slice();
        let bins = BinData::from(vec![1u8, 2, 3]);
        let storage = SparseCategoricalStorage::new(indices, bins, 100);

        // Non-zero entries
        assert_eq!(storage.bin(5), 1);
        assert_eq!(storage.bin(20), 2);
        assert_eq!(storage.bin(50), 3);

        // Zero entries (not in indices)
        assert_eq!(storage.bin(0), 0);
        assert_eq!(storage.bin(10), 0);
        assert_eq!(storage.bin(99), 0);
    }

    #[test]
    fn test_sparse_categorical_sparsity() {
        let indices = vec![1u32, 5].into_boxed_slice();
        let bins = BinData::from(vec![1u8, 2]);
        let storage = SparseCategoricalStorage::new(indices, bins, 10);

        assert!((storage.sparsity() - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_sparse_categorical_size_bytes() {
        let indices = vec![1u32, 5, 10].into_boxed_slice(); // 12 bytes
        let bins = BinData::from(vec![1u8, 2, 3]); // 3 bytes
        let storage = SparseCategoricalStorage::new(indices, bins, 100);

        assert_eq!(storage.size_bytes(), 12 + 3);
    }

    #[test]
    fn test_sparse_categorical_iter() {
        let indices = vec![5u32, 20, 50].into_boxed_slice();
        let bins = BinData::from(vec![1u8, 2, 3]);
        let storage = SparseCategoricalStorage::new(indices, bins, 100);

        let items: Vec<_> = storage.iter().collect();
        assert_eq!(items, vec![(5, 1), (20, 2), (50, 3)]);
    }

    #[test]
    fn test_sparse_categorical_empty() {
        let indices = vec![].into_boxed_slice();
        let bins = BinData::from(vec![0u8; 0]);
        let storage = SparseCategoricalStorage::new(indices, bins, 100);

        assert_eq!(storage.nnz(), 0);
        assert_eq!(storage.bin(50), 0);
        assert!((storage.sparsity() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_sparse_categorical_u16() {
        let indices = vec![5u32, 20].into_boxed_slice();
        let bins = BinData::from(vec![256u16, 512]);
        let storage = SparseCategoricalStorage::new(indices, bins, 100);

        assert_eq!(storage.bin(5), 256);
        assert_eq!(storage.bin(20), 512);
        assert_eq!(storage.bin(0), 0);
    }

    // =========================================================================
    // FeatureStorage tests
    // =========================================================================

    fn make_numeric_storage() -> NumericStorage {
        NumericStorage::new(
            BinData::from(vec![0u8, 1, 2]),
            vec![1.0, 2.0, 3.0].into_boxed_slice(),
        )
    }

    fn make_categorical_storage() -> CategoricalStorage {
        CategoricalStorage::new(BinData::from(vec![0u8, 1, 2]))
    }

    fn make_sparse_numeric_storage() -> SparseNumericStorage {
        SparseNumericStorage::new(
            vec![1u32, 5].into_boxed_slice(),
            BinData::from(vec![1u8, 2]),
            vec![10.0, 20.0].into_boxed_slice(),
            10,
        )
    }

    fn make_sparse_categorical_storage() -> SparseCategoricalStorage {
        SparseCategoricalStorage::new(
            vec![1u32, 5].into_boxed_slice(),
            BinData::from(vec![1u8, 2]),
            10,
        )
    }

    #[test]
    fn test_feature_storage_has_raw_values() {
        let numeric = FeatureStorage::Numeric(make_numeric_storage());
        let categorical = FeatureStorage::Categorical(make_categorical_storage());
        let sparse_numeric = FeatureStorage::SparseNumeric(make_sparse_numeric_storage());
        let sparse_categorical = FeatureStorage::SparseCategorical(make_sparse_categorical_storage());

        assert!(numeric.has_raw_values());
        assert!(!categorical.has_raw_values());
        assert!(sparse_numeric.has_raw_values());
        assert!(!sparse_categorical.has_raw_values());
    }

    #[test]
    fn test_feature_storage_is_categorical() {
        let numeric = FeatureStorage::Numeric(make_numeric_storage());
        let categorical = FeatureStorage::Categorical(make_categorical_storage());
        let sparse_numeric = FeatureStorage::SparseNumeric(make_sparse_numeric_storage());
        let sparse_categorical = FeatureStorage::SparseCategorical(make_sparse_categorical_storage());

        assert!(!numeric.is_categorical());
        assert!(categorical.is_categorical());
        assert!(!sparse_numeric.is_categorical());
        assert!(sparse_categorical.is_categorical());
    }

    #[test]
    fn test_feature_storage_is_sparse() {
        let numeric = FeatureStorage::Numeric(make_numeric_storage());
        let categorical = FeatureStorage::Categorical(make_categorical_storage());
        let sparse_numeric = FeatureStorage::SparseNumeric(make_sparse_numeric_storage());
        let sparse_categorical = FeatureStorage::SparseCategorical(make_sparse_categorical_storage());

        assert!(!numeric.is_sparse());
        assert!(!categorical.is_sparse());
        assert!(sparse_numeric.is_sparse());
        assert!(sparse_categorical.is_sparse());

        assert!(numeric.is_dense());
        assert!(categorical.is_dense());
        assert!(!sparse_numeric.is_dense());
        assert!(!sparse_categorical.is_dense());
    }

    #[test]
    fn test_feature_storage_size_bytes() {
        let numeric = FeatureStorage::Numeric(make_numeric_storage());
        let categorical = FeatureStorage::Categorical(make_categorical_storage());
        let sparse_numeric = FeatureStorage::SparseNumeric(make_sparse_numeric_storage());
        let sparse_categorical = FeatureStorage::SparseCategorical(make_sparse_categorical_storage());

        // Numeric: 3 u8 bins + 3 f32 raw = 3 + 12 = 15
        assert_eq!(numeric.size_bytes(), 15);
        // Categorical: 3 u8 bins = 3
        assert_eq!(categorical.size_bytes(), 3);
        // SparseNumeric: 2 u32 indices + 2 u8 bins + 2 f32 raw = 8 + 2 + 8 = 18
        assert_eq!(sparse_numeric.size_bytes(), 18);
        // SparseCategorical: 2 u32 indices + 2 u8 bins = 8 + 2 = 10
        assert_eq!(sparse_categorical.size_bytes(), 10);
    }

    // Note: test_feature_storage_as_methods removed - as_*() methods were deleted
    // to encourage exhaustive pattern matching.

    #[test]
    fn test_feature_storage_from() {
        // Test From trait implementations using exhaustive matching
        let numeric: FeatureStorage = make_numeric_storage().into();
        assert!(matches!(numeric, FeatureStorage::Numeric(_)));

        let categorical: FeatureStorage = make_categorical_storage().into();
        assert!(matches!(categorical, FeatureStorage::Categorical(_)));

        let sparse_numeric: FeatureStorage = make_sparse_numeric_storage().into();
        assert!(matches!(sparse_numeric, FeatureStorage::SparseNumeric(_)));

        let sparse_categorical: FeatureStorage = make_sparse_categorical_storage().into();
        assert!(matches!(sparse_categorical, FeatureStorage::SparseCategorical(_)));
    }
}

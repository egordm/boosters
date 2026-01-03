//! Storage types for binned feature data.
//!
//! This module provides typed storage for different kinds of features:
//! - `NumericStorage`: Dense numeric features with bins
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
/// let storage = NumericStorage::new(bins);
/// let n_samples = 3;
///
/// // Access bin values
/// assert_eq!(storage.bin(0, 0, n_samples), 0); // Feature 0, sample 0
/// assert_eq!(storage.bin(1, 1, n_samples), 4); // Feature 1, sample 1
///
/// ```
#[derive(Debug, Clone)]
pub struct NumericStorage {
    /// Bin values: [n_features × n_samples], column-major.
    bins: BinData,
}

impl NumericStorage {
    /// Creates a new NumericStorage.
    #[inline]
    pub fn new(bins: BinData) -> Self {
        Self { bins }
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
/// `sample_indices` have implicit bin=0.
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
///
/// let storage = SparseNumericStorage::new(sample_indices, bins, 100);
///
/// // Access non-zero entries
/// assert_eq!(storage.bin(5), 1);
/// assert_eq!(storage.bin(20), 2);
///
/// // Access zero entries (not in sample_indices)
/// assert_eq!(storage.bin(0), 0);
/// assert_eq!(storage.bin(99), 0);
/// ```
#[derive(Debug, Clone)]
pub struct SparseNumericStorage {
    /// Sample indices of non-zero entries (sorted).
    sample_indices: Box<[u32]>,
    /// Bin values for non-zero entries (parallel to sample_indices).
    bins: BinData,
    /// Total number of samples.
    n_samples: usize,
}

impl SparseNumericStorage {
    /// Creates a new SparseNumericStorage.
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

    /// Returns the bin value for the given sample.
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

/// EFB bundle storage: multiple sparse features encoded into one column.
///
/// Bundles are used for Exclusive Feature Bundling (EFB) where multiple sparse
/// features (typically categorical) that rarely have non-zero values simultaneously
/// are encoded into a single column with offset-encoded bins.
///
/// **Key properties**:
/// - Bundles are lossless - no raw values needed
/// - Linear trees skip bundled features for regression (use only for splits)
/// - During histogram building, bundles are treated as regular columns
/// - Decoding only happens at split time (cold path)
///
/// # Bin Encoding Scheme
///
/// For a bundle containing features [A, B, C] with bin counts [5, 10, 8]:
///
/// ```text
/// Bin offsets (computed during bundling):
///   Feature A: offset = 1,  bins 0-4 → encoded as 1-5
///   Feature B: offset = 6,  bins 0-9 → encoded as 6-15  
///   Feature C: offset = 16, bins 0-7 → encoded as 16-23
///
/// Special bin 0: "All features at default value"
///   - Rows where all features have their default (most frequent) bin
///   - Does not represent any actual feature bin
///   - decode(0) returns None
///
/// Total bundle bins: 1 + 5 + 10 + 8 = 24
/// ```
///
/// # Examples
///
/// ```
/// use boosters::data::binned::BundleStorage;
///
/// // Bundle of 3 features with bin counts [5, 10, 8]
/// // Feature indices in original dataset: [2, 7, 12]
/// let storage = BundleStorage::new(
///     vec![0u16, 4, 10, 20].into_boxed_slice(),  // encoded bins for 4 samples
///     vec![2u32, 7, 12].into_boxed_slice(),      // original feature indices
///     vec![1u32, 6, 16].into_boxed_slice(),      // bin offsets  
///     vec![5u32, 10, 8].into_boxed_slice(),      // bins per feature
///     24,                                         // total bins
///     vec![0u32, 0, 0].into_boxed_slice(),       // default bins
///     4,                                          // n_samples
/// );
///
/// // Decode encoded bins
/// assert_eq!(storage.decode(0), None);           // All defaults
/// assert_eq!(storage.decode(4), Some((0, 3)));   // Feature A (idx 0), bin 3
/// assert_eq!(storage.decode(10), Some((1, 4)));  // Feature B (idx 1), bin 4
/// assert_eq!(storage.decode(20), Some((2, 4)));  // Feature C (idx 2), bin 4
/// ```
#[derive(Debug, Clone)]
pub struct BundleStorage {
    /// Encoded bins: [n_samples], always U16.
    /// Bin 0 = all features have default value.
    /// Bin k = offset[i] + original_bin for active feature i.
    encoded_bins: Box<[u16]>,

    /// Original feature indices in this bundle (in original feature space).
    feature_indices: Box<[u32]>,

    /// Bin offset for each feature in the bundle.
    /// Offsets start at 1 (bin 0 is reserved for "all defaults").
    bin_offsets: Box<[u32]>,

    /// Number of bins per feature.
    feature_n_bins: Box<[u32]>,

    /// Total bins in this bundle (sum of all feature bins + 1 for default).
    total_bins: u32,

    /// Default bin for each feature (bin when value is "zero" / most common).
    default_bins: Box<[u32]>,

    /// Number of samples.
    n_samples: usize,
}

impl BundleStorage {
    /// Creates a new BundleStorage.
    ///
    /// # Arguments
    /// - `encoded_bins`: Encoded bin values for each sample
    /// - `feature_indices`: Original feature indices in this bundle
    /// - `bin_offsets`: Bin offset for each feature (starts at 1)
    /// - `feature_n_bins`: Number of bins per feature
    /// - `total_bins`: Total bins including the default bin 0
    /// - `default_bins`: Default bin value for each feature
    /// - `n_samples`: Number of samples
    ///
    /// # Panics
    ///
    /// Panics if array lengths are inconsistent.
    pub fn new(
        encoded_bins: Box<[u16]>,
        feature_indices: Box<[u32]>,
        bin_offsets: Box<[u32]>,
        feature_n_bins: Box<[u32]>,
        total_bins: u32,
        default_bins: Box<[u32]>,
        n_samples: usize,
    ) -> Self {
        let n_features = feature_indices.len();
        assert_eq!(
            bin_offsets.len(),
            n_features,
            "bin_offsets length must match feature_indices"
        );
        assert_eq!(
            feature_n_bins.len(),
            n_features,
            "feature_n_bins length must match feature_indices"
        );
        assert_eq!(
            default_bins.len(),
            n_features,
            "default_bins length must match feature_indices"
        );
        assert_eq!(
            encoded_bins.len(),
            n_samples,
            "encoded_bins length must match n_samples"
        );

        Self {
            encoded_bins,
            feature_indices,
            bin_offsets,
            feature_n_bins,
            total_bins,
            default_bins,
            n_samples,
        }
    }

    /// Decode an encoded bin to (position_in_bundle, original_bin).
    ///
    /// Returns `None` if encoded_bin is 0 (all features at default).
    ///
    /// The returned position is the index within this bundle's `feature_indices`,
    /// NOT the original feature index. To get the original feature index:
    /// ```ignore
    /// if let Some((pos, bin)) = storage.decode(encoded_bin) {
    ///     let original_feature = storage.feature_indices()[pos];
    /// }
    /// ```
    ///
    /// # Implementation Note
    ///
    /// For bundles with ≤4 features, uses linear scan for better cache performance.
    /// For larger bundles, uses binary search.
    #[inline]
    pub fn decode(&self, encoded_bin: u16) -> Option<(usize, u32)> {
        if encoded_bin == 0 {
            return None; // All features at default
        }

        let encoded = encoded_bin as u32;
        let n_features = self.bin_offsets.len();

        // For small bundles, linear scan is faster due to better cache locality
        if n_features <= 4 {
            // Linear scan: find the feature whose offset range contains this bin
            for i in (0..n_features).rev() {
                if encoded >= self.bin_offsets[i] {
                    let original_bin = encoded - self.bin_offsets[i];
                    if original_bin < self.feature_n_bins[i] {
                        return Some((i, original_bin));
                    }
                }
            }
            None
        } else {
            // Binary search for larger bundles
            match self.bin_offsets.binary_search(&encoded) {
                Ok(i) => Some((i, 0)), // Exactly at offset = bin 0 of feature i
                Err(i) => {
                    if i == 0 {
                        return None;
                    }
                    let feature_idx = i - 1;
                    let original_bin = encoded - self.bin_offsets[feature_idx];
                    if original_bin < self.feature_n_bins[feature_idx] {
                        Some((feature_idx, original_bin))
                    } else {
                        None // Out of range
                    }
                }
            }
        }
    }

    /// Decode an encoded bin to (original_feature_index, original_bin).
    ///
    /// This is a convenience method that returns the original feature index
    /// directly, rather than the position within the bundle.
    ///
    /// Returns `None` if encoded_bin is 0 (all features at default).
    #[inline]
    pub fn decode_to_original(&self, encoded_bin: u16) -> Option<(u32, u32)> {
        self.decode(encoded_bin)
            .map(|(pos, bin)| (self.feature_indices[pos], bin))
    }

    /// Returns the encoded bin for the given sample.
    #[inline]
    pub fn bin(&self, sample: usize) -> u16 {
        self.encoded_bins[sample]
    }

    /// Returns the encoded bin without bounds checking.
    ///
    /// # Safety
    ///
    /// `sample` must be less than `n_samples`.
    #[inline]
    pub unsafe fn bin_unchecked(&self, sample: usize) -> u16 {
        // SAFETY: Caller guarantees sample < n_samples
        unsafe { *self.encoded_bins.get_unchecked(sample) }
    }

    /// Returns the original feature indices in this bundle.
    #[inline]
    pub fn feature_indices(&self) -> &[u32] {
        &self.feature_indices
    }

    /// Returns the encoded bins array.
    #[inline]
    pub fn encoded_bins(&self) -> &[u16] {
        &self.encoded_bins
    }

    /// Returns the bin offsets for each feature.
    #[inline]
    pub fn bin_offsets(&self) -> &[u32] {
        &self.bin_offsets
    }

    /// Returns the number of bins per feature.
    #[inline]
    pub fn feature_n_bins(&self) -> &[u32] {
        &self.feature_n_bins
    }

    /// Returns the total number of bins in this bundle.
    #[inline]
    pub fn total_bins(&self) -> u32 {
        self.total_bins
    }

    /// Returns the default bin for each feature.
    #[inline]
    pub fn default_bins(&self) -> &[u32] {
        &self.default_bins
    }

    /// Returns the number of samples.
    #[inline]
    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    /// Returns the number of features in this bundle.
    #[inline]
    pub fn n_features(&self) -> usize {
        self.feature_indices.len()
    }

    /// Returns the total size in bytes used by this storage.
    #[inline]
    pub fn size_bytes(&self) -> usize {
        self.encoded_bins.len() * std::mem::size_of::<u16>()
            + self.feature_indices.len() * std::mem::size_of::<u32>()
            + self.bin_offsets.len() * std::mem::size_of::<u32>()
            + self.feature_n_bins.len() * std::mem::size_of::<u32>()
            + self.default_bins.len() * std::mem::size_of::<u32>()
    }

    /// Returns whether a sample is "all defaults" (encoded bin == 0).
    #[inline]
    pub fn is_all_defaults(&self, sample: usize) -> bool {
        self.encoded_bins[sample] == 0
    }
}

/// Unified feature storage with bins.
///
/// This enum wraps all storage types for homogeneous groups.
/// # Storage Type Properties
///
/// | Variant           | Sparse |
/// |-------------------|--------|
/// | Numeric           | ✗      |
/// | Categorical       | ✗      |
/// | SparseNumeric     | ✓      |
/// | SparseCategorical | ✓      |
/// | Bundle            | ✗      |
///
/// # Examples
///
/// ```
/// use boosters::data::binned::{BinData, NumericStorage, CategoricalStorage, FeatureStorage};
///
/// // Numeric storage
/// let numeric = FeatureStorage::Numeric(NumericStorage::new(
///     BinData::from(vec![0u8, 1, 2]),
/// ));
/// assert!(!numeric.is_categorical());
/// assert!(!numeric.is_sparse());
///
/// // Categorical storage
/// let categorical = FeatureStorage::Categorical(CategoricalStorage::new(
///     BinData::from(vec![0u8, 1, 2]),
/// ));
/// assert!(categorical.is_categorical());
/// ```
#[derive(Debug, Clone)]
pub enum FeatureStorage {
    /// Dense numeric features with bins.
    Numeric(NumericStorage),
    /// Dense categorical features with bins only (lossless).
    Categorical(CategoricalStorage),
    /// Sparse numeric features with bins.
    SparseNumeric(SparseNumericStorage),
    /// Sparse categorical features with bins only (lossless).
    SparseCategorical(SparseCategoricalStorage),
    /// EFB bundle: multiple sparse features encoded into one column.
    Bundle(BundleStorage),
}

impl FeatureStorage {
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

    /// Returns `true` if this storage is a bundle.
    #[inline]
    pub fn is_bundle(&self) -> bool {
        matches!(self, FeatureStorage::Bundle(_))
    }

    /// Returns the total size in bytes used by this storage.
    #[inline]
    pub fn size_bytes(&self) -> usize {
        match self {
            FeatureStorage::Numeric(s) => s.size_bytes(),
            FeatureStorage::Categorical(s) => s.size_bytes(),
            FeatureStorage::SparseNumeric(s) => s.size_bytes(),
            FeatureStorage::SparseCategorical(s) => s.size_bytes(),
            FeatureStorage::Bundle(s) => s.size_bytes(),
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

impl From<BundleStorage> for FeatureStorage {
    fn from(storage: BundleStorage) -> Self {
        FeatureStorage::Bundle(storage)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numeric_storage_new() {
        let bins = BinData::from(vec![0u8, 1, 2, 3, 4, 5]);
        let storage = NumericStorage::new(bins);
        assert_eq!(storage.bins().len(), 6);
    }

    #[test]
    fn test_bin_access() {
        // 2 features, 3 samples
        let bins = BinData::from(vec![
            0u8, 1, 2, // Feature 0
            10, 11, 12, // Feature 1
        ]);
        let storage = NumericStorage::new(bins);
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
    fn test_n_features() {
        // 3 features, 4 samples
        let bins = BinData::from(vec![0u8; 12]);
        let storage = NumericStorage::new(bins);

        assert_eq!(storage.n_features(4), 3);
        assert_eq!(storage.n_features(12), 1);
        assert_eq!(storage.n_features(1), 12);
        assert_eq!(storage.n_features(0), 0);
    }

    #[test]
    fn test_size_bytes() {
        // U8 bins: 6 bytes
        let bins = BinData::from(vec![0u8; 6]);
        let storage = NumericStorage::new(bins);
        assert_eq!(storage.size_bytes(), 6);

        // U16 bins: 6 * 2 = 12 bytes
        let bins = BinData::from(vec![0u16; 6]);
        let storage = NumericStorage::new(bins);
        assert_eq!(storage.size_bytes(), 12);
    }

    #[test]
    fn test_unchecked_access() {
        let bins = BinData::from(vec![5u8, 10, 15]);
        let storage = NumericStorage::new(bins);
        let n_samples = 3;

        unsafe {
            assert_eq!(storage.bin_unchecked(0, 0, n_samples), 5);
            assert_eq!(storage.bin_unchecked(2, 0, n_samples), 15);
        }
    }

    #[test]
    fn test_single_feature_single_sample() {
        let bins = BinData::from(vec![42u8]);
        let storage = NumericStorage::new(bins);

        assert_eq!(storage.n_features(1), 1);
        assert_eq!(storage.bin(0, 0, 1), 42);
    }

    #[test]
    fn test_u16_bins() {
        let bins = BinData::from(vec![256u16, 512, 1000]);
        let storage = NumericStorage::new(bins);
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
            5, 6, 7, // Feature 1: categories
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
        let storage = SparseNumericStorage::new(indices, bins, 100);

        assert_eq!(storage.nnz(), 3);
        assert_eq!(storage.n_samples(), 100);
    }

    #[test]
    #[should_panic(expected = "sample_indices and bins must have the same length")]
    fn test_sparse_numeric_mismatched_bins() {
        let indices = vec![1u32, 5, 10].into_boxed_slice();
        let bins = BinData::from(vec![1u8, 2]);
        SparseNumericStorage::new(indices, bins, 100);
    }

    #[test]
    fn test_sparse_numeric_bin() {
        let indices = vec![5u32, 20, 50].into_boxed_slice();
        let bins = BinData::from(vec![1u8, 2, 3]);
        let storage = SparseNumericStorage::new(indices, bins, 100);

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
    fn test_sparse_numeric_bin_singleton() {
        let indices = vec![5u32].into_boxed_slice();
        let bins = BinData::from(vec![42u8]);
        let storage = SparseNumericStorage::new(indices, bins, 10);

        assert_eq!(storage.bin(5), 42);
        assert_eq!(storage.bin(0), 0);
    }

    #[test]
    fn test_sparse_numeric_sparsity() {
        let indices = vec![1u32, 5].into_boxed_slice();
        let bins = BinData::from(vec![1u8, 2]);
        let storage = SparseNumericStorage::new(indices, bins, 10);

        assert!((storage.sparsity() - 0.8).abs() < 1e-6); // 8/10 zeros
    }

    #[test]
    fn test_sparse_numeric_size_bytes() {
        let indices = vec![1u32, 5, 10].into_boxed_slice(); // 12 bytes
        let bins = BinData::from(vec![1u8, 2, 3]); // 3 bytes
        let storage = SparseNumericStorage::new(indices, bins, 100);

        assert_eq!(storage.size_bytes(), 12 + 3);
    }

    #[test]
    fn test_sparse_numeric_iter() {
        let indices = vec![5u32, 20, 50].into_boxed_slice();
        let bins = BinData::from(vec![1u8, 2, 3]);
        let storage = SparseNumericStorage::new(indices, bins, 100);

        let items: Vec<_> = storage.iter().collect();
        assert_eq!(items, vec![(5, 1), (20, 2), (50, 3)]);
    }

    #[test]
    fn test_sparse_numeric_empty() {
        let indices = vec![].into_boxed_slice();
        let bins = BinData::from(vec![0u8; 0]);
        let storage = SparseNumericStorage::new(indices, bins, 100);

        assert_eq!(storage.nnz(), 0);
        assert_eq!(storage.bin(50), 0);
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
        NumericStorage::new(BinData::from(vec![0u8, 1, 2]))
    }

    fn make_categorical_storage() -> CategoricalStorage {
        CategoricalStorage::new(BinData::from(vec![0u8, 1, 2]))
    }

    fn make_sparse_numeric_storage() -> SparseNumericStorage {
        SparseNumericStorage::new(
            vec![1u32, 5].into_boxed_slice(),
            BinData::from(vec![1u8, 2]),
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
    fn test_feature_storage_is_categorical() {
        let numeric = FeatureStorage::Numeric(make_numeric_storage());
        let categorical = FeatureStorage::Categorical(make_categorical_storage());
        let sparse_numeric = FeatureStorage::SparseNumeric(make_sparse_numeric_storage());
        let sparse_categorical =
            FeatureStorage::SparseCategorical(make_sparse_categorical_storage());

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
        let sparse_categorical =
            FeatureStorage::SparseCategorical(make_sparse_categorical_storage());

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
        let sparse_categorical =
            FeatureStorage::SparseCategorical(make_sparse_categorical_storage());

        // Numeric: 3 u8 bins = 3
        assert_eq!(numeric.size_bytes(), 3);
        // Categorical: 3 u8 bins = 3
        assert_eq!(categorical.size_bytes(), 3);
        // SparseNumeric: 2 u32 indices + 2 u8 bins = 8 + 2 = 10
        assert_eq!(sparse_numeric.size_bytes(), 10);
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
        assert!(matches!(
            sparse_categorical,
            FeatureStorage::SparseCategorical(_)
        ));
    }

    // =========================================================================
    // Property-based tests using randomized inputs
    // =========================================================================

    use rand::{Rng, SeedableRng};
    use rand_xoshiro::Xoshiro256PlusPlus;

    /// Generate random NumericStorage and verify invariants
    #[test]
    fn test_property_numeric_storage_invariants() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(12345);

        for _ in 0..20 {
            let n_features = rng.gen_range(1..=10);
            let n_samples = rng.gen_range(1..=100);
            let total_len = n_features * n_samples;

            // Generate random data
            let bins: Vec<u8> = (0..total_len).map(|_| rng.gen_range(0..=255)).collect();

            let storage = NumericStorage::new(BinData::from(bins.clone()));

            // Property 1: bins.len() == n_features * n_samples
            assert_eq!(storage.bins().len(), total_len);

            // Property 2: All elements accessible and match input
            for f in 0..n_features {
                for s in 0..n_samples {
                    let idx = f * n_samples + s;
                    assert_eq!(storage.bin(s, f, n_samples), bins[idx] as u32);
                }
            }

            // Property 3: n_features computed correctly
            assert_eq!(storage.n_features(n_samples), n_features);
        }
    }

    /// Generate random CategoricalStorage and verify invariants
    #[test]
    fn test_property_categorical_storage_invariants() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(67890);

        for _ in 0..20 {
            let n_features = rng.gen_range(1..=10);
            let n_samples = rng.gen_range(1..=100);
            let max_categories = rng.gen_range(2..=50);
            let total_len = n_features * n_samples;

            // Generate random data with valid category range
            let bins: Vec<u8> = (0..total_len)
                .map(|_| rng.gen_range(0..max_categories) as u8)
                .collect();

            let storage = CategoricalStorage::new(BinData::from(bins.clone()));

            // Property 1: All bin values accessible and match input
            for f in 0..n_features {
                for s in 0..n_samples {
                    let idx = f * n_samples + s;
                    assert_eq!(storage.bin(s, f, n_samples), bins[idx] as u32);
                }
            }

            // Property 2: All values within expected range
            for f in 0..n_features {
                for s in 0..n_samples {
                    let bin = storage.bin(s, f, n_samples);
                    assert!(bin < max_categories as u32);
                }
            }

            // Property 3: n_features computed correctly
            assert_eq!(storage.n_features(n_samples), n_features);
        }
    }

    /// Generate random SparseNumericStorage and verify invariants
    #[test]
    fn test_property_sparse_numeric_storage_invariants() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(11111);

        for _ in 0..20 {
            let n_samples = rng.gen_range(10..=100);
            let nnz = rng.gen_range(1..=n_samples.min(20));

            // Generate sorted unique indices
            let mut indices: Vec<u32> = (0..n_samples as u32).collect();
            use rand::seq::SliceRandom;
            indices.shuffle(&mut rng);
            indices.truncate(nnz);
            indices.sort();

            let bins: Vec<u8> = (0..nnz).map(|_| rng.gen_range(1..=255)).collect();

            let storage = SparseNumericStorage::new(
                indices.clone().into_boxed_slice(),
                BinData::from(bins.clone()),
                n_samples,
            );

            // Property 1: nnz() returns correct count
            assert_eq!(storage.nnz(), nnz);

            // Property 2: n_samples() returns correct count
            assert_eq!(storage.n_samples(), n_samples);

            // Property 3: Indexed samples return correct values
            for (pos, &sample_idx) in indices.iter().enumerate() {
                let bin = storage.bin(sample_idx as usize);
                assert_eq!(bin, bins[pos] as u32);
            }

            // Property 4: Non-indexed samples return 0
            for s in 0..n_samples {
                if !indices.contains(&(s as u32)) {
                    assert_eq!(storage.bin(s), 0);
                }
            }

            // Property 5: Iteration yields all entries in order
            let iter_entries: Vec<_> = storage.iter().collect();
            assert_eq!(iter_entries.len(), nnz);
            for (i, (idx, bin)) in iter_entries.into_iter().enumerate() {
                assert_eq!(idx, indices[i]);
                assert_eq!(bin, bins[i] as u32);
            }
        }
    }

    /// Edge case: empty storages
    #[test]
    fn test_property_empty_storages() {
        // Empty NumericStorage - this would be unusual but valid for 0 samples
        let empty_bins = BinData::from(Vec::<u8>::new());
        let storage = NumericStorage::new(empty_bins);
        assert_eq!(storage.bins().len(), 0);
        assert_eq!(storage.n_features(0), 0); // 0/0 division handled
        assert_eq!(storage.n_features(1), 0); // 0/1 = 0 features

        // Empty SparseNumericStorage
        let sparse = SparseNumericStorage::new(
            vec![].into_boxed_slice(),
            BinData::from(Vec::<u8>::new()),
            10, // 10 samples, all implicit zeros
        );
        assert_eq!(sparse.nnz(), 0);
        assert_eq!(sparse.n_samples(), 10);
        assert_eq!(sparse.sparsity(), 1.0); // 100% sparse

        // Access any sample should return default
        for s in 0..10 {
            assert_eq!(sparse.bin(s), 0);
        }
    }

    /// Edge case: single sample, single feature
    #[test]
    fn test_property_single_element() {
        let storage = NumericStorage::new(BinData::from(vec![42u8]));

        assert_eq!(storage.n_features(1), 1);
        assert_eq!(storage.bin(0, 0, 1), 42);
    }

    /// Test U16 storage with values > 255
    #[test]
    fn test_property_u16_large_bins() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(22222);

        let n_features = 3;
        let n_samples = 10;
        let total_len = n_features * n_samples;

        // Generate bins with values requiring U16
        let bins: Vec<u16> = (0..total_len).map(|_| rng.gen_range(256..=1000)).collect();

        let storage = NumericStorage::new(BinData::from(bins.clone()));

        // Verify all values accessible
        for f in 0..n_features {
            for s in 0..n_samples {
                let idx = f * n_samples + s;
                assert_eq!(storage.bin(s, f, n_samples), bins[idx] as u32);
            }
        }

        // Verify storage type
        assert!(storage.bins().is_u16());
    }

    // =========================================================================
    // BundleStorage tests
    // =========================================================================

    fn make_test_bundle() -> BundleStorage {
        // Bundle of 3 features with bin counts [5, 10, 8]
        // Offsets: [1, 6, 16] (starts at 1, then 1+5=6, then 6+10=16)
        // Total bins: 1 + 5 + 10 + 8 = 24
        BundleStorage::new(
            vec![0u16, 4, 10, 20].into_boxed_slice(), // encoded bins
            vec![2u32, 7, 12].into_boxed_slice(),     // feature indices
            vec![1u32, 6, 16].into_boxed_slice(),     // bin offsets
            vec![5u32, 10, 8].into_boxed_slice(),     // n_bins per feature
            24,                                       // total bins
            vec![0u32, 0, 0].into_boxed_slice(),      // default bins
            4,                                        // n_samples
        )
    }

    #[test]
    fn test_bundle_storage_new() {
        let storage = make_test_bundle();
        assert_eq!(storage.n_samples(), 4);
        assert_eq!(storage.n_features(), 3);
        assert_eq!(storage.total_bins(), 24);
        assert_eq!(storage.feature_indices(), &[2, 7, 12]);
        assert_eq!(storage.bin_offsets(), &[1, 6, 16]);
        assert_eq!(storage.feature_n_bins(), &[5, 10, 8]);
    }

    #[test]
    #[should_panic(expected = "bin_offsets length must match feature_indices")]
    fn test_bundle_storage_mismatched_offsets() {
        BundleStorage::new(
            vec![0u16].into_boxed_slice(),
            vec![1u32, 2].into_boxed_slice(), // 2 features
            vec![1u32].into_boxed_slice(),    // only 1 offset
            vec![5u32, 10].into_boxed_slice(),
            16,
            vec![0u32, 0].into_boxed_slice(),
            1,
        );
    }

    #[test]
    #[should_panic(expected = "encoded_bins length must match n_samples")]
    fn test_bundle_storage_mismatched_samples() {
        BundleStorage::new(
            vec![0u16, 1, 2].into_boxed_slice(), // 3 encoded bins
            vec![1u32, 2].into_boxed_slice(),
            vec![1u32, 6].into_boxed_slice(),
            vec![5u32, 10].into_boxed_slice(),
            16,
            vec![0u32, 0].into_boxed_slice(),
            5, // but says 5 samples
        );
    }

    #[test]
    fn test_bundle_decode_all_defaults() {
        let storage = make_test_bundle();
        // Encoded bin 0 means all features at default
        assert_eq!(storage.decode(0), None);
    }

    #[test]
    fn test_bundle_decode_feature_a() {
        let storage = make_test_bundle();
        // Feature A: offset=1, n_bins=5, so encoded 1-5 → bins 0-4
        assert_eq!(storage.decode(1), Some((0, 0))); // pos 0, bin 0
        assert_eq!(storage.decode(2), Some((0, 1))); // pos 0, bin 1
        assert_eq!(storage.decode(3), Some((0, 2))); // pos 0, bin 2
        assert_eq!(storage.decode(4), Some((0, 3))); // pos 0, bin 3
        assert_eq!(storage.decode(5), Some((0, 4))); // pos 0, bin 4
    }

    #[test]
    fn test_bundle_decode_feature_b() {
        let storage = make_test_bundle();
        // Feature B: offset=6, n_bins=10, so encoded 6-15 → bins 0-9
        assert_eq!(storage.decode(6), Some((1, 0))); // pos 1, bin 0
        assert_eq!(storage.decode(10), Some((1, 4))); // pos 1, bin 4
        assert_eq!(storage.decode(15), Some((1, 9))); // pos 1, bin 9
    }

    #[test]
    fn test_bundle_decode_feature_c() {
        let storage = make_test_bundle();
        // Feature C: offset=16, n_bins=8, so encoded 16-23 → bins 0-7
        assert_eq!(storage.decode(16), Some((2, 0))); // pos 2, bin 0
        assert_eq!(storage.decode(20), Some((2, 4))); // pos 2, bin 4
        assert_eq!(storage.decode(23), Some((2, 7))); // pos 2, bin 7
    }

    #[test]
    fn test_bundle_decode_out_of_range() {
        let storage = make_test_bundle();
        // Encoded bin 24 is out of range (total_bins = 24, valid range is 0-23)
        assert_eq!(storage.decode(24), None);
        assert_eq!(storage.decode(100), None);
    }

    #[test]
    fn test_bundle_decode_to_original() {
        let storage = make_test_bundle();
        // Feature indices are [2, 7, 12]
        assert_eq!(storage.decode_to_original(0), None); // all defaults
        assert_eq!(storage.decode_to_original(4), Some((2, 3))); // feature 2, bin 3
        assert_eq!(storage.decode_to_original(10), Some((7, 4))); // feature 7, bin 4
        assert_eq!(storage.decode_to_original(20), Some((12, 4))); // feature 12, bin 4
    }

    #[test]
    fn test_bundle_bin_access() {
        let storage = make_test_bundle();
        // Encoded bins are [0, 4, 10, 20]
        assert_eq!(storage.bin(0), 0);
        assert_eq!(storage.bin(1), 4);
        assert_eq!(storage.bin(2), 10);
        assert_eq!(storage.bin(3), 20);
    }

    #[test]
    fn test_bundle_is_all_defaults() {
        let storage = make_test_bundle();
        assert!(storage.is_all_defaults(0)); // bin 0
        assert!(!storage.is_all_defaults(1)); // bin 4
        assert!(!storage.is_all_defaults(2)); // bin 10
        assert!(!storage.is_all_defaults(3)); // bin 20
    }

    #[test]
    fn test_bundle_size_bytes() {
        let storage = make_test_bundle();
        // encoded_bins: 4 * 2 = 8
        // feature_indices: 3 * 4 = 12
        // bin_offsets: 3 * 4 = 12
        // feature_n_bins: 3 * 4 = 12
        // default_bins: 3 * 4 = 12
        // Total: 8 + 12 + 12 + 12 + 12 = 56
        assert_eq!(storage.size_bytes(), 56);
    }

    #[test]
    fn test_bundle_single_feature() {
        // Bundle with just 1 feature - should still work
        let storage = BundleStorage::new(
            vec![0u16, 3, 5].into_boxed_slice(),
            vec![42u32].into_boxed_slice(), // original feature index
            vec![1u32].into_boxed_slice(),  // offset starts at 1
            vec![10u32].into_boxed_slice(), // 10 bins
            11,                             // 1 + 10
            vec![0u32].into_boxed_slice(),
            3,
        );

        assert_eq!(storage.n_features(), 1);
        assert_eq!(storage.decode(0), None);
        assert_eq!(storage.decode(1), Some((0, 0)));
        assert_eq!(storage.decode(5), Some((0, 4)));
        assert_eq!(storage.decode(10), Some((0, 9)));
        assert_eq!(storage.decode(11), None); // out of range
    }

    #[test]
    fn test_bundle_large_for_binary_search() {
        // Bundle with 10 features to trigger binary search path
        let n_features = 10;
        let bins_per_feature = 5;
        let feature_indices: Vec<u32> = (0..n_features as u32).collect();
        let mut offsets = vec![1u32];
        for i in 1..n_features {
            offsets.push(offsets[i - 1] + bins_per_feature);
        }
        let n_bins: Vec<u32> = vec![bins_per_feature; n_features];
        let total_bins = 1 + n_features as u32 * bins_per_feature;
        let defaults: Vec<u32> = vec![0; n_features];

        let storage = BundleStorage::new(
            vec![0u16, 3, 8, 25, 47].into_boxed_slice(), // some sample encoded bins
            feature_indices.into_boxed_slice(),
            offsets.into_boxed_slice(),
            n_bins.into_boxed_slice(),
            total_bins,
            defaults.into_boxed_slice(),
            5,
        );

        // Verify decoding works for all features
        // Feature 0: offset=1, bins 0-4 → encoded 1-5
        assert_eq!(storage.decode(1), Some((0, 0)));
        assert_eq!(storage.decode(5), Some((0, 4)));

        // Feature 5: offset=1+5*5=26, bins 0-4 → encoded 26-30
        assert_eq!(storage.decode(26), Some((5, 0)));
        assert_eq!(storage.decode(30), Some((5, 4)));

        // Feature 9: offset=1+5*9=46, bins 0-4 → encoded 46-50
        assert_eq!(storage.decode(46), Some((9, 0)));
        assert_eq!(storage.decode(50), Some((9, 4)));
    }

    #[test]
    fn test_feature_storage_bundle_properties() {
        let bundle = FeatureStorage::Bundle(make_test_bundle());

        assert!(!bundle.is_categorical());
        assert!(!bundle.is_sparse());
        assert!(bundle.is_dense());
        assert!(bundle.is_bundle());
        assert_eq!(bundle.size_bytes(), 56);
    }

    #[test]
    fn test_feature_storage_bundle_from() {
        let bundle: FeatureStorage = make_test_bundle().into();
        assert!(matches!(bundle, FeatureStorage::Bundle(_)));
    }
}

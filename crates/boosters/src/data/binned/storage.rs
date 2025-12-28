//! Storage types for binned data.

/// Storage layout for a feature group.
///
/// Determines how bin values are organized in memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum GroupLayout {
    /// Column-major: bins stored per-feature, features contiguous.
    /// Layout: [f0_row0, f0_row1, ..., f0_rowN, f1_row0, f1_row1, ...]
    /// Good for: feature-parallel histogram building, sparse features
    #[default]
    ColumnMajor,

    /// Row-major: bins stored per-row, rows contiguous.
    /// Layout: [row0_f0, row0_f1, ..., row0_fK, row1_f0, row1_f1, ...]
    /// Good for: row-parallel histogram building, sequential row access
    RowMajor,
}

/// Bin data type for a feature group.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum BinType {
    /// 8-bit bins (max 256 bins per feature).
    #[default]
    U8,
    /// 16-bit bins (max 65536 bins per feature).
    U16,
}

impl BinType {
    /// Maximum number of bins this type can represent.
    #[inline]
    pub const fn max_bins(self) -> u32 {
        match self {
            Self::U8 => 256,
            Self::U16 => 65536,
        }
    }

    /// Size in bytes per bin value.
    #[inline]
    pub const fn size_bytes(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::U16 => 2,
        }
    }

    /// Select appropriate bin type for a given max bin count.
    ///
    /// Returns `None` if max_bins > 65536 (not supported).
    #[inline]
    pub fn for_max_bins(max_bins: u32) -> Option<Self> {
        if max_bins <= 256 {
            Some(Self::U8)
        } else if max_bins <= 65536 {
            Some(Self::U16)
        } else {
            None
        }
    }
}

// =============================================================================
// NEW STORAGE TYPES (RFC-0018)
// =============================================================================

/// Bin data container with typed storage.
///
/// This enum holds the actual bin values and encodes the bin width (U8 or U16)
/// in its variant. It replaces the need for a separate `BinType` enum.
///
/// Part of the RFC-0018 storage hierarchy:
/// - `BinData` - raw bin values (this type)
/// - `NumericStorage` / `CategoricalStorage` - semantic wrappers with optional raw values
/// - `FeatureStorage` - unified enum for all storage types
///
/// # Layout
///
/// All data is stored column-major: `[f0_row0, f0_row1, ..., f0_rowN, f1_row0, ...]`
///
/// # Example
///
/// ```ignore
/// let bins = BinData::from_u8(vec![0, 1, 2, 0, 1, 2]); // 2 features, 3 samples
/// assert!(bins.is_u8());
/// assert_eq!(bins.len(), 6);
/// assert_eq!(bins.get(0), 0);
/// assert_eq!(bins.get(3), 0); // second feature, first sample
/// ```
#[derive(Clone, Debug)]
pub enum BinData {
    /// 8-bit bins (max 256 bins per feature).
    U8(Box<[u8]>),
    /// 16-bit bins (max 65536 bins per feature).
    U16(Box<[u16]>),
}

impl BinData {
    /// Create from u8 vector.
    #[inline]
    pub fn from_u8(data: Vec<u8>) -> Self {
        Self::U8(data.into_boxed_slice())
    }

    /// Create from u16 vector.
    #[inline]
    pub fn from_u16(data: Vec<u16>) -> Self {
        Self::U16(data.into_boxed_slice())
    }

    /// Whether this is U8 storage.
    #[inline]
    pub fn is_u8(&self) -> bool {
        matches!(self, Self::U8(_))
    }

    /// Whether this is U16 storage.
    #[inline]
    pub fn is_u16(&self) -> bool {
        matches!(self, Self::U16(_))
    }

    /// Number of bin values stored.
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Self::U8(data) => data.len(),
            Self::U16(data) => data.len(),
        }
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get bin value at index as u32.
    ///
    /// # Panics
    /// Panics if index is out of bounds.
    #[inline]
    pub fn get(&self, idx: usize) -> u32 {
        match self {
            Self::U8(data) => data[idx] as u32,
            Self::U16(data) => data[idx] as u32,
        }
    }

    /// Get bin value at index without bounds checking.
    ///
    /// # Safety
    /// Caller must ensure idx < self.len().
    #[inline]
    pub unsafe fn get_unchecked(&self, idx: usize) -> u32 {
        match self {
            Self::U8(data) => unsafe { *data.get_unchecked(idx) as u32 },
            Self::U16(data) => unsafe { *data.get_unchecked(idx) as u32 },
        }
    }

    /// Memory size in bytes.
    #[inline]
    pub fn size_bytes(&self) -> usize {
        match self {
            Self::U8(data) => data.len(),
            Self::U16(data) => data.len() * 2,
        }
    }

    /// Get a reference to the underlying u8 slice.
    ///
    /// Returns `None` if this is U16 storage.
    #[inline]
    pub fn as_u8_slice(&self) -> Option<&[u8]> {
        match self {
            Self::U8(data) => Some(data),
            Self::U16(_) => None,
        }
    }

    /// Get a reference to the underlying u16 slice.
    ///
    /// Returns `None` if this is U8 storage.
    #[inline]
    pub fn as_u16_slice(&self) -> Option<&[u16]> {
        match self {
            Self::U8(_) => None,
            Self::U16(data) => Some(data),
        }
    }
}

impl Default for BinData {
    fn default() -> Self {
        Self::U8(Box::new([]))
    }
}

/// Dense numeric storage with raw values.
///
/// Stores both binned and raw feature values for numeric features.
/// This enables linear trees to access original feature values for regression.
///
/// # Layout
///
/// Column-major: `[n_features × n_samples]`
/// For feature `f` at sample `s`: index = `f * n_samples + s`
///
/// Both `bins` and `raw_values` share the same layout.
///
/// # Example
///
/// ```ignore
/// // 2 features, 3 samples
/// let bins = BinData::from_u8(vec![0, 1, 2, 3, 4, 5]);
/// let raw = vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0];
/// let storage = NumericStorage::new(bins, raw);
///
/// // Access feature 0, sample 1
/// assert_eq!(storage.bin(1, 0, 3), 1);
/// assert_eq!(storage.raw(1, 0, 3), 2.0);
///
/// // Get contiguous slice for feature 0
/// assert_eq!(storage.raw_slice(0, 3), &[1.0, 2.0, 3.0]);
/// ```
#[derive(Clone, Debug)]
pub struct NumericStorage {
    /// Bin values (column-major).
    bins: BinData,
    /// Raw f32 values (column-major, same layout as bins).
    /// Stores original feature values including NaN for missing.
    raw_values: Box<[f32]>,
}

impl NumericStorage {
    /// Create new numeric storage.
    ///
    /// # Arguments
    /// * `bins` - Binned values (column-major)
    /// * `raw_values` - Original f32 values (same layout as bins)
    ///
    /// # Panics (debug only)
    /// Panics if bins.len() != raw_values.len()
    pub fn new(bins: BinData, raw_values: Vec<f32>) -> Self {
        debug_assert_eq!(
            bins.len(),
            raw_values.len(),
            "bins and raw_values must have same length"
        );
        Self {
            bins,
            raw_values: raw_values.into_boxed_slice(),
        }
    }

    /// Get bin value at (sample, feature_in_group).
    ///
    /// # Arguments
    /// * `sample` - Sample index (row)
    /// * `feature_in_group` - Feature index within this storage group
    /// * `n_samples` - Total number of samples (for stride calculation)
    #[inline]
    pub fn bin(&self, sample: usize, feature_in_group: usize, n_samples: usize) -> u32 {
        let idx = feature_in_group * n_samples + sample;
        self.bins.get(idx)
    }

    /// Get raw value at (sample, feature_in_group).
    #[inline]
    pub fn raw(&self, sample: usize, feature_in_group: usize, n_samples: usize) -> f32 {
        let idx = feature_in_group * n_samples + sample;
        self.raw_values[idx]
    }

    /// Get contiguous raw value slice for a feature.
    ///
    /// Returns a slice of length `n_samples` containing all raw values
    /// for the specified feature. Efficient for linear trees.
    #[inline]
    pub fn raw_slice(&self, feature_in_group: usize, n_samples: usize) -> &[f32] {
        let start = feature_in_group * n_samples;
        &self.raw_values[start..start + n_samples]
    }

    /// Get reference to underlying bin data.
    #[inline]
    pub fn bins(&self) -> &BinData {
        &self.bins
    }

    /// Get reference to raw values slice.
    #[inline]
    pub fn raw_values(&self) -> &[f32] {
        &self.raw_values
    }

    /// Total number of values stored (bins.len() == raw_values.len()).
    #[inline]
    pub fn len(&self) -> usize {
        self.bins.len()
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.bins.is_empty()
    }

    /// Memory size in bytes (bins + raw values).
    #[inline]
    pub fn size_bytes(&self) -> usize {
        self.bins.size_bytes() + self.raw_values.len() * 4
    }
}

/// Dense categorical storage (no raw values).
///
/// Stores only binned values for categorical features.
/// Categorical encoding is lossless - bin index = category ID.
/// No raw values needed since `bin_to_value()` can recover exact values.
///
/// # Layout
///
/// Column-major: `[n_features × n_samples]`
/// For feature `f` at sample `s`: index = `f * n_samples + s`
///
/// # Example
///
/// ```ignore
/// // 2 features, 3 samples
/// let bins = BinData::from_u8(vec![0, 1, 0, 2, 0, 1]);
/// let storage = CategoricalStorage::new(bins);
///
/// // Access feature 1, sample 2
/// assert_eq!(storage.bin(2, 1, 3), 1);
/// ```
#[derive(Clone, Debug)]
pub struct CategoricalStorage {
    /// Bin values (column-major). Bin index = category ID.
    bins: BinData,
}

impl CategoricalStorage {
    /// Create new categorical storage.
    pub fn new(bins: BinData) -> Self {
        Self { bins }
    }

    /// Get bin value at (sample, feature_in_group).
    ///
    /// For categorical features, bin value = category ID.
    #[inline]
    pub fn bin(&self, sample: usize, feature_in_group: usize, n_samples: usize) -> u32 {
        let idx = feature_in_group * n_samples + sample;
        self.bins.get(idx)
    }

    /// Get reference to underlying bin data.
    #[inline]
    pub fn bins(&self) -> &BinData {
        &self.bins
    }

    /// Total number of values stored.
    #[inline]
    pub fn len(&self) -> usize {
        self.bins.len()
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.bins.is_empty()
    }

    /// Memory size in bytes.
    #[inline]
    pub fn size_bytes(&self) -> usize {
        self.bins.size_bytes()
    }
}

/// Sparse numeric storage: CSC-like, single feature.
///
/// Stores only non-zero samples. Samples not in `sample_indices` have
/// implicit bin=0, raw=0.0. Use when feature sparsity is high.
///
/// **Important**: Sparse storage assumes zeros are meaningful values, not missing.
/// For features where missing should be NaN, use dense storage instead.
///
/// # Layout
///
/// Stores parallel arrays of (sample_index, bin, raw_value) for non-zero entries.
/// Sample indices are sorted for efficient binary search.
///
/// # Example
///
/// ```ignore
/// // 10 samples, only 3 non-zero
/// let indices = vec![2, 5, 8];
/// let bins = BinData::from_u8(vec![1, 2, 3]);
/// let raw = vec![1.5, 2.5, 3.5];
/// let storage = SparseNumericStorage::new(indices, bins, raw, 10);
///
/// // Access via binary search
/// assert_eq!(storage.get(5), (2, 2.5));
/// assert_eq!(storage.get(3), (0, 0.0)); // Not in indices -> implicit zero
/// ```
#[derive(Clone, Debug)]
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
    /// Create new sparse numeric storage.
    ///
    /// # Arguments
    /// * `sample_indices` - Sample indices with non-zero values (must be sorted)
    /// * `bins` - Bin values for non-zero samples
    /// * `raw_values` - Raw values for non-zero samples
    /// * `n_samples` - Total number of samples
    ///
    /// # Panics (debug only)
    /// Panics if arrays have mismatched lengths or indices are unsorted.
    pub fn new(
        sample_indices: Vec<u32>,
        bins: BinData,
        raw_values: Vec<f32>,
        n_samples: usize,
    ) -> Self {
        debug_assert_eq!(
            sample_indices.len(),
            bins.len(),
            "sample_indices and bins must have same length"
        );
        debug_assert_eq!(
            sample_indices.len(),
            raw_values.len(),
            "sample_indices and raw_values must have same length"
        );
        debug_assert!(
            sample_indices.windows(2).all(|w| w[0] < w[1]),
            "sample_indices must be strictly sorted"
        );
        Self {
            sample_indices: sample_indices.into_boxed_slice(),
            bins,
            raw_values: raw_values.into_boxed_slice(),
            n_samples,
        }
    }

    /// Get (bin, raw) value for sample via binary search.
    ///
    /// Returns (0, 0.0) for samples not in indices (implicit zero).
    #[inline]
    pub fn get(&self, sample: usize) -> (u32, f32) {
        match self.sample_indices.binary_search(&(sample as u32)) {
            Ok(pos) => (self.bins.get(pos), self.raw_values[pos]),
            Err(_) => (0, 0.0),
        }
    }

    /// Get bin value for sample (O(log nnz)).
    #[inline]
    pub fn bin(&self, sample: usize) -> u32 {
        self.get(sample).0
    }

    /// Get raw value for sample (O(log nnz)).
    #[inline]
    pub fn raw(&self, sample: usize) -> f32 {
        self.get(sample).1
    }

    /// Number of non-zero entries.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.sample_indices.len()
    }

    /// Total number of samples.
    #[inline]
    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    /// Sparsity ratio (nnz / n_samples).
    #[inline]
    pub fn sparsity(&self) -> f64 {
        if self.n_samples == 0 {
            0.0
        } else {
            self.nnz() as f64 / self.n_samples as f64
        }
    }

    /// Get reference to sample indices slice.
    #[inline]
    pub fn sample_indices(&self) -> &[u32] {
        &self.sample_indices
    }

    /// Get reference to bin data.
    #[inline]
    pub fn bins(&self) -> &BinData {
        &self.bins
    }

    /// Get reference to raw values slice.
    #[inline]
    pub fn raw_values(&self) -> &[f32] {
        &self.raw_values
    }

    /// Check if empty (no non-zero entries).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.sample_indices.is_empty()
    }

    /// Memory size in bytes.
    #[inline]
    pub fn size_bytes(&self) -> usize {
        self.sample_indices.len() * 4 // u32 indices
            + self.bins.size_bytes()
            + self.raw_values.len() * 4 // f32 values
    }
}

/// Sparse categorical storage: CSC-like, single feature.
///
/// Stores only non-zero samples. Samples not in `sample_indices` have
/// implicit bin=0. No raw values needed (categorical encoding is lossless).
///
/// # Layout
///
/// Stores parallel arrays of (sample_index, bin) for non-zero entries.
/// Sample indices are sorted for efficient binary search.
///
/// # Example
///
/// ```ignore
/// // 10 samples, only 3 non-zero
/// let indices = vec![2, 5, 8];
/// let bins = BinData::from_u8(vec![1, 2, 3]);
/// let storage = SparseCategoricalStorage::new(indices, bins, 10);
///
/// assert_eq!(storage.bin(5), 2);
/// assert_eq!(storage.bin(3), 0); // Not in indices -> implicit zero
/// ```
#[derive(Clone, Debug)]
pub struct SparseCategoricalStorage {
    /// Sample indices of non-zero entries (sorted).
    sample_indices: Box<[u32]>,
    /// Bin values for non-zero entries (parallel to sample_indices).
    bins: BinData,
    /// Total number of samples.
    n_samples: usize,
}

impl SparseCategoricalStorage {
    /// Create new sparse categorical storage.
    ///
    /// # Arguments
    /// * `sample_indices` - Sample indices with non-zero values (must be sorted)
    /// * `bins` - Bin values for non-zero samples
    /// * `n_samples` - Total number of samples
    pub fn new(sample_indices: Vec<u32>, bins: BinData, n_samples: usize) -> Self {
        debug_assert_eq!(
            sample_indices.len(),
            bins.len(),
            "sample_indices and bins must have same length"
        );
        debug_assert!(
            sample_indices.windows(2).all(|w| w[0] < w[1]),
            "sample_indices must be strictly sorted"
        );
        Self {
            sample_indices: sample_indices.into_boxed_slice(),
            bins,
            n_samples,
        }
    }

    /// Get bin value for sample via binary search.
    ///
    /// Returns 0 for samples not in indices (implicit zero category).
    #[inline]
    pub fn bin(&self, sample: usize) -> u32 {
        match self.sample_indices.binary_search(&(sample as u32)) {
            Ok(pos) => self.bins.get(pos),
            Err(_) => 0,
        }
    }

    /// Number of non-zero entries.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.sample_indices.len()
    }

    /// Total number of samples.
    #[inline]
    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    /// Sparsity ratio (nnz / n_samples).
    #[inline]
    pub fn sparsity(&self) -> f64 {
        if self.n_samples == 0 {
            0.0
        } else {
            self.nnz() as f64 / self.n_samples as f64
        }
    }

    /// Get reference to sample indices slice.
    #[inline]
    pub fn sample_indices(&self) -> &[u32] {
        &self.sample_indices
    }

    /// Get reference to bin data.
    #[inline]
    pub fn bins(&self) -> &BinData {
        &self.bins
    }

    /// Check if empty (no non-zero entries).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.sample_indices.is_empty()
    }

    /// Memory size in bytes.
    #[inline]
    pub fn size_bytes(&self) -> usize {
        self.sample_indices.len() * 4 + self.bins.size_bytes()
    }
}

/// EFB bundle storage: multiple sparse features encoded into one column.
///
/// Exclusive Feature Bundling (EFB) encodes multiple sparse features that rarely
/// have non-zero values simultaneously into a single column. This is common for:
/// - One-hot encoded categoricals (mutually exclusive by definition)
/// - Sparse indicator features
///
/// Bundles are **lossless** - no raw values needed because bin → value is
/// recoverable via BinMapper. Linear trees skip bundled features for regression
/// and only use direct numeric features.
///
/// # Encoding
///
/// - Bin 0 = all features have default value (typically 0)
/// - Bin k = bin_offsets[i] + original_bin for active feature i
///
/// # Example
///
/// ```ignore
/// // Bundle of 3 sparse features with 4, 3, 5 bins respectively
/// let storage = BundleStorage::new(
///     encoded_bins,        // [n_samples] encoded values
///     vec![0, 1, 2],       // original feature indices
///     vec![1, 5, 8],       // bin offsets (cumulative: 0, 4, 7 + 1 each for default)
///     vec![4, 3, 5],       // bins per feature
///     vec![0, 0, 0],       // default bins
///     1000,                // n_samples
/// );
///
/// // Decode: encoded bin 6 → feature 1, original bin 1 (since 6 - 5 = 1)
/// assert_eq!(storage.decode(6), Some((1, 1)));
/// ```
#[derive(Clone, Debug)]
pub struct BundleStorage {
    /// Encoded bins: [n_samples], always U16.
    /// Bin 0 = all features have default value.
    /// Bin k = offset[i] + original_bin for active feature i.
    encoded_bins: Box<[u16]>,

    /// Original feature indices in this bundle.
    feature_indices: Box<[u32]>,

    /// Bin offset for each feature in the bundle.
    /// bin_offsets[i] = start offset for feature i in encoded space.
    bin_offsets: Box<[u32]>,

    /// Number of bins per feature.
    feature_n_bins: Box<[u32]>,

    /// Total bins in this bundle.
    total_bins: u32,

    /// Default bin for each feature (bin when value is "zero/default").
    default_bins: Box<[u32]>,

    /// Number of samples.
    n_samples: usize,
}

impl BundleStorage {
    /// Create new bundle storage.
    ///
    /// # Arguments
    /// * `encoded_bins` - Offset-encoded bin values for all samples
    /// * `feature_indices` - Original global feature indices in this bundle
    /// * `bin_offsets` - Cumulative bin offsets for each feature
    /// * `feature_n_bins` - Number of bins for each feature
    /// * `default_bins` - Default bin for each feature
    /// * `n_samples` - Total number of samples
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        encoded_bins: Vec<u16>,
        feature_indices: Vec<u32>,
        bin_offsets: Vec<u32>,
        feature_n_bins: Vec<u32>,
        default_bins: Vec<u32>,
        n_samples: usize,
    ) -> Self {
        let n_features = feature_indices.len();
        debug_assert_eq!(bin_offsets.len(), n_features);
        debug_assert_eq!(feature_n_bins.len(), n_features);
        debug_assert_eq!(default_bins.len(), n_features);
        debug_assert_eq!(encoded_bins.len(), n_samples);

        let total_bins = feature_n_bins.iter().sum::<u32>() + 1; // +1 for default bin 0

        Self {
            encoded_bins: encoded_bins.into_boxed_slice(),
            feature_indices: feature_indices.into_boxed_slice(),
            bin_offsets: bin_offsets.into_boxed_slice(),
            feature_n_bins: feature_n_bins.into_boxed_slice(),
            total_bins,
            default_bins: default_bins.into_boxed_slice(),
            n_samples,
        }
    }

    /// Decode encoded bin to (feature_position_in_bundle, original_bin).
    ///
    /// Returns `None` if encoded_bin is 0 (all features at default).
    ///
    /// For bundles with ≤4 features, uses linear scan for better cache performance.
    /// For larger bundles, uses binary search.
    #[inline]
    pub fn decode(&self, encoded_bin: u16) -> Option<(usize, u32)> {
        if encoded_bin == 0 {
            return None; // All features at default
        }

        let encoded = encoded_bin as u32;

        // Use linear scan for small bundles (better cache performance)
        if self.bin_offsets.len() <= 4 {
            for (i, &offset) in self.bin_offsets.iter().enumerate() {
                let end = if i + 1 < self.bin_offsets.len() {
                    self.bin_offsets[i + 1]
                } else {
                    self.total_bins
                };
                if encoded >= offset && encoded < end {
                    return Some((i, encoded - offset));
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
                    Some((i - 1, encoded - self.bin_offsets[i - 1]))
                }
            }
        }
    }

    /// Get encoded bin at sample index.
    #[inline]
    pub fn get(&self, sample: usize) -> u16 {
        self.encoded_bins[sample]
    }

    /// Get reference to encoded bins slice.
    #[inline]
    pub fn encoded_bins(&self) -> &[u16] {
        &self.encoded_bins
    }

    /// Get feature indices in this bundle.
    #[inline]
    pub fn feature_indices(&self) -> &[u32] {
        &self.feature_indices
    }

    /// Get bin offsets for each feature.
    #[inline]
    pub fn bin_offsets(&self) -> &[u32] {
        &self.bin_offsets
    }

    /// Get number of bins per feature.
    #[inline]
    pub fn feature_n_bins(&self) -> &[u32] {
        &self.feature_n_bins
    }

    /// Get default bins for each feature.
    #[inline]
    pub fn default_bins(&self) -> &[u32] {
        &self.default_bins
    }

    /// Total number of bins in this bundle (including default bin 0).
    #[inline]
    pub fn total_bins(&self) -> u32 {
        self.total_bins
    }

    /// Number of features in this bundle.
    #[inline]
    pub fn n_features(&self) -> usize {
        self.feature_indices.len()
    }

    /// Number of samples.
    #[inline]
    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n_samples == 0
    }

    /// Memory size in bytes.
    #[inline]
    pub fn size_bytes(&self) -> usize {
        self.encoded_bins.len() * 2  // u16 bins
            + self.feature_indices.len() * 4
            + self.bin_offsets.len() * 4
            + self.feature_n_bins.len() * 4
            + self.default_bins.len() * 4
    }
}

/// Unified feature storage enum for all storage types.
///
/// This enum wraps all storage variants and provides a common interface for
/// accessing bin data and optional raw values. Groups are homogeneous - all
/// features in a group share the same storage type.
///
/// Part of the RFC-0018 storage hierarchy:
/// - `BinData` - raw bin values
/// - `NumericStorage` / `CategoricalStorage` / `Sparse*Storage` / `BundleStorage`
/// - `FeatureStorage` - this type (unified wrapper)
///
/// # Example
///
/// ```ignore
/// match group.storage() {
///     FeatureStorage::Numeric(s) => {
///         // Access bins and raw values
///         let bin = s.bin(sample, feature, n_samples);
///         let raw = s.raw(sample, feature, n_samples);
///     }
///     FeatureStorage::Categorical(s) => {
///         // Bins only - lossless encoding
///         let bin = s.bin(sample, feature, n_samples);
///     }
///     FeatureStorage::SparseNumeric(s) => {
///         let (bin, raw) = s.get(sample);
///     }
///     FeatureStorage::SparseCategorical(s) => {
///         let bin = s.bin(sample);
///     }
///     FeatureStorage::Bundle(s) => {
///         // Skip for raw access (use only for histogram building)
///         let encoded = s.get(sample);
///     }
/// }
/// ```
#[derive(Clone, Debug)]
pub enum FeatureStorage {
    /// Dense numeric features: bins + raw values.
    Numeric(NumericStorage),

    /// Dense categorical features: bins only (lossless).
    Categorical(CategoricalStorage),

    /// Sparse numeric features: CSC-like storage + raw values.
    SparseNumeric(SparseNumericStorage),

    /// Sparse categorical features: CSC-like storage (lossless).
    SparseCategorical(SparseCategoricalStorage),

    /// EFB bundle: multiple sparse features encoded into one column.
    /// Linear trees skip bundled features for regression.
    Bundle(BundleStorage),
}

impl FeatureStorage {
    /// Whether this storage has raw values available.
    ///
    /// Returns `true` for Numeric and SparseNumeric storage types.
    /// Returns `false` for Categorical, SparseCategorical, and Bundle.
    #[inline]
    pub fn has_raw_values(&self) -> bool {
        matches!(
            self,
            FeatureStorage::Numeric(_) | FeatureStorage::SparseNumeric(_)
        )
    }

    /// Whether this storage is categorical (no raw values needed - lossless).
    #[inline]
    pub fn is_categorical(&self) -> bool {
        matches!(
            self,
            FeatureStorage::Categorical(_) | FeatureStorage::SparseCategorical(_)
        )
    }

    /// Whether this storage is sparse.
    #[inline]
    pub fn is_sparse(&self) -> bool {
        matches!(
            self,
            FeatureStorage::SparseNumeric(_) | FeatureStorage::SparseCategorical(_)
        )
    }

    /// Whether this storage is a bundle.
    #[inline]
    pub fn is_bundle(&self) -> bool {
        matches!(self, FeatureStorage::Bundle(_))
    }

    /// Whether this storage is dense (not sparse or bundle).
    #[inline]
    pub fn is_dense(&self) -> bool {
        matches!(
            self,
            FeatureStorage::Numeric(_) | FeatureStorage::Categorical(_)
        )
    }

    /// Memory size in bytes.
    pub fn size_bytes(&self) -> usize {
        match self {
            FeatureStorage::Numeric(s) => s.size_bytes(),
            FeatureStorage::Categorical(s) => s.size_bytes(),
            FeatureStorage::SparseNumeric(s) => s.size_bytes(),
            FeatureStorage::SparseCategorical(s) => s.size_bytes(),
            FeatureStorage::Bundle(s) => s.size_bytes(),
        }
    }
}

// =============================================================================
// LEGACY STORAGE TYPES (to be removed in Epic 6)
// =============================================================================

/// Bin storage for a feature group.
///
/// Match on this enum to get typed access to the underlying data.
/// This design forces exhaustive handling of all storage types.
///
/// # Example
///
/// ```ignore
/// match group.storage() {
///     BinStorage::DenseU8(data) => {
///         // data: &[u8]
///         for &bin in data { ... }
///     }
///     BinStorage::DenseU16(data) => { ... }
///     BinStorage::SparseU8 { row_indices, bin_values, .. } => { ... }
///     BinStorage::SparseU16 { row_indices, bin_values, .. } => { ... }
/// }
/// ```
#[derive(Clone, Debug)]
pub enum BinStorage {
    /// Dense 8-bit bins.
    DenseU8(Box<[u8]>),
    /// Dense 16-bit bins.
    DenseU16(Box<[u16]>),
    /// Sparse 8-bit bins (CSR-like: row_indices + bin_values).
    /// Only non-zero entries stored.
    SparseU8 {
        /// Row indices of non-zero entries (sorted).
        row_indices: Box<[u32]>,
        /// Bin values for non-zero entries.
        bin_values: Box<[u8]>,
        /// Total number of rows.
        n_rows: usize,
    },
    /// Sparse 16-bit bins.
    SparseU16 {
        row_indices: Box<[u32]>,
        bin_values: Box<[u16]>,
        n_rows: usize,
    },
}

impl BinStorage {
    /// Create from u8 vector.
    pub fn from_u8(data: Vec<u8>) -> Self {
        Self::DenseU8(data.into_boxed_slice())
    }

    /// Create from u16 vector.
    pub fn from_u16(data: Vec<u16>) -> Self {
        Self::DenseU16(data.into_boxed_slice())
    }

    /// Create sparse u8 storage from row indices and bin values.
    ///
    /// # Arguments
    /// * `row_indices` - Sorted row indices of non-zero entries
    /// * `bin_values` - Bin values for non-zero entries
    /// * `n_rows` - Total number of rows in the dataset
    pub fn from_sparse_u8(row_indices: Vec<u32>, bin_values: Vec<u8>, n_rows: usize) -> Self {
        debug_assert_eq!(row_indices.len(), bin_values.len());
        debug_assert!(
            row_indices.windows(2).all(|w| w[0] <= w[1]),
            "row_indices must be sorted"
        );
        Self::SparseU8 {
            row_indices: row_indices.into_boxed_slice(),
            bin_values: bin_values.into_boxed_slice(),
            n_rows,
        }
    }

    /// Create sparse u16 storage from row indices and bin values.
    ///
    /// # Arguments
    /// * `row_indices` - Sorted row indices of non-zero entries
    /// * `bin_values` - Bin values for non-zero entries
    /// * `n_rows` - Total number of rows in the dataset
    pub fn from_sparse_u16(row_indices: Vec<u32>, bin_values: Vec<u16>, n_rows: usize) -> Self {
        debug_assert_eq!(row_indices.len(), bin_values.len());
        Self::SparseU16 {
            row_indices: row_indices.into_boxed_slice(),
            bin_values: bin_values.into_boxed_slice(),
            n_rows,
        }
    }

    /// Get the bin type.
    #[inline]
    pub fn bin_type(&self) -> BinType {
        match self {
            Self::DenseU8(_) | Self::SparseU8 { .. } => BinType::U8,
            Self::DenseU16(_) | Self::SparseU16 { .. } => BinType::U16,
        }
    }

    /// Check if this is sparse storage.
    #[inline]
    pub fn is_sparse(&self) -> bool {
        matches!(self, Self::SparseU8 { .. } | Self::SparseU16 { .. })
    }

    /// Check if this is dense storage.
    #[inline]
    pub fn is_dense(&self) -> bool {
        matches!(self, Self::DenseU8(_) | Self::DenseU16(_))
    }

    /// Total number of bin values stored.
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Self::DenseU8(data) => data.len(),
            Self::DenseU16(data) => data.len(),
            Self::SparseU8 { bin_values, .. } => bin_values.len(),
            Self::SparseU16 { bin_values, .. } => bin_values.len(),
        }
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of non-zero entries (for sparse storage).
    #[inline]
    pub fn nnz(&self) -> Option<usize> {
        match self {
            Self::SparseU8 { bin_values, .. } => Some(bin_values.len()),
            Self::SparseU16 { bin_values, .. } => Some(bin_values.len()),
            _ => None,
        }
    }

    /// Memory size in bytes.
    #[inline]
    pub fn size_bytes(&self) -> usize {
        match self {
            Self::DenseU8(data) => data.len(),
            Self::DenseU16(data) => data.len() * 2,
            Self::SparseU8 {
                row_indices,
                bin_values,
                ..
            } => row_indices.len() * 4 + bin_values.len(),
            Self::SparseU16 {
                row_indices,
                bin_values,
                ..
            } => row_indices.len() * 4 + bin_values.len() * 2,
        }
    }
}

impl Default for BinStorage {
    fn default() -> Self {
        Self::DenseU8(Box::new([]))
    }
}

/// Zero-cost slice view into a single feature's bin data.
///
/// This is the unified access type for feature bins - use this for histogram
/// building, iteration, and any bulk bin access. Match on the variants to get
/// typed access for efficient loops.
///
/// For column-major storage or sparse features, the slice is contiguous.
/// For row-major storage, the slice spans all features with stride access.
///
/// # Example
///
/// ```ignore
/// for f in 0..dataset.n_features() {
///     let slice = dataset.feature_view(f);
///     match slice {
///         FeatureView::U8 { bins, stride } => {
///             // If stride == 1, bins is contiguous for this feature
///             // If stride > 1 (row-major), bins[i * stride] is row i's bin
///             for row in 0..n_rows {
///                 let bin = bins[row * stride];
///             }
///         }
///         FeatureView::U16 { bins, stride } => { ... }
///         FeatureView::SparseU8 { row_indices, bin_values } => {
///             // Only non-zero entries stored
///             for (i, &row) in row_indices.iter().enumerate() {
///                 let bin = bin_values[i];
///             }
///         }
///         FeatureView::SparseU16 { row_indices, bin_values } => { ... }
///     }
/// }
/// ```
#[derive(Clone, Copy, Debug)]
pub enum FeatureView<'a> {
    /// Dense u8 bins with stride.
    /// - Column-major: bins is contiguous feature data, stride=1
    /// - Row-major: bins is full group data, stride=n_features
    U8 { bins: &'a [u8], stride: usize },
    /// Dense u16 bins with stride.
    U16 { bins: &'a [u16], stride: usize },
    /// Sparse u8 bins (always contiguous, stride=1 implicit).
    SparseU8 {
        row_indices: &'a [u32],
        bin_values: &'a [u8],
    },
    /// Sparse u16 bins.
    SparseU16 {
        row_indices: &'a [u32],
        bin_values: &'a [u16],
    },
}

impl<'a> FeatureView<'a> {
    /// Check if sparse.
    #[inline]
    pub fn is_sparse(&self) -> bool {
        matches!(
            self,
            FeatureView::SparseU8 { .. } | FeatureView::SparseU16 { .. }
        )
    }

    /// Check if dense.
    #[inline]
    pub fn is_dense(&self) -> bool {
        matches!(self, FeatureView::U8 { .. } | FeatureView::U16 { .. })
    }

    /// Check if contiguous (stride == 1 for dense, always true for sparse).
    ///
    /// Contiguous slices are optimal for histogram kernels.
    #[inline]
    pub fn is_contiguous(&self) -> bool {
        match self {
            FeatureView::U8 { stride, .. } | FeatureView::U16 { stride, .. } => *stride == 1,
            FeatureView::SparseU8 { .. } | FeatureView::SparseU16 { .. } => true,
        }
    }

    /// Get the stride (1 for contiguous/sparse).
    #[inline]
    pub fn stride(&self) -> usize {
        match self {
            FeatureView::U8 { stride, .. } | FeatureView::U16 { stride, .. } => *stride,
            FeatureView::SparseU8 { .. } | FeatureView::SparseU16 { .. } => 1,
        }
    }

    /// Get bin value at row index for dense slices.
    ///
    /// Returns `None` for sparse storage.
    /// For sparse features, use iteration over row_indices instead.
    #[inline]
    pub fn get_bin(&self, row: usize) -> Option<u32> {
        match self {
            FeatureView::U8 { bins, stride: 1 } => Some(bins[row] as u32),
            FeatureView::U16 { bins, stride: 1 } => Some(bins[row] as u32),
            FeatureView::U8 { bins, stride } => Some(bins[row * stride] as u32),
            FeatureView::U16 { bins, stride } => Some(bins[row * stride] as u32),
            _ => None,
        }
    }

    /// Get bin value at row index without bounds checking (dense only).
    ///
    /// Returns 0 for sparse slices (should not be called for sparse).
    /// This is designed for hot loops where the caller has already validated
    /// the slice type and row bounds.
    ///
    /// # Safety
    /// Caller must ensure:
    /// - `row * stride` is within bounds for strided slices
    /// - `row` is within bounds for contiguous slices
    /// - Not called on sparse slices
    #[inline(always)]
    pub fn get_bin_unchecked(&self, row: usize) -> u32 {
        match self {
            FeatureView::U8 { bins, stride } => bins[row * stride] as u32,
            FeatureView::U16 { bins, stride } => bins[row * stride] as u32,
            FeatureView::SparseU8 { .. } | FeatureView::SparseU16 { .. } => 0,
        }
    }
}

impl<'a> From<&'a BinStorage> for FeatureView<'a> {
    /// Convert from BinStorage to FeatureView.
    ///
    /// This creates a contiguous (stride=1) slice from the storage.
    /// For multi-feature groups with row-major layout, use
    /// `BinnedDataset::feature_view()` instead to get proper stride info.
    fn from(storage: &'a BinStorage) -> Self {
        match storage {
            BinStorage::DenseU8(data) => FeatureView::U8 {
                bins: data,
                stride: 1,
            },
            BinStorage::DenseU16(data) => FeatureView::U16 {
                bins: data,
                stride: 1,
            },
            BinStorage::SparseU8 {
                row_indices,
                bin_values,
                ..
            } => FeatureView::SparseU8 {
                row_indices,
                bin_values,
            },
            BinStorage::SparseU16 {
                row_indices,
                bin_values,
                ..
            } => FeatureView::SparseU16 {
                row_indices,
                bin_values,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bin_type_max_bins() {
        assert_eq!(BinType::U8.max_bins(), 256);
        assert_eq!(BinType::U16.max_bins(), 65536);
    }

    #[test]
    fn test_bin_type_for_max_bins() {
        assert_eq!(BinType::for_max_bins(100), Some(BinType::U8));
        assert_eq!(BinType::for_max_bins(256), Some(BinType::U8));
        assert_eq!(BinType::for_max_bins(257), Some(BinType::U16));
        assert_eq!(BinType::for_max_bins(65536), Some(BinType::U16));
        assert_eq!(BinType::for_max_bins(65537), None);
    }

    #[test]
    fn test_bin_storage_u8() {
        let storage = BinStorage::from_u8(vec![0, 1, 2, 255]);
        assert_eq!(storage.bin_type(), BinType::U8);
        assert_eq!(storage.len(), 4);
        assert!(storage.is_dense());
        assert!(!storage.is_sparse());

        // Access via match
        match &storage {
            BinStorage::DenseU8(data) => {
                assert_eq!(data[0], 0);
                assert_eq!(data[3], 255);
            }
            _ => panic!("expected DenseU8"),
        }
    }

    #[test]
    fn test_bin_storage_u16() {
        let storage = BinStorage::from_u16(vec![0, 256, 1000, 65535]);
        assert_eq!(storage.bin_type(), BinType::U16);
        assert_eq!(storage.len(), 4);
        assert!(storage.is_dense());

        match &storage {
            BinStorage::DenseU16(data) => {
                assert_eq!(data[1], 256);
                assert_eq!(data[3], 65535);
            }
            _ => panic!("expected DenseU16"),
        }
    }

    #[test]
    fn test_bin_storage_size_bytes() {
        let u8_storage = BinStorage::from_u8(vec![0; 100]);
        let u16_storage = BinStorage::from_u16(vec![0; 100]);

        assert_eq!(u8_storage.size_bytes(), 100);
        assert_eq!(u16_storage.size_bytes(), 200);
    }

    #[test]
    fn test_sparse_storage_u8() {
        let storage = BinStorage::from_sparse_u8(vec![0, 2, 5], vec![1u8, 3, 2], 10);

        assert!(storage.is_sparse());
        assert!(!storage.is_dense());
        assert_eq!(storage.bin_type(), BinType::U8);
        assert_eq!(storage.nnz(), Some(3));
        assert_eq!(storage.len(), 3);
        assert_eq!(storage.size_bytes(), 15);

        match &storage {
            BinStorage::SparseU8 {
                row_indices,
                bin_values,
                n_rows,
            } => {
                assert_eq!(row_indices.as_ref(), &[0, 2, 5]);
                assert_eq!(bin_values.as_ref(), &[1, 3, 2]);
                assert_eq!(*n_rows, 10);
            }
            _ => panic!("expected SparseU8"),
        }
    }

    #[test]
    fn test_sparse_storage_u16() {
        let storage = BinStorage::from_sparse_u16(vec![1, 3], vec![256u16, 1000], 5);

        assert!(storage.is_sparse());
        assert_eq!(storage.bin_type(), BinType::U16);
        assert_eq!(storage.nnz(), Some(2));

        match &storage {
            BinStorage::SparseU16 {
                row_indices,
                bin_values,
                ..
            } => {
                assert_eq!(row_indices.as_ref(), &[1, 3]);
                assert_eq!(bin_values.as_ref(), &[256, 1000]);
            }
            _ => panic!("expected SparseU16"),
        }
    }

    #[test]
    fn test_bin_slice_from_storage() {
        let dense = BinStorage::from_u8(vec![1, 2, 3]);
        let slice: FeatureView = (&dense).into();
        assert!(slice.is_dense());

        let sparse = BinStorage::from_sparse_u8(vec![0, 2], vec![1, 3], 4);
        let slice2: FeatureView = (&sparse).into();
        assert!(slice2.is_sparse());
    }

    // =========================================================================
    // BinData tests (RFC-0018)
    // =========================================================================

    #[test]
    fn test_bin_data_u8() {
        let bins = BinData::from_u8(vec![0, 1, 2, 255]);
        assert!(bins.is_u8());
        assert!(!bins.is_u16());
        assert_eq!(bins.len(), 4);
        assert!(!bins.is_empty());
        assert_eq!(bins.size_bytes(), 4);

        // Access via get
        assert_eq!(bins.get(0), 0);
        assert_eq!(bins.get(1), 1);
        assert_eq!(bins.get(2), 2);
        assert_eq!(bins.get(3), 255);

        // Slice access
        assert_eq!(bins.as_u8_slice(), Some(&[0u8, 1, 2, 255][..]));
        assert_eq!(bins.as_u16_slice(), None);
    }

    #[test]
    fn test_bin_data_u16() {
        let bins = BinData::from_u16(vec![0, 256, 1000, 65535]);
        assert!(!bins.is_u8());
        assert!(bins.is_u16());
        assert_eq!(bins.len(), 4);
        assert_eq!(bins.size_bytes(), 8);

        // Access via get
        assert_eq!(bins.get(0), 0);
        assert_eq!(bins.get(1), 256);
        assert_eq!(bins.get(2), 1000);
        assert_eq!(bins.get(3), 65535);

        // Slice access
        assert_eq!(bins.as_u8_slice(), None);
        assert_eq!(bins.as_u16_slice(), Some(&[0u16, 256, 1000, 65535][..]));
    }

    #[test]
    fn test_bin_data_empty() {
        let empty_u8 = BinData::from_u8(vec![]);
        assert!(empty_u8.is_empty());
        assert_eq!(empty_u8.len(), 0);

        let default = BinData::default();
        assert!(default.is_empty());
        assert!(default.is_u8());
    }

    #[test]
    fn test_bin_data_get_unchecked() {
        let bins = BinData::from_u8(vec![10, 20, 30]);
        unsafe {
            assert_eq!(bins.get_unchecked(0), 10);
            assert_eq!(bins.get_unchecked(1), 20);
            assert_eq!(bins.get_unchecked(2), 30);
        }
    }

    // =========================================================================
    // NumericStorage tests (RFC-0018)
    // =========================================================================

    #[test]
    fn test_numeric_storage_basic() {
        // 2 features, 3 samples, column-major layout
        // Feature 0: samples [0,1,2] -> bins [0,1,2], raw [1.0, 2.0, 3.0]
        // Feature 1: samples [0,1,2] -> bins [3,4,5], raw [10.0, 20.0, 30.0]
        let bins = BinData::from_u8(vec![0, 1, 2, 3, 4, 5]);
        let raw = vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0];
        let storage = NumericStorage::new(bins, raw);

        assert_eq!(storage.len(), 6);
        assert!(!storage.is_empty());

        let n_samples = 3;

        // Feature 0 access
        assert_eq!(storage.bin(0, 0, n_samples), 0);
        assert_eq!(storage.bin(1, 0, n_samples), 1);
        assert_eq!(storage.bin(2, 0, n_samples), 2);
        assert_eq!(storage.raw(0, 0, n_samples), 1.0);
        assert_eq!(storage.raw(1, 0, n_samples), 2.0);
        assert_eq!(storage.raw(2, 0, n_samples), 3.0);

        // Feature 1 access
        assert_eq!(storage.bin(0, 1, n_samples), 3);
        assert_eq!(storage.bin(1, 1, n_samples), 4);
        assert_eq!(storage.bin(2, 1, n_samples), 5);
        assert_eq!(storage.raw(0, 1, n_samples), 10.0);
        assert_eq!(storage.raw(1, 1, n_samples), 20.0);
        assert_eq!(storage.raw(2, 1, n_samples), 30.0);
    }

    #[test]
    fn test_numeric_storage_raw_slice() {
        // 2 features, 3 samples
        let bins = BinData::from_u8(vec![0, 1, 2, 3, 4, 5]);
        let raw = vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0];
        let storage = NumericStorage::new(bins, raw);

        let n_samples = 3;

        // raw_slice returns contiguous slice for feature
        assert_eq!(storage.raw_slice(0, n_samples), &[1.0, 2.0, 3.0]);
        assert_eq!(storage.raw_slice(1, n_samples), &[10.0, 20.0, 30.0]);
    }

    #[test]
    fn test_numeric_storage_u16_bins() {
        // Test with u16 bins (>256 bin values)
        let bins = BinData::from_u16(vec![0, 256, 1000, 2000, 3000, 4000]);
        let raw = vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0];
        let storage = NumericStorage::new(bins, raw);

        assert_eq!(storage.len(), 6);

        let n_samples = 3;
        assert_eq!(storage.bin(1, 0, n_samples), 256);
        assert_eq!(storage.bin(0, 1, n_samples), 2000);
    }

    #[test]
    fn test_numeric_storage_size_bytes() {
        // u8 bins: 6 bytes, raw: 6 * 4 = 24 bytes, total = 30
        let bins = BinData::from_u8(vec![0; 6]);
        let raw = vec![0.0; 6];
        let storage = NumericStorage::new(bins, raw);
        assert_eq!(storage.size_bytes(), 6 + 24);

        // u16 bins: 12 bytes, raw: 24 bytes, total = 36
        let bins16 = BinData::from_u16(vec![0; 6]);
        let raw16 = vec![0.0; 6];
        let storage16 = NumericStorage::new(bins16, raw16);
        assert_eq!(storage16.size_bytes(), 12 + 24);
    }

    #[test]
    fn test_numeric_storage_nan_handling() {
        // Raw values can contain NaN for missing data
        let bins = BinData::from_u8(vec![0, 1, 255]); // 255 = missing bin
        let raw = vec![1.0, 2.0, f32::NAN];
        let storage = NumericStorage::new(bins, raw);

        assert!(storage.raw(2, 0, 3).is_nan());
    }

    // =========================================================================
    // CategoricalStorage tests (RFC-0018)
    // =========================================================================

    #[test]
    fn test_categorical_storage_basic() {
        // 2 features, 3 samples, column-major layout
        // Categories are lossless: bin = category ID
        let bins = BinData::from_u8(vec![0, 1, 0, 2, 0, 1]);
        let storage = CategoricalStorage::new(bins);

        assert_eq!(storage.len(), 6);
        assert!(!storage.is_empty());

        let n_samples = 3;

        // Feature 0: categories [0, 1, 0]
        assert_eq!(storage.bin(0, 0, n_samples), 0);
        assert_eq!(storage.bin(1, 0, n_samples), 1);
        assert_eq!(storage.bin(2, 0, n_samples), 0);

        // Feature 1: categories [2, 0, 1]
        assert_eq!(storage.bin(0, 1, n_samples), 2);
        assert_eq!(storage.bin(1, 1, n_samples), 0);
        assert_eq!(storage.bin(2, 1, n_samples), 1);
    }

    #[test]
    fn test_categorical_storage_u16() {
        // High-cardinality categorical with u16
        let bins = BinData::from_u16(vec![0, 300, 500, 1000, 2000, 3000]);
        let storage = CategoricalStorage::new(bins);

        assert_eq!(storage.len(), 6);
        let n_samples = 3;
        assert_eq!(storage.bin(1, 0, n_samples), 300);
        assert_eq!(storage.bin(0, 1, n_samples), 1000);
    }

    #[test]
    fn test_categorical_storage_size_bytes() {
        // No raw values, only bins
        let bins = BinData::from_u8(vec![0; 6]);
        let storage = CategoricalStorage::new(bins);
        assert_eq!(storage.size_bytes(), 6); // Only bin bytes

        let bins16 = BinData::from_u16(vec![0; 6]);
        let storage16 = CategoricalStorage::new(bins16);
        assert_eq!(storage16.size_bytes(), 12); // 6 * 2 bytes
    }

    // =========================================================================
    // SparseNumericStorage tests (RFC-0018)
    // =========================================================================

    #[test]
    fn test_sparse_numeric_storage_basic() {
        // 10 samples, only 3 non-zero at indices [2, 5, 8]
        let indices = vec![2, 5, 8];
        let bins = BinData::from_u8(vec![1, 2, 3]);
        let raw = vec![1.5, 2.5, 3.5];
        let storage = SparseNumericStorage::new(indices, bins, raw, 10);

        assert_eq!(storage.nnz(), 3);
        assert_eq!(storage.n_samples(), 10);
        assert!(!storage.is_empty());
        assert!((storage.sparsity() - 0.3).abs() < 1e-6);

        // Access explicit values
        assert_eq!(storage.get(2), (1, 1.5));
        assert_eq!(storage.get(5), (2, 2.5));
        assert_eq!(storage.get(8), (3, 3.5));

        // Access implicit zeros
        assert_eq!(storage.get(0), (0, 0.0));
        assert_eq!(storage.get(3), (0, 0.0));
        assert_eq!(storage.get(9), (0, 0.0));

        // Individual accessors
        assert_eq!(storage.bin(5), 2);
        assert_eq!(storage.raw(5), 2.5);
        assert_eq!(storage.bin(0), 0);
        assert_eq!(storage.raw(0), 0.0);
    }

    #[test]
    fn test_sparse_numeric_storage_u16() {
        let indices = vec![0, 100];
        let bins = BinData::from_u16(vec![256, 1000]);
        let raw = vec![10.0, 20.0];
        let storage = SparseNumericStorage::new(indices, bins, raw, 200);

        assert_eq!(storage.get(0), (256, 10.0));
        assert_eq!(storage.get(100), (1000, 20.0));
        assert_eq!(storage.get(50), (0, 0.0));
    }

    #[test]
    fn test_sparse_numeric_storage_size_bytes() {
        // 3 entries: 3 * 4 (indices) + 3 (u8 bins) + 3 * 4 (raw) = 27
        let indices = vec![1, 2, 3];
        let bins = BinData::from_u8(vec![1, 2, 3]);
        let raw = vec![1.0, 2.0, 3.0];
        let storage = SparseNumericStorage::new(indices, bins, raw, 10);
        assert_eq!(storage.size_bytes(), 12 + 3 + 12);

        // With u16 bins: 12 + 6 + 12 = 30
        let indices16 = vec![1, 2, 3];
        let bins16 = BinData::from_u16(vec![1, 2, 3]);
        let raw16 = vec![1.0, 2.0, 3.0];
        let storage16 = SparseNumericStorage::new(indices16, bins16, raw16, 10);
        assert_eq!(storage16.size_bytes(), 12 + 6 + 12);
    }

    #[test]
    fn test_sparse_numeric_storage_empty() {
        let storage =
            SparseNumericStorage::new(vec![], BinData::from_u8(vec![]), vec![], 100);
        assert!(storage.is_empty());
        assert_eq!(storage.nnz(), 0);
        assert_eq!(storage.n_samples(), 100);
        assert_eq!(storage.sparsity(), 0.0);
        assert_eq!(storage.get(0), (0, 0.0));
    }

    // =========================================================================
    // SparseCategoricalStorage tests (RFC-0018)
    // =========================================================================

    #[test]
    fn test_sparse_categorical_storage_basic() {
        // 10 samples, only 4 non-zero at indices [1, 3, 5, 7]
        let indices = vec![1, 3, 5, 7];
        let bins = BinData::from_u8(vec![2, 4, 6, 8]);
        let storage = SparseCategoricalStorage::new(indices, bins, 10);

        assert_eq!(storage.nnz(), 4);
        assert_eq!(storage.n_samples(), 10);
        assert!(!storage.is_empty());

        // Access explicit values
        assert_eq!(storage.bin(1), 2);
        assert_eq!(storage.bin(3), 4);
        assert_eq!(storage.bin(5), 6);
        assert_eq!(storage.bin(7), 8);

        // Access implicit zeros
        assert_eq!(storage.bin(0), 0);
        assert_eq!(storage.bin(2), 0);
        assert_eq!(storage.bin(9), 0);
    }

    #[test]
    fn test_sparse_categorical_storage_u16() {
        let indices = vec![0, 50];
        let bins = BinData::from_u16(vec![300, 500]);
        let storage = SparseCategoricalStorage::new(indices, bins, 100);

        assert_eq!(storage.bin(0), 300);
        assert_eq!(storage.bin(50), 500);
        assert_eq!(storage.bin(25), 0);
    }

    #[test]
    fn test_sparse_categorical_storage_size_bytes() {
        // 4 entries: 4 * 4 (indices) + 4 (u8 bins) = 20
        let indices = vec![1, 2, 3, 4];
        let bins = BinData::from_u8(vec![1, 2, 3, 4]);
        let storage = SparseCategoricalStorage::new(indices, bins, 10);
        assert_eq!(storage.size_bytes(), 16 + 4);
    }

    #[test]
    fn test_sparse_categorical_storage_empty() {
        let storage = SparseCategoricalStorage::new(vec![], BinData::from_u8(vec![]), 50);
        assert!(storage.is_empty());
        assert_eq!(storage.nnz(), 0);
        assert_eq!(storage.bin(0), 0);
    }

    // =========================================================================
    // BundleStorage tests (RFC-0018)
    // =========================================================================

    #[test]
    fn test_bundle_storage_basic() {
        // Bundle of 3 features with 4, 3, 5 bins respectively
        // Bin offsets: [1, 5, 8] (1 for default, then cumulative)
        // Total bins: 1 + 4 + 3 + 5 = 13
        //
        // Encoded bin 0 = all default
        // Encoded bins 1-4 = feature 0, bins 0-3
        // Encoded bins 5-7 = feature 1, bins 0-2
        // Encoded bins 8-12 = feature 2, bins 0-4
        let encoded_bins = vec![0, 1, 5, 8, 3, 7, 12];
        let storage = BundleStorage::new(
            encoded_bins,
            vec![10, 20, 30],     // original feature indices
            vec![1, 5, 8],       // bin offsets
            vec![4, 3, 5],       // bins per feature
            vec![0, 0, 0],       // default bins
            7,                   // n_samples
        );

        assert_eq!(storage.n_features(), 3);
        assert_eq!(storage.n_samples(), 7);
        assert_eq!(storage.total_bins(), 13); // 1 + 4 + 3 + 5

        // Decode tests
        assert_eq!(storage.decode(0), None); // Default - all features at default
        assert_eq!(storage.decode(1), Some((0, 0))); // Feature 0, bin 0
        assert_eq!(storage.decode(3), Some((0, 2))); // Feature 0, bin 2
        assert_eq!(storage.decode(5), Some((1, 0))); // Feature 1, bin 0
        assert_eq!(storage.decode(7), Some((1, 2))); // Feature 1, bin 2
        assert_eq!(storage.decode(8), Some((2, 0))); // Feature 2, bin 0
        assert_eq!(storage.decode(12), Some((2, 4))); // Feature 2, bin 4

        // Get encoded value at sample
        assert_eq!(storage.get(0), 0);
        assert_eq!(storage.get(1), 1);
        assert_eq!(storage.get(2), 5);
    }

    #[test]
    fn test_bundle_storage_decode_linear_scan() {
        // Small bundle (≤4 features) - uses linear scan
        let storage = BundleStorage::new(
            vec![0, 1, 3],
            vec![100, 200],       // 2 features
            vec![1, 3],          // offsets
            vec![2, 4],          // bins per feature
            vec![0, 0],
            3,
        );

        assert_eq!(storage.decode(0), None);
        assert_eq!(storage.decode(1), Some((0, 0)));
        assert_eq!(storage.decode(2), Some((0, 1)));
        assert_eq!(storage.decode(3), Some((1, 0)));
        assert_eq!(storage.decode(6), Some((1, 3)));
    }

    #[test]
    fn test_bundle_storage_decode_binary_search() {
        // Large bundle (>4 features) - uses binary search
        let storage = BundleStorage::new(
            vec![0, 1, 5, 10],
            vec![0, 1, 2, 3, 4],  // 5 features
            vec![1, 3, 6, 10, 15], // offsets
            vec![2, 3, 4, 5, 6],   // bins per feature
            vec![0, 0, 0, 0, 0],
            4,
        );

        assert_eq!(storage.decode(0), None);
        assert_eq!(storage.decode(1), Some((0, 0)));
        assert_eq!(storage.decode(2), Some((0, 1)));
        assert_eq!(storage.decode(3), Some((1, 0)));
        assert_eq!(storage.decode(10), Some((3, 0)));
        assert_eq!(storage.decode(15), Some((4, 0)));
        assert_eq!(storage.decode(20), Some((4, 5)));
    }

    #[test]
    fn test_bundle_storage_accessors() {
        let storage = BundleStorage::new(
            vec![0, 1, 2],
            vec![5, 10],
            vec![1, 4],
            vec![3, 5],
            vec![0, 0],
            3,
        );

        assert_eq!(storage.feature_indices(), &[5, 10]);
        assert_eq!(storage.bin_offsets(), &[1, 4]);
        assert_eq!(storage.feature_n_bins(), &[3, 5]);
        assert_eq!(storage.default_bins(), &[0, 0]);
        assert!(!storage.is_empty());
    }

    #[test]
    fn test_bundle_storage_size_bytes() {
        // 3 samples, 2 features
        // encoded_bins: 3 * 2 = 6 bytes
        // feature_indices: 2 * 4 = 8 bytes
        // bin_offsets: 2 * 4 = 8 bytes
        // feature_n_bins: 2 * 4 = 8 bytes
        // default_bins: 2 * 4 = 8 bytes
        // Total: 6 + 8 + 8 + 8 + 8 = 38 bytes
        let storage = BundleStorage::new(
            vec![0, 1, 2],
            vec![0, 1],
            vec![1, 5],
            vec![4, 3],
            vec![0, 0],
            3,
        );
        assert_eq!(storage.size_bytes(), 6 + 8 + 8 + 8 + 8);
    }

    // =========================================================================
    // FeatureStorage tests (RFC-0018)
    // =========================================================================

    #[test]
    fn test_feature_storage_numeric() {
        let bins = BinData::from_u8(vec![0, 1, 2]);
        let raw = vec![1.0, 2.0, 3.0];
        let storage = FeatureStorage::Numeric(NumericStorage::new(bins, raw));

        assert!(storage.has_raw_values());
        assert!(!storage.is_categorical());
        assert!(storage.is_dense());
        assert!(!storage.is_sparse());
        assert!(!storage.is_bundle());
    }

    #[test]
    fn test_feature_storage_categorical() {
        let bins = BinData::from_u8(vec![0, 1, 2]);
        let storage = FeatureStorage::Categorical(CategoricalStorage::new(bins));

        assert!(!storage.has_raw_values());
        assert!(storage.is_categorical());
        assert!(storage.is_dense());
        assert!(!storage.is_sparse());
        assert!(!storage.is_bundle());
    }

    #[test]
    fn test_feature_storage_sparse_numeric() {
        let storage = FeatureStorage::SparseNumeric(SparseNumericStorage::new(
            vec![0, 2],
            BinData::from_u8(vec![1, 2]),
            vec![1.0, 2.0],
            10,
        ));

        assert!(storage.has_raw_values());
        assert!(!storage.is_categorical());
        assert!(!storage.is_dense());
        assert!(storage.is_sparse());
        assert!(!storage.is_bundle());
    }

    #[test]
    fn test_feature_storage_sparse_categorical() {
        let storage = FeatureStorage::SparseCategorical(SparseCategoricalStorage::new(
            vec![0, 2],
            BinData::from_u8(vec![1, 2]),
            10,
        ));

        assert!(!storage.has_raw_values());
        assert!(storage.is_categorical());
        assert!(!storage.is_dense());
        assert!(storage.is_sparse());
        assert!(!storage.is_bundle());
    }

    #[test]
    fn test_feature_storage_bundle() {
        let storage = FeatureStorage::Bundle(BundleStorage::new(
            vec![0, 1, 2],
            vec![0, 1],
            vec![1, 5],
            vec![4, 3],
            vec![0, 0],
            3,
        ));

        assert!(!storage.has_raw_values());
        assert!(!storage.is_categorical());
        assert!(!storage.is_dense());
        assert!(!storage.is_sparse());
        assert!(storage.is_bundle());
    }

    #[test]
    fn test_feature_storage_size_bytes() {
        // Test that size_bytes delegates correctly
        let bins = BinData::from_u8(vec![0, 1, 2]);
        let raw = vec![1.0, 2.0, 3.0];
        let numeric = NumericStorage::new(bins.clone(), raw.clone());
        let expected = numeric.size_bytes();

        let storage = FeatureStorage::Numeric(NumericStorage::new(bins, raw));
        assert_eq!(storage.size_bytes(), expected);
    }
}

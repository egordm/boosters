//! Feature storage types for v2 BinnedDataset.
//!
//! These types encode whether raw feature values are available at the type level.

use super::BinData;

/// Dense storage for numeric features with raw values.
///
/// Stores both binned representation (for histogram building) and raw float values
/// (for linear trees, gblinear, and prediction).
///
/// # Layout
///
/// Both bins and raw_values are column-major: for a group of N features with M samples,
/// feature `f`'s values are at indices `[f*M, (f+1)*M)`.
///
/// # Example
///
/// ```ignore
/// let storage = NumericStorage {
///     bins: BinData::from_u8(vec![0, 1, 2, 0, 1, 2]), // 2 features × 3 samples
///     raw_values: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0].into_boxed_slice(),
/// };
/// assert_eq!(storage.bin(0, 3), 0); // feature 0, sample 0
/// assert_eq!(storage.raw(0, 3), 1.0); // feature 0, sample 0
/// ```
#[derive(Clone, Debug)]
pub struct NumericStorage {
    /// Binned values for histogram building.
    pub bins: BinData,
    /// Raw float values for prediction and linear models.
    pub raw_values: Box<[f32]>,
}

impl NumericStorage {
    /// Create new numeric storage.
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

    /// Get bin value for a specific sample.
    ///
    /// # Arguments
    /// * `offset` - Offset into the storage (feature_in_group * n_samples + sample)
    #[inline]
    pub fn bin(&self, offset: usize) -> u32 {
        self.bins.get(offset)
    }

    /// Get raw value for a specific sample.
    #[inline]
    pub fn raw(&self, offset: usize) -> f32 {
        self.raw_values[offset]
    }

    /// Get raw value slice for a feature.
    ///
    /// # Arguments
    /// * `feature_in_group` - Feature index within this group
    /// * `n_samples` - Number of samples in the dataset
    #[inline]
    pub fn raw_slice(&self, feature_in_group: usize, n_samples: usize) -> &[f32] {
        let start = feature_in_group * n_samples;
        let end = start + n_samples;
        &self.raw_values[start..end]
    }

    /// Total number of bin/raw values stored.
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
        self.bins.size_bytes() + self.raw_values.len() * 4
    }
}

/// Dense storage for categorical features (no raw values).
///
/// Categorical features only need binned representation since the bin IS the category.
/// There's no meaningful "raw value" to store.
#[derive(Clone, Debug)]
pub struct CategoricalStorage {
    /// Binned values (category indices).
    pub bins: BinData,
}

impl CategoricalStorage {
    /// Create new categorical storage.
    pub fn new(bins: BinData) -> Self {
        Self { bins }
    }

    /// Get bin (category) value for a specific sample.
    #[inline]
    pub fn bin(&self, offset: usize) -> u32 {
        self.bins.get(offset)
    }

    /// Total number of bin values stored.
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

/// Sparse storage for numeric features with raw values.
///
/// Uses CSC-like format: stores only non-zero entries with their sample indices.
/// Both bins and raw values are stored for non-zero entries.
#[derive(Clone, Debug)]
pub struct SparseNumericStorage {
    /// Sample indices of non-zero entries (sorted).
    pub sample_indices: Box<[u32]>,
    /// Bin values for non-zero entries.
    pub bins: BinData,
    /// Raw values for non-zero entries.
    pub raw_values: Box<[f32]>,
    /// Total number of samples.
    pub n_samples: usize,
}

impl SparseNumericStorage {
    /// Create new sparse numeric storage.
    pub fn new(
        sample_indices: Vec<u32>,
        bins: BinData,
        raw_values: Vec<f32>,
        n_samples: usize,
    ) -> Self {
        debug_assert_eq!(sample_indices.len(), bins.len());
        debug_assert_eq!(sample_indices.len(), raw_values.len());
        Self {
            sample_indices: sample_indices.into_boxed_slice(),
            bins,
            raw_values: raw_values.into_boxed_slice(),
            n_samples,
        }
    }

    /// Number of non-zero entries.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.sample_indices.len()
    }

    /// Get bin value for a sample using binary search.
    /// Returns 0 if sample not in indices (implicit zero).
    #[inline]
    pub fn bin(&self, sample: usize) -> u32 {
        match self.sample_indices.binary_search(&(sample as u32)) {
            Ok(idx) => self.bins.get(idx),
            Err(_) => 0,
        }
    }

    /// Get raw value for a sample using binary search.
    /// Returns 0.0 if sample not in indices (implicit zero).
    #[inline]
    pub fn raw(&self, sample: usize) -> f32 {
        match self.sample_indices.binary_search(&(sample as u32)) {
            Ok(idx) => self.raw_values[idx],
            Err(_) => 0.0,
        }
    }

    /// Memory size in bytes.
    #[inline]
    pub fn size_bytes(&self) -> usize {
        self.sample_indices.len() * 4 + self.bins.size_bytes() + self.raw_values.len() * 4
    }
}

/// Sparse storage for categorical features (no raw values).
#[derive(Clone, Debug)]
pub struct SparseCategoricalStorage {
    /// Sample indices of non-zero entries (sorted).
    pub sample_indices: Box<[u32]>,
    /// Bin values for non-zero entries.
    pub bins: BinData,
    /// Total number of samples.
    pub n_samples: usize,
}

impl SparseCategoricalStorage {
    /// Create new sparse categorical storage.
    pub fn new(sample_indices: Vec<u32>, bins: BinData, n_samples: usize) -> Self {
        debug_assert_eq!(sample_indices.len(), bins.len());
        Self {
            sample_indices: sample_indices.into_boxed_slice(),
            bins,
            n_samples,
        }
    }

    /// Number of non-zero entries.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.sample_indices.len()
    }

    /// Get bin value for a sample using binary search.
    /// Returns 0 if sample not in indices (implicit zero).
    #[inline]
    pub fn bin(&self, sample: usize) -> u32 {
        match self.sample_indices.binary_search(&(sample as u32)) {
            Ok(idx) => self.bins.get(idx),
            Err(_) => 0,
        }
    }

    /// Memory size in bytes.
    #[inline]
    pub fn size_bytes(&self) -> usize {
        self.sample_indices.len() * 4 + self.bins.size_bytes()
    }
}

/// Unified storage enum for feature groups.
///
/// Each variant encodes the storage type at the type level, making it clear
/// whether raw values are available.
///
/// # Design
///
/// Feature groups are homogeneous: all features in a group have the same storage type.
/// This enum represents the storage for an entire group.
#[derive(Clone, Debug)]
pub enum FeatureStorage {
    /// Dense numeric features with raw values.
    Numeric(NumericStorage),
    /// Dense categorical features (no raw values).
    Categorical(CategoricalStorage),
    /// Sparse numeric features with raw values.
    SparseNumeric(SparseNumericStorage),
    /// Sparse categorical features (no raw values).
    SparseCategorical(SparseCategoricalStorage),
    // TODO: BundleStorage for EFB bundles
}

impl FeatureStorage {
    /// Check if this storage has raw values available.
    #[inline]
    pub fn has_raw_values(&self) -> bool {
        matches!(
            self,
            FeatureStorage::Numeric(_) | FeatureStorage::SparseNumeric(_)
        )
    }

    /// Check if this is categorical storage.
    #[inline]
    pub fn is_categorical(&self) -> bool {
        matches!(
            self,
            FeatureStorage::Categorical(_) | FeatureStorage::SparseCategorical(_)
        )
    }

    /// Check if this is sparse storage.
    #[inline]
    pub fn is_sparse(&self) -> bool {
        matches!(
            self,
            FeatureStorage::SparseNumeric(_) | FeatureStorage::SparseCategorical(_)
        )
    }

    /// Memory size in bytes.
    #[inline]
    pub fn size_bytes(&self) -> usize {
        match self {
            FeatureStorage::Numeric(s) => s.size_bytes(),
            FeatureStorage::Categorical(s) => s.size_bytes(),
            FeatureStorage::SparseNumeric(s) => s.size_bytes(),
            FeatureStorage::SparseCategorical(s) => s.size_bytes(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numeric_storage() {
        let bins = BinData::from_u8(vec![0, 1, 2, 3, 4, 5]); // 2 features × 3 samples
        let raw = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let storage = NumericStorage::new(bins, raw);

        assert_eq!(storage.len(), 6);
        assert_eq!(storage.bin(0), 0);
        assert_eq!(storage.bin(3), 3);
        assert_eq!(storage.raw(0), 1.0);
        assert_eq!(storage.raw(3), 4.0);

        let slice = storage.raw_slice(1, 3); // feature 1, 3 samples
        assert_eq!(slice, &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_categorical_storage() {
        let bins = BinData::from_u8(vec![0, 1, 2, 0, 1, 2]);
        let storage = CategoricalStorage::new(bins);

        assert_eq!(storage.len(), 6);
        assert_eq!(storage.bin(0), 0);
        assert_eq!(storage.bin(2), 2);
    }

    #[test]
    fn test_sparse_numeric_storage() {
        let indices = vec![0, 2, 5];
        let bins = BinData::from_u8(vec![1, 2, 3]);
        let raw = vec![1.0, 2.0, 3.0];
        let storage = SparseNumericStorage::new(indices, bins, raw, 10);

        assert_eq!(storage.nnz(), 3);
        assert_eq!(storage.bin(0), 1);
        assert_eq!(storage.bin(2), 2);
        assert_eq!(storage.bin(1), 0); // not in indices
        assert_eq!(storage.raw(0), 1.0);
        assert_eq!(storage.raw(1), 0.0); // not in indices
    }

    #[test]
    fn test_feature_storage_traits() {
        let numeric = FeatureStorage::Numeric(NumericStorage::new(
            BinData::from_u8(vec![0, 1]),
            vec![1.0, 2.0],
        ));
        assert!(numeric.has_raw_values());
        assert!(!numeric.is_categorical());
        assert!(!numeric.is_sparse());

        let categorical = FeatureStorage::Categorical(CategoricalStorage::new(BinData::from_u8(
            vec![0, 1],
        )));
        assert!(!categorical.has_raw_values());
        assert!(categorical.is_categorical());
        assert!(!categorical.is_sparse());

        let sparse_numeric = FeatureStorage::SparseNumeric(SparseNumericStorage::new(
            vec![0],
            BinData::from_u8(vec![1]),
            vec![1.0],
            10,
        ));
        assert!(sparse_numeric.has_raw_values());
        assert!(!sparse_numeric.is_categorical());
        assert!(sparse_numeric.is_sparse());
    }
}

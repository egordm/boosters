//! Feature group storage for binned datasets.
//!
//! A `FeatureGroup` contains a set of features with homogeneous storage
//! (all numeric or all categorical, all dense or all sparse). Features
//! are grouped together for efficient storage and histogram building.

// Allow dead code during migration - this will be used when we switch over in Epic 7
#![allow(dead_code)]

use super::FeatureStorage;

/// A group of features with homogeneous storage.
///
/// All features in a group share the same storage type (Numeric, Categorical,
/// SparseNumeric, or SparseCategorical). This enables efficient storage and
/// access patterns.
///
/// # Layout
///
/// Features within a group are stored in column-major order:
/// for feature `f` at sample `s`: `index = f * n_samples + s`
///
/// # Examples
///
/// ```
/// use boosters::data::binned::{BinData, FeatureGroup, NumericStorage, FeatureStorage};
///
/// // Create storage for 2 features, 3 samples each
/// let bins = BinData::from(vec![
///     0u8, 1, 2,   // Feature 0
///     3, 4, 5,     // Feature 1
/// ]);
/// let raw = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0].into_boxed_slice();
/// let storage = NumericStorage::new(bins, raw);
///
/// let group = FeatureGroup::new(
///     vec![0, 1].into_boxed_slice(),  // Global feature indices
///     3,                                // n_samples
///     FeatureStorage::Numeric(storage),
///     vec![10, 10].into_boxed_slice(),  // bin_counts per feature
/// );
///
/// assert_eq!(group.n_features(), 2);
/// assert_eq!(group.n_samples(), 3);
/// assert!(group.has_raw_values());
///
/// // Access bins
/// assert_eq!(group.bin(0, 0), 0);  // sample 0, feature_in_group 0
/// assert_eq!(group.bin(1, 1), 4);  // sample 1, feature_in_group 1
///
/// // Access raw values (only for numeric)
/// assert_eq!(group.raw(0, 0), Some(1.0));
/// ```
#[derive(Debug, Clone)]
pub struct FeatureGroup {
    /// Global feature indices in this group.
    feature_indices: Box<[u32]>,
    /// Number of samples.
    n_samples: usize,
    /// Storage (bins + optional raw values).
    storage: FeatureStorage,
    /// Per-feature bin counts.
    bin_counts: Box<[u32]>,
    /// Cumulative bin offsets within group histogram.
    /// bin_offsets[i] = sum of bin_counts[0..i]
    /// Length = n_features + 1 (last element is total bins)
    bin_offsets: Box<[u32]>,
}

impl FeatureGroup {
    /// Creates a new FeatureGroup.
    ///
    /// # Parameters
    /// - `feature_indices`: Global feature indices for features in this group
    /// - `n_samples`: Number of samples
    /// - `storage`: The underlying feature storage
    /// - `bin_counts`: Number of bins per feature
    ///
    /// # Panics
    ///
    /// Panics if `feature_indices.len() != bin_counts.len()`.
    pub fn new(
        feature_indices: Box<[u32]>,
        n_samples: usize,
        storage: FeatureStorage,
        bin_counts: Box<[u32]>,
    ) -> Self {
        assert_eq!(
            feature_indices.len(),
            bin_counts.len(),
            "feature_indices and bin_counts must have the same length"
        );

        // Compute cumulative bin offsets
        let mut bin_offsets = Vec::with_capacity(bin_counts.len() + 1);
        let mut offset = 0u32;
        bin_offsets.push(offset);
        for &count in bin_counts.iter() {
            offset += count;
            bin_offsets.push(offset);
        }

        Self {
            feature_indices,
            n_samples,
            storage,
            bin_counts,
            bin_offsets: bin_offsets.into_boxed_slice(),
        }
    }

    /// Returns the number of features in this group.
    #[inline]
    pub fn n_features(&self) -> usize {
        self.feature_indices.len()
    }

    /// Returns the number of samples.
    #[inline]
    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    /// Returns the global feature indices in this group.
    #[inline]
    pub fn feature_indices(&self) -> &[u32] {
        &self.feature_indices
    }

    /// Returns the bin counts per feature.
    #[inline]
    pub fn bin_counts(&self) -> &[u32] {
        &self.bin_counts
    }

    /// Returns the cumulative bin offsets for histogram allocation.
    ///
    /// `bin_offsets[i]` is the starting bin index for feature `i` in the histogram.
    /// `bin_offsets[n_features]` is the total number of bins in the group.
    #[inline]
    pub fn bin_offsets(&self) -> &[u32] {
        &self.bin_offsets
    }

    /// Returns the total number of bins across all features in this group.
    #[inline]
    pub fn total_bins(&self) -> u32 {
        *self.bin_offsets.last().unwrap_or(&0)
    }

    /// Returns the underlying storage.
    #[inline]
    pub fn storage(&self) -> &FeatureStorage {
        &self.storage
    }

    /// Returns `true` if this group has raw values (numeric features).
    #[inline]
    pub fn has_raw_values(&self) -> bool {
        self.storage.has_raw_values()
    }

    /// Returns `true` if this group contains categorical features.
    #[inline]
    pub fn is_categorical(&self) -> bool {
        self.storage.is_categorical()
    }

    /// Returns `true` if this group uses sparse storage.
    #[inline]
    pub fn is_sparse(&self) -> bool {
        self.storage.is_sparse()
    }

    /// Returns the bin value for a sample and feature within this group.
    ///
    /// # Parameters
    /// - `sample`: Sample index (0..n_samples)
    /// - `feature_in_group`: Feature index within this group (0..n_features)
    ///
    /// # Panics
    ///
    /// Panics if indices are out of bounds.
    #[inline]
    pub fn bin(&self, sample: usize, feature_in_group: usize) -> u32 {
        debug_assert!(sample < self.n_samples, "sample index out of bounds");
        debug_assert!(
            feature_in_group < self.n_features(),
            "feature_in_group index out of bounds"
        );

        match &self.storage {
            FeatureStorage::Numeric(s) => s.bin(sample, feature_in_group, self.n_samples),
            FeatureStorage::Categorical(s) => s.bin(sample, feature_in_group, self.n_samples),
            FeatureStorage::SparseNumeric(s) => s.get(sample).0,
            FeatureStorage::SparseCategorical(s) => s.bin(sample),
        }
    }

    /// Returns the raw value for a sample and feature within this group.
    ///
    /// Returns `None` for categorical features (they don't have raw values).
    ///
    /// # Parameters
    /// - `sample`: Sample index (0..n_samples)
    /// - `feature_in_group`: Feature index within this group (0..n_features)
    #[inline]
    pub fn raw(&self, sample: usize, feature_in_group: usize) -> Option<f32> {
        debug_assert!(sample < self.n_samples, "sample index out of bounds");
        debug_assert!(
            feature_in_group < self.n_features(),
            "feature_in_group index out of bounds"
        );

        match &self.storage {
            FeatureStorage::Numeric(s) => Some(s.raw(sample, feature_in_group, self.n_samples)),
            FeatureStorage::SparseNumeric(s) => Some(s.get(sample).1),
            FeatureStorage::Categorical(_) | FeatureStorage::SparseCategorical(_) => None,
        }
    }

    /// Returns a contiguous slice of raw values for a feature.
    ///
    /// Returns `None` for categorical features or sparse storage.
    ///
    /// # Parameters
    /// - `feature_in_group`: Feature index within this group
    #[inline]
    pub fn raw_slice(&self, feature_in_group: usize) -> Option<&[f32]> {
        debug_assert!(
            feature_in_group < self.n_features(),
            "feature_in_group index out of bounds"
        );

        match &self.storage {
            FeatureStorage::Numeric(s) => Some(s.raw_slice(feature_in_group, self.n_samples)),
            // Sparse storage doesn't have contiguous slices for all samples
            FeatureStorage::SparseNumeric(_) => None,
            FeatureStorage::Categorical(_) | FeatureStorage::SparseCategorical(_) => None,
        }
    }

    /// Returns the total size in bytes used by this group.
    #[inline]
    pub fn size_bytes(&self) -> usize {
        let indices_size = self.feature_indices.len() * std::mem::size_of::<u32>();
        let counts_size = self.bin_counts.len() * std::mem::size_of::<u32>();
        let offsets_size = self.bin_offsets.len() * std::mem::size_of::<u32>();
        let storage_size = self.storage.size_bytes();
        indices_size + counts_size + offsets_size + storage_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::binned::{
        BinData, CategoricalStorage, NumericStorage, SparseCategoricalStorage,
        SparseNumericStorage,
    };

    fn make_numeric_group() -> FeatureGroup {
        // 2 features, 3 samples
        let bins = BinData::from(vec![
            0u8, 1, 2, // Feature 0
            5, 6, 7, // Feature 1
        ]);
        let raw = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0].into_boxed_slice();
        let storage = FeatureStorage::Numeric(NumericStorage::new(bins, raw));

        FeatureGroup::new(
            vec![10, 20].into_boxed_slice(), // Global indices
            3,                                // n_samples
            storage,
            vec![10, 8].into_boxed_slice(), // bin_counts
        )
    }

    fn make_categorical_group() -> FeatureGroup {
        // 2 features, 3 samples
        let bins = BinData::from(vec![
            0u8, 1, 2, // Feature 0
            3, 4, 5, // Feature 1
        ]);
        let storage = FeatureStorage::Categorical(CategoricalStorage::new(bins));

        FeatureGroup::new(
            vec![100, 200].into_boxed_slice(),
            3,
            storage,
            vec![5, 6].into_boxed_slice(),
        )
    }

    fn make_sparse_numeric_group() -> FeatureGroup {
        // Single sparse feature, 10 samples, only 2 non-zero
        let storage = FeatureStorage::SparseNumeric(SparseNumericStorage::new(
            vec![2, 7].into_boxed_slice(),
            BinData::from(vec![1u8, 2]),
            vec![1.5, 2.5].into_boxed_slice(),
            10,
        ));

        FeatureGroup::new(
            vec![50].into_boxed_slice(),
            10,
            storage,
            vec![5].into_boxed_slice(),
        )
    }

    fn make_sparse_categorical_group() -> FeatureGroup {
        // Single sparse feature, 10 samples
        let storage = FeatureStorage::SparseCategorical(SparseCategoricalStorage::new(
            vec![3, 8].into_boxed_slice(),
            BinData::from(vec![1u8, 2]),
            10,
        ));

        FeatureGroup::new(
            vec![75].into_boxed_slice(),
            10,
            storage,
            vec![3].into_boxed_slice(),
        )
    }

    #[test]
    fn test_numeric_group_basic() {
        let group = make_numeric_group();
        assert_eq!(group.n_features(), 2);
        assert_eq!(group.n_samples(), 3);
        assert!(group.has_raw_values());
        assert!(!group.is_categorical());
        assert!(!group.is_sparse());
    }

    #[test]
    fn test_numeric_group_feature_indices() {
        let group = make_numeric_group();
        assert_eq!(group.feature_indices(), &[10, 20]);
    }

    #[test]
    fn test_numeric_group_bin_access() {
        let group = make_numeric_group();

        // Feature 0: bins [0, 1, 2]
        assert_eq!(group.bin(0, 0), 0);
        assert_eq!(group.bin(1, 0), 1);
        assert_eq!(group.bin(2, 0), 2);

        // Feature 1: bins [5, 6, 7]
        assert_eq!(group.bin(0, 1), 5);
        assert_eq!(group.bin(1, 1), 6);
        assert_eq!(group.bin(2, 1), 7);
    }

    #[test]
    fn test_numeric_group_raw_access() {
        let group = make_numeric_group();

        // Feature 0: raw [1.0, 2.0, 3.0]
        assert_eq!(group.raw(0, 0), Some(1.0));
        assert_eq!(group.raw(1, 0), Some(2.0));
        assert_eq!(group.raw(2, 0), Some(3.0));

        // Feature 1: raw [4.0, 5.0, 6.0]
        assert_eq!(group.raw(0, 1), Some(4.0));
        assert_eq!(group.raw(1, 1), Some(5.0));
        assert_eq!(group.raw(2, 1), Some(6.0));
    }

    #[test]
    fn test_numeric_group_raw_slice() {
        let group = make_numeric_group();
        assert_eq!(group.raw_slice(0), Some(&[1.0, 2.0, 3.0][..]));
        assert_eq!(group.raw_slice(1), Some(&[4.0, 5.0, 6.0][..]));
    }

    #[test]
    fn test_numeric_group_bin_offsets() {
        let group = make_numeric_group();
        assert_eq!(group.bin_counts(), &[10, 8]);
        assert_eq!(group.bin_offsets(), &[0, 10, 18]);
        assert_eq!(group.total_bins(), 18);
    }

    #[test]
    fn test_categorical_group_basic() {
        let group = make_categorical_group();
        assert_eq!(group.n_features(), 2);
        assert_eq!(group.n_samples(), 3);
        assert!(!group.has_raw_values());
        assert!(group.is_categorical());
        assert!(!group.is_sparse());
    }

    #[test]
    fn test_categorical_group_no_raw() {
        let group = make_categorical_group();
        assert_eq!(group.raw(0, 0), None);
        assert_eq!(group.raw(1, 1), None);
        assert_eq!(group.raw_slice(0), None);
    }

    #[test]
    fn test_categorical_group_bin_access() {
        let group = make_categorical_group();
        assert_eq!(group.bin(0, 0), 0);
        assert_eq!(group.bin(1, 0), 1);
        assert_eq!(group.bin(2, 1), 5);
    }

    #[test]
    fn test_sparse_numeric_group() {
        let group = make_sparse_numeric_group();
        assert_eq!(group.n_features(), 1);
        assert_eq!(group.n_samples(), 10);
        assert!(group.has_raw_values());
        assert!(group.is_sparse());

        // Sample 2 has bin=1, raw=1.5
        assert_eq!(group.bin(2, 0), 1);
        assert_eq!(group.raw(2, 0), Some(1.5));

        // Sample 7 has bin=2, raw=2.5
        assert_eq!(group.bin(7, 0), 2);
        assert_eq!(group.raw(7, 0), Some(2.5));

        // Sample 0 is not in sparse list, defaults to (0, 0.0)
        assert_eq!(group.bin(0, 0), 0);
        assert_eq!(group.raw(0, 0), Some(0.0));

        // No contiguous raw slice for sparse
        assert_eq!(group.raw_slice(0), None);
    }

    #[test]
    fn test_sparse_categorical_group() {
        let group = make_sparse_categorical_group();
        assert_eq!(group.n_features(), 1);
        assert_eq!(group.n_samples(), 10);
        assert!(!group.has_raw_values());
        assert!(group.is_categorical());
        assert!(group.is_sparse());

        // Sample 3 has bin=1
        assert_eq!(group.bin(3, 0), 1);

        // Sample 8 has bin=2
        assert_eq!(group.bin(8, 0), 2);

        // Sample 0 defaults to bin=0
        assert_eq!(group.bin(0, 0), 0);

        // No raw values for categorical
        assert_eq!(group.raw(0, 0), None);
    }

    #[test]
    fn test_group_size_bytes() {
        let group = make_numeric_group();
        // feature_indices: 2 * 4 = 8
        // bin_counts: 2 * 4 = 8
        // bin_offsets: 3 * 4 = 12
        // storage: 6 bytes bins + 24 bytes raw = 30
        // Total: 8 + 8 + 12 + 30 = 58
        assert_eq!(group.size_bytes(), 58);
    }

    #[test]
    #[should_panic(expected = "feature_indices and bin_counts must have the same length")]
    fn test_mismatched_indices_counts() {
        let bins = BinData::from(vec![0u8, 1, 2]);
        let raw = vec![1.0, 2.0, 3.0].into_boxed_slice();
        let storage = FeatureStorage::Numeric(NumericStorage::new(bins, raw));

        FeatureGroup::new(
            vec![1, 2].into_boxed_slice(), // 2 features
            3,
            storage,
            vec![5].into_boxed_slice(), // Only 1 count - mismatch!
        );
    }

    #[test]
    fn test_single_feature_single_sample() {
        let bins = BinData::from(vec![5u8]);
        let raw = vec![3.14].into_boxed_slice();
        let storage = FeatureStorage::Numeric(NumericStorage::new(bins, raw));

        let group = FeatureGroup::new(
            vec![0].into_boxed_slice(),
            1,
            storage,
            vec![10].into_boxed_slice(),
        );

        assert_eq!(group.n_features(), 1);
        assert_eq!(group.n_samples(), 1);
        assert_eq!(group.bin(0, 0), 5);
        assert_eq!(group.raw(0, 0), Some(3.14));
        assert_eq!(group.raw_slice(0), Some(&[3.14][..]));
        assert_eq!(group.total_bins(), 10);
    }

    #[test]
    fn test_empty_bin_offsets() {
        // Edge case: single feature with 0 bins (shouldn't happen but let's handle it)
        let bins = BinData::from(vec![0u8]);
        let raw = vec![1.0].into_boxed_slice();
        let storage = FeatureStorage::Numeric(NumericStorage::new(bins, raw));

        let group = FeatureGroup::new(
            vec![0].into_boxed_slice(),
            1,
            storage,
            vec![0].into_boxed_slice(), // 0 bins
        );

        assert_eq!(group.bin_offsets(), &[0, 0]);
        assert_eq!(group.total_bins(), 0);
    }
}

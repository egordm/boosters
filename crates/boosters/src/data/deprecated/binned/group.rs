//! Feature group - a collection of features with shared storage layout.
//!
//! # Deprecation Notice
//!
//! This module uses deprecated types (`BinStorage`, `BinType`)
//! that will be migrated to v2 in a future release. See RFC-0018.

#![allow(deprecated)] // Allow internal use of deprecated types during migration

use super::BinMapper;
use super::storage::{BinStorage, BinType};

/// A group of features with shared storage layout.
///
/// Features in a group share:
/// - Storage format (dense or sparse)
/// - Bin width (u8 or u16)
/// - Memory layout (always column-major)
///
/// # Storage Types
///
/// ## Dense (Column-Major)
/// For features with mostly non-zero values.
///
/// ### Column-Major Layout
/// ```text
/// [f0_row0, f0_row1, ..., f0_rowN, f1_row0, f1_row1, ...]
/// ```
///
/// ## Sparse (CSR-like)
/// For highly sparse features (>80% zeros). Single feature per group.
/// Stores only non-zero (row_index, bin_value) pairs.
#[derive(Clone, Debug)]
pub struct FeatureGroup {
    /// Global feature indices in this group.
    feature_indices: Box<[u32]>,
    /// Number of rows in the dataset.
    n_rows: usize,
    /// Bin data.
    storage: BinStorage,
    /// Per-feature bin counts.
    bin_counts: Box<[u32]>,
    /// Cumulative bin offsets within group histogram.
    /// Length = n_features + 1 (last is total).
    bin_offsets: Box<[u32]>,
}

impl FeatureGroup {
    /// Create a new feature group.
    ///
    /// # Arguments
    /// * `feature_indices` - Global feature IDs in this group
    /// * `n_rows` - Number of rows (samples)
    /// * `storage` - Bin data (DenseU8, DenseU16, SparseU8, or SparseU16)
    /// * `bin_counts` - Number of bins per feature
    ///
    /// # Validation
    /// - Dense storage: verifies size = n_rows * n_features
    /// - Sparse storage: requires single feature
    pub fn new(
        feature_indices: Vec<u32>,
        n_rows: usize,
        storage: BinStorage,
        bin_counts: Vec<u32>,
    ) -> Self {
        let n_features = feature_indices.len();

        debug_assert_eq!(
            bin_counts.len(),
            n_features,
            "bin_counts length {} doesn't match n_features {}",
            bin_counts.len(),
            n_features
        );

        if storage.is_dense() {
            let expected_size = n_rows * n_features;
            debug_assert_eq!(
                storage.len(),
                expected_size,
                "Storage size {} doesn't match expected {} (n_rows={} * n_features={})",
                storage.len(),
                expected_size,
                n_rows,
                n_features
            );
        } else {
            // Sparse storage: single feature
            debug_assert_eq!(n_features, 1, "Sparse storage requires single feature");
        }

        Self::new_unchecked(feature_indices, n_rows, storage, bin_counts)
    }

    /// Internal constructor without validation.
    fn new_unchecked(
        feature_indices: Vec<u32>,
        n_rows: usize,
        storage: BinStorage,
        bin_counts: Vec<u32>,
    ) -> Self {
        let n_features = feature_indices.len();

        // Compute cumulative bin offsets
        let mut bin_offsets = Vec::with_capacity(n_features + 1);
        let mut total = 0u32;
        for &count in &bin_counts {
            bin_offsets.push(total);
            total += count;
        }
        bin_offsets.push(total);

        Self {
            feature_indices: feature_indices.into_boxed_slice(),
            n_rows,
            storage,
            bin_counts: bin_counts.into_boxed_slice(),
            bin_offsets: bin_offsets.into_boxed_slice(),
        }
    }

    /// Number of features in this group.
    #[inline]
    pub fn n_features(&self) -> usize {
        self.feature_indices.len()
    }

    /// Number of rows (samples).
    #[inline]
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Bin type for this group.
    #[inline]
    pub fn bin_type(&self) -> BinType {
        self.storage.bin_type()
    }

    /// Global feature indices.
    #[inline]
    pub fn feature_indices(&self) -> &[u32] {
        &self.feature_indices
    }

    /// Total bins in this group (for histogram allocation).
    #[inline]
    pub fn total_bins(&self) -> u32 {
        *self.bin_offsets.last().unwrap_or(&0)
    }

    /// Bin count for a feature in this group.
    #[inline]
    pub fn bin_count(&self, feature_in_group: usize) -> u32 {
        self.bin_counts[feature_in_group]
    }

    /// Bin counts for all features.
    #[inline]
    pub fn bin_counts(&self) -> &[u32] {
        &self.bin_counts
    }

    /// Cumulative bin offsets (length = n_features + 1).
    #[inline]
    pub fn bin_offsets(&self) -> &[u32] {
        &self.bin_offsets
    }

    /// Bin offset for a feature in this group.
    #[inline]
    pub fn bin_offset(&self, feature_in_group: usize) -> u32 {
        self.bin_offsets[feature_in_group]
    }

    /// Access the underlying bin storage directly.
    ///
    /// Match on this to get typed access for efficient loops.
    #[inline]
    pub fn storage(&self) -> &BinStorage {
        &self.storage
    }

    /// Check if this group uses dense storage.
    #[inline]
    pub fn is_dense(&self) -> bool {
        self.storage.is_dense()
    }

    /// Check if this group uses sparse storage.
    #[inline]
    pub fn is_sparse(&self) -> bool {
        self.storage.is_sparse()
    }

    /// Feature stride (number of bins per feature) for column-major layout.
    /// Returns `None` for sparse storage.
    #[inline]
    pub fn feature_stride(&self) -> Option<usize> {
        if self.storage.is_dense() {
            Some(self.n_rows)
        } else {
            None
        }
    }

    /// Memory size in bytes.
    #[inline]
    pub fn size_bytes(&self) -> usize {
        self.storage.size_bytes()
    }

    /// Get a contiguous slice for a feature (single-feature groups only).
    ///
    /// Returns the slice of bins for the given feature.
    /// Returns `None` for multi-feature groups.
    ///
    /// # Arguments
    /// * `_feature_in_group` - Index of the feature within this group (not global feature ID)
    ///   Currently only single-feature groups are supported for direct slice access.
    #[inline]
    pub fn feature_bins_slice(&self, _feature_in_group: usize) -> Option<&BinStorage> {
        if self.storage.is_dense() && self.n_features() == 1 {
            Some(&self.storage)
        } else {
            None
        }
    }

    /// Get feature bins as a range within the storage.
    ///
    /// For column-major layout with multiple features, returns the (start, end) indices
    /// into the storage for the given feature. Returns `None` for sparse storage.
    #[inline]
    pub fn feature_range(&self, feature_in_group: usize) -> Option<(usize, usize)> {
        if self.storage.is_sparse() {
            return None;
        }
        let start = feature_in_group * self.n_rows;
        let end = start + self.n_rows;
        Some((start, end))
    }
}

/// Metadata for a single binned feature.
///
/// This contains binning information (BinMapper) and group assignment
/// for use with BinnedDataset. For simple feature metadata (name + type)
/// used with Dataset, see [`crate::data::BinnedFeatureInfo`].
#[derive(Clone, Debug)]
pub struct BinnedFeatureInfo {
    /// Feature name (optional).
    pub name: Option<String>,
    /// Bin mapper for this feature.
    pub bin_mapper: BinMapper,
    /// Index of the group containing this feature.
    pub group_index: u32,
    /// Index within the group.
    pub index_in_group: u32,
}

impl BinnedFeatureInfo {
    /// Create new feature metadata.
    pub fn new(bin_mapper: BinMapper, group_index: u32, index_in_group: u32) -> Self {
        Self {
            name: None,
            bin_mapper,
            group_index,
            index_in_group,
        }
    }

    /// Create with name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Number of bins.
    #[inline]
    pub fn n_bins(&self) -> u32 {
        self.bin_mapper.n_bins()
    }

    /// Check if categorical.
    #[inline]
    pub fn is_categorical(&self) -> bool {
        self.bin_mapper.is_categorical()
    }

    /// Most frequent bin (for MFB optimization).
    #[inline]
    pub fn most_freq_bin(&self) -> u32 {
        self.bin_mapper.most_freq_bin()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::bin_mapper::MissingType as DeprecatedMissingType;

    fn make_simple_mapper(n_bins: u32) -> BinMapper {
        let bounds: Vec<f64> = (0..n_bins).map(|i| i as f64 + 0.5).collect();
        BinMapper::numerical(
            bounds,
            DeprecatedMissingType::None,
            0,
            0,
            0.0,
            0.0,
            (n_bins - 1) as f64,
        )
    }

    #[test]
    fn test_feature_group_column_major() {
        // 3 rows, 2 features
        // Column-major: [f0r0, f0r1, f0r2, f1r0, f1r1, f1r2]
        let storage = BinStorage::from_u8(vec![0, 1, 2, 3, 4, 5]);
        let group = FeatureGroup::new(
            vec![10, 11], // feature indices (non-zero to test they're stored)
            3,            // n_rows
            storage,
            vec![4, 4], // bin counts
        );

        assert_eq!(group.n_features(), 2);
        assert_eq!(group.n_rows(), 3);
        assert_eq!(group.feature_stride(), Some(3));
        assert_eq!(group.feature_indices(), &[10, 11]);
        assert_eq!(group.total_bins(), 8);
        assert_eq!(group.bin_offsets(), &[0, 4, 8]);

        // Access via match on storage
        match group.storage() {
            BinStorage::DenseU8(data) => {
                // Column-major: feature 0 rows first, then feature 1 rows
                assert_eq!(data[0], 0); // f0r0
                assert_eq!(data[1], 1); // f0r1
                assert_eq!(data[2], 2); // f0r2
                assert_eq!(data[3], 3); // f1r0
            }
            _ => panic!("expected DenseU8"),
        }
    }

    #[test]
    fn test_feature_group_u16_storage() {
        // 2 rows, 2 features - column-major
        let storage = BinStorage::from_u16(vec![0, 256, 1000, 500]);
        let group = FeatureGroup::new(
            vec![0, 1],
            2, // n_rows
            storage,
            vec![1024, 1024],
        );

        assert_eq!(group.bin_type(), BinType::U16);
        match group.storage() {
            BinStorage::DenseU16(data) => {
                // Column-major: [f0r0, f0r1, f1r0, f1r1]
                assert_eq!(data[0], 0);   // f0r0
                assert_eq!(data[1], 256); // f0r1
                assert_eq!(data[2], 1000); // f1r0
                assert_eq!(data[3], 500); // f1r1
            }
            _ => panic!("expected DenseU16"),
        }
    }

    #[test]
    fn test_feature_meta() {
        let mapper = make_simple_mapper(4);
        let meta = BinnedFeatureInfo::new(mapper, 0, 2).with_name("feature_x");

        assert_eq!(meta.name, Some("feature_x".to_string()));
        assert_eq!(meta.group_index, 0);
        assert_eq!(meta.index_in_group, 2);
        assert_eq!(meta.n_bins(), 4);
        assert!(!meta.is_categorical());
    }
}

//! Storage types for binned data.
//!
//! # Deprecation Notice
//!
//! These types are being replaced by the types in [`super::v2`].
//! See the v2 module documentation for migration guide.

#![allow(deprecated)] // Allow internal use of deprecated types during migration

/// Bin data type for a feature group.
#[deprecated(
    since = "0.2.0",
    note = "Use `v2::BinData` instead - it owns the actual data rather than just being a type marker"
)]
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

/// Bin storage for a feature group.
///
/// Match on this enum to get typed access to the underlying data.
/// This design forces exhaustive handling of all storage types.
///
/// # Deprecation
///
/// This type is being replaced by [`super::v2::FeatureStorage`] which
/// encodes whether raw values are available at the type level.
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
#[deprecated(
    since = "0.2.0",
    note = "Use `v2::FeatureStorage` instead - it encodes raw value availability at the type level"
)]
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
/// All dense storage uses column-major layout, so slices are always contiguous.
///
/// # Example
///
/// ```ignore
/// for f in 0..dataset.n_features() {
///     let slice = dataset.feature_view(f);
///     match slice {
///         FeatureView::U8 { bins } => {
///             // bins is contiguous for this feature
///             for row in 0..n_rows {
///                 let bin = bins[row];
///             }
///         }
///         FeatureView::U16 { bins } => { ... }
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
    /// Dense u8 bins (column-major, always contiguous).
    U8 { bins: &'a [u8] },
    /// Dense u16 bins (column-major, always contiguous).
    U16 { bins: &'a [u16] },
    /// Sparse u8 bins.
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

    /// Get bin value at row index for dense slices.
    ///
    /// Returns `None` for sparse storage.
    /// For sparse features, use iteration over row_indices instead.
    #[inline]
    pub fn get_bin(&self, row: usize) -> Option<u32> {
        match self {
            FeatureView::U8 { bins } => Some(bins[row] as u32),
            FeatureView::U16 { bins } => Some(bins[row] as u32),
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
    /// - `row` is within bounds
    /// - Not called on sparse slices
    #[inline(always)]
    pub fn get_bin_unchecked(&self, row: usize) -> u32 {
        match self {
            FeatureView::U8 { bins } => bins[row] as u32,
            FeatureView::U16 { bins } => bins[row] as u32,
            FeatureView::SparseU8 { .. } | FeatureView::SparseU16 { .. } => 0,
        }
    }
}

impl<'a> From<&'a BinStorage> for FeatureView<'a> {
    /// Convert from BinStorage to FeatureView.
    ///
    /// This creates a contiguous slice from the storage.
    fn from(storage: &'a BinStorage) -> Self {
        match storage {
            BinStorage::DenseU8(data) => FeatureView::U8 { bins: data },
            BinStorage::DenseU16(data) => FeatureView::U16 { bins: data },
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
}

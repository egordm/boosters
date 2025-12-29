//! Feature view for zero-cost access to binned data.
//!
//! `FeatureView` provides zero-copy views into feature bins for histogram
//! building. The view is optimized for the hot path: match once on the
//! variant, then iterate the slice directly.

// Allow dead code during migration - this will be used when we switch over in Epic 7
#![allow(dead_code)]

/// Zero-cost view into feature bins.
///
/// No stride field - everything is column-major, contiguous.
/// This is used in the histogram building hot path.
///
/// # Performance
///
/// Match once on the variant, then iterate the slice directly:
///
/// ```ignore
/// match view {
///     FeatureView::U8(bins) => {
///         for sample in samples {
///             let bin = unsafe { *bins.get_unchecked(sample) };
///             // ... update histogram
///         }
///     }
///     FeatureView::U16(bins) => { /* same pattern */ }
///     FeatureView::SparseU8 { sample_indices, bin_values } => { /* sparse */ }
///     FeatureView::SparseU16 { sample_indices, bin_values } => { /* sparse */ }
/// }
/// ```
///
/// # Examples
///
/// ```ignore
/// // Note: Example is marked ignore until Epic 7 when new types are exported publicly.
/// // Currently deprecated types shadow these in boosters::data::binned re-exports.
/// use boosters::data::binned::view::FeatureView;
///
/// // Dense U8 view
/// let bins = &[0u8, 1, 2, 3, 4][..];
/// let view = FeatureView::U8(bins);
/// assert!(view.is_dense());
/// assert!(!view.is_sparse());
///
/// // Sparse U8 view
/// let indices = &[0u32, 2, 4][..];
/// let values = &[1u8, 2, 3][..];
/// let view = FeatureView::SparseU8 {
///     sample_indices: indices,
///     bin_values: values,
/// };
/// assert!(view.is_sparse());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureView<'a> {
    /// Dense U8 bins, contiguous per feature.
    U8(&'a [u8]),
    /// Dense U16 bins, contiguous per feature.
    U16(&'a [u16]),
    /// Sparse U8 bins with sample indices.
    SparseU8 {
        /// Sample indices of non-zero entries (sorted).
        sample_indices: &'a [u32],
        /// Bin values for non-zero entries (parallel to sample_indices).
        bin_values: &'a [u8],
    },
    /// Sparse U16 bins with sample indices.
    SparseU16 {
        /// Sample indices of non-zero entries (sorted).
        sample_indices: &'a [u32],
        /// Bin values for non-zero entries (parallel to sample_indices).
        bin_values: &'a [u16],
    },
}

impl<'a> FeatureView<'a> {
    /// Returns `true` if this is dense storage.
    #[inline]
    pub fn is_dense(&self) -> bool {
        matches!(self, FeatureView::U8(_) | FeatureView::U16(_))
    }

    /// Returns `true` if this is sparse storage.
    #[inline]
    pub fn is_sparse(&self) -> bool {
        !self.is_dense()
    }

    /// Returns `true` if this uses U8 bins.
    #[inline]
    pub fn is_u8(&self) -> bool {
        matches!(self, FeatureView::U8(_) | FeatureView::SparseU8 { .. })
    }

    /// Returns `true` if this uses U16 bins.
    #[inline]
    pub fn is_u16(&self) -> bool {
        matches!(self, FeatureView::U16(_) | FeatureView::SparseU16 { .. })
    }

    /// Returns the number of entries in this view.
    ///
    /// For dense views, this is the number of samples.
    /// For sparse views, this is the number of non-zero entries.
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            FeatureView::U8(bins) => bins.len(),
            FeatureView::U16(bins) => bins.len(),
            FeatureView::SparseU8 { bin_values, .. } => bin_values.len(),
            FeatureView::SparseU16 { bin_values, .. } => bin_values.len(),
        }
    }

    /// Returns `true` if this view is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the bin value for a sample.
    ///
    /// For dense views, this is O(1). For sparse views, this is O(log n).
    /// Returns `None` if the sample is not in the sparse view (default bin).
    #[inline]
    pub fn get_bin(&self, sample: usize) -> Option<u32> {
        match self {
            FeatureView::U8(bins) => bins.get(sample).map(|&b| b as u32),
            FeatureView::U16(bins) => bins.get(sample).map(|&b| b as u32),
            FeatureView::SparseU8 {
                sample_indices,
                bin_values,
            } => {
                // Binary search for sample in sparse indices
                sample_indices
                    .binary_search(&(sample as u32))
                    .ok()
                    .map(|idx| bin_values[idx] as u32)
            }
            FeatureView::SparseU16 {
                sample_indices,
                bin_values,
            } => {
                // Binary search for sample in sparse indices
                sample_indices
                    .binary_search(&(sample as u32))
                    .ok()
                    .map(|idx| bin_values[idx] as u32)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dense_u8() {
        let bins = &[0u8, 1, 2, 3, 4][..];
        let view = FeatureView::U8(bins);

        assert!(view.is_dense());
        assert!(!view.is_sparse());
        assert!(view.is_u8());
        assert!(!view.is_u16());
        assert_eq!(view.len(), 5);
        assert!(!view.is_empty());
    }

    #[test]
    fn test_dense_u16() {
        let bins = &[100u16, 200, 300][..];
        let view = FeatureView::U16(bins);

        assert!(view.is_dense());
        assert!(!view.is_sparse());
        assert!(!view.is_u8());
        assert!(view.is_u16());
        assert_eq!(view.len(), 3);
    }

    #[test]
    fn test_sparse_u8() {
        let indices = &[0u32, 2, 4][..];
        let values = &[1u8, 2, 3][..];
        let view = FeatureView::SparseU8 {
            sample_indices: indices,
            bin_values: values,
        };

        assert!(!view.is_dense());
        assert!(view.is_sparse());
        assert!(view.is_u8());
        assert!(!view.is_u16());
        assert_eq!(view.len(), 3);
    }

    #[test]
    fn test_sparse_u16() {
        let indices = &[1u32, 3, 5, 7][..];
        let values = &[256u16, 512, 768, 1024][..];
        let view = FeatureView::SparseU16 {
            sample_indices: indices,
            bin_values: values,
        };

        assert!(!view.is_dense());
        assert!(view.is_sparse());
        assert!(!view.is_u8());
        assert!(view.is_u16());
        assert_eq!(view.len(), 4);
    }

    #[test]
    fn test_empty_views() {
        let empty_u8: &[u8] = &[];
        let view = FeatureView::U8(empty_u8);
        assert!(view.is_empty());
        assert_eq!(view.len(), 0);

        let empty_indices: &[u32] = &[];
        let empty_values: &[u8] = &[];
        let sparse_view = FeatureView::SparseU8 {
            sample_indices: empty_indices,
            bin_values: empty_values,
        };
        assert!(sparse_view.is_empty());
    }

    #[test]
    fn test_view_copy() {
        let bins = &[1u8, 2, 3][..];
        let view1 = FeatureView::U8(bins);
        let view2 = view1; // Copy
        assert_eq!(view1, view2);
    }

    #[test]
    fn test_view_debug() {
        let bins = &[1u8, 2][..];
        let view = FeatureView::U8(bins);
        let debug_str = format!("{:?}", view);
        assert!(debug_str.contains("U8"));
    }

    #[test]
    fn test_single_element() {
        let bins = &[42u8][..];
        let view = FeatureView::U8(bins);
        assert_eq!(view.len(), 1);
        assert!(!view.is_empty());

        match view {
            FeatureView::U8(b) => assert_eq!(b[0], 42),
            _ => panic!("Expected U8"),
        }
    }
}

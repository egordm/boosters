//! Bin data storage for quantized feature values.
//!
//! `BinData` encapsulates whether bins are stored as `u8` (≤256 bins)
//! or `u16` (≤65536 bins). This replaces the previous `BinType` enum
//! by combining type information with storage.

/// Bin data container. The variant encodes the bin width.
///
/// # Examples
///
/// ```
/// use boosters::data::binned::BinData;
///
/// // Create from vectors
/// let u8_bins = BinData::from(vec![0u8, 1, 2, 3]);
/// assert!(u8_bins.is_u8());
/// assert_eq!(u8_bins.len(), 4);
/// assert_eq!(u8_bins.get(2), Some(2));
///
/// let u16_bins = BinData::from(vec![0u16, 256, 512]);
/// assert!(u16_bins.is_u16());
/// assert_eq!(u16_bins.max_bins(), 65536);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BinData {
    /// Bins stored as u8 (supports up to 256 bins per feature).
    U8(Box<[u8]>),
    /// Bins stored as u16 (supports up to 65536 bins per feature).
    U16(Box<[u16]>),
}

impl BinData {
    /// Returns `true` if this is U8 storage.
    #[inline]
    pub fn is_u8(&self) -> bool {
        matches!(self, BinData::U8(_))
    }

    /// Returns `true` if this is U16 storage.
    #[inline]
    pub fn is_u16(&self) -> bool {
        matches!(self, BinData::U16(_))
    }

    /// Returns the number of bin values stored.
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            BinData::U8(data) => data.len(),
            BinData::U16(data) => data.len(),
        }
    }

    /// Returns `true` if no bin values are stored.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the bin value at the given index, or `None` if out of bounds.
    ///
    /// # Performance Note
    ///
    /// This method matches on the variant for every access. **Do not use in hot
    /// paths** like histogram building. Instead, match once on the `BinData` variant
    /// using [`as_u8()`](Self::as_u8) or [`as_u16()`](Self::as_u16), then iterate
    /// the slice directly:
    ///
    /// ```ignore
    /// match bin_data {
    ///     BinData::U8(bins) => {
    ///         for sample in samples {
    ///             let bin = bins[sample];  // Direct slice access, no branch
    ///             // ... use bin
    ///         }
    ///     }
    ///     BinData::U16(bins) => { /* same pattern */ }
    /// }
    /// ```
    #[inline]
    pub fn get(&self, idx: usize) -> Option<u32> {
        match self {
            BinData::U8(data) => data.get(idx).map(|&b| b as u32),
            BinData::U16(data) => data.get(idx).map(|&b| b as u32),
        }
    }

    /// Returns the bin value at the given index without bounds checking.
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is undefined behavior.
    ///
    /// # Performance Note
    ///
    /// Despite being `unsafe`, this method still matches on the variant for every
    /// access. **Do not use in hot paths** like histogram building. Instead, match
    /// once on the variant using [`as_u8()`](Self::as_u8) or [`as_u16()`](Self::as_u16),
    /// then use `get_unchecked` on the raw slice:
    ///
    /// ```ignore
    /// match bin_data {
    ///     BinData::U8(bins) => {
    ///         for sample in samples {
    ///             let bin = unsafe { *bins.get_unchecked(sample) };
    ///             // ... use bin
    ///         }
    ///     }
    ///     BinData::U16(bins) => { /* same pattern */ }
    /// }
    /// ```
    #[inline]
    pub unsafe fn get_unchecked(&self, idx: usize) -> u32 {
        // SAFETY: Caller must ensure idx is within bounds
        unsafe {
            match self {
                BinData::U8(data) => *data.get_unchecked(idx) as u32,
                BinData::U16(data) => *data.get_unchecked(idx) as u32,
            }
        }
    }

    /// Returns the size in bytes of the stored data.
    #[inline]
    pub fn size_bytes(&self) -> usize {
        match self {
            BinData::U8(data) => data.len(),
            BinData::U16(data) => data.len() * 2,
        }
    }

    /// Returns the maximum number of bins this storage type can represent.
    ///
    /// - `U8`: 256 bins (values 0-255)
    /// - `U16`: 65536 bins (values 0-65535)
    #[inline]
    pub fn max_bins(&self) -> u32 {
        match self {
            BinData::U8(_) => 256,
            BinData::U16(_) => 65536,
        }
    }

    /// Returns `true` if the given number of bins requires U16 storage.
    ///
    /// U8 can store up to 256 bins, so any value > 256 requires U16.
    #[inline]
    pub fn needs_u16(n_bins: u32) -> bool {
        n_bins > 256
    }

    /// Returns a slice of U8 bins if this is U8 storage, or `None` otherwise.
    #[inline]
    pub fn as_u8(&self) -> Option<&[u8]> {
        match self {
            BinData::U8(data) => Some(data),
            BinData::U16(_) => None,
        }
    }

    /// Returns a slice of U16 bins if this is U16 storage, or `None` otherwise.
    #[inline]
    pub fn as_u16(&self) -> Option<&[u16]> {
        match self {
            BinData::U16(data) => Some(data),
            BinData::U8(_) => None,
        }
    }
}

impl Default for BinData {
    /// Returns an empty U8 bin data.
    fn default() -> Self {
        BinData::U8(Box::new([]))
    }
}

impl From<Vec<u8>> for BinData {
    fn from(data: Vec<u8>) -> Self {
        BinData::U8(data.into_boxed_slice())
    }
}

impl From<Vec<u16>> for BinData {
    fn from(data: Vec<u16>) -> Self {
        BinData::U16(data.into_boxed_slice())
    }
}

impl From<Box<[u8]>> for BinData {
    fn from(data: Box<[u8]>) -> Self {
        BinData::U8(data)
    }
}

impl From<Box<[u16]>> for BinData {
    fn from(data: Box<[u16]>) -> Self {
        BinData::U16(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_checks() {
        let u8_data = BinData::from(vec![1u8, 2, 3]);
        assert!(u8_data.is_u8());
        assert!(!u8_data.is_u16());

        let u16_data = BinData::from(vec![1u16, 2, 3]);
        assert!(u16_data.is_u16());
        assert!(!u16_data.is_u8());
    }

    #[test]
    fn test_len_and_empty() {
        let empty = BinData::default();
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);

        let u8_data = BinData::from(vec![1u8, 2, 3]);
        assert!(!u8_data.is_empty());
        assert_eq!(u8_data.len(), 3);

        let u16_data = BinData::from(vec![100u16, 200, 300, 400]);
        assert_eq!(u16_data.len(), 4);
    }

    #[test]
    fn test_get() {
        let u8_data = BinData::from(vec![10u8, 20, 30]);
        assert_eq!(u8_data.get(0), Some(10));
        assert_eq!(u8_data.get(1), Some(20));
        assert_eq!(u8_data.get(2), Some(30));
        assert_eq!(u8_data.get(3), None);

        let u16_data = BinData::from(vec![1000u16, 2000]);
        assert_eq!(u16_data.get(0), Some(1000));
        assert_eq!(u16_data.get(1), Some(2000));
        assert_eq!(u16_data.get(2), None);
    }

    #[test]
    fn test_get_unchecked() {
        let u8_data = BinData::from(vec![10u8, 20, 30]);
        unsafe {
            assert_eq!(u8_data.get_unchecked(0), 10);
            assert_eq!(u8_data.get_unchecked(2), 30);
        }

        let u16_data = BinData::from(vec![500u16, 1000]);
        unsafe {
            assert_eq!(u16_data.get_unchecked(0), 500);
            assert_eq!(u16_data.get_unchecked(1), 1000);
        }
    }

    #[test]
    fn test_size_bytes() {
        let u8_data = BinData::from(vec![1u8, 2, 3, 4, 5]);
        assert_eq!(u8_data.size_bytes(), 5);

        let u16_data = BinData::from(vec![1u16, 2, 3, 4, 5]);
        assert_eq!(u16_data.size_bytes(), 10);

        let empty = BinData::default();
        assert_eq!(empty.size_bytes(), 0);
    }

    #[test]
    fn test_max_bins() {
        let u8_data = BinData::from(vec![1u8]);
        assert_eq!(u8_data.max_bins(), 256);

        let u16_data = BinData::from(vec![1u16]);
        assert_eq!(u16_data.max_bins(), 65536);
    }

    #[test]
    fn test_needs_u16() {
        // Edge cases around 256
        assert!(!BinData::needs_u16(0));
        assert!(!BinData::needs_u16(1));
        assert!(!BinData::needs_u16(255));
        assert!(!BinData::needs_u16(256));
        assert!(BinData::needs_u16(257));
        assert!(BinData::needs_u16(1000));
        assert!(BinData::needs_u16(65536));
    }

    #[test]
    fn test_as_slices() {
        let u8_data = BinData::from(vec![1u8, 2, 3]);
        assert_eq!(u8_data.as_u8(), Some(&[1u8, 2, 3][..]));
        assert_eq!(u8_data.as_u16(), None);

        let u16_data = BinData::from(vec![100u16, 200]);
        assert_eq!(u16_data.as_u8(), None);
        assert_eq!(u16_data.as_u16(), Some(&[100u16, 200][..]));
    }

    #[test]
    fn test_default() {
        let default = BinData::default();
        assert!(default.is_u8());
        assert!(default.is_empty());
    }

    #[test]
    fn test_from_boxed_slice() {
        let boxed_u8: Box<[u8]> = vec![1u8, 2, 3].into_boxed_slice();
        let bin_data = BinData::from(boxed_u8);
        assert!(bin_data.is_u8());
        assert_eq!(bin_data.len(), 3);

        let boxed_u16: Box<[u16]> = vec![1u16, 2].into_boxed_slice();
        let bin_data = BinData::from(boxed_u16);
        assert!(bin_data.is_u16());
        assert_eq!(bin_data.len(), 2);
    }

    #[test]
    fn test_clone_and_eq() {
        let original = BinData::from(vec![1u8, 2, 3]);
        let cloned = original.clone();
        assert_eq!(original, cloned);

        let different = BinData::from(vec![1u8, 2, 4]);
        assert_ne!(original, different);

        let different_type = BinData::from(vec![1u16, 2, 3]);
        assert_ne!(original, different_type);
    }

    #[test]
    fn test_max_values() {
        // U8 can store values 0-255
        let u8_max = BinData::from(vec![0u8, 128, 255]);
        assert_eq!(u8_max.get(0), Some(0));
        assert_eq!(u8_max.get(1), Some(128));
        assert_eq!(u8_max.get(2), Some(255));

        // U16 can store values 0-65535
        let u16_max = BinData::from(vec![0u16, 256, 1000, 65535]);
        assert_eq!(u16_max.get(0), Some(0));
        assert_eq!(u16_max.get(1), Some(256));
        assert_eq!(u16_max.get(2), Some(1000));
        assert_eq!(u16_max.get(3), Some(65535));
    }
}

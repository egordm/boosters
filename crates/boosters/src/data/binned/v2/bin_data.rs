//! BinData - Typed bin storage replacing BinType.
//!
//! This type stores the actual bin values rather than just describing the type.
//! It replaces both `BinType` (type indicator) and the storage aspects of `BinStorage`.

/// Typed bin data storage.
///
/// This is the foundational storage type for binned features. It stores the actual
/// bin values in a contiguous boxed slice. The enum variants encode the bit width.
///
/// # Design
///
/// Unlike the old `BinType` which only indicated the type, `BinData` owns the data.
/// This makes the storage type self-contained and eliminates the need for separate
/// type tracking.
///
/// # Example
///
/// ```ignore
/// let bins = BinData::U8(vec![0, 1, 2, 0, 1].into_boxed_slice());
/// assert!(bins.is_u8());
/// assert_eq!(bins.len(), 5);
/// assert_eq!(bins.get(0), 0);
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

    /// Check if this is U8 storage.
    #[inline]
    pub fn is_u8(&self) -> bool {
        matches!(self, Self::U8(_))
    }

    /// Check if this is U16 storage.
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

    /// Get bin value at index (returns as u32 for uniformity).
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
    /// Caller must ensure `idx < self.len()`.
    #[inline]
    pub unsafe fn get_unchecked(&self, idx: usize) -> u32 {
        match self {
            Self::U8(data) => *data.get_unchecked(idx) as u32,
            Self::U16(data) => *data.get_unchecked(idx) as u32,
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

    /// Maximum number of bins this storage type supports.
    #[inline]
    pub fn max_bins(&self) -> u32 {
        match self {
            Self::U8(_) => 256,
            Self::U16(_) => 65536,
        }
    }

    /// Select appropriate bin type for a given max bin count.
    ///
    /// Returns `true` if U8 is sufficient, `false` if U16 is needed.
    /// Panics if max_bins > 65536.
    #[inline]
    pub fn needs_u16(max_bins: u32) -> bool {
        assert!(max_bins <= 65536, "max_bins exceeds u16 capacity");
        max_bins > 256
    }

    /// Get underlying u8 slice if this is U8 storage.
    #[inline]
    pub fn as_u8(&self) -> Option<&[u8]> {
        match self {
            Self::U8(data) => Some(data),
            Self::U16(_) => None,
        }
    }

    /// Get underlying u16 slice if this is U16 storage.
    #[inline]
    pub fn as_u16(&self) -> Option<&[u16]> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bin_data_u8() {
        let bins = BinData::from_u8(vec![0, 1, 2, 255]);
        assert!(bins.is_u8());
        assert!(!bins.is_u16());
        assert_eq!(bins.len(), 4);
        assert_eq!(bins.get(0), 0);
        assert_eq!(bins.get(3), 255);
        assert_eq!(bins.size_bytes(), 4);
        assert_eq!(bins.max_bins(), 256);
    }

    #[test]
    fn test_bin_data_u16() {
        let bins = BinData::from_u16(vec![0, 256, 1000, 65535]);
        assert!(!bins.is_u8());
        assert!(bins.is_u16());
        assert_eq!(bins.len(), 4);
        assert_eq!(bins.get(1), 256);
        assert_eq!(bins.get(3), 65535);
        assert_eq!(bins.size_bytes(), 8);
        assert_eq!(bins.max_bins(), 65536);
    }

    #[test]
    fn test_needs_u16() {
        assert!(!BinData::needs_u16(100));
        assert!(!BinData::needs_u16(256));
        assert!(BinData::needs_u16(257));
        assert!(BinData::needs_u16(65536));
    }

    #[test]
    fn test_as_slice() {
        let u8_bins = BinData::from_u8(vec![1, 2, 3]);
        assert_eq!(u8_bins.as_u8(), Some(&[1u8, 2, 3][..]));
        assert_eq!(u8_bins.as_u16(), None);

        let u16_bins = BinData::from_u16(vec![1, 2, 3]);
        assert_eq!(u16_bins.as_u8(), None);
        assert_eq!(u16_bins.as_u16(), Some(&[1u16, 2, 3][..]));
    }

    #[test]
    fn test_get_unchecked() {
        let bins = BinData::from_u8(vec![10, 20, 30]);
        unsafe {
            assert_eq!(bins.get_unchecked(0), 10);
            assert_eq!(bins.get_unchecked(2), 30);
        }
    }
}

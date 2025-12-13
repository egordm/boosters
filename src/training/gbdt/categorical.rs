//! Categorical feature utilities.
//!
//! This module provides data structures for handling categorical features
//! in gradient boosted trees.

/// Compact bitset for categorical membership (up to 64 categories inline).
///
/// For splits on categorical features, this tracks which categories go left.
/// Categories beyond 64 use heap-allocated overflow storage.
#[derive(Clone, Debug, Default)]
pub struct CatBitset {
    /// Inline bits for categories 0..63.
    bits: u64,
    /// Heap storage for categories 64+.
    overflow: Option<Box<[u64]>>,
}

impl CatBitset {
    /// Create an empty bitset.
    #[inline]
    pub fn empty() -> Self {
        Self::default()
    }

    /// Create a bitset with a single category.
    #[inline]
    pub fn singleton(cat: u32) -> Self {
        let mut s = Self::empty();
        s.insert(cat);
        s
    }

    /// Check if a category is in the set.
    #[inline]
    pub fn contains(&self, cat: u32) -> bool {
        if cat < 64 {
            (self.bits >> cat) & 1 != 0
        } else {
            let idx = ((cat - 64) / 64) as usize;
            let bit = (cat - 64) % 64;
            self.overflow
                .as_ref()
                .and_then(|o| o.get(idx))
                .map_or(false, |&w| (w >> bit) & 1 != 0)
        }
    }

    /// Insert a category into the set.
    pub fn insert(&mut self, cat: u32) {
        if cat < 64 {
            self.bits |= 1u64 << cat;
        } else {
            let idx = ((cat - 64) / 64) as usize;
            let bit = (cat - 64) % 64;

            let overflow = self.overflow.get_or_insert_with(|| vec![0u64; idx + 1].into_boxed_slice());

            // Grow if needed
            if idx >= overflow.len() {
                let mut new_overflow = vec![0u64; idx + 1];
                new_overflow[..overflow.len()].copy_from_slice(overflow);
                *overflow = new_overflow.into_boxed_slice();
            }

            overflow[idx] |= 1u64 << bit;
        }
    }

    /// Number of categories in the set.
    pub fn count(&self) -> u32 {
        let mut count = self.bits.count_ones();
        if let Some(ref overflow) = self.overflow {
            for word in overflow.iter() {
                count += word.count_ones();
            }
        }
        count
    }

    /// Check if the bitset is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.bits == 0 && self.overflow.as_ref().map_or(true, |o| o.iter().all(|&w| w == 0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cat_bitset_inline() {
        let mut bs = CatBitset::empty();
        assert!(!bs.contains(0));
        assert!(bs.is_empty());

        bs.insert(0);
        bs.insert(5);
        bs.insert(63);

        assert!(bs.contains(0));
        assert!(bs.contains(5));
        assert!(bs.contains(63));
        assert!(!bs.contains(1));
        assert!(!bs.contains(64));
        assert_eq!(bs.count(), 3);
    }

    #[test]
    fn test_cat_bitset_overflow() {
        let mut bs = CatBitset::empty();
        bs.insert(100);
        bs.insert(200);

        assert!(bs.contains(100));
        assert!(bs.contains(200));
        assert!(!bs.contains(0));
        assert!(!bs.contains(99));
        assert_eq!(bs.count(), 2);
    }

    #[test]
    fn test_cat_bitset_singleton() {
        let bs = CatBitset::singleton(42);
        assert!(bs.contains(42));
        assert!(!bs.contains(0));
        assert_eq!(bs.count(), 1);
    }
}

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
                .is_some_and(|&w| (w >> bit) & 1 != 0)
        }
    }

    /// Insert a category into the set.
    pub fn insert(&mut self, cat: u32) {
        if cat < 64 {
            self.bits |= 1u64 << cat;
        } else {
            let idx = ((cat - 64) / 64) as usize;
            let bit = (cat - 64) % 64;

            let overflow = self
                .overflow
                .get_or_insert_with(|| vec![0u64; idx + 1].into_boxed_slice());

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
        self.bits == 0
            && self
                .overflow
                .as_ref()
                .is_none_or(|o| o.iter().all(|&w| w == 0))
    }

    /// Iterate over all categories contained in the set.
    ///
    /// Categories are returned in ascending order.
    pub fn iter(&self) -> CatBitsetIter<'_> {
        CatBitsetIter {
            inline_remaining: self.bits,
            overflow: self.overflow.as_deref().unwrap_or(&[]),
            overflow_word_idx: 0,
            overflow_word_remaining: 0,
        }
    }
}

/// Iterator over categories contained in a [`CatBitset`].
#[derive(Clone, Debug)]
pub struct CatBitsetIter<'a> {
    inline_remaining: u64,
    overflow: &'a [u64],
    overflow_word_idx: usize,
    overflow_word_remaining: u64,
}

impl<'a> Iterator for CatBitsetIter<'a> {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.inline_remaining != 0 {
            let tz = self.inline_remaining.trailing_zeros();
            let cat = tz;
            self.inline_remaining &= self.inline_remaining - 1;
            return Some(cat);
        }

        loop {
            if self.overflow_word_remaining != 0 {
                let tz = self.overflow_word_remaining.trailing_zeros();
                // overflow_word_idx was already incremented AFTER loading this word,
                // so subtract 1 for the category calculation.
                let cat = 64u32 + (self.overflow_word_idx as u32 - 1) * 64u32 + tz;
                self.overflow_word_remaining &= self.overflow_word_remaining - 1;
                return Some(cat);
            }

            if self.overflow_word_idx >= self.overflow.len() {
                return None;
            }

            self.overflow_word_remaining = self.overflow[self.overflow_word_idx];
            self.overflow_word_idx += 1;
        }
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

    #[test]
    fn iter_yields_categories_in_order() {
        let mut bs = CatBitset::empty();
        bs.insert(5);
        bs.insert(0);
        bs.insert(63);
        bs.insert(64);
        bs.insert(130);

        let cats: Vec<u32> = bs.iter().collect();
        assert_eq!(cats, vec![0, 5, 63, 64, 130]);
    }
}

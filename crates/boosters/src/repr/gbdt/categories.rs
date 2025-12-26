//! Categorical split storage for tree nodes.
//!
//! XGBoost stores categorical splits as bitsets indicating which categories
//! go right (the "chosen" categories). Categories NOT in the set go left.
//!
//! This module provides efficient packed bitset storage that matches
//! XGBoost's format while being cache-friendly for traversal.

use super::NodeId;

/// Storage for categorical split bitsets in a tree.
///
/// Stores category sets as packed u32 bitsets, where each bit represents
/// whether a category value should go RIGHT during tree traversal.
///
/// # Format
///
/// - `categories`: flat array of u32 bitset words for all nodes
/// - `segments`: per-node `(start_index, size)` into categories array
///
/// # Decision Rule
///
/// For a categorical split on a node with feature value `c`:
/// - If bit `c` is SET in the bitset → go RIGHT
/// - If bit `c` is NOT set → go LEFT
/// - If feature value is NaN → use default direction
///
/// This matches XGBoost's partition-based categorical split behavior.
#[derive(Debug, Clone, Default)]
pub struct CategoriesStorage {
    /// Flat array of bitset words (32 categories per word).
    categories: Box<[u32]>,
    /// Per-node segment: `(start_index, size)` into categories array.
    /// Indexed by node_idx. Nodes without categorical splits have `(0, 0)`.
    segments: Box<[(u32, u32)]>,
}

use std::sync::LazyLock;

/// Static empty CategoriesStorage for returning references.
static EMPTY_CATEGORIES: LazyLock<CategoriesStorage> = LazyLock::new(CategoriesStorage::empty);

impl CategoriesStorage {
    /// Get a static reference to an empty storage.
    ///
    /// Useful for returning `&CategoriesStorage` when no categories exist.
    #[inline]
    pub fn empty_ref() -> &'static Self {
        &EMPTY_CATEGORIES
    }

    /// Create empty categories storage.
    #[inline]
    pub fn empty() -> Self {
        Self::default()
    }

    /// Create categories storage from raw data.
    ///
    /// # Arguments
    ///
    /// * `categories` - Flat bitset data for all categorical nodes
    /// * `segments` - Per-node `(start, size)` into categories. Length must equal num_nodes.
    pub fn new(categories: Vec<u32>, segments: Vec<(u32, u32)>) -> Self {
        Self {
            categories: categories.into_boxed_slice(),
            segments: segments.into_boxed_slice(),
        }
    }

    /// Check if a category is in the "right" set for a given node.
    ///
    /// Returns `true` if the category is in the set (should go right),
    /// `false` if not in the set (should go left).
    #[inline]
    pub fn category_goes_right(&self, node_idx: NodeId, category: u32) -> bool {
        let (start, size) = self.segments[node_idx as usize];
        if size == 0 {
            return false;
        }

        let word_idx = category >> 5;
        let bit_idx = category & 31;

        if word_idx >= size {
            return false;
        }

        let word = self.categories[(start + word_idx) as usize];
        (word >> bit_idx) & 1 != 0
    }

    /// Whether this storage has any categorical data.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.categories.is_empty()
    }

    /// Get the segments array.
    #[inline]
    pub fn segments(&self) -> &[(u32, u32)] {
        &self.segments
    }

    /// Get the raw bitsets array.
    #[inline]
    pub fn bitsets(&self) -> &[u32] {
        &self.categories
    }

    /// Get the bitset slice for a specific node (for testing/debugging).
    #[inline]
    pub fn bitset_for_node(&self, node_idx: NodeId) -> &[u32] {
        let (start, size) = self.segments[node_idx as usize];
        &self.categories[start as usize..(start + size) as usize]
    }
}

/// Convert a feature value (f32) to a category index (u32).
///
/// XGBoost stores categorical features as f32 values representing integer
/// category indices. This function converts them back to integers.
#[inline]
pub fn float_to_category(value: f32) -> u32 {
    debug_assert!(
        !value.is_nan(),
        "NaN should be handled as missing value before category conversion"
    );
    debug_assert!(
        value >= 0.0,
        "Categorical feature value must be non-negative, got {}",
        value
    );
    debug_assert!(
        value == value.trunc(),
        "Categorical feature value must be an integer, got {}",
        value
    );
    value as u32
}

/// Build a packed u32 bitset from a list of category values.
///
/// Sets bit `c` for each category value `c` in the input.
pub fn categories_to_bitset(categories: &[u32]) -> Vec<u32> {
    if categories.is_empty() {
        return vec![];
    }

    let max_cat = categories.iter().copied().max().unwrap_or(0);
    let num_words = ((max_cat >> 5) + 1) as usize;
    let mut bitset = vec![0u32; num_words];

    for &cat in categories {
        let word_idx = (cat >> 5) as usize;
        let bit_idx = cat & 31;
        bitset[word_idx] |= 1u32 << bit_idx;
    }

    bitset
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_storage() {
        let storage = CategoriesStorage::empty();
        assert!(storage.is_empty());
    }

    #[test]
    fn category_goes_right_basic() {
        let categories = vec![0b1010u32];
        let segments = vec![(0, 1)];
        let storage = CategoriesStorage::new(categories, segments);

        assert!(storage.category_goes_right(0, 1));
        assert!(storage.category_goes_right(0, 3));
        assert!(!storage.category_goes_right(0, 0));
        assert!(!storage.category_goes_right(0, 2));
    }

    #[test]
    fn category_beyond_bitset() {
        let categories = vec![0b1010u32];
        let segments = vec![(0, 1)];
        let storage = CategoriesStorage::new(categories, segments);

        assert!(!storage.category_goes_right(0, 64));
    }

    #[test]
    fn categories_to_bitset_single_word() {
        let bitset = categories_to_bitset(&[1, 3, 5]);
        assert_eq!(bitset, vec![0b00101010]);
    }

    #[test]
    fn categories_to_bitset_multi_word() {
        let bitset = categories_to_bitset(&[1, 33]);
        assert_eq!(bitset, vec![0b10, 0b10]);
    }

    #[test]
    fn categories_to_bitset_empty() {
        let bitset = categories_to_bitset(&[]);
        assert!(bitset.is_empty());
    }

    #[test]
    fn float_to_category_validation() {
        assert_eq!(float_to_category(0.0), 0);
        assert_eq!(float_to_category(5.0), 5);
    }
}

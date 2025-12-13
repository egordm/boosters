//! Categorical split storage for tree nodes.
//!
//! XGBoost stores categorical splits as bitsets indicating which categories
//! go right (the "chosen" categories). Categories NOT in the set go left.
//!
//! This module provides efficient packed bitset storage that matches
//! XGBoost's format while being cache-friendly for inference.

// =============================================================================
// CategoriesStorage
// =============================================================================

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

impl CategoriesStorage {
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
    ///
    /// # Arguments
    ///
    /// * `node_idx` - The tree node index
    /// * `category` - The category value (as integer)
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `category` exceeds `u32::MAX`.
    #[inline]
    pub fn category_goes_right(&self, node_idx: u32, category: u32) -> bool {
        let (start, size) = self.segments[node_idx as usize];
        if size == 0 {
            // No categories stored - not a categorical node or all go left
            return false;
        }

        // Determine which word contains this category:
        //   word_idx = category / 32
        // Using bit shift: >> 5 is equivalent to / 32 (since 32 = 2^5)
        let word_idx = category >> 5;

        // Determine which bit within the word:
        //   bit_idx = category % 32
        // Using bitwise AND: & 31 is equivalent to % 32 (since 31 = 0b11111)
        let bit_idx = category & 31;

        // Check if within bounds
        if word_idx >= size {
            // Category is beyond stored bitset - treat as not in set (go left)
            return false;
        }

        // Extract the bit: shift the word right by bit_idx, then mask to get LSB
        let word = self.categories[(start + word_idx) as usize];
        (word >> bit_idx) & 1 != 0
    }

    /// Whether this storage has any categorical data.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.categories.is_empty()
    }

    /// Get the segments array (for copying to array layout).
    #[inline]
    pub fn segments(&self) -> &[(u32, u32)] {
        &self.segments
    }

    /// Get the raw bitsets array (for copying to array layout).
    #[inline]
    pub fn bitsets(&self) -> &[u32] {
        &self.categories
    }

    /// Get the bitset slice for a specific node (for testing/debugging).
    #[inline]
    pub fn bitset_for_node(&self, node_idx: u32) -> &[u32] {
        let (start, size) = self.segments[node_idx as usize];
        &self.categories[start as usize..(start + size) as usize]
    }
}

// =============================================================================
// Bitset Builder Utilities
// =============================================================================

/// Convert a feature value (f32) to a category index (u32).
///
/// XGBoost stores categorical features as f32 values representing integer
/// category indices. This function converts them back to integers.
///
/// # Panics
///
/// In debug builds, panics if:
/// - `value` is NaN (should be handled by missing value logic before this)
/// - `value` is negative
/// - `value` is not a whole number
#[inline]
pub fn float_to_category(value: f32) -> u32 {
    debug_assert!(
        !value.is_nan(),
        "NaN should be handled as missing value before category conversion"
    );
    debug_assert!(
        value >= 0.0,
        "Categorical feature value must be non-negative, got {value}"
    );
    debug_assert!(
        value == value.trunc(),
        "Categorical feature value must be an integer, got {value}"
    );
    value as u32
}

/// Build a packed u32 bitset from a list of category values.
///
/// Sets bit `c` for each category value `c` in the input.
///
/// # Bitset Layout
///
/// Categories are packed into u32 words, 32 categories per word:
/// - Categories 0-31 are stored in word 0
/// - Categories 32-63 are stored in word 1
/// - And so on...
///
/// Within each word, bit `i` represents category `word_index * 32 + i`:
/// ```text
/// Word 0: [cat 31] [cat 30] ... [cat 1] [cat 0]   (LSB = cat 0)
/// Word 1: [cat 63] [cat 62] ... [cat 33] [cat 32]
/// ```
///
/// # Example
///
/// ```ignore
/// use booste_rs::inference::gbdt::categories::categories_to_bitset;
///
/// // Categories {1, 3, 5} → bits 1, 3, 5 set
/// let bitset = categories_to_bitset(&[1, 3, 5]);
/// assert_eq!(bitset, vec![0b00101010]); // bits 1, 3, 5
/// ```
pub fn categories_to_bitset(categories: &[u32]) -> Vec<u32> {
    if categories.is_empty() {
        return vec![];
    }

    // Find the maximum category to determine how many u32 words we need.
    // Each word stores 32 categories, so we need ceil((max_cat + 1) / 32) words.
    let max_cat = categories.iter().copied().max().unwrap_or(0);
    let num_words = ((max_cat >> 5) + 1) as usize; // max_cat / 32 + 1
    let mut bitset = vec![0u32; num_words];

    for &cat in categories {
        // Determine which word this category belongs to:
        //   word_idx = cat / 32
        // Using bit shift: >> 5 is equivalent to / 32 (since 32 = 2^5)
        let word_idx = (cat >> 5) as usize;

        // Determine which bit within the word:
        //   bit_idx = cat % 32
        // Using bitwise AND: & 31 is equivalent to % 32 (since 31 = 0b11111)
        let bit_idx = cat & 31;

        // Set the bit: shift 1 to the correct position and OR it in
        bitset[word_idx] |= 1u32 << bit_idx;
    }

    bitset
}

// =============================================================================
// Tests
// =============================================================================

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
        // Node 0 has categories {1, 3} (bits 1 and 3 set = 0b1010)
        let categories = vec![0b1010u32];
        let segments = vec![(0, 1)];
        let storage = CategoriesStorage::new(categories, segments);

        // Category 1 is in set → go right
        assert!(storage.category_goes_right(0, 1));
        // Category 3 is in set → go right
        assert!(storage.category_goes_right(0, 3));
        // Category 0 is NOT in set → go left
        assert!(!storage.category_goes_right(0, 0));
        // Category 2 is NOT in set → go left
        assert!(!storage.category_goes_right(0, 2));
    }

    #[test]
    fn category_beyond_bitset() {
        let categories = vec![0b1010u32]; // Only 1 word (categories 0-31)
        let segments = vec![(0, 1)];
        let storage = CategoriesStorage::new(categories, segments);

        // Category 100 is beyond stored bitset → go left
        assert!(!storage.category_goes_right(0, 100));
    }

    #[test]
    fn multi_word_bitset() {
        // Categories {35, 64}: word 1 bit 3, word 2 bit 0
        let categories = vec![0u32, 0b1000u32, 0b1u32];
        let segments = vec![(0, 3)];
        let storage = CategoriesStorage::new(categories, segments);

        assert!(storage.category_goes_right(0, 35)); // word 1, bit 3
        assert!(storage.category_goes_right(0, 64)); // word 2, bit 0
        assert!(!storage.category_goes_right(0, 0));
        assert!(!storage.category_goes_right(0, 32));
    }

    #[test]
    fn multiple_nodes() {
        // Node 0: categories {0, 1} at offset 0, size 1
        // Node 1: no categorical (offset 0, size 0)
        // Node 2: categories {2} at offset 1, size 1
        let categories = vec![0b11u32, 0b100u32];
        let segments = vec![(0, 1), (0, 0), (1, 1)];
        let storage = CategoriesStorage::new(categories, segments);

        // Node 0
        assert!(storage.category_goes_right(0, 0));
        assert!(storage.category_goes_right(0, 1));
        assert!(!storage.category_goes_right(0, 2));

        // Node 1 (not categorical) - always returns false
        assert!(!storage.category_goes_right(1, 0));
        assert!(!storage.category_goes_right(1, 1));

        // Node 2
        assert!(!storage.category_goes_right(2, 0));
        assert!(!storage.category_goes_right(2, 1));
        assert!(storage.category_goes_right(2, 2));
    }

    #[test]
    fn categories_to_bitset_empty() {
        let bitset = categories_to_bitset(&[]);
        assert!(bitset.is_empty());
    }

    #[test]
    fn categories_to_bitset_single_word() {
        let bitset = categories_to_bitset(&[0, 1, 3, 7]);
        assert_eq!(bitset, vec![0b10001011]); // bits 0, 1, 3, 7
    }

    #[test]
    fn categories_to_bitset_multi_word() {
        let bitset = categories_to_bitset(&[0, 35, 64]);
        assert_eq!(bitset.len(), 3);
        assert_eq!(bitset[0], 0b1); // bit 0
        assert_eq!(bitset[1], 0b1000); // bit 3 (35 % 32 = 3)
        assert_eq!(bitset[2], 0b1); // bit 0 (64 % 32 = 0)
    }
}

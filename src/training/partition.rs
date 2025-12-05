//! Row partitioning for tree building.
//!
//! This module implements RFC-0014: tracking and updating row-to-node assignments.
//!
//! # Overview
//!
//! During tree building, we need to know which rows belong to each node:
//! - **Histogram building**: Only aggregate gradients for rows in the current node
//! - **Split application**: After finding a split, reassign rows to children
//! - **Leaf prediction**: Final node assignment determines predictions
//!
//! # Position List Representation
//!
//! Uses contiguous position lists (like XGBoost):
//! ```text
//! positions: [row_ids for node 0...][row_ids for node 1...][row_ids for node 2...]
//!             ↑                      ↑                      ↑
//! node_bounds: [(0,n0),             (n0,n0+n1),            (n0+n1,total)]
//! ```
//!
//! This enables O(1) access to a node's rows as a contiguous slice.
//!
//! # Example
//!
//! ```ignore
//! use booste_rs::training::partition::RowPartitioner;
//!
//! let mut partitioner = RowPartitioner::new(1000);
//!
//! // All rows start in root (node 0)
//! let root_rows = partitioner.node_rows(0);
//! assert_eq!(root_rows.len(), 1000);
//!
//! // After a split
//! let (left, right) = partitioner.apply_split(0, &split_info, &quantized);
//! let left_rows = partitioner.node_rows(left);
//! let right_rows = partitioner.node_rows(right);
//! assert_eq!(left_rows.len() + right_rows.len(), 1000);
//! ```
//!
//! See RFC-0014 for design rationale.

use super::quantize::{BinCuts, BinIndex, QuantizedMatrix};
use super::split::SplitInfo;

// ============================================================================
// RowPartitioner
// ============================================================================

/// Row partitioner using contiguous position lists.
///
/// Each node has a contiguous slice of row indices in the `positions` array.
/// After a split, rows are partitioned in-place using Dutch flag algorithm.
///
/// # Memory Usage
///
/// - `positions`: ~4 bytes per row
/// - `node_bounds`: ~8 bytes per node (start, end pair)
/// - For 1M rows, 1000 nodes: ~4 MB + 8 KB = ~4 MB
///
/// # Performance
///
/// - `node_rows()`: O(1) - just returns a slice
/// - `apply_split()`: O(rows_in_node) - in-place partition
#[derive(Debug, Clone)]
pub struct RowPartitioner {
    /// Row indices, grouped by node.
    /// `positions[bounds.0..bounds.1]` = rows in node n
    positions: Vec<u32>,

    /// (start, end) bounds for each node's rows.
    /// Length: num_nodes
    node_bounds: Vec<(u32, u32)>,

    /// Number of nodes created (including split parents)
    num_nodes: usize,
}

impl RowPartitioner {
    /// Create a partitioner with all rows in the root node.
    ///
    /// # Arguments
    ///
    /// * `num_rows` - Total number of rows in the dataset
    pub fn new(num_rows: u32) -> Self {
        Self {
            positions: (0..num_rows).collect(),
            node_bounds: vec![(0, num_rows)],
            num_nodes: 1,
        }
    }

    /// Create a partitioner with a subset of rows.
    ///
    /// Useful for sampling (e.g., GOSS, random sampling).
    pub fn with_rows(rows: Vec<u32>) -> Self {
        let num_rows = rows.len() as u32;
        Self {
            positions: rows,
            node_bounds: vec![(0, num_rows)],
            num_nodes: 1,
        }
    }

    /// Number of active nodes.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Get row indices for a node.
    ///
    /// Returns a contiguous slice of row indices belonging to the node.
    /// This is efficient for histogram building.
    #[inline]
    pub fn node_rows(&self, node: u32) -> &[u32] {
        let (start, end) = self.node_bounds[node as usize];
        &self.positions[start as usize..end as usize]
    }

    /// Number of rows in a node.
    #[inline]
    pub fn node_size(&self, node: u32) -> u32 {
        let (start, end) = self.node_bounds[node as usize];
        end - start
    }

    /// Check if a node is empty.
    #[inline]
    pub fn is_empty(&self, node: u32) -> bool {
        self.node_size(node) == 0
    }

    /// Apply a split: partition rows into left and right children.
    ///
    /// Performs in-place Dutch flag partition based on the split decision.
    /// Creates two new node entries for the children.
    ///
    /// # Arguments
    ///
    /// * `node` - Node to split
    /// * `split` - Split information (feature, threshold, default direction)
    /// * `quantized` - Quantized feature matrix for evaluating split condition
    ///
    /// # Returns
    ///
    /// `(left_node_id, right_node_id)` - IDs of the newly created child nodes
    pub fn apply_split<B: BinIndex>(
        &mut self,
        node: u32,
        split: &SplitInfo,
        quantized: &QuantizedMatrix<B>,
    ) -> (u32, u32) {
        let (start, end) = self.node_bounds[node as usize];
        let start = start as usize;
        let end = end as usize;
        let rows = &mut self.positions[start..end];

        // Partition rows: left rows first, then right rows
        let mid = if split.is_categorical {
            partition_categorical(rows, split, quantized)
        } else {
            partition_numerical(rows, split, quantized)
        };

        // Create two new node entries
        let left_node = self.num_nodes as u32;
        let right_node = left_node + 1;

        // Add bounds for new nodes
        // Left: start..start+mid
        // Right: start+mid..end
        self.node_bounds.push((start as u32, (start + mid) as u32));
        self.node_bounds.push(((start + mid) as u32, end as u32));

        self.num_nodes += 2;

        (left_node, right_node)
    }

    /// Get the total number of rows being tracked.
    #[inline]
    pub fn total_rows(&self) -> usize {
        self.positions.len()
    }

    /// Reset to initial state (all rows in root).
    pub fn reset(&mut self) {
        let num_rows = self.positions.len() as u32;
        self.positions = (0..num_rows).collect();
        self.node_bounds = vec![(0, num_rows)];
        self.num_nodes = 1;
    }
}

// ============================================================================
// Partition functions (standalone for borrow checker)
// ============================================================================

/// Partition rows for a numerical split using Dutch flag algorithm.
///
/// Rows with bin <= split_bin go left.
/// Missing values (bin 0) go according to `default_left`.
///
/// Returns the number of rows going left.
fn partition_numerical<B: BinIndex>(
    rows: &mut [u32],
    split: &SplitInfo,
    quantized: &QuantizedMatrix<B>,
) -> usize {
    let feature = split.feature;
    let split_bin = split.split_bin;
    let default_left = split.default_left;

    // Dutch National Flag partition algorithm:
    // - Maintains invariant: [0..left) are "left" rows, [right..len) are "right" rows
    // - Scans from left, swapping "right" elements to the end
    // - Terminates when left == right (all elements classified)
    let mut left = 0;
    let mut right = rows.len();

    while left < right {
        let row = rows[left];
        let bin = quantized.get(row, feature);

        // Bin 0 = missing value (NaN was mapped to bin 0 during quantization)
        let goes_left = if bin.to_usize() == 0 {
            // Missing value - use learned default direction
            default_left
        } else {
            // Non-missing: bin <= split_bin goes left
            bin.to_usize() <= split_bin as usize
        };

        if goes_left {
            // Element belongs on left, advance left pointer
            left += 1;
        } else {
            // Element belongs on right, swap to end and shrink right region
            right -= 1;
            rows.swap(left, right);
        }
    }

    left
}

/// Partition rows for a categorical split using bitset lookup.
///
/// Rows with category in `categories_left` go left.
/// Missing values (bin 0) go according to `default_left`.
///
/// Returns the number of rows going left.
fn partition_categorical<B: BinIndex>(
    rows: &mut [u32],
    split: &SplitInfo,
    quantized: &QuantizedMatrix<B>,
) -> usize {
    let feature = split.feature;
    let default_left = split.default_left;

    // Build inline bitset for O(1) category membership test.
    // Uses 4 × 64-bit words = 256 bits, supporting up to 256 categories.
    //
    // For category `cat`:
    //   - word index = cat / 64 (which 64-bit chunk)
    //   - bit position = cat % 64 (which bit within that chunk)
    //
    // These divisions can be optimized to shifts: cat >> 6 and cat & 63
    // but the compiler does this automatically for power-of-2 divisors.
    let mut left_cats = [0u64; 4];
    for &cat in &split.categories_left {
        let word = (cat / 64) as usize; // Which 64-bit word (0-3)
        let bit = cat % 64;             // Which bit within word (0-63)
        if word < 4 {
            left_cats[word] |= 1u64 << bit;
        }
    }

    // Dutch National Flag partition (same as numerical, but using bitset lookup)
    let mut left = 0;
    let mut right = rows.len();

    while left < right {
        let row = rows[left];
        let bin = quantized.get(row, feature).to_usize();

        // Bin 0 = missing value (NaN was mapped to bin 0 during quantization)
        let goes_left = if bin == 0 {
            // Missing value
            default_left
        } else {
            // Check if category is in left set using bitset lookup
            let word = bin / 64;  // Which 64-bit word
            let bit = bin % 64;   // Which bit position
            // Extract single bit: shift right then mask with 1
            word < 4 && (left_cats[word] >> bit) & 1 == 1
        };

        if goes_left {
            left += 1;
        } else {
            right -= 1;
            rows.swap(left, right);
        }
    }

    left
}

// ============================================================================
// Utility functions
// ============================================================================

/// Find the bin index corresponding to a threshold value.
///
/// Uses binary search on the bin cuts.
/// Returns the bin index where `value <= threshold` would be true.
pub fn find_threshold_bin(cuts: &BinCuts, feature: u32, threshold: f32) -> u32 {
    let feature_cuts = cuts.feature_cuts(feature);

    // Binary search for threshold position
    match feature_cuts.binary_search_by(|cut| {
        cut.partial_cmp(&threshold).unwrap_or(std::cmp::Ordering::Equal)
    }) {
        Ok(idx) => idx as u32 + 1, // Exact match: values up to and including this bin go left
        Err(idx) => idx as u32,    // Not found: idx is where it would be inserted
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_partitioner() {
        let part = RowPartitioner::new(10);

        assert_eq!(part.num_nodes(), 1);
        assert_eq!(part.node_size(0), 10);
        assert_eq!(part.total_rows(), 10);

        let rows = part.node_rows(0);
        assert_eq!(rows.len(), 10);
        assert_eq!(rows, &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_with_rows() {
        let part = RowPartitioner::with_rows(vec![0, 2, 4, 6, 8]);

        assert_eq!(part.num_nodes(), 1);
        assert_eq!(part.node_size(0), 5);
        assert_eq!(part.node_rows(0), &[0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_is_empty() {
        let part = RowPartitioner::new(10);
        assert!(!part.is_empty(0));

        let empty = RowPartitioner::with_rows(vec![]);
        assert!(empty.is_empty(0));
    }

    #[test]
    fn test_reset() {
        let mut part = RowPartitioner::new(5);
        // Modify state (simulate splits)
        part.positions = vec![4, 3, 2, 1, 0]; // Scrambled
        part.node_bounds = vec![(0, 2), (2, 5)]; // Two nodes
        part.num_nodes = 2;

        part.reset();

        assert_eq!(part.num_nodes(), 1);
        assert_eq!(part.node_rows(0), &[0, 1, 2, 3, 4]);
    }

    mod integration {
        use super::*;
        use crate::data::ColMatrix;
        use crate::training::quantize::{ExactQuantileCuts, Quantizer};
        use crate::training::split::SplitInfo;

        fn make_test_quantized() -> (QuantizedMatrix<u8>, crate::training::quantize::BinCuts) {
            // 10 rows, 2 features
            // Feature 0: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
            // Feature 1: 5, 5, 5, 5, 5, 0, 0, 0, 0, 0 (two groups)
            let data: Vec<f32> = vec![
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, // feat 0
                5.0, 5.0, 5.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, // feat 1
            ];
            let matrix = ColMatrix::from_vec(data, 10, 2);

            let quantizer = Quantizer::from_data(&matrix, &ExactQuantileCuts::default(), 256);
            let quantized = quantizer.quantize_u8(&matrix);
            let cuts = (*quantizer.cuts()).clone();

            (quantized, cuts)
        }

        #[test]
        fn test_apply_split_basic() {
            let (quantized, _cuts) = make_test_quantized();
            let mut part = RowPartitioner::new(10);

            // Create a split on feature 0, split_bin=4 (values 0-4 go left, 5-9 go right)
            let mut split = SplitInfo::none();
            split.feature = 0;
            split.split_bin = 5; // Rows with bin <= 5 go left (approximately values 0-4)
            split.default_left = true;
            split.is_categorical = false;

            let (left, right) = part.apply_split(0, &split, &quantized);

            // Should create two new nodes
            assert_eq!(part.num_nodes(), 3);
            assert_eq!(left, 1);
            assert_eq!(right, 2);

            // Original node should still exist but now children have the rows
            let left_rows = part.node_rows(left);
            let right_rows = part.node_rows(right);

            // Total rows should be preserved
            assert_eq!(left_rows.len() + right_rows.len(), 10);

            // No duplicates
            let mut all_rows: Vec<u32> = left_rows.iter().chain(right_rows.iter()).cloned().collect();
            all_rows.sort();
            assert_eq!(all_rows, (0..10).collect::<Vec<_>>());
        }

        #[test]
        fn test_apply_split_missing_default_left() {
            // Create data with missing values
            let data: Vec<f32> = vec![
                f32::NAN, 1.0, 2.0, f32::NAN, 4.0, 5.0, 6.0, f32::NAN, 8.0, 9.0,
            ];
            let matrix = ColMatrix::from_vec(data, 10, 1);

            let quantizer = Quantizer::from_data(&matrix, &ExactQuantileCuts::default(), 256);
            let quantized = quantizer.quantize_u8(&matrix);

            let mut part = RowPartitioner::new(10);

            // Split: missing goes left
            let mut split = SplitInfo::none();
            split.feature = 0;
            split.split_bin = 4;
            split.default_left = true;
            split.is_categorical = false;

            let (left, right) = part.apply_split(0, &split, &quantized);

            let left_rows = part.node_rows(left);
            let _right_rows = part.node_rows(right);

            // Missing rows (0, 3, 7) should be in left
            assert!(left_rows.contains(&0));
            assert!(left_rows.contains(&3));
            assert!(left_rows.contains(&7));
        }

        #[test]
        fn test_apply_split_missing_default_right() {
            let data: Vec<f32> = vec![
                f32::NAN, 1.0, 2.0, f32::NAN, 4.0, 5.0, 6.0, f32::NAN, 8.0, 9.0,
            ];
            let matrix = ColMatrix::from_vec(data, 10, 1);

            let quantizer = Quantizer::from_data(&matrix, &ExactQuantileCuts::default(), 256);
            let quantized = quantizer.quantize_u8(&matrix);

            let mut part = RowPartitioner::new(10);

            // Split: missing goes right
            let mut split = SplitInfo::none();
            split.feature = 0;
            split.split_bin = 4;
            split.default_left = false;
            split.is_categorical = false;

            let (left, right) = part.apply_split(0, &split, &quantized);

            let _left_rows = part.node_rows(left);
            let right_rows = part.node_rows(right);

            // Missing rows (0, 3, 7) should be in right
            assert!(right_rows.contains(&0));
            assert!(right_rows.contains(&3));
            assert!(right_rows.contains(&7));
        }

        #[test]
        fn test_apply_split_categorical() {
            // Feature values: categories 1, 2, 3, 4, 5 (after quantization, bins 1-5)
            let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0];
            let matrix = ColMatrix::from_vec(data, 10, 1);

            let quantizer = Quantizer::from_data(&matrix, &ExactQuantileCuts::default(), 256);
            let quantized = quantizer.quantize_u8(&matrix);

            let mut part = RowPartitioner::new(10);

            // Categorical split: categories 1, 2 go left (bins 1, 2 after quantization)
            // We need to find the actual bin indices from quantization
            let col = quantized.feature_column(0);
            let bin1 = col[0]; // bin for value 1.0
            let bin2 = col[1]; // bin for value 2.0

            let mut split = SplitInfo::none();
            split.feature = 0;
            split.is_categorical = true;
            split.categories_left = vec![bin1 as u32, bin2 as u32];
            split.default_left = true;

            let (left, right) = part.apply_split(0, &split, &quantized);

            let left_rows = part.node_rows(left);
            let right_rows = part.node_rows(right);

            // Rows 0, 5 have value 1.0 (should go left)
            // Rows 1, 6 have value 2.0 (should go left)
            // Others should go right
            assert!(left_rows.contains(&0) || left_rows.contains(&5));
            assert!(left_rows.contains(&1) || left_rows.contains(&6));
            assert_eq!(left_rows.len() + right_rows.len(), 10);
        }

        #[test]
        fn test_multiple_splits() {
            let (quantized, _cuts) = make_test_quantized();
            let mut part = RowPartitioner::new(10);

            // First split on feature 0
            let mut split1 = SplitInfo::none();
            split1.feature = 0;
            split1.split_bin = 5;
            split1.default_left = true;
            split1.is_categorical = false;

            let (left1, _right1) = part.apply_split(0, &split1, &quantized);
            assert_eq!(part.num_nodes(), 3);

            let left1_size = part.node_size(left1);

            // Split the left child again
            if left1_size >= 2 {
                let mut split2 = SplitInfo::none();
                split2.feature = 0;
                split2.split_bin = 3;
                split2.default_left = true;
                split2.is_categorical = false;

                let (left2, right2) = part.apply_split(left1, &split2, &quantized);
                assert_eq!(part.num_nodes(), 5);

                let left2_rows = part.node_rows(left2);
                let right2_rows = part.node_rows(right2);

                // All rows from left1 should now be in left2 or right2
                assert_eq!(left2_rows.len() + right2_rows.len(), left1_size as usize);
            }
        }

        #[test]
        fn test_no_rows_lost_or_duplicated() {
            let (quantized, _cuts) = make_test_quantized();
            let mut part = RowPartitioner::new(10);

            // Do several splits
            let mut split = SplitInfo::none();
            split.feature = 0;
            split.split_bin = 5;
            split.default_left = true;
            split.is_categorical = false;

            let (left, right) = part.apply_split(0, &split, &quantized);

            // Collect all rows from children
            let mut all_rows: Vec<u32> = Vec::new();
            all_rows.extend(part.node_rows(left));
            all_rows.extend(part.node_rows(right));

            // Sort and check
            all_rows.sort();
            assert_eq!(all_rows, (0..10).collect::<Vec<_>>(), "Rows should not be lost or duplicated");
        }
    }

    #[test]
    fn test_find_threshold_bin() {
        use super::super::quantize::BinCuts;

        // Cuts at [0.5, 1.5, 2.5] for feature 0
        let cut_values = vec![0.5, 1.5, 2.5];
        let cut_ptrs = vec![0, 3];
        let cuts = BinCuts::new(cut_values, cut_ptrs);

        // Threshold 0.0 -> bin 0 (below all cuts)
        assert_eq!(find_threshold_bin(&cuts, 0, 0.0), 0);

        // Threshold 0.5 -> bin 1 (at first cut)
        assert_eq!(find_threshold_bin(&cuts, 0, 0.5), 1);

        // Threshold 1.0 -> bin 1 (between cuts 0 and 1)
        assert_eq!(find_threshold_bin(&cuts, 0, 1.0), 1);

        // Threshold 1.5 -> bin 2
        assert_eq!(find_threshold_bin(&cuts, 0, 1.5), 2);

        // Threshold 3.0 -> bin 3 (above all cuts)
        assert_eq!(find_threshold_bin(&cuts, 0, 3.0), 3);
    }
}

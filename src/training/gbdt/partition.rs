//! Row partitioning for tree training.
//!
//! Manages row indices per leaf, enabling efficient partitioning when applying splits.
//! Uses a single contiguous buffer with ranges per leaf to avoid allocations during training.
//!
//! # Design (following LightGBM)
//!
//! The partitioner stores:
//! - `indices`: Contiguous buffer of row indices, ordered by leaf
//! - `leaf_begin`: Start position for each leaf in `indices`
//! - `leaf_count`: Number of rows in each leaf
//!
//! When splitting a leaf, rows are partitioned in-place and the new leaf ranges are updated.
//!
//! # Requirements
//!
//! The dataset must provide dense storage (no missing bins). When evaluating splits,
//! we call `dataset.get_bin(row, feature)` which returns `Option<u32>`. If the dataset
//! has sparse storage, this may return `None`, causing a panic. GBDT training assumes
//! all bin values are available for partitioning.

use super::split::{SplitInfo, SplitType};
use crate::data::BinnedDataset;

/// Leaf identifier (index during training).
pub type LeafId = u32;

/// Manages row indices per leaf during tree training.
///
/// Uses a single contiguous buffer containing all row indices. Each leaf owns
/// a range within this buffer. When a leaf is split, its range is partitioned
/// in-place into two child ranges.
///
/// ```text
/// Initial (all rows in leaf 0):
///   indices: [0, 1, 2, 3, 4, 5, 6, 7]
///   leaf_begin: [0], leaf_count: [8]
///
/// After splitting leaf 0 (rows 0,2,4,6 go left to leaf 0, rows 1,3,5,7 go right to leaf 1):
///   indices: [0, 2, 4, 6, 1, 3, 5, 7]
///   leaf_begin: [0, 4], leaf_count: [4, 4]
/// ```
pub struct RowPartitioner {
    /// Row indices buffer. Partitioned in-place.
    indices: Box<[u32]>,
    /// Start position for each leaf in `indices`.
    leaf_begin: Vec<u32>,
    /// Number of rows in each leaf.
    leaf_count: Vec<u32>,
    /// Number of leaves currently allocated.
    n_leaves: usize,
}

impl RowPartitioner {
    /// Create a new partitioner.
    ///
    /// # Arguments
    /// * `n_samples` - Number of rows (samples) in the dataset
    /// * `max_leaves` - Maximum number of leaves to support
    pub fn new(n_samples: usize, max_leaves: usize) -> Self {
        let indices: Box<[u32]> = (0..n_samples as u32).collect();

        Self {
            indices,
            leaf_begin: vec![0; max_leaves],
            leaf_count: vec![0; max_leaves],
            n_leaves: 0,
        }
    }

    /// Reset the partitioner for a new tree.
    ///
    /// # Arguments
    /// * `n_samples` - Total number of rows in the dataset
    /// * `sampled` - Optional sampled row indices (None = use all rows)
    ///
    /// Initializes with all rows (or sampled rows) in leaf 0.
    pub fn reset(&mut self, n_samples: usize, sampled: Option<&[u32]>) {
        match sampled {
            None => {
                // Reset indices to sequential order
                if self.indices.len() != n_samples {
                    self.indices = (0..n_samples as u32).collect();
                } else {
                    for (i, idx) in self.indices.iter_mut().enumerate() {
                        *idx = i as u32;
                    }
                }
                
                // Reset leaf tracking
                self.leaf_begin.fill(0);
                self.leaf_count.fill(0);
                
                // Root leaf owns all rows
                self.leaf_begin[0] = 0;
                self.leaf_count[0] = n_samples as u32;
                self.n_leaves = 1;
            }
            Some(sampled_indices) => {
                // Initialize with only sampled rows
                let n_sampled = sampled_indices.len();
                if self.indices.len() != n_sampled {
                    self.indices = sampled_indices.to_vec().into_boxed_slice();
                } else {
                    self.indices[..n_sampled].copy_from_slice(sampled_indices);
                }
                
                // Reset leaf tracking
                self.leaf_begin.fill(0);
                self.leaf_count.fill(0);
                
                // Root leaf owns all sampled rows
                self.leaf_begin[0] = 0;
                self.leaf_count[0] = n_sampled as u32;
                self.n_leaves = 1;
            }
        }
    }

    /// Get the row indices for a leaf.
    #[inline]
    pub fn get_leaf_indices(&self, leaf: LeafId) -> &[u32] {
        let begin = self.leaf_begin[leaf as usize] as usize;
        let count = self.leaf_count[leaf as usize] as usize;
        &self.indices[begin..begin + count]
    }

    /// Get the number of rows in a leaf.
    #[inline]
    pub fn leaf_count(&self, leaf: LeafId) -> u32 {
        self.leaf_count[leaf as usize]
    }

    /// Get the start position for a leaf.
    #[inline]
    pub fn leaf_begin(&self, leaf: LeafId) -> u32 {
        self.leaf_begin[leaf as usize]
    }

    /// Number of allocated leaves.
    #[inline]
    pub fn n_leaves(&self) -> usize {
        self.n_leaves
    }

    /// Split a leaf according to a split decision.
    ///
    /// The original leaf keeps the left-going rows.
    /// A new leaf is allocated for the right-going rows.
    ///
    /// # Returns
    /// `(right_leaf, left_count, right_count)` where:
    /// - `right_leaf`: ID of the newly allocated right leaf
    /// - `left_count`: Number of rows going left (remaining in original leaf)
    /// - `right_count`: Number of rows going right (in new leaf)
    pub fn split(
        &mut self,
        leaf: LeafId,
        split: &SplitInfo,
        dataset: &BinnedDataset,
    ) -> (LeafId, u32, u32) {
        let begin = self.leaf_begin[leaf as usize] as usize;
        let count = self.leaf_count[leaf as usize] as usize;
        let end = begin + count;

        // Partition in place: left elements move to front
        let mut left_end = begin;

        for i in begin..end {
            let row = self.indices[i];
            let goes_left = self.evaluate_split(row, split, dataset);

            if goes_left {
                self.indices.swap(i, left_end);
                left_end += 1;
            }
        }

        let left_count = (left_end - begin) as u32;
        let right_count = (end - left_end) as u32;

        // Update original leaf (now left only)
        self.leaf_count[leaf as usize] = left_count;

        // Allocate new leaf for right
        let right_leaf = self.n_leaves as LeafId;
        self.n_leaves += 1;
        self.leaf_begin[right_leaf as usize] = left_end as u32;
        self.leaf_count[right_leaf as usize] = right_count;

        (right_leaf, left_count, right_count)
    }

    /// Evaluate whether a row goes left according to the split.
    ///
    /// # Panics
    ///
    /// Panics if `dataset.get_bin(row, feature)` returns `None`. This can occur
    /// when using sparse storage where some bins are not present. GBDT training
    /// requires dense storage for all rows to be available during partitioning.
    #[inline]
    fn evaluate_split(&self, row: u32, split: &SplitInfo, dataset: &BinnedDataset) -> bool {
        let bin = dataset.get_bin(row as usize, split.feature as usize)
            .expect("partition requires dense storage");

        // Handle missing/default bin
        let bin_mapper = dataset.bin_mapper(split.feature as usize);
        let default_bin = bin_mapper.default_bin();
        if bin == default_bin && bin_mapper.missing_type() != crate::data::MissingType::None {
            return split.default_left;
        }

        match &split.split_type {
            SplitType::Numerical { bin: threshold } => bin <= *threshold as u32,
            SplitType::Categorical { left_cats } => left_cats.contains(bin),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{BinMapper, BinnedDataset, BinnedDatasetBuilder, GroupLayout, GroupStrategy, MissingType};

    fn make_test_dataset() -> BinnedDataset {
        // 8 samples, 2 features
        // Feature 0: bins [0,1,0,1,0,1,0,1] - alternating
        // Feature 1: bins [0,0,0,0,1,1,1,1] - first half 0, second half 1
        let f0_bins = vec![0, 1, 0, 1, 0, 1, 0, 1];
        let f0_mapper =
            BinMapper::numerical(vec![0.5, 1.5], MissingType::None, 0, 0, 0.0, 0.0, 1.0);

        let f1_bins = vec![0, 0, 0, 0, 1, 1, 1, 1];
        let f1_mapper =
            BinMapper::numerical(vec![0.5, 1.5], MissingType::None, 0, 0, 0.0, 0.0, 1.0);

        BinnedDatasetBuilder::new()
            .add_binned(f0_bins, f0_mapper)
            .add_binned(f1_bins, f1_mapper)
            .group_strategy(GroupStrategy::SingleGroup { layout: GroupLayout::ColumnMajor })
            .build()
            .unwrap()
    }

    #[test]
    fn test_partitioner_init() {
        let mut partitioner = RowPartitioner::new(100, 16);
        partitioner.reset(100, None);

        assert_eq!(partitioner.leaf_count(0), 100);
        assert_eq!(partitioner.get_leaf_indices(0).len(), 100);
        assert_eq!(partitioner.n_leaves(), 1);

        // Check indices are sequential
        let indices = partitioner.get_leaf_indices(0);
        for i in 0..100 {
            assert_eq!(indices[i], i as u32);
        }
    }

    #[test]
    fn test_split_numerical() {
        let dataset = make_test_dataset();
        let mut partitioner = RowPartitioner::new(8, 16);
        partitioner.reset(8, None);

        // Split on feature 1 at bin 0 (rows 0-3 go left, rows 4-7 go right)
        let split = SplitInfo::numerical(1, 0, 1.0, false);
        let (right_leaf, left_count, right_count) = partitioner.split(0, &split, &dataset);

        assert_eq!(left_count, 4);
        assert_eq!(right_count, 4);
        assert_eq!(right_leaf, 1);
        assert_eq!(partitioner.n_leaves(), 2);

        // Left (leaf 0) should have rows 0,1,2,3
        let left_indices: Vec<u32> = partitioner.get_leaf_indices(0).to_vec();
        assert_eq!(left_indices.len(), 4);
        for row in &left_indices {
            assert!(*row < 4);
        }

        // Right (leaf 1) should have rows 4,5,6,7
        let right_indices: Vec<u32> = partitioner.get_leaf_indices(right_leaf).to_vec();
        assert_eq!(right_indices.len(), 4);
        for row in &right_indices {
            assert!(*row >= 4);
        }
    }

    #[test]
    fn test_split_alternating() {
        let dataset = make_test_dataset();
        let mut partitioner = RowPartitioner::new(8, 16);
        partitioner.reset(8, None);

        // Split on feature 0 at bin 0 (even indices go left, odd go right)
        let split = SplitInfo::numerical(0, 0, 1.0, false);
        let (right_leaf, left_count, right_count) = partitioner.split(0, &split, &dataset);

        assert_eq!(left_count, 4);
        assert_eq!(right_count, 4);

        // Left should have even rows (0,2,4,6)
        let left_indices: Vec<u32> = partitioner.get_leaf_indices(0).to_vec();
        for row in &left_indices {
            assert_eq!(row % 2, 0);
        }

        // Right should have odd rows (1,3,5,7)
        let right_indices: Vec<u32> = partitioner.get_leaf_indices(right_leaf).to_vec();
        for row in &right_indices {
            assert_eq!(row % 2, 1);
        }
    }

    #[test]
    fn test_multiple_splits() {
        let dataset = make_test_dataset();
        let mut partitioner = RowPartitioner::new(8, 32);
        partitioner.reset(8, None);

        // First split: feature 1 at bin 0 → leaf 0 has 0-3, leaf 1 has 4-7
        let split1 = SplitInfo::numerical(1, 0, 1.0, false);
        let (leaf1, _, _) = partitioner.split(0, &split1, &dataset);

        // Split leaf 0 on feature 0 → even (0,2) stay, odd (1,3) go to new leaf
        let split2 = SplitInfo::numerical(0, 0, 1.0, false);
        let (leaf2, left_count, right_count) = partitioner.split(0, &split2, &dataset);

        assert_eq!(left_count, 2);
        assert_eq!(right_count, 2);

        // Verify correct rows
        let leaf0_indices: Vec<u32> = partitioner.get_leaf_indices(0).to_vec();
        let leaf2_indices: Vec<u32> = partitioner.get_leaf_indices(leaf2).to_vec();

        for row in &leaf0_indices {
            assert!(*row < 4 && row % 2 == 0);
        }
        for row in &leaf2_indices {
            assert!(*row < 4 && row % 2 == 1);
        }

        // Leaf 1 should still have rows 4-7
        let leaf1_indices = partitioner.get_leaf_indices(leaf1);
        assert_eq!(leaf1_indices.len(), 4);
    }
}

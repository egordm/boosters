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
//! # Important invariants
//!
//! - The tree grower’s **ordered-gradient** histogram builder assumes a 1:1 correspondence
//!   between the per-leaf `indices` slice and the gathered `ordered_grad/ordered_hess` buffers:
//!   `ordered_grad[i]` must correspond to `indices[i]`.
//! - This partitioner uses **stable partitioning** when splitting leaves. Starting from the
//!   root’s sequential indices, this ensures each leaf’s indices remain in ascending order.
//!   This improves locality for downstream gradient gathering and histogram building.
//! - Note: “ascending order” is not the same as being a contiguous range; most leaves will
//!   still require gradient gathering.
//!
//! # Requirements
//!
//! The dataset must provide dense storage (no missing bins). When evaluating splits,
//! we call `dataset.get_bin(row, feature)` which returns `Option<u32>`. If the dataset
//! has sparse storage, this may return `None`, causing a panic. GBDT training assumes
//! all bin values are available for partitioning.

use super::split::{SplitInfo, SplitType};
use crate::data::binned::FeatureView;
use crate::data::{BinnedDataset, BinnedFeatureColumnSource, BinnedFeatureViews, MissingType};

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
    /// Scratch buffer used for stable partitioning.
    scratch: Box<[u32]>,
    /// Start position for each leaf in `indices`.
    leaf_begin: Vec<u32>,
    /// Number of rows in each leaf.
    leaf_count: Vec<u32>,
    /// If present, the leaf's row indices are strictly sequential in leaf order: [k, k+1, ...].
    ///
    /// This allows histogram building to avoid an O(n) contiguity scan per node.
    leaf_sequential_start: Vec<Option<u32>>,
    /// Number of leaves currently allocated.
    n_leaves: usize,
}

impl RowPartitioner {
    fn stable_partition_range(
        indices: &mut [u32],
        scratch: &mut [u32],
        begin: usize,
        end: usize,
        mut goes_left: impl FnMut(u32) -> bool,
    ) -> (u32, Option<u32>, bool, Option<u32>, bool) {
        // Single-pass stable partition: write left from start, right from end of scratch,
        // then reverse right portion in-place for stable ordering.
        let mut left_write = begin;
        let mut right_write = end; // Points past the end, we decrement before writing

        // Track whether each child partition is strictly sequential in the produced order.
        // This stays correct even when the root indices come from an arbitrary sampled order.
        let mut left_seq_start: Option<u32> = None;
        let mut left_prev: u32 = 0;
        let mut left_is_seq = true;
        let mut right_seq_start: Option<u32> = None;
        let mut right_prev: u32 = 0;
        let mut right_is_seq = true;

        #[inline]
        fn update_seq(start: &mut Option<u32>, prev: &mut u32, is_seq: &mut bool, idx: u32) {
            match *start {
                None => {
                    *start = Some(idx);
                    *prev = idx;
                }
                Some(_) => {
                    if *is_seq {
                        if idx == prev.wrapping_add(1) {
                            *prev = idx;
                        } else {
                            *is_seq = false;
                        }
                    }
                }
            }
        }

        // Iterate over the original indices and write the partition into scratch.
        for &idx in &indices[begin..end] {
            if goes_left(idx) {
                scratch[left_write] = idx;
                left_write += 1;
                update_seq(&mut left_seq_start, &mut left_prev, &mut left_is_seq, idx);
            } else {
                right_write -= 1;
                scratch[right_write] = idx;
                update_seq(
                    &mut right_seq_start,
                    &mut right_prev,
                    &mut right_is_seq,
                    idx,
                );
            }
        }

        let left_count = (left_write - begin) as u32;
        debug_assert_eq!(left_write, right_write, "partition pointers should meet");

        // Right portion was written in reverse. Reverse it to restore stable ordering.
        scratch[left_write..end].reverse();
        indices[begin..end].copy_from_slice(&scratch[begin..end]);

        (
            left_count,
            left_seq_start,
            left_is_seq,
            right_seq_start,
            right_is_seq,
        )
    }

    #[inline]
    fn sequential_start(indices: &[u32]) -> Option<u32> {
        let (first, rest) = indices.split_first()?;
        let mut prev = *first;
        for &idx in rest {
            if idx != prev.wrapping_add(1) {
                return None;
            }
            prev = idx;
        }
        Some(*first)
    }

    /// Create a new partitioner.
    ///
    /// # Arguments
    /// * `n_samples` - Number of rows (samples) in the dataset
    /// * `max_leaves` - Maximum number of leaves to support
    pub fn new(n_samples: usize, max_leaves: usize) -> Self {
        let indices: Box<[u32]> = (0..n_samples as u32).collect();
        let scratch: Box<[u32]> = vec![0u32; n_samples].into_boxed_slice();

        Self {
            indices,
            scratch,
            leaf_begin: vec![0; max_leaves],
            leaf_count: vec![0; max_leaves],
            leaf_sequential_start: vec![None; max_leaves],
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
                    self.scratch = vec![0u32; n_samples].into_boxed_slice();
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
                self.leaf_sequential_start.fill(None);
                self.leaf_sequential_start[0] = Some(0);
                self.n_leaves = 1;
            }
            Some(sampled_indices) => {
                // Initialize with only sampled rows
                let n_sampled = sampled_indices.len();
                if self.indices.len() != n_sampled {
                    self.indices = sampled_indices.to_vec().into_boxed_slice();
                    self.scratch = vec![0u32; n_sampled].into_boxed_slice();
                } else {
                    self.indices[..n_sampled].copy_from_slice(sampled_indices);
                }

                // Reset leaf tracking
                self.leaf_begin.fill(0);
                self.leaf_count.fill(0);

                // Root leaf owns all sampled rows
                self.leaf_begin[0] = 0;
                self.leaf_count[0] = n_sampled as u32;
                self.leaf_sequential_start.fill(None);
                self.leaf_sequential_start[0] = Self::sequential_start(&self.indices);
                self.n_leaves = 1;
            }
        }
    }

    #[inline]
    pub(super) fn leaf_sequential_start(&self, leaf: LeafId) -> Option<u32> {
        self.leaf_sequential_start[leaf as usize]
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

    #[inline]
    pub(super) fn leaf_range(&self, leaf: LeafId) -> (usize, usize) {
        let begin = self.leaf_begin[leaf as usize] as usize;
        let count = self.leaf_count[leaf as usize] as usize;
        (begin, begin + count)
    }

    #[inline]
    pub(super) fn indices(&self) -> &[u32] {
        &self.indices
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
    /// # Arguments
    /// * `leaf` - Leaf to split
    /// * `split` - Split decision (feature index refers to bundled column if n_bundles > 0)
    /// * `dataset` - The binned dataset
    /// * `n_bundles` - Number of bundled columns (0 if no bundling)
    ///
    /// # Returns
    /// `(right_leaf, left_count, right_count)` where:
    /// - `right_leaf`: ID of the newly allocated right leaf
    /// - `left_count`: Number of rows going left (remaining in original leaf)
    /// - `right_count`: Number of rows going right (in new leaf)
    ///
    /// Note: With LightGBM-style bundling, `split.feature` is an effective column index.
    /// We use effective feature views for partitioning (bins are already encoded correctly).
    pub fn split(
        &mut self,
        leaf: LeafId,
        split: &SplitInfo,
        dataset: &BinnedDataset,
        effective: &BinnedFeatureViews<'_>,
    ) -> (LeafId, u32, u32) {
        let begin = self.leaf_begin[leaf as usize] as usize;
        let count = self.leaf_count[leaf as usize] as usize;
        let end = begin + count;

        // Get feature view and metadata from effective layout.
        // split.feature is an effective column index (bundles first, then standalone).
        let effective_col = split.feature as usize;
        let view = effective.views[effective_col];

        // Bundles: default_bin is 0 (all defaults), has_missing is true.
        // Standalone: use the original feature's bin mapper.
        let (default_bin, has_missing) = match effective.sources[effective_col] {
            BinnedFeatureColumnSource::Bundle { .. } => (0u32, true),
            BinnedFeatureColumnSource::Standalone { feature_idx } => {
                let bin_mapper = dataset.bin_mapper(feature_idx);
                (
                    bin_mapper.default_bin(),
                    bin_mapper.missing_type() != MissingType::None,
                )
            }
        };
        let default_left = split.default_left;

        // Common decision predicate on *bin* (already default-filled for sparse).
        // Missing bins are represented as default_bin when has_missing is true.
        let goes_left_for_bin = |bin: u32| {
            if has_missing && bin == default_bin {
                default_left
            } else {
                match &split.split_type {
                    SplitType::Numerical { bin: threshold } => bin <= (*threshold as u32),
                    SplitType::Categorical { left_cats } => left_cats.contains(bin),
                }
            }
        };

        let (left_count, left_seq_start, left_is_seq, right_seq_start, right_is_seq) = match view {
            FeatureView::U8(bins) => Self::stable_partition_range(
                &mut self.indices,
                &mut self.scratch,
                begin,
                end,
                |idx| goes_left_for_bin(bins[idx as usize] as u32),
            ),
            FeatureView::U16(bins) => Self::stable_partition_range(
                &mut self.indices,
                &mut self.scratch,
                begin,
                end,
                |idx| goes_left_for_bin(bins[idx as usize] as u32),
            ),
            _ => Self::stable_partition_range(
                &mut self.indices,
                &mut self.scratch,
                begin,
                end,
                |idx| {
                    let row = idx as usize;
                    let bin = view.get_bin(row).unwrap_or(default_bin);
                    goes_left_for_bin(bin)
                },
            ),
        };

        let right_count = count as u32 - left_count;

        // Update original leaf (now left only)
        self.leaf_count[leaf as usize] = left_count;
        self.leaf_sequential_start[leaf as usize] = if left_is_seq { left_seq_start } else { None };

        // Allocate new leaf for right
        let right_leaf = self.n_leaves as LeafId;
        self.n_leaves += 1;
        self.leaf_begin[right_leaf as usize] = (begin as u32) + left_count;
        self.leaf_count[right_leaf as usize] = right_count;
        self.leaf_sequential_start[right_leaf as usize] =
            if right_is_seq { right_seq_start } else { None };

        (right_leaf, left_count, right_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{BinnedDataset, BinningConfig, Dataset};
    use ndarray::array;

    fn make_test_dataset() -> BinnedDataset {
        // 8 samples, 2 features
        // Feature 0: bins [0,1,0,1,0,1,0,1] - alternating
        // Feature 1: bins [0,0,0,0,1,1,1,1] - first half 0, second half 1
        //
        // With default binning (256 bins), we need raw values that bin appropriately.
        // Using values 0.0 and 1.0 that will bin to different bins.
        let raw = Dataset::builder()
            .add_feature_unnamed(array![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
            .add_feature_unnamed(array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
            .build()
            .unwrap();
        BinnedDataset::from_dataset(&raw, &BinningConfig::default()).unwrap()
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
        for (i, &idx) in indices.iter().enumerate().take(100) {
            assert_eq!(idx, i as u32);
        }
    }

    #[test]
    fn test_leaf_sequential_start_sampled_order() {
        let mut partitioner = RowPartitioner::new(4, 8);
        partitioner.reset(0, Some(&[0, 2, 1, 3]));
        assert_eq!(partitioner.leaf_sequential_start(0), None);

        partitioner.reset(0, Some(&[5, 6, 7]));
        assert_eq!(partitioner.leaf_sequential_start(0), Some(5));
    }

    #[test]
    fn test_split_numerical() {
        let dataset = make_test_dataset();
        let effective = dataset.feature_views();
        let mut partitioner = RowPartitioner::new(8, 16);
        partitioner.reset(8, None);

        // Split on feature 1 at bin 0 (rows 0-3 go left, rows 4-7 go right)
        let split = SplitInfo::numerical(1, 0, 1.0, false);
        let (right_leaf, left_count, right_count) =
            partitioner.split(0, &split, &dataset, &effective);

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
        let effective = dataset.feature_views();
        let mut partitioner = RowPartitioner::new(8, 16);
        partitioner.reset(8, None);

        // Split on feature 0 at bin 0 (even indices go left, odd go right)
        let split = SplitInfo::numerical(0, 0, 1.0, false);
        let (right_leaf, left_count, right_count) =
            partitioner.split(0, &split, &dataset, &effective);

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
        let effective = dataset.feature_views();
        let mut partitioner = RowPartitioner::new(8, 32);
        partitioner.reset(8, None);

        // First split: feature 1 at bin 0 → leaf 0 has 0-3, leaf 1 has 4-7
        let split1 = SplitInfo::numerical(1, 0, 1.0, false);
        let (leaf1, _, _) = partitioner.split(0, &split1, &dataset, &effective);

        // Split leaf 0 on feature 0 → even (0,2) stay, odd (1,3) go to new leaf
        let split2 = SplitInfo::numerical(0, 0, 1.0, false);
        let (leaf2, left_count, right_count) = partitioner.split(0, &split2, &dataset, &effective);

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

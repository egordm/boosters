//! Row block iterator for efficient prediction.
//!
//! Provides buffered access to dataset rows in blocks,
//! optimized for prediction where we need row-major access to features.
//!
//! # Buffer Reuse
//!
//! For maximum efficiency, use the callback-based methods (`for_each`, `for_each_with`)
//! which reuse a single buffer across all blocks. The `Iterator` implementation
//! allocates a new array for each block.
//!
//! # Parallelism
//!
//! Use `for_each_with` or `flat_map_with` with `Parallelism::Parallel` to process
//! blocks in parallel. Each parallel worker gets its own buffer.

// Allow dead code during migration - this will be used when we switch over in Epic 7
#![allow(dead_code)]

use super::dataset::BinnedDataset;
use super::view::FeatureView;
use crate::utils::Parallelism;
use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;

/// Buffered row block iterator for efficient prediction.
///
/// Transposes column-major storage to row-major blocks on demand.
/// Provides ~2x speedup for prediction vs random column access.
///
/// # Design
///
/// - Block size: configurable, default 64 samples
/// - For numeric features: returns raw values (accessed via column slice)
/// - For categorical features: casts bin index to f32 (via FeatureView)
/// - Iteration pattern: column-major access internally, row-major output
///
/// # Usage Patterns
///
/// ## Sequential with buffer reuse (most efficient for single-threaded)
///
/// ```ignore
/// let blocks = RowBlocks::new(&dataset, 64);
/// blocks.for_each(|start_idx, block| {
///     // block is ArrayView2<f32> with shape [block_size, n_features]
///     for row in block.outer_iter() {
///         // Process each sample
///     }
/// });
/// ```
///
/// ## Parallel processing
///
/// ```ignore
/// let blocks = RowBlocks::new(&dataset, 256);
/// let predictions = blocks.flat_map_with(Parallelism::Parallel, |start_idx, block| {
///     block.outer_iter()
///         .map(|row| model.predict_one(row.as_slice().unwrap()))
///         .collect()
/// });
/// ```
pub struct RowBlocks<'a> {
    dataset: &'a BinnedDataset,
    block_size: usize,
}

impl<'a> RowBlocks<'a> {
    /// Create a new row block iterator.
    ///
    /// # Parameters
    /// - `dataset`: The dataset to iterate over
    /// - `block_size`: Number of samples per block (default: 64)
    pub fn new(dataset: &'a BinnedDataset, block_size: usize) -> Self {
        Self {
            dataset,
            block_size,
        }
    }

    /// Create with default block size (64).
    pub fn with_default_block_size(dataset: &'a BinnedDataset) -> Self {
        Self::new(dataset, 64)
    }

    /// Number of blocks.
    pub fn n_blocks(&self) -> usize {
        let n_samples = self.dataset.n_samples();
        if n_samples == 0 {
            return 0;
        }
        (n_samples + self.block_size - 1) / self.block_size
    }

    /// Process blocks sequentially, calling `f` for each block.
    ///
    /// This reuses a single buffer across all blocks for maximum efficiency.
    ///
    /// # Parameters
    /// - `f`: Callback receiving (start_sample_idx, block: ArrayView2<f32>)
    pub fn for_each<F>(&self, mut f: F)
    where
        F: FnMut(usize, ArrayView2<f32>),
    {
        let n_samples = self.dataset.n_samples();
        let n_features = self.dataset.n_features();

        if n_samples == 0 || n_features == 0 {
            return;
        }

        // Allocate buffer once, reuse for all blocks
        let mut buffer = Array2::zeros((self.block_size, n_features));
        let mut current_sample = 0;

        while current_sample < n_samples {
            let end_sample = (current_sample + self.block_size).min(n_samples);
            let actual_block_size = end_sample - current_sample;

            // Fill buffer
            Self::fill_block(
                &mut buffer,
                self.dataset,
                current_sample,
                actual_block_size,
            );

            // Call with view of actual size
            let view = buffer.slice(ndarray::s![0..actual_block_size, ..]);
            f(current_sample, view);

            current_sample = end_sample;
        }
    }

    /// Process blocks with parallelism control.
    ///
    /// When `Parallelism::Parallel`, blocks are processed in parallel using rayon.
    /// Each parallel worker allocates its own buffer.
    /// When `Parallelism::Sequential`, uses `for_each` with buffer reuse.
    ///
    /// # Parameters
    /// - `parallelism`: Whether to use parallel processing
    /// - `f`: Callback receiving (start_sample_idx, block: ArrayView2<f32>)
    pub fn for_each_with<F>(&self, parallelism: Parallelism, f: F)
    where
        F: Fn(usize, ArrayView2<f32>) + Sync + Send,
    {
        if parallelism.is_parallel() {
            let n_blocks = self.n_blocks();
            (0..n_blocks).into_par_iter().for_each(|block_idx| {
                let (start_idx, block) = self.get_block(block_idx);
                f(start_idx, block.view());
            });
        } else {
            self.for_each(|start_idx, block| f(start_idx, block));
        }
    }

    /// Collect results from each block with parallelism control.
    ///
    /// # Parameters
    /// - `parallelism`: Whether to use parallel processing
    /// - `f`: Callback receiving (start_sample_idx, block: ArrayView2<f32>) -> Vec<T>
    ///
    /// # Returns
    /// Results concatenated in block order.
    pub fn flat_map_with<T, F>(&self, parallelism: Parallelism, f: F) -> Vec<T>
    where
        T: Send,
        F: Fn(usize, ArrayView2<f32>) -> Vec<T> + Sync + Send,
    {
        if parallelism.is_parallel() {
            let n_blocks = self.n_blocks();
            let mut results: Vec<(usize, Vec<T>)> = (0..n_blocks)
                .into_par_iter()
                .map(|block_idx| {
                    let (start_idx, block) = self.get_block(block_idx);
                    (block_idx, f(start_idx, block.view()))
                })
                .collect();

            // Sort by block index to maintain order
            results.sort_by_key(|(idx, _)| *idx);

            // Flatten results
            results.into_iter().flat_map(|(_, v)| v).collect()
        } else {
            let mut results = Vec::new();
            self.for_each(|start_idx, block| {
                results.extend(f(start_idx, block));
            });
            results
        }
    }

    /// Get a specific block by index.
    ///
    /// Returns (start_sample_idx, block: Array2<f32>).
    /// Allocates a new array - use `for_each` for buffer reuse.
    pub fn get_block(&self, block_idx: usize) -> (usize, Array2<f32>) {
        let n_samples = self.dataset.n_samples();
        let n_features = self.dataset.n_features();

        let start_sample = block_idx * self.block_size;
        let end_sample = (start_sample + self.block_size).min(n_samples);
        let actual_block_size = end_sample - start_sample;

        let mut block = Array2::zeros((actual_block_size, n_features));
        Self::fill_block(&mut block, self.dataset, start_sample, actual_block_size);

        (start_sample, block)
    }

    /// Fill a block buffer with data from the dataset.
    fn fill_block(
        block: &mut Array2<f32>,
        dataset: &BinnedDataset,
        start_sample: usize,
        block_size: usize,
    ) {
        let n_features = dataset.n_features();

        for feature_idx in 0..n_features {
            // Try raw slice first (most efficient for numeric)
            if let Some(raw_slice) = dataset.raw_feature_slice(feature_idx) {
                Self::fill_from_raw_slice(block, feature_idx, raw_slice, start_sample, block_size);
            } else {
                // Fall back to FeatureView for categorical or sparse
                let view = dataset.original_feature_view(feature_idx);
                Self::fill_from_feature_view(block, feature_idx, &view, start_sample, block_size);
            }
        }
    }

    /// Fill a column in the block from a numeric feature's raw slice.
    #[inline]
    fn fill_from_raw_slice(
        block: &mut Array2<f32>,
        feature_idx: usize,
        raw_slice: &[f32],
        start_sample: usize,
        block_size: usize,
    ) {
        // Direct slice copy - efficient
        for (row_in_block, value) in raw_slice[start_sample..start_sample + block_size]
            .iter()
            .enumerate()
        {
            block[[row_in_block, feature_idx]] = *value;
        }
    }

    /// Fill a column in the block from a FeatureView (for categorical or missing raw).
    #[inline]
    fn fill_from_feature_view(
        block: &mut Array2<f32>,
        feature_idx: usize,
        view: &FeatureView<'_>,
        start_sample: usize,
        block_size: usize,
    ) {
        // Match once, then iterate efficiently
        match view {
            FeatureView::U8(bins) => {
                for (row_in_block, sample) in (start_sample..start_sample + block_size).enumerate()
                {
                    block[[row_in_block, feature_idx]] = bins[sample] as f32;
                }
            }
            FeatureView::U16(bins) => {
                for (row_in_block, sample) in (start_sample..start_sample + block_size).enumerate()
                {
                    block[[row_in_block, feature_idx]] = bins[sample] as f32;
                }
            }
            FeatureView::SparseU8 {
                sample_indices,
                bin_values,
            } => {
                // Zero-fill first, then set sparse values
                for row in 0..block_size {
                    block[[row, feature_idx]] = 0.0;
                }
                // Find entries in this block's range
                let end_sample = start_sample + block_size;
                for (idx, &sample) in sample_indices.iter().enumerate() {
                    let sample = sample as usize;
                    if sample >= start_sample && sample < end_sample {
                        block[[sample - start_sample, feature_idx]] = bin_values[idx] as f32;
                    }
                }
            }
            FeatureView::SparseU16 {
                sample_indices,
                bin_values,
            } => {
                // Zero-fill first, then set sparse values
                for row in 0..block_size {
                    block[[row, feature_idx]] = 0.0;
                }
                // Find entries in this block's range
                let end_sample = start_sample + block_size;
                for (idx, &sample) in sample_indices.iter().enumerate() {
                    let sample = sample as usize;
                    if sample >= start_sample && sample < end_sample {
                        block[[sample - start_sample, feature_idx]] = bin_values[idx] as f32;
                    }
                }
            }
        }
    }

    /// Create an iterator that yields owned blocks.
    ///
    /// Note: This allocates a new array for each block.
    /// For buffer reuse, use `for_each` instead.
    pub fn iter(&self) -> RowBlocksIter<'a> {
        RowBlocksIter {
            dataset: self.dataset,
            block_size: self.block_size,
            current_sample: 0,
        }
    }
}

/// Iterator over row blocks that yields owned arrays.
///
/// Note: Allocates a new array for each block.
/// For buffer reuse, use `RowBlocks::for_each` instead.
pub struct RowBlocksIter<'a> {
    dataset: &'a BinnedDataset,
    block_size: usize,
    current_sample: usize,
}

impl<'a> Iterator for RowBlocksIter<'a> {
    type Item = Array2<f32>;

    fn next(&mut self) -> Option<Self::Item> {
        let n_samples = self.dataset.n_samples();
        let n_features = self.dataset.n_features();

        if self.current_sample >= n_samples {
            return None;
        }

        let end_sample = (self.current_sample + self.block_size).min(n_samples);
        let actual_block_size = end_sample - self.current_sample;

        let mut block = Array2::zeros((actual_block_size, n_features));
        RowBlocks::fill_block(&mut block, self.dataset, self.current_sample, actual_block_size);

        self.current_sample = end_sample;
        Some(block)
    }
}

impl<'a> IntoIterator for RowBlocks<'a> {
    type Item = Array2<f32>;
    type IntoIter = RowBlocksIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::binned::builder::DatasetBuilder;
    use crate::data::binned::feature_analysis::BinningConfig;
    use ndarray::{array, Array2};

    fn make_array(values: &[f32], rows: usize, cols: usize) -> Array2<f32> {
        Array2::from_shape_vec((rows, cols), values.to_vec()).unwrap()
    }

    #[test]
    fn test_row_blocks_single_block() {
        let data = make_array(&[1.5, 2.5, 3.5, 4.5, 5.5], 5, 1);
        let config = BinningConfig::default();

        let built = DatasetBuilder::from_array(data.view(), &config)
            .unwrap()
            .build_groups()
            .unwrap();

        let dataset = BinnedDataset::from_built_groups(built);

        // Block size larger than dataset - should return single block
        let blocks: Vec<_> = RowBlocks::new(&dataset, 10).iter().collect();
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].shape(), &[5, 1]);

        // Check values
        assert_eq!(blocks[0][[0, 0]], 1.5);
        assert_eq!(blocks[0][[4, 0]], 5.5);
    }

    #[test]
    fn test_row_blocks_multiple_blocks() {
        let data = make_array(&[1.5, 2.5, 3.5, 4.5, 5.5], 5, 1);
        let config = BinningConfig::default();

        let built = DatasetBuilder::from_array(data.view(), &config)
            .unwrap()
            .build_groups()
            .unwrap();

        let dataset = BinnedDataset::from_built_groups(built);

        // Block size 2 - should return 3 blocks (2, 2, 1)
        let blocks: Vec<_> = RowBlocks::new(&dataset, 2).iter().collect();
        assert_eq!(blocks.len(), 3);
        assert_eq!(blocks[0].shape(), &[2, 1]);
        assert_eq!(blocks[1].shape(), &[2, 1]);
        assert_eq!(blocks[2].shape(), &[1, 1]); // Last partial block

        // Check values
        assert_eq!(blocks[0][[0, 0]], 1.5);
        assert_eq!(blocks[0][[1, 0]], 2.5);
        assert_eq!(blocks[2][[0, 0]], 5.5);
    }

    #[test]
    fn test_row_blocks_multiple_features() {
        let built = DatasetBuilder::new()
            .add_numeric("x", array![1.5, 2.5, 3.5].view())
            .add_numeric("y", array![10.5, 20.5, 30.5].view())
            .build_groups()
            .unwrap();

        let dataset = BinnedDataset::from_built_groups(built);

        let blocks: Vec<_> = RowBlocks::new(&dataset, 10).iter().collect();
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].shape(), &[3, 2]);

        // Check row-major values
        assert_eq!(blocks[0][[0, 0]], 1.5); // sample 0, feature 0
        assert_eq!(blocks[0][[0, 1]], 10.5); // sample 0, feature 1
        assert_eq!(blocks[0][[1, 0]], 2.5); // sample 1, feature 0
        assert_eq!(blocks[0][[2, 1]], 30.5); // sample 2, feature 1
    }

    #[test]
    fn test_row_blocks_mixed_features() {
        let built = DatasetBuilder::new()
            .add_numeric("num", array![1.5, 2.5, 3.5].view())
            .add_categorical("cat", array![0.0, 1.0, 2.0].view())
            .build_groups()
            .unwrap();

        let dataset = BinnedDataset::from_built_groups(built);

        let blocks: Vec<_> = RowBlocks::new(&dataset, 10).iter().collect();
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].shape(), &[3, 2]);

        // Numeric feature should have raw values
        assert_eq!(blocks[0][[0, 0]], 1.5);
        assert_eq!(blocks[0][[1, 0]], 2.5);

        // Categorical feature should have bin values as f32
        // Bins are 0, 1, 2 (category IDs)
        assert_eq!(blocks[0][[0, 1]], 0.0);
        assert_eq!(blocks[0][[1, 1]], 1.0);
        assert_eq!(blocks[0][[2, 1]], 2.0);
    }

    #[test]
    fn test_row_blocks_for_each_buffer_reuse() {
        let data = make_array(&[1.5, 2.5, 3.5, 4.5, 5.5], 5, 1);
        let config = BinningConfig::default();

        let built = DatasetBuilder::from_array(data.view(), &config)
            .unwrap()
            .build_groups()
            .unwrap();

        let dataset = BinnedDataset::from_built_groups(built);

        // Test for_each with buffer reuse
        let mut results: Vec<(usize, f32)> = Vec::new();
        RowBlocks::new(&dataset, 2).for_each(|start_idx, block| {
            for (i, row) in block.outer_iter().enumerate() {
                results.push((start_idx + i, row[0]));
            }
        });

        assert_eq!(results.len(), 5);
        assert_eq!(results[0], (0, 1.5));
        assert_eq!(results[1], (1, 2.5));
        assert_eq!(results[2], (2, 3.5));
        assert_eq!(results[3], (3, 4.5));
        assert_eq!(results[4], (4, 5.5));
    }

    #[test]
    fn test_row_blocks_flat_map_sequential() {
        let data = make_array(&[1.5, 2.5, 3.5, 4.5, 5.5], 5, 1);
        let config = BinningConfig::default();

        let built = DatasetBuilder::from_array(data.view(), &config)
            .unwrap()
            .build_groups()
            .unwrap();

        let dataset = BinnedDataset::from_built_groups(built);

        // Test flat_map_with sequential
        let results: Vec<f32> =
            RowBlocks::new(&dataset, 2).flat_map_with(Parallelism::Sequential, |_, block| {
                block.outer_iter().map(|row| row[0] * 2.0).collect()
            });

        assert_eq!(results.len(), 5);
        assert_eq!(results[0], 3.0); // 1.5 * 2
        assert_eq!(results[1], 5.0); // 2.5 * 2
        assert_eq!(results[4], 11.0); // 5.5 * 2
    }

    #[test]
    fn test_row_blocks_flat_map_parallel() {
        let data = make_array(&[1.5, 2.5, 3.5, 4.5, 5.5], 5, 1);
        let config = BinningConfig::default();

        let built = DatasetBuilder::from_array(data.view(), &config)
            .unwrap()
            .build_groups()
            .unwrap();

        let dataset = BinnedDataset::from_built_groups(built);

        // Test flat_map_with parallel - should give same results
        let results: Vec<f32> =
            RowBlocks::new(&dataset, 2).flat_map_with(Parallelism::Parallel, |_, block| {
                block.outer_iter().map(|row| row[0] * 2.0).collect()
            });

        assert_eq!(results.len(), 5);
        // Results should be in order
        assert_eq!(results[0], 3.0);
        assert_eq!(results[1], 5.0);
        assert_eq!(results[4], 11.0);
    }

    #[test]
    fn test_row_blocks_n_blocks() {
        let data = make_array(&[1.5, 2.5, 3.5, 4.5, 5.5], 5, 1);
        let config = BinningConfig::default();

        let built = DatasetBuilder::from_array(data.view(), &config)
            .unwrap()
            .build_groups()
            .unwrap();

        let dataset = BinnedDataset::from_built_groups(built);

        assert_eq!(RowBlocks::new(&dataset, 2).n_blocks(), 3); // 2+2+1
        assert_eq!(RowBlocks::new(&dataset, 5).n_blocks(), 1); // exact fit
        assert_eq!(RowBlocks::new(&dataset, 10).n_blocks(), 1); // larger than data
        assert_eq!(RowBlocks::new(&dataset, 1).n_blocks(), 5); // one per sample
    }

    #[test]
    fn test_row_blocks_get_block() {
        let data = make_array(&[1.5, 2.5, 3.5, 4.5, 5.5], 5, 1);
        let config = BinningConfig::default();

        let built = DatasetBuilder::from_array(data.view(), &config)
            .unwrap()
            .build_groups()
            .unwrap();

        let dataset = BinnedDataset::from_built_groups(built);
        let blocks = RowBlocks::new(&dataset, 2);

        let (start0, block0) = blocks.get_block(0);
        assert_eq!(start0, 0);
        assert_eq!(block0.shape(), &[2, 1]);
        assert_eq!(block0[[0, 0]], 1.5);

        let (start1, block1) = blocks.get_block(1);
        assert_eq!(start1, 2);
        assert_eq!(block1.shape(), &[2, 1]);
        assert_eq!(block1[[0, 0]], 3.5);

        let (start2, block2) = blocks.get_block(2);
        assert_eq!(start2, 4);
        assert_eq!(block2.shape(), &[1, 1]); // partial
        assert_eq!(block2[[0, 0]], 5.5);
    }

    #[test]
    fn test_row_blocks_into_iter() {
        let data = make_array(&[1.5, 2.5], 2, 1);
        let config = BinningConfig::default();

        let built = DatasetBuilder::from_array(data.view(), &config)
            .unwrap()
            .build_groups()
            .unwrap();

        let dataset = BinnedDataset::from_built_groups(built);

        // Test IntoIterator trait
        let mut count = 0;
        for block in RowBlocks::new(&dataset, 1) {
            count += 1;
            assert_eq!(block.shape(), &[1, 1]);
        }
        assert_eq!(count, 2);
    }
}

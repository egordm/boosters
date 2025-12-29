//! Sample block iterator for efficient prediction.
//!
//! Provides buffered access to dataset samples in blocks,
//! optimized for prediction where we need sample-major access to features.
//!
//! # Terminology
//!
//! We use "sample blocks" rather than "row blocks" to avoid confusion with
//! ndarray's row/column terminology. In our domain:
//! - **Sample**: A single data point (one row in the original data matrix)
//! - **Sample block**: A contiguous range of samples buffered together
//!
//! # Buffer Reuse with Thread-Local Storage
//!
//! The `for_each_with` method uses thread-local buffers for maximum efficiency:
//! - In sequential mode: single buffer reused across all blocks
//! - In parallel mode: one buffer per thread, reused across blocks processed by that thread
//!
//! This follows the same pattern as `Predictor::predict_into` in the inference module.
//!
//! # Usage
//!
//! ```ignore
//! use boosters::utils::Parallelism;
//!
//! // Process blocks with thread-local buffer reuse
//! let blocks = SampleBlocks::new(&dataset, 64);
//! blocks.for_each_with(Parallelism::Parallel, |start_idx, block| {
//!     // block is ArrayView2<f32> with shape [block_size, n_features]
//!     for sample in block.outer_iter() {
//!         // Process each sample
//!     }
//! });
//! ```

// Allow dead code during migration - this will be used when we switch over in Epic 7
#![allow(dead_code)]

use super::dataset::BinnedDataset;
use super::view::FeatureView;
use crate::utils::Parallelism;
use ndarray::{Array2, ArrayView2};
use std::cell::RefCell;

/// Buffered sample block iterator for efficient prediction.
///
/// Transposes column-major storage to sample-major blocks on demand.
/// Provides ~2x speedup for prediction vs random column access.
///
/// # Design
///
/// - Block size: configurable, default 64 samples
/// - For numeric features: returns raw values (accessed via column slice)
/// - For categorical features: casts bin index to f32 (via FeatureView)
/// - Buffer reuse: thread-local buffers for both sequential and parallel processing
///
/// # Usage
///
/// ```ignore
/// let blocks = SampleBlocks::new(&dataset, 64);
/// blocks.for_each_with(Parallelism::Parallel, |start_idx, block| {
///     // block is ArrayView2<f32> with shape [block_size, n_features]
///     for sample in block.outer_iter() {
///         // Process each sample
///     }
/// });
/// ```
pub struct SampleBlocks<'a> {
    dataset: &'a BinnedDataset,
    block_size: usize,
}

impl<'a> SampleBlocks<'a> {
    /// Create a new sample block iterator.
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

    /// Block size.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Process blocks with parallelism control and thread-local buffer reuse.
    ///
    /// Uses thread-local `Array2` buffers sized at `[block_size, n_features]`.
    /// Each thread gets/reuses its own buffer, providing efficient memory usage
    /// in both sequential and parallel modes.
    ///
    /// This follows the same pattern as `Predictor::predict_into` in the inference module.
    ///
    /// # Parameters
    /// - `parallelism`: Whether to use parallel processing
    /// - `f`: Callback receiving (start_sample_idx, block: ArrayView2<f32>)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let blocks = SampleBlocks::new(&dataset, 64);
    /// blocks.for_each_with(Parallelism::Parallel, |start_idx, block| {
    ///     for (i, sample) in block.outer_iter().enumerate() {
    ///         let sample_idx = start_idx + i;
    ///         // Process sample...
    ///     }
    /// });
    /// ```
    pub fn for_each_with<F>(&self, parallelism: Parallelism, f: F)
    where
        F: Fn(usize, ArrayView2<f32>) + Sync + Send,
    {
        let n_samples = self.dataset.n_samples();
        let n_features = self.dataset.n_features();

        if n_samples == 0 || n_features == 0 {
            return;
        }

        // Thread-local Array2 buffer for block [block_size, n_features]
        thread_local! {
            static BLOCK_BUFFER: RefCell<Option<Array2<f32>>> = const { RefCell::new(None) };
        }

        let block_size = self.block_size;
        let n_blocks = self.n_blocks();
        let dataset = self.dataset;

        parallelism.maybe_par_bridge_for_each(0..n_blocks, move |block_idx| {
            let start_sample = block_idx * block_size;
            let end_sample = (start_sample + block_size).min(n_samples);
            let actual_block_size = end_sample - start_sample;

            BLOCK_BUFFER.with(|buf| {
                let mut buffer_ref = buf.borrow_mut();

                // Get or create buffer with correct size
                let buffer = buffer_ref
                    .get_or_insert_with(|| Array2::zeros((block_size, n_features)));

                // Resize buffer if shape changed (different block_size or n_features)
                if buffer.nrows() < actual_block_size || buffer.ncols() != n_features {
                    *buffer = Array2::zeros((block_size.max(actual_block_size), n_features));
                }

                // Get mutable view of only the rows we need
                let mut block_view = buffer.slice_mut(ndarray::s![0..actual_block_size, ..]);

                // Fill the block
                Self::fill_block_view(&mut block_view, dataset, start_sample, actual_block_size);

                // Call with immutable view
                f(start_sample, block_view.view());
            });
        });
    }

    /// Fill a block buffer with data from the dataset.
    fn fill_block_view(
        block: &mut ndarray::ArrayViewMut2<f32>,
        dataset: &BinnedDataset,
        start_sample: usize,
        block_size: usize,
    ) {
        let n_features = dataset.n_features();

        for feature_idx in 0..n_features {
            // Get mutable column view
            let col = block.column_mut(feature_idx);

            // Try raw slice first (most efficient for numeric)
            if let Some(raw_slice) = dataset.raw_feature_slice(feature_idx) {
                Self::fill_col_from_raw_slice(col, raw_slice, start_sample, block_size);
            } else {
                // Fall back to FeatureView for categorical or sparse
                let view = dataset.original_feature_view(feature_idx);
                Self::fill_col_from_feature_view(col, &view, start_sample, block_size);
            }
        }
    }

    /// Fill a column from a numeric feature's raw slice.
    ///
    /// Uses ndarray's assign for efficient strided copy.
    #[inline]
    fn fill_col_from_raw_slice(
        mut col: ndarray::ArrayViewMut1<f32>,
        raw_slice: &[f32],
        start_sample: usize,
        block_size: usize,
    ) {
        let src = &raw_slice[start_sample..start_sample + block_size];
        // Use ndarray's assign which handles strides efficiently
        col.assign(&ndarray::ArrayView1::from(src));
    }

    /// Fill a column from a FeatureView (for categorical or missing raw).
    ///
    /// Uses ndarray's iterator for efficient strided access.
    #[inline]
    fn fill_col_from_feature_view(
        mut col: ndarray::ArrayViewMut1<f32>,
        view: &FeatureView<'_>,
        start_sample: usize,
        block_size: usize,
    ) {
        // Match once, then iterate efficiently
        match view {
            FeatureView::U8(bins) => {
                let src = &bins[start_sample..start_sample + block_size];
                for (dst, &bin) in col.iter_mut().zip(src.iter()) {
                    *dst = bin as f32;
                }
            }
            FeatureView::U16(bins) => {
                let src = &bins[start_sample..start_sample + block_size];
                for (dst, &bin) in col.iter_mut().zip(src.iter()) {
                    *dst = bin as f32;
                }
            }
            FeatureView::SparseU8 {
                sample_indices,
                bin_values,
            } => {
                // Zero-fill first
                col.fill(0.0);
                // Set sparse values
                let end_sample = start_sample + block_size;
                for (idx, &sample) in sample_indices.iter().enumerate() {
                    let sample = sample as usize;
                    if sample >= start_sample && sample < end_sample {
                        col[sample - start_sample] = bin_values[idx] as f32;
                    }
                }
            }
            FeatureView::SparseU16 {
                sample_indices,
                bin_values,
            } => {
                // Zero-fill first
                col.fill(0.0);
                // Set sparse values
                let end_sample = start_sample + block_size;
                for (idx, &sample) in sample_indices.iter().enumerate() {
                    let sample = sample as usize;
                    if sample >= start_sample && sample < end_sample {
                        col[sample - start_sample] = bin_values[idx] as f32;
                    }
                }
            }
        }
    }

    /// Create an iterator that yields owned blocks.
    ///
    /// **Note**: This allocates a new array for each block.
    /// For buffer reuse, use `for_each_with` instead.
    pub fn iter(&self) -> SampleBlocksIter<'a> {
        SampleBlocksIter {
            dataset: self.dataset,
            block_size: self.block_size,
            current_sample: 0,
        }
    }
}

/// Iterator over sample blocks that yields owned arrays.
///
/// **Note**: Allocates a new array for each block.
/// For buffer reuse, use `SampleBlocks::for_each_with` instead.
pub struct SampleBlocksIter<'a> {
    dataset: &'a BinnedDataset,
    block_size: usize,
    current_sample: usize,
}

impl<'a> Iterator for SampleBlocksIter<'a> {
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
        SampleBlocks::fill_block_view(
            &mut block.view_mut(),
            self.dataset,
            self.current_sample,
            actual_block_size,
        );

        self.current_sample = end_sample;
        Some(block)
    }
}

impl<'a> IntoIterator for SampleBlocks<'a> {
    type Item = Array2<f32>;
    type IntoIter = SampleBlocksIter<'a>;

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
    fn test_sample_blocks_single_block() {
        let data = make_array(&[1.5, 2.5, 3.5, 4.5, 5.5], 5, 1);
        let config = BinningConfig::default();

        let built = DatasetBuilder::from_array(data.view(), &config)
            .unwrap()
            .build_groups()
            .unwrap();

        let dataset = BinnedDataset::from_built_groups(built);

        // Block size larger than dataset - should return single block
        let blocks: Vec<_> = SampleBlocks::new(&dataset, 10).iter().collect();
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].shape(), &[5, 1]);

        // Check values
        assert_eq!(blocks[0][[0, 0]], 1.5);
        assert_eq!(blocks[0][[4, 0]], 5.5);
    }

    #[test]
    fn test_sample_blocks_multiple_blocks() {
        let data = make_array(&[1.5, 2.5, 3.5, 4.5, 5.5], 5, 1);
        let config = BinningConfig::default();

        let built = DatasetBuilder::from_array(data.view(), &config)
            .unwrap()
            .build_groups()
            .unwrap();

        let dataset = BinnedDataset::from_built_groups(built);

        // Block size 2 - should return 3 blocks (2, 2, 1)
        let blocks: Vec<_> = SampleBlocks::new(&dataset, 2).iter().collect();
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
    fn test_sample_blocks_multiple_features() {
        let built = DatasetBuilder::new()
            .add_numeric("x", array![1.5, 2.5, 3.5].view())
            .add_numeric("y", array![10.5, 20.5, 30.5].view())
            .build_groups()
            .unwrap();

        let dataset = BinnedDataset::from_built_groups(built);

        let blocks: Vec<_> = SampleBlocks::new(&dataset, 10).iter().collect();
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].shape(), &[3, 2]);

        // Check sample-major values
        assert_eq!(blocks[0][[0, 0]], 1.5); // sample 0, feature 0
        assert_eq!(blocks[0][[0, 1]], 10.5); // sample 0, feature 1
        assert_eq!(blocks[0][[1, 0]], 2.5); // sample 1, feature 0
        assert_eq!(blocks[0][[2, 1]], 30.5); // sample 2, feature 1
    }

    #[test]
    fn test_sample_blocks_mixed_features() {
        let built = DatasetBuilder::new()
            .add_numeric("num", array![1.5, 2.5, 3.5].view())
            .add_categorical("cat", array![0.0, 1.0, 2.0].view())
            .build_groups()
            .unwrap();

        let dataset = BinnedDataset::from_built_groups(built);

        let blocks: Vec<_> = SampleBlocks::new(&dataset, 10).iter().collect();
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
    fn test_sample_blocks_for_each_with_sequential() {
        let data = make_array(&[1.5, 2.5, 3.5, 4.5, 5.5], 5, 1);
        let config = BinningConfig::default();

        let built = DatasetBuilder::from_array(data.view(), &config)
            .unwrap()
            .build_groups()
            .unwrap();

        let dataset = BinnedDataset::from_built_groups(built);

        // Test for_each_with sequential
        use std::sync::Mutex;
        let results = Mutex::new(Vec::new());
        SampleBlocks::new(&dataset, 2).for_each_with(Parallelism::Sequential, |start_idx, block| {
            let mut results = results.lock().unwrap();
            for (i, row) in block.outer_iter().enumerate() {
                results.push((start_idx + i, row[0]));
            }
        });

        let results = results.into_inner().unwrap();
        assert_eq!(results.len(), 5);
        assert_eq!(results[0], (0, 1.5));
        assert_eq!(results[1], (1, 2.5));
        assert_eq!(results[2], (2, 3.5));
        assert_eq!(results[3], (3, 4.5));
        assert_eq!(results[4], (4, 5.5));
    }

    #[test]
    fn test_sample_blocks_for_each_with_parallel() {
        let data = make_array(&[1.5, 2.5, 3.5, 4.5, 5.5], 5, 1);
        let config = BinningConfig::default();

        let built = DatasetBuilder::from_array(data.view(), &config)
            .unwrap()
            .build_groups()
            .unwrap();

        let dataset = BinnedDataset::from_built_groups(built);

        // Test for_each_with parallel - results may be out of order
        use std::sync::Mutex;
        let results = Mutex::new(Vec::new());
        SampleBlocks::new(&dataset, 2).for_each_with(Parallelism::Parallel, |start_idx, block| {
            let mut results = results.lock().unwrap();
            for (i, row) in block.outer_iter().enumerate() {
                results.push((start_idx + i, row[0]));
            }
        });

        let mut results = results.into_inner().unwrap();
        // Sort by sample index since parallel order is non-deterministic
        results.sort_by_key(|(idx, _)| *idx);

        assert_eq!(results.len(), 5);
        assert_eq!(results[0], (0, 1.5));
        assert_eq!(results[1], (1, 2.5));
        assert_eq!(results[2], (2, 3.5));
        assert_eq!(results[3], (3, 4.5));
        assert_eq!(results[4], (4, 5.5));
    }

    #[test]
    fn test_sample_blocks_parallel_matches_sequential() {
        let data = make_array(&[1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5], 8, 1);
        let config = BinningConfig::default();

        let built = DatasetBuilder::from_array(data.view(), &config)
            .unwrap()
            .build_groups()
            .unwrap();

        let dataset = BinnedDataset::from_built_groups(built);

        use std::sync::Mutex;

        // Sequential
        let seq_results = Mutex::new(Vec::new());
        SampleBlocks::new(&dataset, 3).for_each_with(Parallelism::Sequential, |start_idx, block| {
            let mut results = seq_results.lock().unwrap();
            for (i, row) in block.outer_iter().enumerate() {
                results.push((start_idx + i, row[0]));
            }
        });

        // Parallel
        let par_results = Mutex::new(Vec::new());
        SampleBlocks::new(&dataset, 3).for_each_with(Parallelism::Parallel, |start_idx, block| {
            let mut results = par_results.lock().unwrap();
            for (i, row) in block.outer_iter().enumerate() {
                results.push((start_idx + i, row[0]));
            }
        });

        let mut seq = seq_results.into_inner().unwrap();
        let mut par = par_results.into_inner().unwrap();

        seq.sort_by_key(|(idx, _)| *idx);
        par.sort_by_key(|(idx, _)| *idx);

        assert_eq!(seq, par);
    }

    #[test]
    fn test_sample_blocks_n_blocks() {
        let data = make_array(&[1.5, 2.5, 3.5, 4.5, 5.5], 5, 1);
        let config = BinningConfig::default();

        let built = DatasetBuilder::from_array(data.view(), &config)
            .unwrap()
            .build_groups()
            .unwrap();

        let dataset = BinnedDataset::from_built_groups(built);

        assert_eq!(SampleBlocks::new(&dataset, 2).n_blocks(), 3); // 2+2+1
        assert_eq!(SampleBlocks::new(&dataset, 5).n_blocks(), 1); // exact fit
        assert_eq!(SampleBlocks::new(&dataset, 10).n_blocks(), 1); // larger than data
        assert_eq!(SampleBlocks::new(&dataset, 1).n_blocks(), 5); // one per sample
    }

    #[test]
    fn test_sample_blocks_into_iter() {
        let data = make_array(&[1.5, 2.5], 2, 1);
        let config = BinningConfig::default();

        let built = DatasetBuilder::from_array(data.view(), &config)
            .unwrap()
            .build_groups()
            .unwrap();

        let dataset = BinnedDataset::from_built_groups(built);

        // Test IntoIterator trait
        let mut count = 0;
        for block in SampleBlocks::new(&dataset, 1) {
            count += 1;
            assert_eq!(block.shape(), &[1, 1]);
        }
        assert_eq!(count, 2);
    }

    #[test]
    fn test_sample_blocks_single_sample() {
        // Test edge case: single sample dataset
        let data = make_array(&[1.5], 1, 1);
        let config = BinningConfig::default();

        let built = DatasetBuilder::from_array(data.view(), &config)
            .unwrap()
            .build_groups()
            .unwrap();

        let dataset = BinnedDataset::from_built_groups(built);

        assert_eq!(SampleBlocks::new(&dataset, 10).n_blocks(), 1);

        // Should produce exactly one block with one sample
        use std::sync::Mutex;
        let results = Mutex::new(Vec::new());
        SampleBlocks::new(&dataset, 10).for_each_with(Parallelism::Sequential, |start_idx, block| {
            let mut results = results.lock().unwrap();
            results.push((start_idx, block.shape()[0], block[[0, 0]]));
        });

        let results = results.into_inner().unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], (0, 1, 1.5));
    }
}

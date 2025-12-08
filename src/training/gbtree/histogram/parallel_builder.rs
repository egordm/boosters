//! Row-parallel histogram builder.
//!
//! This module provides row-parallel histogram building as an alternative
//! to the feature-parallel approach in [`HistogramBuilder`]. Row-parallel
//! building is more efficient for "tall" datasets (many rows, few features).
//!
//! See RFC-0025 for design rationale and performance analysis.
//!
//! # Algorithm Overview
//!
//! ```text
//! 1. Partition rows across threads
//! 2. Each thread accumulates into its local scratch buffer
//! 3. Reduce all thread buffers into target histogram
//! ```
//!
//! # When to Use
//!
//! - **Row-parallel (this module)**: Tall data (rows >> features × bins)
//! - **Feature-parallel ([`HistogramBuilder::build_parallel`])**: Wide data
//!
//! The builder provides an automatic threshold selection method.

use super::pool::{ContiguousHistogramPool, HistogramSlotMut};
use super::scratch::RowParallelScratch;
use super::types::NodeId;
use crate::training::gbtree::quantize::{BinCuts, BinIndex, QuantizedMatrix};

/// Wrapper to make raw pointers Send + Sync.
///
/// # Safety
///
/// The caller must ensure that the pointer is valid and that concurrent
/// access to the pointed-to data is safe (i.e., each thread accesses
/// a disjoint region).
#[derive(Clone, Copy)]
struct SendSyncPtr<T>(*mut T);

impl<T> SendSyncPtr<T> {
    /// Get the inner raw pointer.
    #[inline]
    fn ptr(self) -> *mut T {
        self.0
    }
}

// SAFETY: We guarantee disjoint access in build_row_parallel.
// Each thread_id maps to a unique scratch region.
unsafe impl<T> Send for SendSyncPtr<T> {}
unsafe impl<T> Sync for SendSyncPtr<T> {}

/// Configuration for parallel histogram building.
#[derive(Debug, Clone)]
pub struct ParallelHistogramConfig {
    /// Number of threads to use.
    /// Defaults to `rayon::current_num_threads()`.
    pub num_threads: usize,

    /// Minimum rows per chunk to avoid excessive overhead.
    /// Chunks smaller than this will be merged.
    pub min_chunk_size: usize,

    /// Whether to use parallel reduction for large histograms.
    /// Parallel reduction is beneficial when bins_per_hist > 1000.
    pub parallel_reduce: bool,

    /// Threshold for choosing row-parallel vs feature-parallel.
    /// Row-parallel is used when: rows > threshold × (num_features × bins_per_feature)
    pub row_parallel_threshold: f32,
}

impl Default for ParallelHistogramConfig {
    fn default() -> Self {
        Self {
            num_threads: rayon::current_num_threads(),
            min_chunk_size: 256,
            parallel_reduce: false,
            row_parallel_threshold: 4.0,
        }
    }
}

impl ParallelHistogramConfig {
    /// Create config optimized for tall datasets (many rows).
    pub fn for_tall_data() -> Self {
        Self {
            num_threads: rayon::current_num_threads(),
            min_chunk_size: 512,
            parallel_reduce: false,
            row_parallel_threshold: 2.0,
        }
    }

    /// Create config optimized for wide datasets (many features).
    pub fn for_wide_data() -> Self {
        Self {
            num_threads: rayon::current_num_threads(),
            min_chunk_size: 128,
            parallel_reduce: true,
            row_parallel_threshold: 8.0,
        }
    }

    /// Should we use row-parallel for the given dataset shape?
    pub fn should_use_row_parallel(&self, num_rows: usize, bins_per_hist: usize) -> bool {
        num_rows as f32 > self.row_parallel_threshold * bins_per_hist as f32
    }
}

/// Row-parallel histogram builder.
///
/// Builds histograms by partitioning rows across threads and reducing.
/// This is more efficient than feature-parallel building for tall datasets.
///
/// # Example
///
/// ```ignore
/// let mut builder = ParallelHistogramBuilder::new(config, &cuts);
///
/// // Build histogram for a node
/// let node_id = NodeId(0);
/// builder.build_into_pool(&mut pool, node_id, &quantized, &grads, &hess, &rows);
///
/// // Access result
/// let hist = pool.get(node_id).unwrap();
/// ```
pub struct ParallelHistogramBuilder {
    /// Configuration.
    config: ParallelHistogramConfig,

    /// Per-thread scratch buffers.
    scratch: RowParallelScratch,

    /// Total bins across all features.
    bins_per_hist: usize,

    /// Feature offsets for indexing into flat histogram.
    feature_offsets: Box<[usize]>,
}

impl ParallelHistogramBuilder {
    /// Create a new row-parallel histogram builder.
    ///
    /// # Arguments
    ///
    /// * `config` - Builder configuration
    /// * `cuts` - Quantization cuts defining bin counts per feature
    pub fn new(config: ParallelHistogramConfig, cuts: &BinCuts) -> Self {
        let bins_per_hist = cuts.total_bins();
        let num_threads = config.num_threads;
        let num_features = cuts.num_features() as usize;

        // Compute feature offsets
        let mut offsets = Vec::with_capacity(num_features);
        let mut offset = 0;
        for feat in 0..cuts.num_features() {
            offsets.push(offset);
            offset += cuts.num_bins(feat);
        }
        let feature_offsets = offsets.into_boxed_slice();

        let scratch = RowParallelScratch::new(num_threads, bins_per_hist);

        Self {
            config,
            scratch,
            bins_per_hist,
            feature_offsets,
        }
    }

    /// Build histogram directly into the pool.
    ///
    /// This is the main entry point for row-parallel histogram building.
    /// The histogram is allocated (or retrieved from cache) in the pool,
    /// then populated using row-parallel accumulation.
    ///
    /// # Arguments
    ///
    /// * `pool` - Histogram pool for allocation/caching
    /// * `node_id` - Node identifier for the histogram
    /// * `index` - Quantized feature matrix
    /// * `grads` - Gradient values (length = n_samples)
    /// * `hess` - Hessian values (length = n_samples)
    /// * `rows` - Row indices belonging to this node
    pub fn build_into_pool<B: BinIndex>(
        &mut self,
        pool: &mut ContiguousHistogramPool,
        node_id: NodeId,
        index: &QuantizedMatrix<B>,
        grads: &[f32],
        hess: &[f32],
        rows: &[u32],
    ) {
        debug_assert_eq!(grads.len(), hess.len());
        debug_assert_eq!(pool.bins_per_hist(), self.bins_per_hist);

        // Reset scratch buffers
        self.scratch.reset_all();

        // Get or allocate target histogram
        let mut target = pool.get_or_allocate(node_id);
        target.reset();

        // Build using row-parallel strategy
        self.build_row_parallel(index, grads, hess, rows);

        // Reduce scratch into target
        if self.config.parallel_reduce && self.bins_per_hist > 1000 {
            self.scratch.reduce_into_parallel(&mut target);
        } else {
            self.scratch.reduce_into(&mut target);
        }
    }

    /// Build histogram into a mutable slot directly.
    ///
    /// Lower-level API for when you already have a slot reference.
    pub fn build_into_slot<B: BinIndex>(
        &mut self,
        target: &mut HistogramSlotMut<'_>,
        index: &QuantizedMatrix<B>,
        grads: &[f32],
        hess: &[f32],
        rows: &[u32],
    ) {
        debug_assert_eq!(grads.len(), hess.len());
        debug_assert_eq!(target.num_bins(), self.bins_per_hist);

        // Reset scratch buffers
        self.scratch.reset_all();

        // Reset target
        target.reset();

        // Build using row-parallel strategy
        self.build_row_parallel(index, grads, hess, rows);

        // Reduce scratch into target
        if self.config.parallel_reduce && self.bins_per_hist > 1000 {
            self.scratch.reduce_into_parallel(target);
        } else {
            self.scratch.reduce_into(target);
        }
    }

    /// Core row-parallel building algorithm.
    ///
    /// Partitions rows across threads and each thread accumulates into
    /// its local scratch buffer.
    fn build_row_parallel<B: BinIndex>(
        &mut self,
        index: &QuantizedMatrix<B>,
        grads: &[f32],
        hess: &[f32],
        rows: &[u32],
    ) {
        let num_threads = self.config.num_threads;
        let num_features = self.feature_offsets.len();
        let bins_per_hist = self.bins_per_hist;

        // Divide rows into exactly num_threads chunks
        let chunk_size = (rows.len() + num_threads - 1) / num_threads;
        let chunk_size = chunk_size.max(1);

        // Clone feature offsets for thread-local access
        let feature_offsets: &[usize] = &self.feature_offsets;

        // Get raw pointers BEFORE parallel iteration, wrapped for Send+Sync
        let sg_base = SendSyncPtr(self.scratch.sum_grads_ptr());
        let sh_base = SendSyncPtr(self.scratch.sum_hess_ptr());
        let sc_base = SendSyncPtr(self.scratch.counts_ptr());

        // Use rayon::scope for parallel execution with proper borrowing
        rayon::scope(|s| {
            for thread_id in 0..num_threads {
                let start = thread_id * chunk_size;
                if start >= rows.len() {
                    continue; // This thread has no work
                }
                let end = (start + chunk_size).min(rows.len());
                let chunk_rows = &rows[start..end];

                // Copy the wrapped pointers for this spawn
                let sg_ptr = sg_base;
                let sh_ptr = sh_base;
                let sc_ptr = sc_base;

                // Spawn a task for this thread
                s.spawn(move |_| {
                    let scratch_offset = thread_id * bins_per_hist;

                    // Extract raw pointers inside the spawn using method (not field access)
                    let sg_base_raw = sg_ptr.ptr();
                    let sh_base_raw = sh_ptr.ptr();
                    let sc_base_raw = sc_ptr.ptr();

                    // SAFETY: Each thread_id maps to a unique, disjoint region
                    unsafe {
                        let sg = sg_base_raw.add(scratch_offset);
                        let sh = sh_base_raw.add(scratch_offset);
                        let sc = sc_base_raw.add(scratch_offset);

                        // Process each row in this chunk
                        for &row in chunk_rows {
                            let row_idx = row as usize;
                            let grad = *grads.get_unchecked(row_idx);
                            let hess_val = *hess.get_unchecked(row_idx);

                            // Accumulate into each feature's bins
                            for feat in 0..num_features {
                                let bin = index.get(row, feat as u32).to_usize();
                                let global_bin = *feature_offsets.get_unchecked(feat) + bin;

                                *sg.add(global_bin) += grad;
                                *sh.add(global_bin) += hess_val;
                                *sc.add(global_bin) += 1;
                            }
                        }
                    }
                });
            }
        });
    }

    /// Get reference to configuration.
    pub fn config(&self) -> &ParallelHistogramConfig {
        &self.config
    }

    /// Get total bins per histogram.
    pub fn bins_per_hist(&self) -> usize {
        self.bins_per_hist
    }
}

impl std::fmt::Debug for ParallelHistogramBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParallelHistogramBuilder")
            .field("config", &self.config)
            .field("bins_per_hist", &self.bins_per_hist)
            .field("num_features", &self.feature_offsets.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::ColMatrix;
    use crate::training::gbtree::histogram::HistogramBuilder;
    use crate::training::gbtree::quantize::{ExactQuantileCuts, Quantizer};

    fn make_test_data() -> (QuantizedMatrix<u8>, Vec<f32>, Vec<f32>, BinCuts) {
        // Create simple data: 100 rows, 3 features
        let n_rows = 100;
        let n_features = 3;

        let mut data = Vec::with_capacity(n_rows * n_features);
        for feat in 0..n_features {
            for row in 0..n_rows {
                data.push((row * (feat + 1)) as f32 % 10.0);
            }
        }
        let matrix = ColMatrix::from_vec(data, n_rows, n_features);

        // Quantize
        let quantizer = Quantizer::from_data(&matrix, &ExactQuantileCuts::default(), 16);
        let quantized: QuantizedMatrix<u8> = quantizer.quantize(&matrix);
        let cuts = quantized.cuts().clone();

        // Create gradients: grad = row_id % 5, hess = 1.0
        let grads: Vec<f32> = (0..n_rows).map(|i| (i % 5) as f32).collect();
        let hess: Vec<f32> = vec![1.0; n_rows];

        (quantized, grads, hess, cuts)
    }

    #[test]
    fn test_parallel_builder_basic() {
        let (quantized, grads, hess, cuts) = make_test_data();
        let rows: Vec<u32> = (0..100).collect();

        let config = ParallelHistogramConfig::default();
        let mut builder = ParallelHistogramBuilder::new(config, &cuts);

        // Create pool and build
        let mut pool = ContiguousHistogramPool::new(1, cuts.total_bins());
        let node_id = NodeId(0);
        builder.build_into_pool(&mut pool, node_id, &quantized, &grads, &hess, &rows);

        // Verify we got a histogram
        let hist = pool.get(node_id).expect("histogram should exist");

        // Total count should be 100 (using first feature only)
        let bins_feat0 = cuts.num_bins(0);
        assert_eq!(hist.total_count_first_feature(bins_feat0), 100);
    }

    #[test]
    fn test_parallel_builder_matches_sequential() {
        let (quantized, grads, hess, cuts) = make_test_data();
        let rows: Vec<u32> = (0..100).collect();

        // Build with sequential builder
        let mut hist_seq =
            crate::training::gbtree::histogram::NodeHistogram::new(quantized.cuts());
        HistogramBuilder.build(&mut hist_seq, &quantized, &grads, &hess, &rows);

        // Build with parallel builder
        let config = ParallelHistogramConfig::default();
        let mut builder = ParallelHistogramBuilder::new(config, &cuts);
        let mut pool = ContiguousHistogramPool::new(1, cuts.total_bins());
        let node_id = NodeId(0);
        builder.build_into_pool(&mut pool, node_id, &quantized, &grads, &hess, &rows);
        let hist_par = pool.get(node_id).unwrap();

        // Compare totals (use first feature's bins for count comparison)
        // NodeHistogram computes totals from feature 0 only
        let bins_feat0 = cuts.num_bins(0);
        assert_eq!(
            hist_seq.total_count(),
            hist_par.total_count_first_feature(bins_feat0),
            "count mismatch"
        );

        // Compare gradients/hessians using first feature only
        let total_grad = hist_par.total_grad_first_feature(bins_feat0);
        let total_hess = hist_par.total_hess_first_feature(bins_feat0);
        let seq_total_grad = hist_seq.total_grad();
        let seq_total_hess = hist_seq.total_hess();

        assert!(
            (seq_total_grad - total_grad).abs() < 1e-4,
            "grad mismatch: {} vs {}",
            seq_total_grad,
            total_grad
        );
        assert!(
            (seq_total_hess - total_hess).abs() < 1e-4,
            "hess mismatch: {} vs {}",
            seq_total_hess,
            total_hess
        );
    }

    #[test]
    fn test_parallel_builder_subset() {
        let (quantized, grads, hess, cuts) = make_test_data();

        // Build for subset of rows
        let rows: Vec<u32> = (0..50).collect();

        let config = ParallelHistogramConfig::default();
        let mut builder = ParallelHistogramBuilder::new(config, &cuts);
        let mut pool = ContiguousHistogramPool::new(1, cuts.total_bins());
        let node_id = NodeId(0);
        builder.build_into_pool(&mut pool, node_id, &quantized, &grads, &hess, &rows);

        let hist = pool.get(node_id).unwrap();
        let bins_feat0 = cuts.num_bins(0);
        assert_eq!(hist.total_count_first_feature(bins_feat0), 50);
    }

    #[test]
    fn test_parallel_builder_single_thread() {
        let (quantized, grads, hess, cuts) = make_test_data();
        let rows: Vec<u32> = (0..100).collect();

        // Use single thread config
        let config = ParallelHistogramConfig {
            num_threads: 1,
            ..Default::default()
        };
        let mut builder = ParallelHistogramBuilder::new(config, &cuts);

        let mut pool = ContiguousHistogramPool::new(1, cuts.total_bins());
        let node_id = NodeId(0);
        builder.build_into_pool(&mut pool, node_id, &quantized, &grads, &hess, &rows);

        let hist = pool.get(node_id).unwrap();
        let bins_feat0 = cuts.num_bins(0);
        assert_eq!(hist.total_count_first_feature(bins_feat0), 100);
    }

    #[test]
    fn test_parallel_builder_rebuild() {
        let (quantized, grads, hess, cuts) = make_test_data();

        let config = ParallelHistogramConfig::default();
        let mut builder = ParallelHistogramBuilder::new(config, &cuts);
        let mut pool = ContiguousHistogramPool::new(2, cuts.total_bins());

        // Build first histogram
        let rows1: Vec<u32> = (0..50).collect();
        let node1 = NodeId(0);
        builder.build_into_pool(&mut pool, node1, &quantized, &grads, &hess, &rows1);

        // Build second histogram
        let rows2: Vec<u32> = (50..100).collect();
        let node2 = NodeId(1);
        builder.build_into_pool(&mut pool, node2, &quantized, &grads, &hess, &rows2);

        // Both should be in pool
        assert!(pool.contains(node1));
        assert!(pool.contains(node2));

        // Check counts separately to avoid borrow issues
        let bins_feat0 = cuts.num_bins(0);
        let count1 = pool.get(node1).unwrap().total_count_first_feature(bins_feat0);
        let count2 = pool.get(node2).unwrap().total_count_first_feature(bins_feat0);

        assert_eq!(count1, 50);
        assert_eq!(count2, 50);
    }

    #[test]
    fn test_config_should_use_row_parallel() {
        let config = ParallelHistogramConfig {
            row_parallel_threshold: 4.0,
            ..Default::default()
        };

        // 10000 rows, 100 bins -> 10000 > 4 * 100 = 400 -> should use row parallel
        assert!(config.should_use_row_parallel(10000, 100));

        // 100 rows, 100 bins -> 100 > 4 * 100 = 400 -> should NOT use row parallel
        assert!(!config.should_use_row_parallel(100, 100));
    }

    #[test]
    fn test_parallel_builder_many_threads() {
        let (quantized, grads, hess, cuts) = make_test_data();
        let rows: Vec<u32> = (0..100).collect();

        // More threads than reasonable chunks
        let config = ParallelHistogramConfig {
            num_threads: 16,
            min_chunk_size: 8,
            ..Default::default()
        };
        let mut builder = ParallelHistogramBuilder::new(config, &cuts);

        let mut pool = ContiguousHistogramPool::new(1, cuts.total_bins());
        let node_id = NodeId(0);
        builder.build_into_pool(&mut pool, node_id, &quantized, &grads, &hess, &rows);

        let hist = pool.get(node_id).unwrap();
        let bins_feat0 = cuts.num_bins(0);
        assert_eq!(hist.total_count_first_feature(bins_feat0), 100);
    }
}

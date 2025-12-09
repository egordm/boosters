//! Unified histogram building with multiple parallelization strategies.
//!
//! This module provides [`HistogramBuilder`], the unified interface for building
//! histograms from quantized features and gradients. It supports three strategies:
//!
//! - **Sequential**: Single-threaded baseline
//! - **Feature-parallel**: Parallelizes across features (good for wide data)
//! - **Row-parallel**: Parallelizes across rows (good for tall data)
//!
//! # Usage
//!
//! ```ignore
//! // Create builder with row-parallel support
//! let mut builder = HistogramBuilder::new(&cuts, HistogramConfig::default());
//!
//! // Build into a pool slot (the main API)
//! let mut slot = pool.get_or_allocate(node_id);
//! builder.build(&mut slot, &layout, strategy, &quantized, &grads, &hess, &rows);
//! ```
//!
//! See RFC-0025 for design rationale and performance analysis.

use rayon::prelude::*;

use super::pool::{ContiguousHistogramPool, HistogramSlotMut};
use super::scratch::RowParallelScratch;
use super::types::NodeId;
use crate::training::gbtree::quantize::{BinCuts, BinIndex, QuantizedMatrix};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for histogram building strategies.
#[derive(Debug, Clone)]
pub struct HistogramConfig {
    /// Number of threads to use for parallel strategies.
    /// Defaults to `rayon::current_num_threads()`.
    pub num_threads: usize,

    /// Minimum rows per chunk for row-parallel building.
    /// Chunks smaller than this will be merged.
    pub min_chunk_size: usize,

    /// Whether to use parallel reduction for large histograms.
    /// Parallel reduction is beneficial when bins_per_hist > 1000.
    pub parallel_reduce: bool,

    /// Threshold for auto-selecting row-parallel vs feature-parallel.
    /// Row-parallel is used when: rows > threshold × bins_per_hist
    pub row_parallel_threshold: f32,
}

impl Default for HistogramConfig {
    fn default() -> Self {
        Self {
            num_threads: rayon::current_num_threads(),
            min_chunk_size: 256,
            parallel_reduce: false,
            row_parallel_threshold: 4.0,
        }
    }
}

impl HistogramConfig {
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

// ============================================================================
// HistogramBuilder
// ============================================================================

/// Unified histogram builder with multiple parallelization strategies.
///
/// # Strategies
///
/// - **Sequential**: Single-threaded, builds all features in one pass
/// - **Feature-parallel**: Parallelizes across features (Rayon)
/// - **Row-parallel**: Parallelizes across rows with thread-local scratch
///
/// # Creating a Builder
///
/// ```ignore
/// // With row-parallel support (requires cuts for scratch allocation)
/// let builder = HistogramBuilder::new(&cuts, HistogramConfig::default());
/// ```
///
/// # Main API
///
/// The primary method is [`build`](Self::build), which dispatches to the
/// appropriate implementation based on the strategy parameter.
pub struct HistogramBuilder {
    /// Configuration.
    config: HistogramConfig,

    /// Per-thread scratch buffers for row-parallel building.
    scratch: RowParallelScratch,

    /// Total bins across all features.
    bins_per_hist: usize,

    /// Feature offsets for indexing into flat histogram.
    feature_offsets: Box<[usize]>,
}

impl HistogramBuilder {
    /// Create a builder with full strategy support.
    ///
    /// # Arguments
    ///
    /// * `cuts` - Quantization cuts (needed to allocate scratch buffers)
    /// * `config` - Builder configuration
    pub fn new(cuts: &BinCuts, config: HistogramConfig) -> Self {
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

    /// Get reference to configuration.
    #[inline]
    pub fn config(&self) -> &HistogramConfig {
        &self.config
    }

    /// Get total bins per histogram.
    #[inline]
    pub fn bins_per_hist(&self) -> usize {
        self.bins_per_hist
    }

    // ========================================================================
    // Main Build API
    // ========================================================================

    /// Build histogram into a pool slot using the specified strategy.
    ///
    /// This is the main entry point for histogram building. It dispatches
    /// to the appropriate implementation based on the strategy.
    ///
    /// # Arguments
    ///
    /// * `slot` - Pool slot to build into (will be reset)
    /// * `layout` - Histogram layout for feature offsets
    /// * `strategy` - Parallelization strategy
    /// * `index` - Quantized feature matrix
    /// * `grads` - Gradient slice for all rows
    /// * `hess` - Hessian slice for all rows
    /// * `rows` - Row indices belonging to this node
    pub fn build<B: BinIndex>(
        &mut self,
        slot: &mut HistogramSlotMut<'_>,
        layout: &super::types::HistogramLayout,
        strategy: super::super::grower::ParallelStrategy,
        index: &QuantizedMatrix<B>,
        grads: &[f32],
        hess: &[f32],
        rows: &[u32],
    ) {
        use super::super::grower::ParallelStrategy;

        // Reset slot first
        slot.reset();

        match strategy {
            ParallelStrategy::Sequential | ParallelStrategy::Auto => {
                self.build_sequential(slot, layout, index, grads, hess, rows);
            }
            ParallelStrategy::FeatureParallel => {
                self.build_feature_parallel(slot, layout, index, grads, hess, rows);
            }
            ParallelStrategy::RowParallel => {
                self.build_row_parallel_internal(slot, index, grads, hess, rows);
            }
        }
    }

    /// Build histogram into a pool, allocating the slot automatically.
    ///
    /// Convenience method that handles pool allocation and building in one call.
    pub fn build_into_pool<B: BinIndex>(
        &mut self,
        pool: &mut ContiguousHistogramPool,
        node_id: NodeId,
        layout: &super::types::HistogramLayout,
        strategy: super::super::grower::ParallelStrategy,
        index: &QuantizedMatrix<B>,
        grads: &[f32],
        hess: &[f32],
        rows: &[u32],
    ) {
        let mut slot = pool.get_or_allocate(node_id);
        self.build(&mut slot, layout, strategy, index, grads, hess, rows);
    }

    // ========================================================================
    // Strategy Implementations
    // ========================================================================

    /// Build histogram sequentially (single-threaded).
    fn build_sequential<B: BinIndex>(
        &self,
        slot: &mut HistogramSlotMut<'_>,
        layout: &super::types::HistogramLayout,
        index: &QuantizedMatrix<B>,
        grads: &[f32],
        hess: &[f32],
        rows: &[u32],
    ) {
        debug_assert_eq!(grads.len(), hess.len());
        let num_features = layout.num_features();

        for &row in rows {
            let row_idx = row as usize;
            let grad = grads[row_idx];
            let hess_val = hess[row_idx];

            for feat in 0..num_features {
                let bin = index.get(row, feat).to_usize();
                let global_bin = layout.feature_offset(feat) + bin;
                slot.add(global_bin, grad, hess_val);
            }
        }
    }

    /// Build histogram with per-feature parallelism.
    fn build_feature_parallel<B: BinIndex>(
        &self,
        slot: &mut HistogramSlotMut<'_>,
        layout: &super::types::HistogramLayout,
        index: &QuantizedMatrix<B>,
        grads: &[f32],
        hess: &[f32],
        rows: &[u32],
    ) {
        debug_assert_eq!(grads.len(), hess.len());
        let num_features = layout.num_features();

        // Use DisjointSlotWriter for safe parallel writes to disjoint regions
        let writer = DisjointSlotWriter::new(slot);

        (0..num_features).into_par_iter().for_each(|feat| {
            let (feat_start, feat_end) = layout.feature_range(feat);

            // SAFETY: Each feature writes to a disjoint range of bins.
            // DisjointSlotWriter encapsulates this invariant.
            unsafe {
                let num_bins = feat_end - feat_start;

                for (&row, bin) in rows
                    .iter()
                    .zip(index.iter_rows_for_feature(feat, rows))
                {
                    let row_idx = row as usize;
                    let bin_idx = bin.to_usize();
                    debug_assert!(bin_idx < num_bins);

                    let global_bin = feat_start + bin_idx;
                    writer.add_unchecked(global_bin, grads[row_idx], hess[row_idx]);
                }
            }
        });
    }

    /// Build histogram with row-parallel strategy (internal implementation).
    fn build_row_parallel_internal<B: BinIndex>(
        &mut self,
        slot: &mut HistogramSlotMut<'_>,
        index: &QuantizedMatrix<B>,
        grads: &[f32],
        hess: &[f32],
        rows: &[u32],
    ) {
        debug_assert_eq!(grads.len(), hess.len());

        // Reset scratch buffers
        self.scratch.reset_all();

        // Build into scratch buffers in parallel
        self.build_into_scratch(index, grads, hess, rows);

        // Reduce scratch into target
        if self.config.parallel_reduce && self.bins_per_hist > 1000 {
            self.scratch.reduce_into_parallel(slot);
        } else {
            self.scratch.reduce_into(slot);
        }
    }

    /// Core row-parallel building into scratch buffers.
    fn build_into_scratch<B: BinIndex>(
        &mut self,
        index: &QuantizedMatrix<B>,
        grads: &[f32],
        hess: &[f32],
        rows: &[u32],
    ) {
        let num_threads = self.config.num_threads;
        let num_features = self.feature_offsets.len();
        let bins_per_hist = self.bins_per_hist;
        let feature_offsets: &[usize] = &self.feature_offsets;

        // Divide rows into chunks for each thread
        let chunk_size = (rows.len() + num_threads - 1) / num_threads;
        let chunk_size = chunk_size.max(1);

        // Create thread-safe scratch accessor
        let scratch_writer = ScratchWriter::new(&mut self.scratch, bins_per_hist);

        // Use rayon::scope for parallel execution
        rayon::scope(|s| {
            for thread_id in 0..num_threads {
                let start = thread_id * chunk_size;
                if start >= rows.len() {
                    continue;
                }
                let end = (start + chunk_size).min(rows.len());
                let chunk_rows = &rows[start..end];

                // Clone writer reference for this thread
                let writer = scratch_writer;

                s.spawn(move |_| {
                    // SAFETY: Each thread_id accesses a disjoint region of scratch.
                    // ScratchWriter encapsulates this invariant.
                    unsafe {
                        for &row in chunk_rows {
                            let row_idx = row as usize;
                            let grad = *grads.get_unchecked(row_idx);
                            let hess_val = *hess.get_unchecked(row_idx);

                            for feat in 0..num_features {
                                let bin = index.get(row, feat as u32).to_usize();
                                let global_bin = *feature_offsets.get_unchecked(feat) + bin;
                                writer.add_unchecked(thread_id, global_bin, grad, hess_val);
                            }
                        }
                    }
                });
            }
        });
    }
}

impl std::fmt::Debug for HistogramBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HistogramBuilder")
            .field("config", &self.config)
            .field("bins_per_hist", &self.bins_per_hist)
            .field("num_features", &self.feature_offsets.len())
            .finish()
    }
}

// ============================================================================
// Thread-Safe Writers (encapsulating unsafe invariants)
// ============================================================================

/// Safe wrapper for parallel writes to disjoint histogram slot regions.
///
/// This type encapsulates the invariant that parallel threads write to
/// disjoint bin ranges (one range per feature). The raw pointer access
/// is hidden behind this abstraction.
#[derive(Clone, Copy)]
struct DisjointSlotWriter {
    sum_grad: *mut f32,
    sum_hess: *mut f32,
    count: *mut u32,
}

// SAFETY: DisjointSlotWriter is only used with disjoint writes.
// Each parallel task writes to a distinct feature's bin range.
unsafe impl Send for DisjointSlotWriter {}
unsafe impl Sync for DisjointSlotWriter {}

impl DisjointSlotWriter {
    fn new(slot: &mut HistogramSlotMut<'_>) -> Self {
        Self {
            sum_grad: slot.sum_grad.as_mut_ptr(),
            sum_hess: slot.sum_hess.as_mut_ptr(),
            count: slot.count.as_mut_ptr(),
        }
    }

    /// Add a sample to a bin.
    ///
    /// # Safety
    ///
    /// Caller must ensure `global_bin` is within this writer's valid range
    /// and that no other thread is writing to the same bin.
    #[inline]
    unsafe fn add_unchecked(&self, global_bin: usize, grad: f32, hess: f32) {
        // SAFETY: Caller guarantees global_bin is valid and we have exclusive access
        unsafe {
            *self.sum_grad.add(global_bin) += grad;
            *self.sum_hess.add(global_bin) += hess;
            *self.count.add(global_bin) += 1;
        }
    }
}

/// Safe wrapper for parallel writes to thread-local scratch regions.
///
/// This type encapsulates the invariant that each thread_id maps to a
/// disjoint region of the scratch buffer.
#[derive(Clone, Copy)]
struct ScratchWriter {
    sum_grad: *mut f32,
    sum_hess: *mut f32,
    count: *mut u32,
    bins_per_hist: usize,
}

// SAFETY: ScratchWriter is only used with disjoint writes.
// Each thread_id accesses a distinct scratch region.
unsafe impl Send for ScratchWriter {}
unsafe impl Sync for ScratchWriter {}

impl ScratchWriter {
    fn new(scratch: &mut RowParallelScratch, bins_per_hist: usize) -> Self {
        Self {
            sum_grad: scratch.sum_grads_ptr(),
            sum_hess: scratch.sum_hess_ptr(),
            count: scratch.counts_ptr(),
            bins_per_hist,
        }
    }

    /// Add a sample to a bin in a specific thread's scratch region.
    ///
    /// # Safety
    ///
    /// Caller must ensure:
    /// - `thread_id` is unique per concurrent caller
    /// - `global_bin < bins_per_hist`
    #[inline]
    unsafe fn add_unchecked(&self, thread_id: usize, global_bin: usize, grad: f32, hess: f32) {
        let offset = thread_id * self.bins_per_hist + global_bin;
        // SAFETY: Caller guarantees thread_id and global_bin are valid
        unsafe {
            *self.sum_grad.add(offset) += grad;
            *self.sum_hess.add(offset) += hess;
            *self.count.add(offset) += 1;
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::ColMatrix;
    use crate::training::gbtree::histogram::types::HistogramLayout;
    use crate::training::gbtree::quantize::{ExactQuantileCuts, Quantizer};

    fn make_test_data() -> (QuantizedMatrix<u8>, Vec<f32>, Vec<f32>, BinCuts) {
        // Create simple data: 10 rows, 2 features
        let data: Vec<f32> = vec![
            // Feature 0: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
            // Feature 1: all same value
            5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
        ];
        let matrix = ColMatrix::from_vec(data, 10, 2);

        let quantizer = Quantizer::from_data(&matrix, &ExactQuantileCuts::default(), 256);
        let quantized: QuantizedMatrix<u8> = quantizer.quantize(&matrix);
        let cuts = quantized.cuts().clone();

        let grads: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let hess: Vec<f32> = vec![1.0; 10];

        (quantized, grads, hess, cuts)
    }

    fn make_test_data_large() -> (QuantizedMatrix<u8>, Vec<f32>, Vec<f32>, BinCuts) {
        let n_rows = 100;
        let n_features = 3;

        let mut data = Vec::with_capacity(n_rows * n_features);
        for feat in 0..n_features {
            for row in 0..n_rows {
                data.push((row * (feat + 1)) as f32 % 10.0);
            }
        }
        let matrix = ColMatrix::from_vec(data, n_rows, n_features);

        let quantizer = Quantizer::from_data(&matrix, &ExactQuantileCuts::default(), 16);
        let quantized: QuantizedMatrix<u8> = quantizer.quantize(&matrix);
        let cuts = quantized.cuts().clone();

        let grads: Vec<f32> = (0..n_rows).map(|i| (i % 5) as f32).collect();
        let hess: Vec<f32> = vec![1.0; n_rows];

        (quantized, grads, hess, cuts)
    }

    #[test]
    fn test_histogram_builder_sequential() {
        let (quantized, grads, hess, cuts) = make_test_data();
        let rows: Vec<u32> = (0..10).collect();
        let layout = HistogramLayout::from_cuts(&cuts);

        let mut builder = HistogramBuilder::new(&cuts, HistogramConfig::default());
        let mut pool = ContiguousHistogramPool::new(1, layout.total_bins());

        use super::super::super::grower::ParallelStrategy;
        builder.build_into_pool(
            &mut pool,
            NodeId(0),
            &layout,
            ParallelStrategy::Sequential,
            &quantized,
            &grads,
            &hess,
            &rows,
        );

        let slot = pool.get(NodeId(0)).unwrap();
        let bins_feat0 = cuts.num_bins(0);

        // Total grad should be 0+1+2+...+9 = 45
        assert!((slot.total_grad_first_feature(bins_feat0) - 45.0).abs() < 1e-5);
        assert!((slot.total_hess_first_feature(bins_feat0) - 10.0).abs() < 1e-5);
        assert_eq!(slot.total_count_first_feature(bins_feat0), 10);
    }

    #[test]
    fn test_histogram_builder_feature_parallel() {
        let (quantized, grads, hess, cuts) = make_test_data();
        let rows: Vec<u32> = (0..10).collect();
        let layout = HistogramLayout::from_cuts(&cuts);

        let mut builder = HistogramBuilder::new(&cuts, HistogramConfig::default());
        let mut pool = ContiguousHistogramPool::new(1, layout.total_bins());

        use super::super::super::grower::ParallelStrategy;
        builder.build_into_pool(
            &mut pool,
            NodeId(0),
            &layout,
            ParallelStrategy::FeatureParallel,
            &quantized,
            &grads,
            &hess,
            &rows,
        );

        let slot = pool.get(NodeId(0)).unwrap();
        let bins_feat0 = cuts.num_bins(0);

        assert!((slot.total_grad_first_feature(bins_feat0) - 45.0).abs() < 1e-5);
        assert!((slot.total_hess_first_feature(bins_feat0) - 10.0).abs() < 1e-5);
        assert_eq!(slot.total_count_first_feature(bins_feat0), 10);
    }

    #[test]
    fn test_histogram_builder_row_parallel() {
        let (quantized, grads, hess, cuts) = make_test_data_large();
        let rows: Vec<u32> = (0..100).collect();
        let layout = HistogramLayout::from_cuts(&cuts);

        let mut builder = HistogramBuilder::new(&cuts, HistogramConfig::default());
        let mut pool = ContiguousHistogramPool::new(1, layout.total_bins());

        use super::super::super::grower::ParallelStrategy;
        builder.build_into_pool(
            &mut pool,
            NodeId(0),
            &layout,
            ParallelStrategy::RowParallel,
            &quantized,
            &grads,
            &hess,
            &rows,
        );

        let slot = pool.get(NodeId(0)).unwrap();
        let bins_feat0 = cuts.num_bins(0);
        assert_eq!(slot.total_count_first_feature(bins_feat0), 100);
    }

    #[test]
    fn test_strategies_produce_same_results() {
        let (quantized, grads, hess, cuts) = make_test_data_large();
        let rows: Vec<u32> = (0..100).collect();
        let layout = HistogramLayout::from_cuts(&cuts);

        use super::super::super::grower::ParallelStrategy;
        let strategies = [
            ParallelStrategy::Sequential,
            ParallelStrategy::FeatureParallel,
            ParallelStrategy::RowParallel,
        ];

        let mut results = Vec::new();

        for (i, &strategy) in strategies.iter().enumerate() {
            let mut builder = HistogramBuilder::new(&cuts, HistogramConfig::default());
            let mut pool = ContiguousHistogramPool::new(1, layout.total_bins());

            builder.build_into_pool(
                &mut pool,
                NodeId(0),
                &layout,
                strategy,
                &quantized,
                &grads,
                &hess,
                &rows,
            );

            let slot = pool.get(NodeId(0)).unwrap();
            let bins_feat0 = cuts.num_bins(0);
            results.push((
                slot.total_grad_first_feature(bins_feat0),
                slot.total_hess_first_feature(bins_feat0),
                slot.total_count_first_feature(bins_feat0),
            ));

            if i > 0 {
                assert!(
                    (results[i].0 - results[0].0).abs() < 1e-4,
                    "Strategy {:?} grad mismatch",
                    strategy
                );
                assert!(
                    (results[i].1 - results[0].1).abs() < 1e-4,
                    "Strategy {:?} hess mismatch",
                    strategy
                );
                assert_eq!(results[i].2, results[0].2, "Strategy {:?} count mismatch", strategy);
            }
        }
    }

    #[test]
    fn test_config_should_use_row_parallel() {
        let config = HistogramConfig {
            row_parallel_threshold: 4.0,
            ..Default::default()
        };

        assert!(config.should_use_row_parallel(10000, 100));
        assert!(!config.should_use_row_parallel(100, 100));
    }

    #[test]
    fn test_row_parallel_single_thread() {
        let (quantized, grads, hess, cuts) = make_test_data_large();
        let rows: Vec<u32> = (0..100).collect();
        let layout = HistogramLayout::from_cuts(&cuts);

        let config = HistogramConfig {
            num_threads: 1,
            ..Default::default()
        };
        let mut builder = HistogramBuilder::new(&cuts, config);
        let mut pool = ContiguousHistogramPool::new(1, layout.total_bins());

        use super::super::super::grower::ParallelStrategy;
        builder.build_into_pool(
            &mut pool,
            NodeId(0),
            &layout,
            ParallelStrategy::RowParallel,
            &quantized,
            &grads,
            &hess,
            &rows,
        );

        let slot = pool.get(NodeId(0)).unwrap();
        let bins_feat0 = cuts.num_bins(0);
        assert_eq!(slot.total_count_first_feature(bins_feat0), 100);
    }

    #[test]
    fn test_row_parallel_many_threads() {
        let (quantized, grads, hess, cuts) = make_test_data_large();
        let rows: Vec<u32> = (0..100).collect();
        let layout = HistogramLayout::from_cuts(&cuts);

        let config = HistogramConfig {
            num_threads: 16,
            min_chunk_size: 8,
            ..Default::default()
        };
        let mut builder = HistogramBuilder::new(&cuts, config);
        let mut pool = ContiguousHistogramPool::new(1, layout.total_bins());

        use super::super::super::grower::ParallelStrategy;
        builder.build_into_pool(
            &mut pool,
            NodeId(0),
            &layout,
            ParallelStrategy::RowParallel,
            &quantized,
            &grads,
            &hess,
            &rows,
        );

        let slot = pool.get(NodeId(0)).unwrap();
        let bins_feat0 = cuts.num_bins(0);
        assert_eq!(slot.total_count_first_feature(bins_feat0), 100);
    }
}

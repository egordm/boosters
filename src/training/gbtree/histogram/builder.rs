//! Unified histogram building with multiple parallelization strategies.
//!
//! This module provides [`HistogramBuilder`], the unified interface for building
//! histograms from quantized features and gradients. It supports three strategies:
//!
//! - **Sequential**: Single-threaded baseline
//! - **Feature-parallel**: Parallelizes across features (good for wide data)
//! - **Row-parallel**: Parallelizes across rows (good for tall data)
//!
//! See RFC-0025 for design rationale and performance analysis.

use rayon::prelude::*;

use super::node::NodeHistogram;
use super::pool::{ContiguousHistogramPool, HistogramSlotMut};
use super::scratch::RowParallelScratch;
use super::types::NodeId;
use crate::training::gbtree::quantize::{BinCuts, BinIndex, QuantizedMatrix};

/// Wrapper to make raw pointers Send + Sync for row-parallel building.
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

/// Unified histogram builder with multiple parallelization strategies.
///
/// Provides three building strategies:
/// - [`build_sequential`](Self::build_sequential): Single-threaded
/// - [`build_feature_parallel`](Self::build_feature_parallel): Per-feature parallelism
/// - [`build_row_parallel`](Self::build_row_parallel): Per-row parallelism
/// - [`build`](Self::build): Auto-selects best strategy based on data shape
///
/// # Creating a Builder
///
/// ```ignore
/// // Simple (uses defaults, no row-parallel support)
/// let builder = HistogramBuilder::default();
///
/// // With row-parallel support (requires cuts for scratch allocation)
/// let builder = HistogramBuilder::with_config(HistogramConfig::default(), &cuts);
/// ```
///
/// # Algorithm
///
/// 1. Reset histogram to zero
/// 2. For each row in the node:
///    - Look up gradient and hessian
///    - For each feature, add (grad, hess) to the corresponding bin
/// 3. Update cached totals
///
/// # Parallelization Strategy Selection
///
/// - **Sequential**: Small nodes (< 1000 rows)
/// - **Feature-parallel**: Wide data (many features, fewer rows)
/// - **Row-parallel**: Tall data (many rows, fewer bins)
///
/// The [`build`](Self::build) method auto-selects based on heuristics.
pub struct HistogramBuilder {
    /// Configuration.
    config: HistogramConfig,

    /// Per-thread scratch buffers for row-parallel building.
    /// `None` if row-parallel is not supported (created without cuts).
    scratch: Option<RowParallelScratch>,

    /// Total bins across all features (for row-parallel).
    bins_per_hist: usize,

    /// Feature offsets for indexing into flat histogram (for row-parallel).
    feature_offsets: Box<[usize]>,
}

impl Default for HistogramBuilder {
    /// Create a builder that supports sequential and feature-parallel strategies only.
    ///
    /// To use row-parallel, use [`with_config`](Self::with_config) instead.
    fn default() -> Self {
        Self {
            config: HistogramConfig::default(),
            scratch: None,
            bins_per_hist: 0,
            feature_offsets: Box::new([]),
        }
    }
}

impl HistogramBuilder {
    /// Create a builder with full strategy support including row-parallel.
    ///
    /// # Arguments
    ///
    /// * `config` - Builder configuration
    /// * `cuts` - Quantization cuts (needed to allocate scratch buffers)
    pub fn with_config(config: HistogramConfig, cuts: &BinCuts) -> Self {
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

        let scratch = Some(RowParallelScratch::new(num_threads, bins_per_hist));

        Self {
            config,
            scratch,
            bins_per_hist,
            feature_offsets,
        }
    }

    /// Get reference to configuration.
    pub fn config(&self) -> &HistogramConfig {
        &self.config
    }

    /// Get total bins per histogram (0 if row-parallel not configured).
    pub fn bins_per_hist(&self) -> usize {
        self.bins_per_hist
    }

    /// Check if row-parallel building is available.
    pub fn supports_row_parallel(&self) -> bool {
        self.scratch.is_some()
    }

    // ========================================================================
    // Sequential Strategy (into NodeHistogram - legacy API)
    // ========================================================================

    /// Build histogram (single-threaded, into NodeHistogram).
    ///
    /// This is the legacy API that builds into [`NodeHistogram`].
    /// For new code, prefer [`build_sequential_into_pool`](Self::build_sequential_into_pool).
    ///
    /// # Arguments
    ///
    /// * `hist` - Histogram to fill (will be reset first)
    /// * `index` - Quantized feature matrix
    /// * `grads` - Gradient slice for all rows (length = n_samples)
    /// * `hess` - Hessian slice for all rows (length = n_samples)
    /// * `rows` - Row indices belonging to this node
    pub fn build_sequential<B: BinIndex>(
        &self,
        hist: &mut NodeHistogram,
        index: &QuantizedMatrix<B>,
        grads: &[f32],
        hess: &[f32],
        rows: &[u32],
    ) {
        debug_assert_eq!(grads.len(), hess.len());
        hist.reset();
        let num_features = hist.num_features();

        for &row in rows {
            let row_idx = row as usize;
            let grad = grads[row_idx];
            let hess_val = hess[row_idx];

            for feat in 0..num_features {
                let bin = index.get(row, feat as u32).to_usize();
                hist.feature_mut(feat).add(bin, grad, hess_val);
            }
        }

        hist.update_totals();
    }

    // ========================================================================
    // Feature-Parallel Strategy (into NodeHistogram)
    // ========================================================================

    /// Build histogram with per-feature parallelism (into NodeHistogram).
    ///
    /// Each feature histogram is built independently in parallel using Rayon.
    /// This is the preferred method for datasets with many features.
    ///
    /// # Arguments
    ///
    /// Same as [`build_sequential`](Self::build_sequential).
    pub fn build_feature_parallel<B: BinIndex>(
        &self,
        hist: &mut NodeHistogram,
        index: &QuantizedMatrix<B>,
        grads: &[f32],
        hess: &[f32],
        rows: &[u32],
    ) {
        debug_assert_eq!(grads.len(), hess.len());
        hist.features_mut()
            .par_iter_mut()
            .enumerate()
            .for_each(|(feat, feat_hist)| {
                feat_hist.reset();

                for (row, bin) in rows
                    .iter()
                    .zip(index.iter_rows_for_feature(feat as u32, rows))
                {
                    let row_idx = *row as usize;
                    feat_hist.add(bin.to_usize(), grads[row_idx], hess[row_idx]);
                }
            });

        hist.update_totals();
    }

    /// Build histogram using column iteration (cache-friendly, sequential).
    ///
    /// Processes one feature at a time, which is cache-friendly for
    /// column-major quantized matrices.
    pub fn build_column_wise<B: BinIndex>(
        &self,
        hist: &mut NodeHistogram,
        index: &QuantizedMatrix<B>,
        grads: &[f32],
        hess: &[f32],
        rows: &[u32],
    ) {
        debug_assert_eq!(grads.len(), hess.len());
        hist.reset();
        let num_features = hist.num_features();

        for feat in 0..num_features {
            let feat_hist = hist.feature_mut(feat);

            for (&row, bin) in rows
                .iter()
                .zip(index.iter_rows_for_feature(feat as u32, rows))
            {
                let row_idx = row as usize;
                feat_hist.add(bin.to_usize(), grads[row_idx], hess[row_idx]);
            }
        }

        hist.update_totals();
    }

    // ========================================================================
    // Row-Parallel Strategy (into ContiguousHistogramPool)
    // ========================================================================

    /// Build histogram using row-parallel strategy into a pool slot.
    ///
    /// Partitions rows across threads, each thread accumulates into local
    /// scratch buffers, then reduces all scratch into the target.
    ///
    /// # Requirements
    ///
    /// Builder must be created with [`with_config`](Self::with_config) to
    /// have scratch buffers allocated.
    ///
    /// # Arguments
    ///
    /// * `pool` - Histogram pool for allocation
    /// * `node_id` - Node identifier for the histogram
    /// * `index` - Quantized feature matrix
    /// * `grads` - Gradient values (length = n_samples)
    /// * `hess` - Hessian values (length = n_samples)
    /// * `rows` - Row indices belonging to this node
    ///
    /// # Panics
    ///
    /// Panics if row-parallel is not supported (no scratch buffers).
    pub fn build_row_parallel<B: BinIndex>(
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

        let scratch = self
            .scratch
            .as_mut()
            .expect("row-parallel requires HistogramBuilder::with_config()");

        // Reset scratch buffers
        scratch.reset_all();

        // Get or allocate target histogram
        let mut target = pool.get_or_allocate(node_id);
        target.reset();

        // Build using row-parallel strategy
        self.build_row_parallel_core(index, grads, hess, rows);

        // Reduce scratch into target
        let scratch = self.scratch.as_ref().unwrap();
        if self.config.parallel_reduce && self.bins_per_hist > 1000 {
            scratch.reduce_into_parallel(&mut target);
        } else {
            scratch.reduce_into(&mut target);
        }
    }

    /// Build histogram into a mutable slot directly (row-parallel).
    ///
    /// Lower-level API for when you already have a slot reference.
    pub fn build_row_parallel_into_slot<B: BinIndex>(
        &mut self,
        target: &mut HistogramSlotMut<'_>,
        index: &QuantizedMatrix<B>,
        grads: &[f32],
        hess: &[f32],
        rows: &[u32],
    ) {
        debug_assert_eq!(grads.len(), hess.len());
        debug_assert_eq!(target.num_bins(), self.bins_per_hist);

        let scratch = self
            .scratch
            .as_mut()
            .expect("row-parallel requires HistogramBuilder::with_config()");

        // Reset scratch buffers and target
        scratch.reset_all();
        target.reset();

        // Build using row-parallel strategy
        self.build_row_parallel_core(index, grads, hess, rows);

        // Reduce scratch into target
        let scratch = self.scratch.as_ref().unwrap();
        if self.config.parallel_reduce && self.bins_per_hist > 1000 {
            scratch.reduce_into_parallel(target);
        } else {
            scratch.reduce_into(target);
        }
    }

    /// Core row-parallel building algorithm.
    ///
    /// Partitions rows across threads and each thread accumulates into
    /// its local scratch buffer.
    fn build_row_parallel_core<B: BinIndex>(
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
        let scratch = self.scratch.as_mut().unwrap();
        let sg_base = SendSyncPtr(scratch.sum_grads_ptr());
        let sh_base = SendSyncPtr(scratch.sum_hess_ptr());
        let sc_base = SendSyncPtr(scratch.counts_ptr());

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

                    // Extract raw pointers inside the spawn
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
}

impl std::fmt::Debug for HistogramBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HistogramBuilder")
            .field("config", &self.config)
            .field("bins_per_hist", &self.bins_per_hist)
            .field("num_features", &self.feature_offsets.len())
            .field("supports_row_parallel", &self.scratch.is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::ColMatrix;
    use crate::training::gbtree::quantize::{ExactQuantileCuts, Quantizer};

    fn make_test_data() -> (QuantizedMatrix<u8>, Vec<f32>, Vec<f32>) {
        // Create simple data: 10 rows, 2 features
        let data: Vec<f32> = vec![
            // Feature 0: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
            // Feature 1: all same value
            5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
        ];
        let matrix = ColMatrix::from_vec(data, 10, 2);

        // Quantize
        let quantizer = Quantizer::from_data(&matrix, &ExactQuantileCuts::default(), 256);
        let quantized: QuantizedMatrix<u8> = quantizer.quantize(&matrix);

        // Create gradients: grad = row_id, hess = 1.0
        let grads: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let hess: Vec<f32> = vec![1.0; 10];

        (quantized, grads, hess)
    }

    fn make_test_data_large() -> (QuantizedMatrix<u8>, Vec<f32>, Vec<f32>, BinCuts) {
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
    fn test_histogram_builder_basic() {
        let (quantized, grads, hess) = make_test_data();
        let rows: Vec<u32> = (0..10).collect();

        let builder = HistogramBuilder::default();
        let mut hist = NodeHistogram::new(quantized.cuts());
        builder.build_sequential(&mut hist, &quantized, &grads, &hess, &rows);

        // Total grad should be 0+1+2+...+9 = 45
        // Total hess should be 10 (each row contributes 1.0)
        assert_eq!(hist.total_count(), 10);
        assert!((hist.total_grad() - 45.0).abs() < 1e-5);
        assert!((hist.total_hess() - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_histogram_builder_subset() {
        let (quantized, grads, hess) = make_test_data();

        // Build histogram for subset of rows
        let rows: Vec<u32> = vec![0, 2, 4, 6, 8]; // Even rows

        let builder = HistogramBuilder::default();
        let mut hist = NodeHistogram::new(quantized.cuts());
        builder.build_sequential(&mut hist, &quantized, &grads, &hess, &rows);

        // Grad sum: 0+2+4+6+8 = 20, hess: 5
        assert_eq!(hist.total_count(), 5);
        assert!((hist.total_grad() - 20.0).abs() < 1e-5);
        assert!((hist.total_hess() - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_feature_parallel_matches_sequential() {
        let (quantized, grads, hess) = make_test_data();
        let rows: Vec<u32> = (0..10).collect();

        let builder = HistogramBuilder::default();

        let mut hist_seq = NodeHistogram::new(quantized.cuts());
        builder.build_sequential(&mut hist_seq, &quantized, &grads, &hess, &rows);

        let mut hist_par = NodeHistogram::new(quantized.cuts());
        builder.build_feature_parallel(&mut hist_par, &quantized, &grads, &hess, &rows);

        // Compare totals
        assert_eq!(hist_seq.total_count(), hist_par.total_count());
        assert!((hist_seq.total_grad() - hist_par.total_grad()).abs() < 1e-5);
        assert!((hist_seq.total_hess() - hist_par.total_hess()).abs() < 1e-5);

        // Compare per-feature histograms
        for feat in 0..hist_seq.num_features() {
            let seq = hist_seq.feature(feat);
            let par = hist_par.feature(feat);
            for bin in 0..seq.num_bins() as usize {
                let (sg, sh, sc) = seq.bin_stats(bin);
                let (pg, ph, pc) = par.bin_stats(bin);
                assert!(
                    (sg - pg).abs() < 1e-5,
                    "Feature {} bin {} grad mismatch",
                    feat,
                    bin
                );
                assert!(
                    (sh - ph).abs() < 1e-5,
                    "Feature {} bin {} hess mismatch",
                    feat,
                    bin
                );
                assert_eq!(sc, pc, "Feature {} bin {} count mismatch", feat, bin);
            }
        }
    }

    #[test]
    fn test_histogram_subtraction_correctness() {
        let (quantized, grads, hess) = make_test_data();

        let builder = HistogramBuilder::default();

        // Build parent histogram (all rows)
        let all_rows: Vec<u32> = (0..10).collect();
        let mut parent = NodeHistogram::new(quantized.cuts());
        builder.build_sequential(&mut parent, &quantized, &grads, &hess, &all_rows);

        // Build left child histogram (first 6 rows)
        let left_rows: Vec<u32> = vec![0, 1, 2, 3, 4, 5];
        let mut left = NodeHistogram::new(quantized.cuts());
        builder.build_sequential(&mut left, &quantized, &grads, &hess, &left_rows);

        // Build right child directly for comparison
        let right_rows: Vec<u32> = vec![6, 7, 8, 9];
        let mut right_direct = NodeHistogram::new(quantized.cuts());
        builder.build_sequential(&mut right_direct, &quantized, &grads, &hess, &right_rows);

        // Compute right via subtraction: parent - left = right
        let right_subtracted = &parent - &left;

        // Compare
        assert_eq!(right_direct.total_count(), right_subtracted.total_count());
        assert!((right_direct.total_grad() - right_subtracted.total_grad()).abs() < 1e-5);
        assert!((right_direct.total_hess() - right_subtracted.total_hess()).abs() < 1e-5);

        // Verify actual values: right should have rows 6,7,8,9
        // Grad sum: 6+7+8+9 = 30
        assert!((right_subtracted.total_grad() - 30.0).abs() < 1e-5);
        assert_eq!(right_subtracted.total_count(), 4);
    }

    #[test]
    fn test_column_wise_matches_row_wise() {
        let (quantized, grads, hess) = make_test_data();
        let rows: Vec<u32> = (0..10).collect();

        let builder = HistogramBuilder::default();

        let mut hist_row = NodeHistogram::new(quantized.cuts());
        builder.build_sequential(&mut hist_row, &quantized, &grads, &hess, &rows);

        let mut hist_col = NodeHistogram::new(quantized.cuts());
        builder.build_column_wise(&mut hist_col, &quantized, &grads, &hess, &rows);

        // Compare totals
        assert_eq!(hist_row.total_count(), hist_col.total_count());
        assert!((hist_row.total_grad() - hist_col.total_grad()).abs() < 1e-5);
        assert!((hist_row.total_hess() - hist_col.total_hess()).abs() < 1e-5);
    }

    // ========================================================================
    // Row-parallel tests
    // ========================================================================

    #[test]
    fn test_row_parallel_basic() {
        let (quantized, grads, hess, cuts) = make_test_data_large();
        let rows: Vec<u32> = (0..100).collect();

        let config = HistogramConfig::default();
        let mut builder = HistogramBuilder::with_config(config, &cuts);

        // Create pool and build
        let mut pool = ContiguousHistogramPool::new(1, cuts.total_bins());
        let node_id = NodeId(0);
        builder.build_row_parallel(&mut pool, node_id, &quantized, &grads, &hess, &rows);

        // Verify we got a histogram
        let hist = pool.get(node_id).expect("histogram should exist");

        // Total count should be 100 (using first feature only)
        let bins_feat0 = cuts.num_bins(0);
        assert_eq!(hist.total_count_first_feature(bins_feat0), 100);
    }

    #[test]
    fn test_row_parallel_matches_sequential() {
        let (quantized, grads, hess, cuts) = make_test_data_large();
        let rows: Vec<u32> = (0..100).collect();

        // Build with sequential builder
        let builder_seq = HistogramBuilder::default();
        let mut hist_seq = NodeHistogram::new(quantized.cuts());
        builder_seq.build_sequential(&mut hist_seq, &quantized, &grads, &hess, &rows);

        // Build with row-parallel builder
        let config = HistogramConfig::default();
        let mut builder = HistogramBuilder::with_config(config, &cuts);
        let mut pool = ContiguousHistogramPool::new(1, cuts.total_bins());
        let node_id = NodeId(0);
        builder.build_row_parallel(&mut pool, node_id, &quantized, &grads, &hess, &rows);
        let hist_par = pool.get(node_id).unwrap();

        // Compare totals (use first feature's bins for count comparison)
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
    fn test_row_parallel_subset() {
        let (quantized, grads, hess, cuts) = make_test_data_large();

        // Build for subset of rows
        let rows: Vec<u32> = (0..50).collect();

        let config = HistogramConfig::default();
        let mut builder = HistogramBuilder::with_config(config, &cuts);
        let mut pool = ContiguousHistogramPool::new(1, cuts.total_bins());
        let node_id = NodeId(0);
        builder.build_row_parallel(&mut pool, node_id, &quantized, &grads, &hess, &rows);

        let hist = pool.get(node_id).unwrap();
        let bins_feat0 = cuts.num_bins(0);
        assert_eq!(hist.total_count_first_feature(bins_feat0), 50);
    }

    #[test]
    fn test_row_parallel_single_thread() {
        let (quantized, grads, hess, cuts) = make_test_data_large();
        let rows: Vec<u32> = (0..100).collect();

        // Use single thread config
        let config = HistogramConfig {
            num_threads: 1,
            ..Default::default()
        };
        let mut builder = HistogramBuilder::with_config(config, &cuts);

        let mut pool = ContiguousHistogramPool::new(1, cuts.total_bins());
        let node_id = NodeId(0);
        builder.build_row_parallel(&mut pool, node_id, &quantized, &grads, &hess, &rows);

        let hist = pool.get(node_id).unwrap();
        let bins_feat0 = cuts.num_bins(0);
        assert_eq!(hist.total_count_first_feature(bins_feat0), 100);
    }

    #[test]
    fn test_row_parallel_rebuild() {
        let (quantized, grads, hess, cuts) = make_test_data_large();

        let config = HistogramConfig::default();
        let mut builder = HistogramBuilder::with_config(config, &cuts);
        let mut pool = ContiguousHistogramPool::new(2, cuts.total_bins());

        // Build first histogram
        let rows1: Vec<u32> = (0..50).collect();
        let node1 = NodeId(0);
        builder.build_row_parallel(&mut pool, node1, &quantized, &grads, &hess, &rows1);

        // Build second histogram
        let rows2: Vec<u32> = (50..100).collect();
        let node2 = NodeId(1);
        builder.build_row_parallel(&mut pool, node2, &quantized, &grads, &hess, &rows2);

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
        let config = HistogramConfig {
            row_parallel_threshold: 4.0,
            ..Default::default()
        };

        // 10000 rows, 100 bins -> 10000 > 4 * 100 = 400 -> should use row parallel
        assert!(config.should_use_row_parallel(10000, 100));

        // 100 rows, 100 bins -> 100 > 4 * 100 = 400 -> should NOT use row parallel
        assert!(!config.should_use_row_parallel(100, 100));
    }

    #[test]
    fn test_row_parallel_many_threads() {
        let (quantized, grads, hess, cuts) = make_test_data_large();
        let rows: Vec<u32> = (0..100).collect();

        // More threads than reasonable chunks
        let config = HistogramConfig {
            num_threads: 16,
            min_chunk_size: 8,
            ..Default::default()
        };
        let mut builder = HistogramBuilder::with_config(config, &cuts);

        let mut pool = ContiguousHistogramPool::new(1, cuts.total_bins());
        let node_id = NodeId(0);
        builder.build_row_parallel(&mut pool, node_id, &quantized, &grads, &hess, &rows);

        let hist = pool.get(node_id).unwrap();
        let bins_feat0 = cuts.num_bins(0);
        assert_eq!(hist.total_count_first_feature(bins_feat0), 100);
    }

    #[test]
    fn test_supports_row_parallel() {
        // Default builder doesn't support row-parallel
        let default_builder = HistogramBuilder::default();
        assert!(!default_builder.supports_row_parallel());

        // Builder with config does support it
        let (_, _, _, cuts) = make_test_data_large();
        let builder = HistogramBuilder::with_config(HistogramConfig::default(), &cuts);
        assert!(builder.supports_row_parallel());
    }
}

//! Per-thread scratch buffers for row-parallel histogram building.
//!
//! This module provides thread-local scratch space for accumulating gradients
//! during row-parallel histogram construction, plus efficient reduction to
//! merge results into the target histogram.
//!
//! # Architecture
//!
//! ```text
//! Thread 0: [scratch_hist_0]  ──┐
//! Thread 1: [scratch_hist_1]  ──┼──► reduce_into() ──► [target histogram]
//! Thread 2: [scratch_hist_2]  ──┤
//! ...                           │
//! Thread N: [scratch_hist_N]  ──┘
//! ```
//!
//! Each thread accumulates into its own scratch buffer (no synchronization),
//! then a single reduce step sums all buffers into the target.
//!
//! See RFC-0025 for design rationale.

use rayon::prelude::*;

use super::pool::HistogramSlotMut;

/// Per-thread scratch space for row-parallel histogram building.
///
/// Each thread accumulates gradients into its local scratch buffer,
/// then results are merged into the target histogram via `reduce_into()`.
///
/// Uses SoA layout matching our `FeatureHistogram` and `ContiguousHistogramPool`.
///
/// # Memory Usage
///
/// Total: `num_threads × bins_per_hist × 12` bytes
///
/// For 8 threads, 100 features, 256 bins = 8 × 25,600 × 12 = ~2.4 MB
///
/// # Example
///
/// ```ignore
/// let mut scratch = RowParallelScratch::new(8, 25600);
///
/// // In parallel region, each thread uses its buffer
/// rows.par_chunks(chunk_size).enumerate().for_each(|(tid, chunk)| {
///     let buf = scratch.get_buffer(tid);
///     for &row in chunk {
///         // accumulate into buf
///     }
/// });
///
/// // Merge all thread results into target
/// scratch.reduce_into(&mut target_hist);
/// ```
pub struct RowParallelScratch {
    /// Per-thread gradient sums: flattened [thread][bins]
    sum_grads: Box<[f32]>,

    /// Per-thread hessian sums: flattened [thread][bins]
    sum_hess: Box<[f32]>,

    /// Per-thread counts: flattened [thread][bins]
    counts: Box<[u32]>,

    /// Number of bins per histogram.
    bins_per_hist: usize,

    /// Number of threads.
    num_threads: usize,
}

/// Mutable view into a thread's scratch buffer.
#[derive(Debug)]
#[allow(dead_code)]
pub struct ScratchSlotMut<'a> {
    /// Sum of gradients per bin.
    pub sum_grad: &'a mut [f32],
    /// Sum of hessians per bin.
    pub sum_hess: &'a mut [f32],
    /// Count per bin.
    pub count: &'a mut [u32],
}

#[allow(dead_code)]
impl ScratchSlotMut<'_> {
    /// Reset this buffer to zero.
    #[inline]
    pub fn reset(&mut self) {
        self.sum_grad.fill(0.0);
        self.sum_hess.fill(0.0);
        self.count.fill(0);
    }

    /// Number of bins in this buffer.
    #[inline]
    pub fn num_bins(&self) -> usize {
        self.sum_grad.len()
    }

    /// Add a sample to a bin.
    #[inline]
    pub fn add(&mut self, bin: usize, grad: f32, hess: f32) {
        debug_assert!(bin < self.sum_grad.len());
        unsafe {
            *self.sum_grad.get_unchecked_mut(bin) += grad;
            *self.sum_hess.get_unchecked_mut(bin) += hess;
            *self.count.get_unchecked_mut(bin) += 1;
        }
    }
}

#[allow(dead_code)]
impl RowParallelScratch {
    /// Create scratch space for the given number of threads.
    ///
    /// # Arguments
    ///
    /// * `num_threads` - Number of threads (typically `rayon::current_num_threads()`)
    /// * `bins_per_hist` - Total bins per histogram (sum across all features)
    ///
    /// # Memory
    ///
    /// Allocates `num_threads × bins_per_hist × 12` bytes.
    pub fn new(num_threads: usize, bins_per_hist: usize) -> Self {
        let total = num_threads * bins_per_hist;
        Self {
            sum_grads: vec![0.0f32; total].into_boxed_slice(),
            sum_hess: vec![0.0f32; total].into_boxed_slice(),
            counts: vec![0u32; total].into_boxed_slice(),
            bins_per_hist,
            num_threads,
        }
    }

    /// Get mutable scratch buffer for a specific thread.
    ///
    /// # Safety
    ///
    /// The caller must ensure that each thread uses a unique `thread_id`
    /// in the range `[0, num_threads)`. This is naturally satisfied when
    /// using `enumerate()` with Rayon's parallel iterators.
    ///
    /// # Panics
    ///
    /// Panics in debug mode if `thread_id >= num_threads`.
    #[inline]
    pub fn get_buffer(&mut self, thread_id: usize) -> ScratchSlotMut<'_> {
        debug_assert!(thread_id < self.num_threads);
        let start = thread_id * self.bins_per_hist;

        // SAFETY: We need to split the mutable references.
        // This is safe because each thread accesses a disjoint range.
        unsafe {
            let sum_grad =
                std::slice::from_raw_parts_mut(self.sum_grads.as_ptr().add(start).cast_mut(), self.bins_per_hist);
            let sum_hess =
                std::slice::from_raw_parts_mut(self.sum_hess.as_ptr().add(start).cast_mut(), self.bins_per_hist);
            let count =
                std::slice::from_raw_parts_mut(self.counts.as_ptr().add(start).cast_mut(), self.bins_per_hist);

            ScratchSlotMut {
                sum_grad,
                sum_hess,
                count,
            }
        }
    }

    /// Get a raw pointer to the scratch buffer for a thread.
    ///
    /// This is used internally for parallel access where we need
    /// to work around Rust's borrow checker.
    ///
    /// # Safety
    ///
    /// Caller must ensure no data races. Unsafe operations within use
    /// raw pointer arithmetic.
    #[inline]
    pub(crate) unsafe fn buffer_ptrs(&self, thread_id: usize) -> (*mut f32, *mut f32, *mut u32) {
        let start = thread_id * self.bins_per_hist;
        unsafe {
            (
                self.sum_grads.as_ptr().add(start).cast_mut(),
                self.sum_hess.as_ptr().add(start).cast_mut(),
                self.counts.as_ptr().add(start).cast_mut(),
            )
        }
    }

    /// Get raw pointer to sum_grads array for parallel access.
    ///
    /// # Safety
    ///
    /// Caller must ensure no data races.
    #[inline]
    pub(crate) fn sum_grads_ptr(&self) -> *mut f32 {
        self.sum_grads.as_ptr() as *mut f32
    }

    /// Get raw pointer to sum_hess array for parallel access.
    ///
    /// # Safety
    ///
    /// Caller must ensure no data races.
    #[inline]
    pub(crate) fn sum_hess_ptr(&self) -> *mut f32 {
        self.sum_hess.as_ptr() as *mut f32
    }

    /// Get raw pointer to counts array for parallel access.
    ///
    /// # Safety
    ///
    /// Caller must ensure no data races.
    #[inline]
    pub(crate) fn counts_ptr(&self) -> *mut u32 {
        self.counts.as_ptr() as *mut u32
    }

    /// Reset all thread buffers to zero.
    pub fn reset_all(&mut self) {
        self.sum_grads.fill(0.0);
        self.sum_hess.fill(0.0);
        self.counts.fill(0);
    }

    /// Reset a specific thread's buffer to zero.
    pub fn reset(&mut self, thread_id: usize) {
        debug_assert!(thread_id < self.num_threads);
        let start = thread_id * self.bins_per_hist;
        let end = start + self.bins_per_hist;
        self.sum_grads[start..end].fill(0.0);
        self.sum_hess[start..end].fill(0.0);
        self.counts[start..end].fill(0);
    }

    /// Number of threads this scratch supports.
    #[inline]
    pub fn num_threads(&self) -> usize {
        self.num_threads
    }

    /// Number of bins per histogram.
    #[inline]
    pub fn bins_per_hist(&self) -> usize {
        self.bins_per_hist
    }

    /// Reduce all thread buffers into the target histogram.
    ///
    /// This sums all thread-local accumulations into the target.
    /// The target is NOT reset first - caller should reset if needed.
    ///
    /// Uses linear reduction which is efficient for typical thread counts (≤16).
    pub fn reduce_into(&self, target: &mut HistogramSlotMut<'_>) {
        debug_assert_eq!(target.num_bins(), self.bins_per_hist);
        reduce_histograms(
            target.sum_grad,
            target.sum_hess,
            target.count,
            &self.sum_grads,
            &self.sum_hess,
            &self.counts,
            self.bins_per_hist,
            self.num_threads,
        );
    }

    /// Reduce using parallel SIMD-friendly loops.
    ///
    /// More efficient for large histograms (>1000 bins).
    pub fn reduce_into_parallel(&self, target: &mut HistogramSlotMut<'_>) {
        debug_assert_eq!(target.num_bins(), self.bins_per_hist);
        reduce_histograms_parallel(
            target.sum_grad,
            target.sum_hess,
            target.count,
            &self.sum_grads,
            &self.sum_hess,
            &self.counts,
            self.bins_per_hist,
            self.num_threads,
        );
    }
}

impl std::fmt::Debug for RowParallelScratch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RowParallelScratch")
            .field("num_threads", &self.num_threads)
            .field("bins_per_hist", &self.bins_per_hist)
            .field(
                "memory_bytes",
                &(self.sum_grads.len() * 4 + self.sum_hess.len() * 4 + self.counts.len() * 4),
            )
            .finish()
    }
}

// ============================================================================
// Reduction Functions
// ============================================================================

/// Merge thread-local histograms into target (linear reduction).
///
/// This is the core reduction operation. For each bin, we sum the contributions
/// from all threads.
///
/// # Algorithm
///
/// ```text
/// for each bin b:
///     target.grad[b] = sum(thread[t].grad[b] for t in 0..num_threads)
///     target.hess[b] = sum(thread[t].hess[b] for t in 0..num_threads)
///     target.count[b] = sum(thread[t].count[b] for t in 0..num_threads)
/// ```
///
/// # Complexity
///
/// O(num_threads × bins_per_hist)
#[inline]
fn reduce_histograms(
    target_grad: &mut [f32],
    target_hess: &mut [f32],
    target_count: &mut [u32],
    source_grads: &[f32],
    source_hess: &[f32],
    source_counts: &[u32],
    bins_per_hist: usize,
    num_threads: usize,
) {
    // Zero targets first
    target_grad.fill(0.0);
    target_hess.fill(0.0);
    target_count.fill(0);

    // Sum contributions from each thread
    for t in 0..num_threads {
        let start = t * bins_per_hist;
        let end = start + bins_per_hist;

        let tg = &source_grads[start..end];
        let th = &source_hess[start..end];
        let tc = &source_counts[start..end];

        // Process grads
        for (dst, src) in target_grad.iter_mut().zip(tg.iter()) {
            *dst += *src;
        }

        // Process hess
        for (dst, src) in target_hess.iter_mut().zip(th.iter()) {
            *dst += *src;
        }

        // Process counts
        for (dst, src) in target_count.iter_mut().zip(tc.iter()) {
            *dst += *src;
        }
    }
}

/// Parallel reduction for large histograms.
///
/// Uses Rayon to parallelize across bins, which is beneficial when
/// `bins_per_hist` is large (>1000).
#[inline]
fn reduce_histograms_parallel(
    target_grad: &mut [f32],
    target_hess: &mut [f32],
    target_count: &mut [u32],
    source_grads: &[f32],
    source_hess: &[f32],
    source_counts: &[u32],
    bins_per_hist: usize,
    num_threads: usize,
) {
    // Process grads in parallel
    target_grad
        .par_iter_mut()
        .enumerate()
        .for_each(|(bin, dst)| {
            let mut sum = 0.0f32;
            for t in 0..num_threads {
                sum += source_grads[t * bins_per_hist + bin];
            }
            *dst = sum;
        });

    // Process hess in parallel
    target_hess
        .par_iter_mut()
        .enumerate()
        .for_each(|(bin, dst)| {
            let mut sum = 0.0f32;
            for t in 0..num_threads {
                sum += source_hess[t * bins_per_hist + bin];
            }
            *dst = sum;
        });

    // Process counts in parallel
    target_count
        .par_iter_mut()
        .enumerate()
        .for_each(|(bin, dst)| {
            let mut sum = 0u32;
            for t in 0..num_threads {
                sum += source_counts[t * bins_per_hist + bin];
            }
            *dst = sum;
        });
}

/// Subtract two histogram slices: target = parent - child.
///
/// Used for histogram subtraction optimization.
#[inline]
#[allow(dead_code)]
pub fn subtract_histograms(
    target_grad: &mut [f32],
    target_hess: &mut [f32],
    target_count: &mut [u32],
    parent_grad: &[f32],
    parent_hess: &[f32],
    parent_count: &[u32],
    child_grad: &[f32],
    child_hess: &[f32],
    child_count: &[u32],
) {
    debug_assert_eq!(target_grad.len(), parent_grad.len());
    debug_assert_eq!(target_grad.len(), child_grad.len());

    for i in 0..target_grad.len() {
        target_grad[i] = parent_grad[i] - child_grad[i];
        target_hess[i] = parent_hess[i] - child_hess[i];
        target_count[i] = parent_count[i] - child_count[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::gbtree::histogram::pool::ContiguousHistogramPool;

    #[test]
    fn test_scratch_basic() {
        let mut scratch = RowParallelScratch::new(4, 10);

        assert_eq!(scratch.num_threads(), 4);
        assert_eq!(scratch.bins_per_hist(), 10);

        // Get buffer for thread 0
        let mut buf = scratch.get_buffer(0);
        assert_eq!(buf.num_bins(), 10);

        // Add some data
        buf.add(0, 1.0, 0.5);
        buf.add(0, 2.0, 1.0);

        // Verify via getting buffer again
        // Note: We need to drop the first borrow
        drop(buf);
        let _buf = scratch.get_buffer(0);
        // Can't read directly from ScratchSlotMut, but we can test via reduce
    }

    #[test]
    fn test_scratch_reset() {
        let mut scratch = RowParallelScratch::new(2, 10);

        // Add data to thread 0
        let mut buf = scratch.get_buffer(0);
        buf.add(0, 1.0, 0.5);
        drop(buf);

        // Reset thread 0
        scratch.reset(0);

        // Reduce to verify it's zero
        let mut pool = ContiguousHistogramPool::new(1, 10);
        let mut target = pool.get_or_allocate(super::super::types::NodeId(0));
        target.reset();
        scratch.reduce_into(&mut target);

        // Should be all zeros
        let hist = pool.get(super::super::types::NodeId(0)).unwrap();
        assert_eq!(hist.total_count(), 0);
    }

    #[test]
    fn test_scratch_reset_all() {
        let mut scratch = RowParallelScratch::new(2, 10);

        // Add data to both threads
        {
            let mut buf = scratch.get_buffer(0);
            buf.add(0, 1.0, 0.5);
        }
        {
            let mut buf = scratch.get_buffer(1);
            buf.add(1, 2.0, 1.0);
        }

        // Reset all
        scratch.reset_all();

        // Reduce to verify all zeros
        let mut pool = ContiguousHistogramPool::new(1, 10);
        let mut target = pool.get_or_allocate(super::super::types::NodeId(0));
        target.reset();
        scratch.reduce_into(&mut target);

        let hist = pool.get(super::super::types::NodeId(0)).unwrap();
        assert_eq!(hist.total_count(), 0);
    }

    #[test]
    fn test_reduce_single_thread() {
        let mut scratch = RowParallelScratch::new(1, 10);

        // Add data to single thread
        let mut buf = scratch.get_buffer(0);
        buf.add(0, 1.0, 0.5);
        buf.add(0, 2.0, 1.0);
        buf.add(1, 3.0, 1.5);
        drop(buf);

        // Reduce
        let mut pool = ContiguousHistogramPool::new(1, 10);
        let mut target = pool.get_or_allocate(super::super::types::NodeId(0));
        target.reset();
        scratch.reduce_into(&mut target);

        // Verify
        let hist = pool.get(super::super::types::NodeId(0)).unwrap();
        assert_eq!(hist.bin_stats(0), (3.0, 1.5, 2));
        assert_eq!(hist.bin_stats(1), (3.0, 1.5, 1));
        assert_eq!(hist.total_count(), 3);
    }

    #[test]
    fn test_reduce_multiple_threads() {
        let mut scratch = RowParallelScratch::new(3, 10);

        // Thread 0: bin 0 = (1, 0.5, 1)
        {
            let mut buf = scratch.get_buffer(0);
            buf.add(0, 1.0, 0.5);
        }

        // Thread 1: bin 0 = (2, 1.0, 1), bin 1 = (3, 1.5, 1)
        {
            let mut buf = scratch.get_buffer(1);
            buf.add(0, 2.0, 1.0);
            buf.add(1, 3.0, 1.5);
        }

        // Thread 2: bin 0 = (4, 2.0, 1)
        {
            let mut buf = scratch.get_buffer(2);
            buf.add(0, 4.0, 2.0);
        }

        // Reduce
        let mut pool = ContiguousHistogramPool::new(1, 10);
        let mut target = pool.get_or_allocate(super::super::types::NodeId(0));
        target.reset();
        scratch.reduce_into(&mut target);

        // Verify sums
        let hist = pool.get(super::super::types::NodeId(0)).unwrap();
        // bin 0: 1+2+4=7, 0.5+1.0+2.0=3.5, 3
        assert_eq!(hist.bin_stats(0), (7.0, 3.5, 3));
        // bin 1: 3, 1.5, 1
        assert_eq!(hist.bin_stats(1), (3.0, 1.5, 1));
        assert_eq!(hist.total_count(), 4);
    }

    #[test]
    fn test_reduce_empty_threads() {
        let mut scratch = RowParallelScratch::new(4, 10);

        // Only thread 1 has data
        {
            let mut buf = scratch.get_buffer(1);
            buf.add(0, 5.0, 2.5);
        }

        // Reduce
        let mut pool = ContiguousHistogramPool::new(1, 10);
        let mut target = pool.get_or_allocate(super::super::types::NodeId(0));
        target.reset();
        scratch.reduce_into(&mut target);

        // Verify
        let hist = pool.get(super::super::types::NodeId(0)).unwrap();
        assert_eq!(hist.bin_stats(0), (5.0, 2.5, 1));
        assert_eq!(hist.total_count(), 1);
    }

    #[test]
    fn test_reduce_large_histogram() {
        let num_bins = 1000;
        let mut scratch = RowParallelScratch::new(4, num_bins);

        // Each thread adds to different bins
        for t in 0..4 {
            let mut buf = scratch.get_buffer(t);
            for b in (t..num_bins).step_by(4) {
                buf.add(b, 1.0, 0.5);
            }
        }

        // Reduce
        let mut pool = ContiguousHistogramPool::new(1, num_bins);
        let mut target = pool.get_or_allocate(super::super::types::NodeId(0));
        target.reset();
        scratch.reduce_into(&mut target);

        // Verify: each bin should have exactly 1 sample
        let hist = pool.get(super::super::types::NodeId(0)).unwrap();
        assert_eq!(hist.total_count(), num_bins as u32);
        for b in 0..num_bins {
            assert_eq!(hist.bin_stats(b).2, 1, "bin {} should have count 1", b);
        }
    }

    #[test]
    fn test_reduce_parallel_matches_sequential() {
        let mut scratch = RowParallelScratch::new(4, 100);

        // Add varied data
        for t in 0..4 {
            let mut buf = scratch.get_buffer(t);
            for b in 0..100 {
                buf.add(b, (t * 100 + b) as f32, 1.0);
            }
        }

        // Reduce sequentially
        let mut pool1 = ContiguousHistogramPool::new(1, 100);
        let mut target1 = pool1.get_or_allocate(super::super::types::NodeId(0));
        target1.reset();
        scratch.reduce_into(&mut target1);

        // Reduce in parallel
        let mut pool2 = ContiguousHistogramPool::new(1, 100);
        let mut target2 = pool2.get_or_allocate(super::super::types::NodeId(1));
        target2.reset();
        scratch.reduce_into_parallel(&mut target2);

        // Compare
        let hist1 = pool1.get(super::super::types::NodeId(0)).unwrap();
        let hist2 = pool2.get(super::super::types::NodeId(1)).unwrap();

        for b in 0..100 {
            let (g1, h1, c1) = hist1.bin_stats(b);
            let (g2, h2, c2) = hist2.bin_stats(b);
            assert!(
                (g1 - g2).abs() < 1e-5,
                "bin {} grad mismatch: {} vs {}",
                b,
                g1,
                g2
            );
            assert!(
                (h1 - h2).abs() < 1e-5,
                "bin {} hess mismatch: {} vs {}",
                b,
                h1,
                h2
            );
            assert_eq!(c1, c2, "bin {} count mismatch", b);
        }
    }

    #[test]
    fn test_subtract_histograms() {
        let mut target_grad = vec![0.0f32; 4];
        let mut target_hess = vec![0.0f32; 4];
        let mut target_count = vec![0u32; 4];

        let parent_grad = vec![10.0, 20.0, 30.0, 40.0];
        let parent_hess = vec![5.0, 10.0, 15.0, 20.0];
        let parent_count = vec![2, 4, 6, 8];

        let child_grad = vec![3.0, 8.0, 12.0, 15.0];
        let child_hess = vec![1.5, 4.0, 6.0, 7.5];
        let child_count = vec![1, 2, 3, 4];

        subtract_histograms(
            &mut target_grad,
            &mut target_hess,
            &mut target_count,
            &parent_grad,
            &parent_hess,
            &parent_count,
            &child_grad,
            &child_hess,
            &child_count,
        );

        assert_eq!(target_grad, vec![7.0, 12.0, 18.0, 25.0]);
        assert_eq!(target_hess, vec![3.5, 6.0, 9.0, 12.5]);
        assert_eq!(target_count, vec![1, 2, 3, 4]);
    }
}

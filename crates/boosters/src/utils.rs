//! Common utilities used across the crate.
//!
//! This module provides slice utilities, parallelism configuration, and
//! iterator helpers that are used by various subsystems.

use rayon::prelude::*;

// =============================================================================
// Statistical Utilities
// =============================================================================

/// Compute unweighted quantile using partial sort (O(n) average case).
///
/// For uniform weights, we only need to find the k-th smallest element,
/// which is much faster than full sorting.
#[inline]
pub fn unweighted_quantile_indexed<V>(
    indices: &[u32],
    value_fn: V,
    alpha: f32,
    scratch: &mut Vec<u32>,
) -> f32
where
    V: Fn(u32) -> f32,
{
    let n = indices.len();
    if n == 0 {
        return f32::NAN;
    }
    if n == 1 {
        return value_fn(indices[0]);
    }

    // Prepare scratch buffer: copy indices for partial sorting
    scratch.clear();
    scratch.extend_from_slice(indices);

    // For uniform weights, the k-th element where cumulative weight >= alpha * n
    // is at position ceil(alpha * n) - 1 (0-indexed)
    let k = ((alpha * n as f32).ceil() as usize)
        .saturating_sub(1)
        .min(n - 1);

    // Partial sort: O(n) average case
    // After this, scratch[k] is the k-th smallest value (by value_fn)
    scratch.select_nth_unstable_by(k, |&a, &b| {
        value_fn(a)
            .partial_cmp(&value_fn(b))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    value_fn(scratch[k])
}

/// Compute the weighted quantile over indexed data using a step function.
///
/// This implements the same algorithm as XGBoost's `WeightedQuantile` function:
/// no interpolation, returns the value at the point where the cumulative weight
/// first exceeds `alpha * total_weight`.
///
/// This zero-allocation version works directly with row indices into data arrays,
/// avoiding the need to collect values/weights into intermediate vectors.
///
/// # Arguments
/// * `indices` - Row indices into the data arrays
/// * `value_fn` - Function to get the value for a given index
/// * `weight_fn` - Function to get the weight for a given index (return 1.0 for uniform)
/// * `alpha` - The quantile level in (0, 1)
/// * `scratch` - Mutable scratch space for sorting indices (will be resized if needed)
///
/// # Returns
/// The weighted quantile value. Returns `f32::NAN` if indices is empty.
///
/// # Algorithm
/// 1. Copy indices to scratch buffer
/// 2. Sort by value (using provided accessor)
/// 3. Compute cumulative weights in sorted order
/// 4. Find first index where cumulative weight >= alpha * total_weight
/// 5. Return value at that index
#[inline]
pub fn weighted_quantile_indexed<V, W>(
    indices: &[u32],
    value_fn: V,
    weight_fn: W,
    alpha: f32,
    scratch: &mut Vec<u32>,
) -> f32
where
    V: Fn(u32) -> f32,
    W: Fn(u32) -> f32,
{
    let n = indices.len();
    if n == 0 {
        return f32::NAN;
    }
    if n == 1 {
        return value_fn(indices[0]);
    }

    // Prepare scratch buffer: copy indices for sorting
    scratch.clear();
    scratch.extend_from_slice(indices);

    // Sort indices by value
    scratch.sort_by(|&a, &b| {
        value_fn(a)
            .partial_cmp(&value_fn(b))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Compute total weight and threshold
    let total_weight: f32 = indices.iter().map(|&i| weight_fn(i)).sum();
    let threshold = total_weight * alpha;

    // Find the quantile using cumulative weights
    let mut cumulative = 0.0f32;
    for &idx in scratch.iter() {
        cumulative += weight_fn(idx);
        if cumulative >= threshold {
            return value_fn(idx);
        }
    }

    // Edge case: return last value
    value_fn(scratch[n - 1])
}

// =============================================================================
// Parallelism Configuration
// =============================================================================

/// Whether parallel execution is allowed.
///
/// This is a simple boolean flag passed through training components.
/// When `true`, components may use `rayon` parallel iterators.
/// When `false`, components must use sequential iteration.
///
/// The actual thread pool is set up at the model API level via `n_threads`.
/// Components don't manage thread pools - they just respect this flag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Parallelism {
    Sequential,
    Parallel,
}

impl Parallelism {
    /// Create from thread count semantics.
    ///
    /// - 0 = auto (parallel if rayon pool has multiple threads, sequential otherwise)
    /// - 1 = sequential
    /// - >1 = parallel
    #[inline]
    pub fn from_threads(n_threads: usize) -> Self {
        if n_threads == 1 || (n_threads == 0 && rayon::current_num_threads() == 1) {
            Parallelism::Sequential
        } else {
            Parallelism::Parallel
        }
    }

    /// Returns `true` if parallel execution is allowed.
    #[inline]
    pub fn is_parallel(self) -> bool {
        matches!(self, Parallelism::Parallel)
    }

    #[inline]
    pub fn maybe_par_for_each<T, I, F>(self, iter: I, f: F)
    where
        T: Send,
        I: IntoIterator<Item = T> + IntoParallelIterator<Item = T>,
        F: Fn(T) + Sync + Send,
    {
        if self.is_parallel() {
            iter.into_par_iter().for_each(f);
        } else {
            iter.into_iter().for_each(f);
        }
    }

    #[inline]
    pub fn maybe_par_bridge_for_each<T, I, F>(self, iter: I, f: F)
    where
        T: Send,
        I: Iterator<Item = T> + Send,
        F: Fn(T) + Sync + Send,
    {
        if self.is_parallel() {
            iter.par_bridge().for_each(f);
        } else {
            iter.for_each(f);
        }
    }

    /// Parallel bridge for_each with per-thread initialization.
    ///
    /// Like [`maybe_par_bridge_for_each`](Self::maybe_par_bridge_for_each) but with
    /// per-thread state initialization. The `init` closure is called once per worker
    /// thread (in parallel mode) or once total (in sequential mode).
    ///
    /// This is ideal for iterators that don't implement `IntoParallelIterator`
    /// (like `axis_chunks_iter_mut`) but need thread-local buffers.
    #[inline]
    pub fn maybe_par_bridge_for_each_init<T, I, INIT, S, F>(self, iter: I, init: INIT, f: F)
    where
        T: Send,
        I: Iterator<Item = T> + Send,
        INIT: Fn() -> S + Sync + Send,
        F: Fn(&mut S, T) + Sync + Send,
    {
        if self.is_parallel() {
            iter.par_bridge().for_each_init(init, f);
        } else {
            let mut state = init();
            iter.for_each(|item| f(&mut state, item));
        }
    }

    #[inline]
    pub fn maybe_par_map<T, B, I, F>(self, iter: I, f: F) -> Vec<B>
    where
        T: Send,
        B: Send,
        I: IntoIterator<Item = T> + IntoParallelIterator<Item = T>,
        F: Fn(T) -> B + Sync + Send,
    {
        if self.is_parallel() {
            iter.into_par_iter().map(f).collect()
        } else {
            iter.into_iter().map(f).collect()
        }
    }

    /// Parallel for_each with per-thread initialization.
    ///
    /// The `init` closure is called once per worker thread (in parallel mode)
    /// or once total (in sequential mode). The resulting value is passed to `f`
    /// and reused across iterations on the same thread.
    ///
    /// This is ideal for thread-local buffers that should persist across iterations.
    #[inline]
    pub fn maybe_par_for_each_init<T, I, INIT, S, F>(self, iter: I, init: INIT, f: F)
    where
        T: Send,
        I: IntoIterator<Item = T> + IntoParallelIterator<Item = T>,
        INIT: Fn() -> S + Sync + Send,
        F: Fn(&mut S, T) + Sync + Send,
    {
        if self.is_parallel() {
            iter.into_par_iter().for_each_init(init, f);
        } else {
            let mut state = init();
            iter.into_iter().for_each(|item| f(&mut state, item));
        }
    }
}

// =============================================================================
// Thread Pool Setup
// =============================================================================

/// Run a closure with the appropriate thread pool.
///
/// Thread count semantics:
/// - `0` = auto (use all available cores)
/// - `1` = sequential (no thread pool)
/// - `n > 1` = use exactly `n` threads
///
/// # Example
///
/// ```ignore
/// use boosters::run_with_threads;
///
/// // Auto-detect threads
/// let result = run_with_threads(0, || expensive_computation());
///
/// // Sequential
/// let result = run_with_threads(1, || expensive_computation());
///
/// // Exactly 4 threads
/// let result = run_with_threads(4, || expensive_computation());
/// ```
#[inline]
pub fn run_with_threads<T: Send>(n_threads: usize, f: impl FnOnce(Parallelism) -> T + Send) -> T {
    let parallelism = Parallelism::from_threads(n_threads);

    match parallelism {
        Parallelism::Sequential => f(Parallelism::Sequential),
        Parallelism::Parallel => {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n_threads)
                .build()
                .expect("Failed to create thread pool");
            pool.install(|| f(Parallelism::Parallel))
        }
    }
}

// =============================================================================
// Slice Utilities
// =============================================================================

/// Get two disjoint mutable slices from a single slice.
///
/// Returns `(slice[a_start..a_start+len], slice[b_start..b_start+len])`
/// with the first element mutable and second immutable.
///
/// # Panics
/// Panics if the ranges overlap or are out of bounds.
#[inline]
pub fn disjoint_slices_mut<T>(
    slice: &mut [T],
    a_start: usize,
    b_start: usize,
    len: usize,
) -> (&mut [T], &[T]) {
    debug_assert!(
        a_start + len <= b_start || b_start + len <= a_start,
        "Ranges overlap: [{}..{}] and [{}..{}]",
        a_start,
        a_start + len,
        b_start,
        b_start + len
    );
    debug_assert!(
        a_start + len <= slice.len() && b_start + len <= slice.len(),
        "Range out of bounds"
    );

    if a_start < b_start {
        let (left, right) = slice.split_at_mut(b_start);
        (&mut left[a_start..a_start + len], &right[..len])
    } else {
        let (left, right) = slice.split_at_mut(a_start);
        (&mut right[..len], &left[b_start..b_start + len])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper for weighted quantile tests
    fn test_weighted_quantile(
        values: &[f32],
        weights: &[f32],
        alpha: f32,
        scratch: &mut Vec<u32>,
    ) -> f32 {
        let indices: Vec<u32> = (0..values.len() as u32).collect();
        let value_fn = |i: u32| values[i as usize];
        let weight_fn = |i: u32| weights[i as usize];
        weighted_quantile_indexed(&indices, value_fn, weight_fn, alpha, scratch)
    }

    // Helper for unweighted quantile tests
    fn test_unweighted_quantile(values: &[f32], alpha: f32, scratch: &mut Vec<u32>) -> f32 {
        let indices: Vec<u32> = (0..values.len() as u32).collect();
        let value_fn = |i: u32| values[i as usize];
        unweighted_quantile_indexed(&indices, value_fn, alpha, scratch)
    }

    #[test]
    fn test_weighted_quantile_empty() {
        let mut scratch = Vec::new();
        let result = weighted_quantile_indexed::<_, fn(u32) -> f32>(
            &[],
            |_| 0.0,
            |_| 1.0,
            0.5,
            &mut scratch,
        );
        assert!(result.is_nan());
    }

    #[test]
    fn test_unweighted_quantile_empty() {
        let mut scratch = Vec::new();
        let result = unweighted_quantile_indexed::<fn(u32) -> f32>(&[], |_| 0.0, 0.5, &mut scratch);
        assert!(result.is_nan());
    }

    #[test]
    fn test_unweighted_quantile_single() {
        let mut scratch = Vec::new();
        let values = [42.0];
        let result = test_unweighted_quantile(&values, 0.5, &mut scratch);
        assert!((result - 42.0).abs() < 1e-6);
    }

    #[test]
    fn test_unweighted_quantile_median() {
        let mut scratch = Vec::new();
        // Values: 1, 2, 3, 4, 5 - median should be 3
        let values = [1.0, 2.0, 3.0, 4.0, 5.0];
        let result = test_unweighted_quantile(&values, 0.5, &mut scratch);
        assert!((result - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_unweighted_quantile_q25() {
        let mut scratch = Vec::new();
        // Values: 1, 2, 3, 4 - 25th percentile
        // k = ceil(0.25 * 4) - 1 = 0, so returns smallest
        let values = [1.0, 2.0, 3.0, 4.0];
        let result = test_unweighted_quantile(&values, 0.25, &mut scratch);
        assert!((result - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_unweighted_quantile_q75() {
        let mut scratch = Vec::new();
        // Values: 1, 2, 3, 4 - 75th percentile
        // k = ceil(0.75 * 4) - 1 = 2
        let values = [1.0, 2.0, 3.0, 4.0];
        let result = test_unweighted_quantile(&values, 0.75, &mut scratch);
        assert!((result - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_unweighted_quantile_unsorted_input() {
        let mut scratch = Vec::new();
        // Values in random order, but should still find correct median
        let values = [5.0, 1.0, 3.0, 2.0, 4.0];
        let result = test_unweighted_quantile(&values, 0.5, &mut scratch);
        assert!((result - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_weighted_quantile_basic() {
        let mut scratch = Vec::new();
        // Values: 1, 2, 3 with weights 1, 2, 1
        // Total weight = 4, threshold for median = 2.0
        // Sorted: 1 (w=1), 2 (w=2), 3 (w=1)
        // Cumulative: 1, 3, 4
        // At index 0: cumulative = 1 < 2
        // At index 1: cumulative = 3 >= 2 -> return 2.0
        let values = [1.0, 2.0, 3.0];
        let weights = [1.0, 2.0, 1.0];
        let result = test_weighted_quantile(&values, &weights, 0.5, &mut scratch);
        assert!((result - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_weighted_quantile_high_weight_pulls_quantile() {
        let mut scratch = Vec::new();
        // Values: 1, 10 with weights 1, 9
        // Total weight = 10, threshold for 0.5 = 5.0
        // Sorted: 1 (w=1), 10 (w=9)
        // Cumulative: 1 < 5, then 10 >= 5 -> return 10.0
        let values = [1.0, 10.0];
        let weights = [1.0, 9.0];
        let result = test_weighted_quantile(&values, &weights, 0.5, &mut scratch);
        assert!((result - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_disjoint_slices_mut() {
        let mut data = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let (a, b) = disjoint_slices_mut(&mut data, 0, 5, 3);
        assert_eq!(a, &mut [0, 1, 2]);
        assert_eq!(b, &[5, 6, 7]);

        // Modify a
        a[0] = 100;
        assert_eq!(data[0], 100);
    }

    #[test]
    fn test_disjoint_slices_mut_reversed() {
        let mut data = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let (a, b) = disjoint_slices_mut(&mut data, 6, 2, 3);
        assert_eq!(a, &mut [6, 7, 8]);
        assert_eq!(b, &[2, 3, 4]);
    }

    #[test]
    fn test_parallelism_from_threads() {
        assert!(Parallelism::from_threads(0).is_parallel()); // auto = parallel
        assert!(!Parallelism::from_threads(1).is_parallel()); // 1 = sequential
        assert!(Parallelism::from_threads(2).is_parallel()); // >1 = parallel
        assert!(Parallelism::from_threads(8).is_parallel());
    }

    #[test]
    fn test_parallelism_constants() {
        assert!(Parallelism::Parallel.is_parallel());
        assert!(!Parallelism::Sequential.is_parallel());
    }

    #[test]
    fn test_run_with_threads_sequential() {
        let result = run_with_threads(1, |_| 42);
        assert_eq!(result, 42);
    }

    #[test]
    fn test_run_with_threads_auto() {
        let result = run_with_threads(0, |_| 42);
        assert_eq!(result, 42);
    }

    #[test]
    fn test_run_with_threads_explicit() {
        let result = run_with_threads(2, |_| rayon::current_num_threads());
        assert_eq!(result, 2);
    }

    #[test]
    fn test_maybe_par_for_each() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let sum = AtomicUsize::new(0);
        Parallelism::Sequential.maybe_par_for_each(0..10usize, |i| {
            sum.fetch_add(i, Ordering::Relaxed);
        });
        assert_eq!(sum.load(Ordering::Relaxed), 45);

        sum.store(0, Ordering::Relaxed);
        Parallelism::Parallel.maybe_par_for_each(0..10usize, |i| {
            sum.fetch_add(i, Ordering::Relaxed);
        });
        assert_eq!(sum.load(Ordering::Relaxed), 45);
    }

    #[test]
    fn test_maybe_par_map() {
        let result: Vec<_> = Parallelism::Sequential.maybe_par_map(0..5usize, |i| i * 2);
        assert_eq!(result, vec![0, 2, 4, 6, 8]);

        let result: Vec<_> = Parallelism::Parallel.maybe_par_map(0..5usize, |i| i * 2);
        assert_eq!(result, vec![0, 2, 4, 6, 8]);
    }
}

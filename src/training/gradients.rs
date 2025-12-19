//! Interleaved gradient buffer with column-major layout.
//!
//! Provides a `Gradients` struct that stores `(grad, hess)` pairs in a single
//! contiguous array using **column-major** (output-major) order.
//!
//! # Design Rationale
//!
//! Column-major layout optimizes for the histogram building hot path:
//!
//! 1. **Zero-copy output slicing**: `output_pairs(k)` returns a contiguous slice of all
//!    samples for output `k`, enabling efficient histogram building
//! 2. **Cache-friendly histograms**: Histogram builder iterates samples for one output,
//!    which now has perfect cache locality
//! 3. **Auto-vectorization**: Contiguous f32 arrays are SIMD-friendly
//! 4. **Eliminated gradient copy**: Trainer can pass slices directly to grower
//!
//! # Layout
//!
//! For `n_samples` samples and `n_outputs` outputs (1 for regression, K for multiclass):
//!
//! ```text
//! data:  [(s0_o0), (s1_o0), ..., (sN_o0), (s0_o1), (s1_o1), ..., (sN_o1), ...]
//!        |------ output 0 -------|      |------ output 1 -------|
//! ```
//!
//! Index formula: `data[output * n_samples + sample]`
//!
//! # Performance Impact
//!
//! This layout keeps per-output slices contiguous (important for training), while
//! improving locality in hot loops that need both gradient and hessian together.

/// Structure-of-Arrays gradient buffer with column-major layout.
///
/// Stores gradients and hessians in separate contiguous arrays, organized
/// by output (column-major) for cache-efficient histogram building.
///
/// # Example
///
/// ```
/// use boosters::training::Gradients;
///
/// // Single-output regression: 100 samples, 1 output
/// let mut buffer = Gradients::new(100, 1);
///
/// // Set gradient for sample 0
/// buffer.set(0, 0, -0.5, 1.0);
/// // Get gradient and hessian
/// let (grad, hess) = buffer.get(0, 0);
/// assert_eq!(grad, -0.5);
/// assert_eq!(hess, 1.0);
/// ```
///
/// ```
/// use boosters::training::Gradients;
///
/// // Multiclass: 100 samples, 3 classes
/// let mut buffer = Gradients::new(100, 3);
///
/// // Set gradients for sample 0, all classes
/// buffer.set(0, 0, 0.2, 0.16);   // class 0
/// buffer.set(0, 1, 0.3, 0.21);   // class 1
/// buffer.set(0, 2, -0.5, 0.25);  // class 2 (true class)
///
/// // Get contiguous slice for class 0 (all samples) - zero-copy!
/// let class0 = buffer.output_pairs(0);
/// assert_eq!(class0[0].grad, 0.2);  // sample 0's class 0 gradient
/// ```
/// Interleaved gradient/hessian pair.
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct GradsTuple {
    pub grad: f32,
    pub hess: f32,
}

/// Interleaved gradient buffer with column-major layout.
#[derive(Debug, Clone)]
pub struct Gradients {
    /// Interleaved gradient/hessian pairs.
    data: Vec<GradsTuple>,
    /// Number of samples.
    n_samples: usize,
    /// Number of outputs per sample (1 for regression, K for K-class).
    n_outputs: usize,
}

impl Gradients {
    /// Create a new gradient buffer initialized to zeros.
    ///
    /// # Arguments
    ///
    /// * `n_samples` - Number of training samples
    /// * `n_outputs` - Number of outputs per sample (1 for regression/binary, K for K-class)
    ///
    /// # Panics
    ///
    /// Panics if `n_samples` or `n_outputs` is zero.
    pub fn new(n_samples: usize, n_outputs: usize) -> Self {
        assert!(n_samples > 0, "n_samples must be positive");
        assert!(n_outputs > 0, "n_outputs must be positive");

        let size = n_samples * n_outputs;
        Self {
            data: vec![GradsTuple::default(); size],
            n_samples,
            n_outputs,
        }
    }

    /// Number of samples in the buffer.
    #[inline]
    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    /// Number of outputs per sample.
    #[inline]
    pub fn n_outputs(&self) -> usize {
        self.n_outputs
    }

    /// Total number of gradient pairs (n_samples × n_outputs).
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Reset all gradients and hessians to zero.
    pub fn reset(&mut self) {
        self.data.fill(GradsTuple::default());
    }

    // =========================================================================
    // Single element access
    // =========================================================================

    /// Get gradient and hessian for a (sample, output) pair.
    ///
    /// # Arguments
    ///
    /// * `sample` - Sample index (0 to n_samples-1)
    /// * `output` - Output index (0 to n_outputs-1)
    #[inline]
    pub fn get(&self, sample: usize, output: usize) -> (f32, f32) {
        let idx = self.index(sample, output);
        let p = self.data[idx];
        (p.grad, p.hess)
    }

    /// Set gradient and hessian for a (sample, output) pair.
    #[inline]
    pub fn set(&mut self, sample: usize, output: usize, grad: f32, hess: f32) {
        let idx = self.index(sample, output);
        self.data[idx] = GradsTuple { grad, hess };
    }

    // =========================================================================
    // Slice access for bulk operations
    // =========================================================================

    /// Get the full interleaved buffer.
    #[inline]
    pub fn pairs(&self) -> &[GradsTuple] {
        &self.data
    }

    /// Get the full interleaved buffer (mutable).
    #[inline]
    pub fn pairs_mut(&mut self) -> &mut [GradsTuple] {
        &mut self.data
    }

    // =========================================================================
    // Per-sample access (for loss functions)
    // =========================================================================

    /// Get gradients for all outputs of a single sample.
    ///
    /// **Note**: With column-major layout, this requires strided access.
    /// For bulk per-sample operations, iterate manually for better cache use.
    ///
    /// Returns a newly allocated Vec (not a slice) because data is non-contiguous.
    #[inline]
    pub fn sample_grads(&self, sample: usize) -> Vec<f32> {
        (0..self.n_outputs)
            .map(|output| self.data[self.index(sample, output)].grad)
            .collect()
    }

    /// Get hessians for all outputs of a single sample.
    ///
    /// Returns a newly allocated Vec (not a slice) because data is non-contiguous.
    #[inline]
    pub fn sample_hess(&self, sample: usize) -> Vec<f32> {
        (0..self.n_outputs)
            .map(|output| self.data[self.index(sample, output)].hess)
            .collect()
    }

    // =========================================================================
    // Per-output access (for histogram building - the hot path)
    // =========================================================================

    /// Get contiguous `(grad, hess)` pairs for a specific output (all samples).
    #[inline]
    pub fn output_pairs(&self, output: usize) -> &[GradsTuple] {
        debug_assert!(output < self.n_outputs);
        let start = output * self.n_samples;
        &self.data[start..start + self.n_samples]
    }

    /// Get mutable contiguous `(grad, hess)` pairs for a specific output.
    #[inline]
    pub fn output_pairs_mut(&mut self, output: usize) -> &mut [GradsTuple] {
        debug_assert!(output < self.n_outputs);
        let start = output * self.n_samples;
        &mut self.data[start..start + self.n_samples]
    }

    /// Iterator over (sample_idx, grad, hess) for a specific output.
    ///
    /// This is the key access pattern for coordinate descent: iterating
    /// over all samples for one output (class).
    ///
    /// # Example
    ///
    /// ```
    /// use boosters::training::Gradients;
    ///
    /// let mut buffer = Gradients::new(3, 2);
    /// buffer.set(0, 0, 1.0, 0.5);
    /// buffer.set(1, 0, 2.0, 0.5);
    /// buffer.set(2, 0, 3.0, 0.5);
    ///
    /// // Iterate over output 0
    /// let grads: Vec<f32> = buffer.output_iter(0).map(|(_, g, _)| g).collect();
    /// assert_eq!(grads, vec![1.0, 2.0, 3.0]);
    /// ```
    pub fn output_iter(&self, output: usize) -> OutputIter<'_> {
        debug_assert!(output < self.n_outputs);
        OutputIter {
            pairs: self.output_pairs(output),
            sample: 0,
        }
    }

    // =========================================================================
    // Aggregation helpers
    // =========================================================================

    /// Sum gradients and hessians for a specific output.
    ///
    /// # Arguments
    /// * `output` - Which output to sum
    /// * `rows` - Optional subset of row indices. If `None`, sums all samples.
    ///
    /// This intentionally accumulates into `f64` to reduce numerical drift in
    /// gain / Newton-step style computations, while keeping the underlying storage
    /// in `f32` for performance.
    #[inline]
    pub fn sum(&self, output: usize, rows: Option<&[u32]>) -> (f64, f64) {
        let pairs = self.output_pairs(output);

        let mut sum_grad = 0.0f64;
        let mut sum_hess = 0.0f64;

        match rows {
            None => {
                for i in 0..self.n_samples {
                    sum_grad += pairs[i].grad as f64;
                    sum_hess += pairs[i].hess as f64;
                }
            }
            Some(rows) => {
                for &row in rows {
                    let row = row as usize;
                    sum_grad += pairs[row].grad as f64;
                    sum_hess += pairs[row].hess as f64;
                }
            }
        }
        (sum_grad, sum_hess)
    }

    /// Compute Newton step for bias update: -sum(grad) / sum(hess).
    ///
    /// # Arguments
    ///
    /// * `output` - Which output to compute for
    /// * `min_hess` - Minimum hessian to avoid division by zero
    pub fn bias_update(&self, output: usize, min_hess: f32) -> f32 {
        let (sum_grad, sum_hess) = self.sum(output, None);
        if sum_hess.abs() < min_hess as f64 {
            0.0
        } else {
            (-sum_grad / sum_hess) as f32
        }
    }

    // =========================================================================
    // Private helpers
    // =========================================================================

    /// Convert (sample, output) to linear index (column-major).
    #[inline]
    fn index(&self, sample: usize, output: usize) -> usize {
        debug_assert!(sample < self.n_samples);
        debug_assert!(output < self.n_outputs);
        output * self.n_samples + sample
    }
}

/// Iterator over all samples for a specific output.
///
/// With column-major layout, this iterates over contiguous memory.
pub struct OutputIter<'a> {
    pairs: &'a [GradsTuple],
    sample: usize,
}

impl<'a> Iterator for OutputIter<'a> {
    /// (sample_index, gradient, hessian)
    type Item = (usize, f32, f32);

    fn next(&mut self) -> Option<Self::Item> {
        if self.sample >= self.pairs.len() {
            return None;
        }

        let p = self.pairs[self.sample];
        let result = (self.sample, p.grad, p.hess);
        self.sample += 1;
        Some(result)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.pairs.len() - self.sample;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for OutputIter<'_> {}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_buffer() {
        let buffer = Gradients::new(100, 3);
        assert_eq!(buffer.n_samples(), 100);
        assert_eq!(buffer.n_outputs(), 3);
        assert_eq!(buffer.len(), 300);
    }

    #[test]
    fn get_set_single() {
        let mut buffer = Gradients::new(10, 1);

        buffer.set(0, 0, 1.5, 0.25);
        let (g, h) = buffer.get(0, 0);
        assert_eq!(g, 1.5);
        assert_eq!(h, 0.25);

        buffer.set(5, 0, -0.5, 1.0);
        let (g, h) = buffer.get(5, 0);
        assert_eq!(g, -0.5);
        assert_eq!(h, 1.0);
    }

    #[test]
    fn get_set_multiclass() {
        let mut buffer = Gradients::new(10, 3);

        // Sample 0: gradients for 3 classes
        buffer.set(0, 0, 0.2, 0.16);
        buffer.set(0, 1, 0.3, 0.21);
        buffer.set(0, 2, -0.5, 0.25);

        // Verify via slices: sample 0 is at index 0 of each output slice
        assert_eq!(buffer.output_pairs(0)[0].grad, 0.2);
        assert_eq!(buffer.output_pairs(1)[0].grad, 0.3);
        assert_eq!(buffer.output_pairs(2)[0].grad, -0.5);
    }

    #[test]
    fn output_slices() {
        let mut buffer = Gradients::new(5, 3);

        // Set output 1's gradients (for all samples)
        let pairs = buffer.output_pairs_mut(1);
        pairs[0].grad = 1.0; // sample 0, output 1
        pairs[1].grad = 2.0; // sample 1, output 1
        pairs[2].grad = 3.0; // sample 2, output 1

        // Read back via output slice
        let pairs = buffer.output_pairs(1);
        assert_eq!(pairs[0].grad, 1.0);
        assert_eq!(pairs[1].grad, 2.0);
        assert_eq!(pairs[2].grad, 3.0);
    }

    #[test]
    fn sample_grads_returns_vec() {
        let mut buffer = Gradients::new(5, 3);

        // Set sample 2's gradients (across outputs)
        buffer.set(2, 0, 1.0, 0.5);
        buffer.set(2, 1, 2.0, 0.5);
        buffer.set(2, 2, 3.0, 0.5);

        // sample_grads returns a Vec (not a slice) because data is non-contiguous
        let grads = buffer.sample_grads(2);
        assert_eq!(grads, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn output_iter() {
        let mut buffer = Gradients::new(3, 2);

        // Output 0: grads = [1, 2, 3]
        buffer.set(0, 0, 1.0, 0.5);
        buffer.set(1, 0, 2.0, 0.5);
        buffer.set(2, 0, 3.0, 0.5);

        // Output 1: grads = [10, 20, 30]
        buffer.set(0, 1, 10.0, 0.5);
        buffer.set(1, 1, 20.0, 0.5);
        buffer.set(2, 1, 30.0, 0.5);

        // Iterate output 0
        let collected: Vec<_> = buffer.output_iter(0).collect();
        assert_eq!(collected.len(), 3);
        assert_eq!(collected[0], (0, 1.0, 0.5));
        assert_eq!(collected[1], (1, 2.0, 0.5));
        assert_eq!(collected[2], (2, 3.0, 0.5));

        // Iterate output 1
        let grads1: Vec<f32> = buffer.output_iter(1).map(|(_, g, _)| g).collect();
        assert_eq!(grads1, vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn sum_gradients() {
        let mut buffer = Gradients::new(4, 2);

        // Output 0: grads = [1, 2, 3, 4], hess = [1, 1, 1, 1]
        for i in 0..4 {
            buffer.set(i, 0, (i + 1) as f32, 1.0);
        }

        // Sum all
        let (sum_g, sum_h) = buffer.sum(0, None);
        assert_eq!(sum_g, 10.0); // 1+2+3+4
        assert_eq!(sum_h, 4.0);

        // Sum subset of rows
        let (sum_g, sum_h) = buffer.sum(0, Some(&[1, 3]));
        assert_eq!(sum_g, 6.0); // 2+4
        assert_eq!(sum_h, 2.0);
    }

    #[test]
    fn bias_update() {
        let mut buffer = Gradients::new(4, 1);

        // grad = [1, 2, 3, 4] → sum = 10
        // hess = [1, 1, 1, 1] → sum = 4
        // bias_update = -10/4 = -2.5
        for i in 0..4 {
            buffer.set(i, 0, (i + 1) as f32, 1.0);
        }

        let update = buffer.bias_update(0, 1e-6);
        assert!((update - (-2.5)).abs() < 1e-6);
    }

    #[test]
    fn reset() {
        let mut buffer = Gradients::new(3, 2);
        buffer.set(0, 0, 1.0, 2.0);
        buffer.set(1, 1, 3.0, 4.0);

        buffer.reset();

        // Verify all values are zero
        assert!(buffer.pairs().iter().all(|p| p.grad == 0.0 && p.hess == 0.0));
    }

    #[test]
    #[should_panic(expected = "n_samples must be positive")]
    fn zero_samples_panics() {
        Gradients::new(0, 1);
    }

    #[test]
    #[should_panic(expected = "n_outputs must be positive")]
    fn zero_outputs_panics() {
        Gradients::new(10, 0);
    }
}

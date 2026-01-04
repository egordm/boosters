//! Interleaved gradient buffer with output-major layout.
//!
//! Provides a `Gradients` struct that stores `(grad, hess)` pairs in a 2D array
//! with shape `[n_outputs, n_samples]`.
//!
//! # Design Rationale
//!
//! Output-major layout optimizes for the histogram building hot path:
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
//! Shape: [n_outputs, n_samples]
//! data[output, sample] = GradsTuple { grad, hess }
//! ```
//!
//! Row `k` contains all samples for output `k`, which is contiguous in memory.

use ndarray::{Array2, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2};

/// Interleaved gradient buffer with output-major layout.
///
/// Stores gradients and hessians as `Array2<GradsTuple>` with shape
/// `[n_outputs, n_samples]` for cache-efficient histogram building.
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

/// Interleaved gradient buffer with output-major layout.
///
/// Shape: `[n_outputs, n_samples]`
#[derive(Debug, Clone)]
pub struct Gradients {
    /// 2D array of gradient pairs: shape `[n_outputs, n_samples]`.
    data: Array2<GradsTuple>,
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

        Self {
            data: Array2::default((n_outputs, n_samples)),
        }
    }

    /// Number of samples in the buffer.
    #[inline]
    pub fn n_samples(&self) -> usize {
        self.data.ncols()
    }

    /// Number of outputs per sample.
    #[inline]
    pub fn n_outputs(&self) -> usize {
        self.data.nrows()
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
        let p = self.data[[output, sample]];
        (p.grad, p.hess)
    }

    /// Set gradient and hessian for a (sample, output) pair.
    #[inline]
    pub fn set(&mut self, sample: usize, output: usize, grad: f32, hess: f32) {
        self.data[[output, sample]] = GradsTuple { grad, hess };
    }

    // =========================================================================
    // Array view access for bulk operations
    // =========================================================================

    /// Get the gradient buffer as a 2D array view.
    ///
    /// Shape: `[n_outputs, n_samples]`
    #[inline]
    pub fn pairs_array(&self) -> ArrayView2<'_, GradsTuple> {
        self.data.view()
    }

    /// Get the gradient buffer as a mutable 2D array view.
    ///
    /// Shape: `[n_outputs, n_samples]`
    #[inline]
    pub fn pairs_array_mut(&mut self) -> ArrayViewMut2<'_, GradsTuple> {
        self.data.view_mut()
    }

    // =========================================================================
    // Per-output access (for histogram building - the hot path)
    // =========================================================================

    /// Get row view for a specific output (all samples).
    ///
    /// Returns an `ArrayView1<GradsTuple>` of length `n_samples`.
    #[inline]
    pub fn output_view(&self, output: usize) -> ArrayView1<'_, GradsTuple> {
        self.data.row(output)
    }

    /// Get mutable row view for a specific output (all samples).
    ///
    /// Returns an `ArrayViewMut1<GradsTuple>` of length `n_samples`.
    #[inline]
    pub fn output_view_mut(&mut self, output: usize) -> ArrayViewMut1<'_, GradsTuple> {
        self.data.row_mut(output)
    }

    /// Get contiguous `(grad, hess)` pairs for a specific output (all samples).
    ///
    /// This is a slice view into the row, valid because the array is row-major.
    #[inline]
    pub fn output_pairs(&self, output: usize) -> &[GradsTuple] {
        debug_assert!(output < self.n_outputs());
        // For C-contiguous row-major array, each row is contiguous
        // Row `output` starts at offset `output * ncols` in the flat slice
        let ncols = self.data.ncols();
        let start = output * ncols;
        &self.data.as_slice().expect("array should be contiguous")[start..start + ncols]
    }

    /// Get mutable contiguous `(grad, hess)` pairs for a specific output.
    #[inline]
    pub fn output_pairs_mut(&mut self, output: usize) -> &mut [GradsTuple] {
        debug_assert!(output < self.n_outputs());
        let ncols = self.data.ncols();
        let start = output * ncols;
        &mut self
            .data
            .as_slice_mut()
            .expect("array should be contiguous")[start..start + ncols]
    }
}

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
    fn output_view() {
        let mut buffer = Gradients::new(3, 2);

        // Output 0: grads = [1, 2, 3]
        buffer.set(0, 0, 1.0, 0.5);
        buffer.set(1, 0, 2.0, 0.5);
        buffer.set(2, 0, 3.0, 0.5);

        // Output 1: grads = [10, 20, 30]
        buffer.set(0, 1, 10.0, 0.5);
        buffer.set(1, 1, 20.0, 0.5);
        buffer.set(2, 1, 30.0, 0.5);

        // Use view for output 0
        let view = buffer.output_view(0);
        assert_eq!(view.len(), 3);
        assert_eq!(view[0].grad, 1.0);
        assert_eq!(view[1].grad, 2.0);
        assert_eq!(view[2].grad, 3.0);

        // Use view for output 1
        let view = buffer.output_view(1);
        assert_eq!(view[0].grad, 10.0);
    }

    #[test]
    fn reset() {
        let mut buffer = Gradients::new(3, 2);
        buffer.set(0, 0, 1.0, 2.0);
        buffer.set(1, 1, 3.0, 4.0);

        buffer.reset();

        // Verify all values are zero via array view
        let view = buffer.pairs_array();
        assert!(view.iter().all(|p| p.grad == 0.0 && p.hess == 0.0));
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

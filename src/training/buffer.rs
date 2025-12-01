//! Structure-of-Arrays gradient buffer.
//!
//! Provides a `GradientBuffer` struct that stores gradients and hessians in separate
//! contiguous arrays rather than interleaved `GradientPair` structs.
//!
//! # Design Rationale
//!
//! Structure-of-Arrays (SoA) layout offers several advantages over Array-of-Structures (AoS):
//!
//! 1. **Cache efficiency**: Gradient-only or hessian-only operations touch only one array
//! 2. **Auto-vectorization**: Contiguous f32 arrays are SIMD-friendly
//! 3. **Cleaner multi-output**: Shape is explicit `[n_samples, n_outputs]`, no stride hacks
//! 4. **Memory layout**: Better alignment for SIMD operations
//!
//! # Layout
//!
//! For `n_samples` samples and `n_outputs` outputs (1 for regression, K for multiclass):
//!
//! ```text
//! grads: [s0_o0, s0_o1, ..., s0_oK, s1_o0, s1_o1, ..., sN_oK]
//! hess:  [s0_o0, s0_o1, ..., s0_oK, s1_o0, s1_o1, ..., sN_oK]
//! ```
//!
//! Index formula: `grads[sample * n_outputs + output]`

/// Structure-of-Arrays gradient buffer.
///
/// Stores gradients and hessians in separate contiguous arrays for better
/// cache efficiency and SIMD-friendly access patterns.
///
/// # Example
///
/// ```
/// use booste_rs::training::GradientBuffer;
///
/// // Single-output regression: 100 samples, 1 output
/// let mut buffer = GradientBuffer::new(100, 1);
///
/// // Set gradient for sample 0
/// buffer.set(0, 0, -0.5, 1.0);
///
/// // Get gradient and hessian
/// let (grad, hess) = buffer.get(0, 0);
/// assert_eq!(grad, -0.5);
/// assert_eq!(hess, 1.0);
/// ```
///
/// ```
/// use booste_rs::training::GradientBuffer;
///
/// // Multiclass: 100 samples, 3 classes
/// let mut buffer = GradientBuffer::new(100, 3);
///
/// // Set gradients for sample 0, all classes
/// buffer.set(0, 0, 0.2, 0.16);   // class 0
/// buffer.set(0, 1, 0.3, 0.21);   // class 1
/// buffer.set(0, 2, -0.5, 0.25);  // class 2 (true class)
/// ```
#[derive(Debug, Clone)]
pub struct GradientBuffer {
    /// Gradient values (∂L/∂pred).
    grads: Vec<f32>,
    /// Hessian values (∂²L/∂pred²).
    hess: Vec<f32>,
    /// Number of samples.
    n_samples: usize,
    /// Number of outputs per sample (1 for regression, K for K-class).
    n_outputs: usize,
}

impl GradientBuffer {
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
            grads: vec![0.0; size],
            hess: vec![0.0; size],
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
        self.grads.len()
    }

    /// Whether the buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.grads.is_empty()
    }

    /// Reset all gradients and hessians to zero.
    pub fn reset(&mut self) {
        self.grads.fill(0.0);
        self.hess.fill(0.0);
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
        (self.grads[idx], self.hess[idx])
    }

    /// Set gradient and hessian for a (sample, output) pair.
    #[inline]
    pub fn set(&mut self, sample: usize, output: usize, grad: f32, hess: f32) {
        let idx = self.index(sample, output);
        self.grads[idx] = grad;
        self.hess[idx] = hess;
    }

    /// Get only the gradient for a (sample, output) pair.
    #[inline]
    pub fn grad(&self, sample: usize, output: usize) -> f32 {
        self.grads[self.index(sample, output)]
    }

    /// Get only the hessian for a (sample, output) pair.
    #[inline]
    pub fn hess(&self, sample: usize, output: usize) -> f32 {
        self.hess[self.index(sample, output)]
    }

    // =========================================================================
    // Slice access for bulk operations
    // =========================================================================

    /// Get the full gradients array as a slice.
    #[inline]
    pub fn grads(&self) -> &[f32] {
        &self.grads
    }

    /// Get the full gradients array as a mutable slice.
    #[inline]
    pub fn grads_mut(&mut self) -> &mut [f32] {
        &mut self.grads
    }

    /// Get the full hessians array as a slice.
    #[inline]
    pub fn hess_slice(&self) -> &[f32] {
        &self.hess
    }

    /// Get the full hessians array as a mutable slice.
    #[inline]
    pub fn hess_mut(&mut self) -> &mut [f32] {
        &mut self.hess
    }

    /// Get gradients and hessians as mutable slice pair.
    ///
    /// Useful for loss functions that need to write both arrays.
    #[inline]
    pub fn as_mut_slices(&mut self) -> (&mut [f32], &mut [f32]) {
        (&mut self.grads, &mut self.hess)
    }

    // =========================================================================
    // Per-sample access (for multiclass)
    // =========================================================================

    /// Get gradients for all outputs of a single sample.
    ///
    /// Returns a slice of length `n_outputs`.
    #[inline]
    pub fn sample_grads(&self, sample: usize) -> &[f32] {
        let start = sample * self.n_outputs;
        &self.grads[start..start + self.n_outputs]
    }

    /// Get mutable gradients for all outputs of a single sample.
    #[inline]
    pub fn sample_grads_mut(&mut self, sample: usize) -> &mut [f32] {
        let start = sample * self.n_outputs;
        &mut self.grads[start..start + self.n_outputs]
    }

    /// Get hessians for all outputs of a single sample.
    #[inline]
    pub fn sample_hess(&self, sample: usize) -> &[f32] {
        let start = sample * self.n_outputs;
        &self.hess[start..start + self.n_outputs]
    }

    /// Get mutable hessians for all outputs of a single sample.
    #[inline]
    pub fn sample_hess_mut(&mut self, sample: usize) -> &mut [f32] {
        let start = sample * self.n_outputs;
        &mut self.hess[start..start + self.n_outputs]
    }

    // =========================================================================
    // Per-output access (for coordinate descent)
    // =========================================================================

    /// Iterator over (sample_idx, grad, hess) for a specific output.
    ///
    /// This is the key access pattern for coordinate descent: iterating
    /// over all samples for one output (class).
    ///
    /// # Example
    ///
    /// ```
    /// use booste_rs::training::GradientBuffer;
    ///
    /// let mut buffer = GradientBuffer::new(3, 2);
    /// buffer.set(0, 0, 1.0, 0.5);
    /// buffer.set(1, 0, 2.0, 0.5);
    /// buffer.set(2, 0, 3.0, 0.5);
    ///
    /// // Iterate over output 0
    /// let grads: Vec<f32> = buffer.output_iter(0).map(|(_, g, _)| g).collect();
    /// assert_eq!(grads, vec![1.0, 2.0, 3.0]);
    /// ```
    pub fn output_iter(&self, output: usize) -> OutputIter<'_> {
        OutputIter {
            buffer: self,
            output,
            sample: 0,
        }
    }

    // =========================================================================
    // Aggregation helpers
    // =========================================================================

    /// Sum gradients and hessians for a specific output across all samples.
    ///
    /// Returns (sum_grad, sum_hess).
    pub fn sum_for_output(&self, output: usize) -> (f32, f32) {
        let mut sum_grad = 0.0f32;
        let mut sum_hess = 0.0f32;

        for sample in 0..self.n_samples {
            let idx = sample * self.n_outputs + output;
            sum_grad += self.grads[idx];
            sum_hess += self.hess[idx];
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
        let (sum_grad, sum_hess) = self.sum_for_output(output);
        if sum_hess.abs() < min_hess {
            0.0
        } else {
            -sum_grad / sum_hess
        }
    }

    // =========================================================================
    // Private helpers
    // =========================================================================

    /// Convert (sample, output) to linear index.
    #[inline]
    fn index(&self, sample: usize, output: usize) -> usize {
        debug_assert!(sample < self.n_samples);
        debug_assert!(output < self.n_outputs);
        sample * self.n_outputs + output
    }
}

/// Iterator over all samples for a specific output.
pub struct OutputIter<'a> {
    buffer: &'a GradientBuffer,
    output: usize,
    sample: usize,
}

impl<'a> Iterator for OutputIter<'a> {
    /// (sample_index, gradient, hessian)
    type Item = (usize, f32, f32);

    fn next(&mut self) -> Option<Self::Item> {
        if self.sample >= self.buffer.n_samples {
            return None;
        }

        let idx = self.sample * self.buffer.n_outputs + self.output;
        let result = (
            self.sample,
            self.buffer.grads[idx],
            self.buffer.hess[idx],
        );
        self.sample += 1;
        Some(result)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.buffer.n_samples - self.sample;
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
        let buffer = GradientBuffer::new(100, 3);
        assert_eq!(buffer.n_samples(), 100);
        assert_eq!(buffer.n_outputs(), 3);
        assert_eq!(buffer.len(), 300);
    }

    #[test]
    fn get_set_single() {
        let mut buffer = GradientBuffer::new(10, 1);

        buffer.set(0, 0, 1.5, 0.25);
        let (g, h) = buffer.get(0, 0);
        assert_eq!(g, 1.5);
        assert_eq!(h, 0.25);

        buffer.set(5, 0, -0.5, 1.0);
        assert_eq!(buffer.grad(5, 0), -0.5);
        assert_eq!(buffer.hess(5, 0), 1.0);
    }

    #[test]
    fn get_set_multiclass() {
        let mut buffer = GradientBuffer::new(10, 3);

        // Sample 0: gradients for 3 classes
        buffer.set(0, 0, 0.2, 0.16);
        buffer.set(0, 1, 0.3, 0.21);
        buffer.set(0, 2, -0.5, 0.25);

        assert_eq!(buffer.grad(0, 0), 0.2);
        assert_eq!(buffer.grad(0, 1), 0.3);
        assert_eq!(buffer.grad(0, 2), -0.5);
    }

    #[test]
    fn sample_slices() {
        let mut buffer = GradientBuffer::new(5, 3);

        // Set sample 2's gradients
        let grads = buffer.sample_grads_mut(2);
        grads[0] = 1.0;
        grads[1] = 2.0;
        grads[2] = 3.0;

        // Read back
        let grads = buffer.sample_grads(2);
        assert_eq!(grads, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn output_iter() {
        let mut buffer = GradientBuffer::new(3, 2);

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
    fn sum_for_output() {
        let mut buffer = GradientBuffer::new(4, 2);

        // Output 0: grads = [1, 2, 3, 4], hess = [1, 1, 1, 1]
        for i in 0..4 {
            buffer.set(i, 0, (i + 1) as f32, 1.0);
        }

        let (sum_g, sum_h) = buffer.sum_for_output(0);
        assert_eq!(sum_g, 10.0); // 1+2+3+4
        assert_eq!(sum_h, 4.0);
    }

    #[test]
    fn bias_update() {
        let mut buffer = GradientBuffer::new(4, 1);

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
        let mut buffer = GradientBuffer::new(3, 2);
        buffer.set(0, 0, 1.0, 2.0);
        buffer.set(1, 1, 3.0, 4.0);

        buffer.reset();

        assert_eq!(buffer.grad(0, 0), 0.0);
        assert_eq!(buffer.hess(0, 0), 0.0);
        assert_eq!(buffer.grad(1, 1), 0.0);
        assert_eq!(buffer.hess(1, 1), 0.0);
    }

    #[test]
    fn as_mut_slices() {
        let mut buffer = GradientBuffer::new(3, 1);

        let (grads, hess) = buffer.as_mut_slices();
        grads[0] = 1.0;
        grads[1] = 2.0;
        grads[2] = 3.0;
        hess[0] = 0.5;
        hess[1] = 0.5;
        hess[2] = 0.5;

        assert_eq!(buffer.grads(), &[1.0, 2.0, 3.0]);
        assert_eq!(buffer.hess_slice(), &[0.5, 0.5, 0.5]);
    }

    #[test]
    #[should_panic(expected = "n_samples must be positive")]
    fn zero_samples_panics() {
        GradientBuffer::new(0, 1);
    }

    #[test]
    #[should_panic(expected = "n_outputs must be positive")]
    fn zero_outputs_panics() {
        GradientBuffer::new(10, 0);
    }
}

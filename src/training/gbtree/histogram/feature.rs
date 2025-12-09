//! Per-feature gradient histogram.

use std::ops::{Sub, SubAssign};

use super::slice::HistogramBins;

/// Gradient histogram for a single feature.
///
/// Each bin stores sum of gradients, sum of hessians, and sample count.
///
/// # Memory Layout
///
/// For 256 bins:
/// - `sum_grad`: 1024 bytes (256 × f32)
/// - `sum_hess`: 1024 bytes (256 × f32)
/// - `count`:    1024 bytes (256 × u32)
/// - Total: ~3 KB per feature
#[derive(Debug, Clone)]
pub struct FeatureHistogram {
    /// Sum of gradients per bin: [num_bins]
    sum_grad: Box<[f32]>,
    /// Sum of hessians per bin: [num_bins]
    sum_hess: Box<[f32]>,
    /// Count of samples per bin: [num_bins]
    count: Box<[u32]>,
    /// Number of bins (including bin 0 for missing)
    num_bins: u16,
}

impl FeatureHistogram {
    /// Create a new histogram with the specified number of bins.
    ///
    /// All bins are initialized to zero.
    pub fn new(num_bins: u16) -> Self {
        let n = num_bins as usize;
        Self {
            sum_grad: vec![0.0; n].into_boxed_slice(),
            sum_hess: vec![0.0; n].into_boxed_slice(),
            count: vec![0; n].into_boxed_slice(),
            num_bins,
        }
    }

    /// Number of bins in this histogram.
    #[inline]
    pub fn num_bins(&self) -> u16 {
        self.num_bins
    }

    /// Add a sample to a bin.
    ///
    /// Accumulates gradient, hessian, and increments count.
    #[inline]
    pub fn add(&mut self, bin: usize, grad: f32, hess: f32) {
        debug_assert!(
            bin < self.num_bins as usize,
            "bin {} >= num_bins {}",
            bin,
            self.num_bins
        );
        // SAFETY: bounds checked by debug_assert
        unsafe {
            *self.sum_grad.get_unchecked_mut(bin) += grad;
            *self.sum_hess.get_unchecked_mut(bin) += hess;
            *self.count.get_unchecked_mut(bin) += 1;
        }
    }

    /// Get statistics for a bin.
    ///
    /// Returns (sum_grad, sum_hess, count).
    #[inline]
    pub fn bin_stats(&self, bin: usize) -> (f32, f32, u32) {
        debug_assert!(bin < self.num_bins as usize);
        unsafe {
            (
                *self.sum_grad.get_unchecked(bin),
                *self.sum_hess.get_unchecked(bin),
                *self.count.get_unchecked(bin),
            )
        }
    }

    /// Get sum of gradients for a bin.
    #[inline]
    pub fn grad(&self, bin: usize) -> f32 {
        debug_assert!(bin < self.num_bins as usize);
        unsafe { *self.sum_grad.get_unchecked(bin) }
    }

    /// Get sum of hessians for a bin.
    #[inline]
    pub fn hess(&self, bin: usize) -> f32 {
        debug_assert!(bin < self.num_bins as usize);
        unsafe { *self.sum_hess.get_unchecked(bin) }
    }

    /// Get count for a bin.
    #[inline]
    pub fn count(&self, bin: usize) -> u32 {
        debug_assert!(bin < self.num_bins as usize);
        unsafe { *self.count.get_unchecked(bin) }
    }

    /// Reset all bins to zero.
    ///
    /// Faster than allocating a new histogram.
    pub fn reset(&mut self) {
        self.sum_grad.fill(0.0);
        self.sum_hess.fill(0.0);
        self.count.fill(0);
    }

    /// Copy from another histogram.
    ///
    /// Overwrites current contents with source.
    pub fn copy_from(&mut self, src: &Self) {
        debug_assert_eq!(self.num_bins, src.num_bins);
        self.sum_grad.copy_from_slice(&src.sum_grad);
        self.sum_hess.copy_from_slice(&src.sum_hess);
        self.count.copy_from_slice(&src.count);
    }

    /// Get slice of gradient sums (all bins).
    #[inline]
    pub fn grads(&self) -> &[f32] {
        &self.sum_grad
    }

    /// Get slice of hessian sums (all bins).
    #[inline]
    pub fn hesses(&self) -> &[f32] {
        &self.sum_hess
    }

    /// Get slice of counts (per bin).
    #[inline]
    pub fn counts(&self) -> &[u32] {
        &self.count
    }

    /// Compute total gradient sum across all bins.
    pub fn total_grad(&self) -> f32 {
        self.sum_grad.iter().sum()
    }

    /// Compute total hessian sum across all bins.
    pub fn total_hess(&self) -> f32 {
        self.sum_hess.iter().sum()
    }

    /// Compute total count across all bins.
    pub fn total_count(&self) -> u32 {
        self.count.iter().sum()
    }
}

/// Subtract two histograms: `&self - &rhs` (allocating).
///
/// Used for histogram subtraction optimization:
/// `parent - smaller_child = larger_child`.
impl Sub<&FeatureHistogram> for &FeatureHistogram {
    type Output = FeatureHistogram;

    fn sub(self, rhs: &FeatureHistogram) -> FeatureHistogram {
        debug_assert_eq!(self.num_bins, rhs.num_bins);

        let n = self.num_bins as usize;

        let mut sum_grad = vec![0.0f32; n].into_boxed_slice();
        let mut sum_hess = vec![0.0f32; n].into_boxed_slice();
        let mut count = vec![0u32; n].into_boxed_slice();

        for i in 0..n {
            sum_grad[i] = self.sum_grad[i] - rhs.sum_grad[i];
            sum_hess[i] = self.sum_hess[i] - rhs.sum_hess[i];
            count[i] = self.count[i] - rhs.count[i];
        }

        FeatureHistogram {
            sum_grad,
            sum_hess,
            count,
            num_bins: self.num_bins,
        }
    }
}

/// Subtract in place: `self -= &rhs`.
///
/// More efficient than allocating when you can mutate the left operand.
impl SubAssign<&FeatureHistogram> for FeatureHistogram {
    #[inline]
    fn sub_assign(&mut self, rhs: &FeatureHistogram) {
        debug_assert_eq!(self.num_bins, rhs.num_bins);

        for i in 0..self.sum_grad.len() {
            self.sum_grad[i] -= rhs.sum_grad[i];
            self.sum_hess[i] -= rhs.sum_hess[i];
        }
        for i in 0..self.count.len() {
            self.count[i] -= rhs.count[i];
        }
    }
}

/// Consuming subtract: `self - &rhs` (reuses self's memory).
///
/// Most efficient when you own the left operand and don't need it afterward.
/// Avoids allocation by reusing self's storage for the result.
impl Sub<&FeatureHistogram> for FeatureHistogram {
    type Output = FeatureHistogram;

    #[inline]
    fn sub(mut self, rhs: &FeatureHistogram) -> FeatureHistogram {
        self -= rhs;
        self
    }
}

impl HistogramBins for FeatureHistogram {
    #[inline]
    fn num_bins(&self) -> u16 {
        self.num_bins
    }

    #[inline]
    fn bin_stats(&self, bin: usize) -> (f32, f32, u32) {
        debug_assert!(bin < self.num_bins as usize);
        unsafe {
            (
                *self.sum_grad.get_unchecked(bin),
                *self.sum_hess.get_unchecked(bin),
                *self.count.get_unchecked(bin),
            )
        }
    }
}

impl<H: HistogramBins + ?Sized> HistogramBins for &H {
    #[inline]
    fn num_bins(&self) -> u16 {
        (*self).num_bins()
    }

    #[inline]
    fn bin_stats(&self, bin: usize) -> (f32, f32, u32) {
        (*self).bin_stats(bin)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_histogram_basic() {
        let mut hist = FeatureHistogram::new(4);

        assert_eq!(hist.num_bins(), 4);
        assert_eq!(hist.bin_stats(0), (0.0, 0.0, 0));

        // Add some samples
        hist.add(0, 1.0, 0.5);
        hist.add(0, 2.0, 1.0);
        hist.add(1, 3.0, 1.5);

        assert_eq!(hist.bin_stats(0), (3.0, 1.5, 2));
        assert_eq!(hist.bin_stats(1), (3.0, 1.5, 1));
        assert_eq!(hist.bin_stats(2), (0.0, 0.0, 0));

        // Check individual accessors
        assert_eq!(hist.grad(0), 3.0);
        assert_eq!(hist.hess(0), 1.5);
        assert_eq!(hist.count(0), 2);
    }

    #[test]
    fn test_feature_histogram_totals() {
        let mut hist = FeatureHistogram::new(4);

        hist.add(0, 1.0, 0.5);
        hist.add(1, 2.0, 1.0);
        hist.add(2, 3.0, 1.5);
        hist.add(3, 4.0, 2.0);

        assert_eq!(hist.total_grad(), 10.0);
        assert_eq!(hist.total_hess(), 5.0);
        assert_eq!(hist.total_count(), 4);
    }

    #[test]
    fn test_feature_histogram_reset() {
        let mut hist = FeatureHistogram::new(4);

        hist.add(0, 1.0, 0.5);
        hist.add(1, 2.0, 1.0);

        hist.reset();

        assert_eq!(hist.bin_stats(0), (0.0, 0.0, 0));
        assert_eq!(hist.bin_stats(1), (0.0, 0.0, 0));
        assert_eq!(hist.total_count(), 0);
    }

    #[test]
    fn test_feature_histogram_subtraction() {
        // Parent histogram
        let mut parent = FeatureHistogram::new(4);
        parent.add(0, 10.0, 5.0); // bin 0: 2 samples
        parent.add(0, 10.0, 5.0);
        parent.add(1, 6.0, 3.0); // bin 1: 1 sample
        parent.add(2, 4.0, 2.0); // bin 2: 1 sample

        // Child (left) histogram - subset of parent
        let mut child = FeatureHistogram::new(4);
        child.add(0, 10.0, 5.0); // bin 0: 1 sample
        child.add(1, 6.0, 3.0); // bin 1: 1 sample

        // Compute sibling via subtraction: parent - child = sibling
        let sibling = &parent - &child;

        // Sibling should have: bin 0: 1 sample, bin 2: 1 sample
        assert_eq!(sibling.bin_stats(0), (10.0, 5.0, 1)); // 20-10=10
        assert_eq!(sibling.bin_stats(1), (0.0, 0.0, 0)); // 6-6=0
        assert_eq!(sibling.bin_stats(2), (4.0, 2.0, 1)); // 4-0=4
    }

    #[test]
    fn test_feature_histogram_sub_assign() {
        // Parent histogram
        let mut parent = FeatureHistogram::new(4);
        parent.add(0, 10.0, 5.0);
        parent.add(0, 10.0, 5.0);
        parent.add(1, 6.0, 3.0);
        parent.add(2, 4.0, 2.0);

        // Child histogram
        let mut child = FeatureHistogram::new(4);
        child.add(0, 10.0, 5.0);
        child.add(1, 6.0, 3.0);

        // In-place subtraction: parent -= child
        parent -= &child;

        // parent now contains sibling values
        assert_eq!(parent.bin_stats(0), (10.0, 5.0, 1));
        assert_eq!(parent.bin_stats(1), (0.0, 0.0, 0));
        assert_eq!(parent.bin_stats(2), (4.0, 2.0, 1));
    }

    #[test]
    fn test_feature_histogram_consuming_sub() {
        // Parent histogram (will be consumed)
        let mut parent = FeatureHistogram::new(4);
        parent.add(0, 10.0, 5.0);
        parent.add(0, 10.0, 5.0);
        parent.add(1, 6.0, 3.0);
        parent.add(2, 4.0, 2.0);

        // Child histogram
        let mut child = FeatureHistogram::new(4);
        child.add(0, 10.0, 5.0);
        child.add(1, 6.0, 3.0);

        // Consuming subtraction: parent - &child (reuses parent's memory)
        let sibling = parent - &child;

        assert_eq!(sibling.bin_stats(0), (10.0, 5.0, 1));
        assert_eq!(sibling.bin_stats(1), (0.0, 0.0, 0));
        assert_eq!(sibling.bin_stats(2), (4.0, 2.0, 1));
    }
}

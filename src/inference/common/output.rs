//! Prediction output types.

use approx::{AbsDiffEq, RelativeEq};

/// Prediction output: flat storage with shape metadata.
///
/// Stores predictions in row-major layout for cache efficiency.
/// Each row contains `num_groups` values (1 for regression, K for K-class).
///
/// # Memory Layout
///
/// ```text
/// data[row * num_groups + group] = prediction for (row, group)
/// ```
///
/// # Example
///
/// ```
/// use booste_rs::inference::common::PredictionOutput;
///
/// // 3 rows, 2 groups (binary classification logits)
/// let output = PredictionOutput::new(vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6], 3, 2);
///
/// assert_eq!(output.row(0), &[0.1, -0.2]);
/// assert_eq!(output.row(1), &[0.3, -0.4]);
/// assert_eq!(output.row(2), &[0.5, -0.6]);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct PredictionOutput {
    /// Flat data in row-major layout.
    data: Vec<f32>,
    /// Number of rows (samples).
    num_rows: usize,
    /// Number of groups (output dimensions).
    num_groups: usize,
}

impl PredictionOutput {
    /// Create a new prediction output.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != num_rows * num_groups`.
    pub fn new(data: Vec<f32>, num_rows: usize, num_groups: usize) -> Self {
        assert_eq!(
            data.len(),
            num_rows * num_groups,
            "Data length {} does not match shape {}x{}",
            data.len(),
            num_rows,
            num_groups
        );
        Self {
            data,
            num_rows,
            num_groups,
        }
    }

    /// Create an output initialized to zeros.
    pub fn zeros(num_rows: usize, num_groups: usize) -> Self {
        Self {
            data: vec![0.0; num_rows * num_groups],
            num_rows,
            num_groups,
        }
    }

    /// Number of rows (samples).
    #[inline]
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// Number of groups (output dimensions).
    #[inline]
    pub fn num_groups(&self) -> usize {
        self.num_groups
    }

    /// Shape as (rows, groups).
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.num_rows, self.num_groups)
    }

    /// Get prediction for a single row.
    ///
    /// # Panics
    ///
    /// Panics if `row_idx >= num_rows`.
    #[inline]
    pub fn row(&self, row_idx: usize) -> &[f32] {
        let start = row_idx * self.num_groups;
        &self.data[start..start + self.num_groups]
    }

    /// Get mutable prediction for a single row.
    ///
    /// # Panics
    ///
    /// Panics if `row_idx >= num_rows`.
    #[inline]
    pub fn row_mut(&mut self, row_idx: usize) -> &mut [f32] {
        let start = row_idx * self.num_groups;
        &mut self.data[start..start + self.num_groups]
    }

    /// Iterate over rows.
    pub fn rows(&self) -> impl Iterator<Item = &[f32]> {
        self.data.chunks_exact(self.num_groups)
    }

    /// Get raw flat data.
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Get mutable raw flat data.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Consume and return raw data.
    pub fn into_vec(self) -> Vec<f32> {
        self.data
    }

    /// Convert to `Vec<Vec<f32>>` (allocates).
    pub fn to_nested(&self) -> Vec<Vec<f32>> {
        self.rows().map(|r| r.to_vec()).collect()
    }
}

// =============================================================================
// Approx Trait Implementations
// =============================================================================

impl AbsDiffEq for PredictionOutput {
    type Epsilon = f32;

    fn default_epsilon() -> Self::Epsilon {
        f32::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.num_rows == other.num_rows
            && self.num_groups == other.num_groups
            && self
                .data
                .iter()
                .zip(other.data.iter())
                .all(|(a, b)| a.abs_diff_eq(b, epsilon))
    }
}

impl RelativeEq for PredictionOutput {
    fn default_max_relative() -> Self::Epsilon {
        f32::default_max_relative()
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.num_rows == other.num_rows
            && self.num_groups == other.num_groups
            && self
                .data
                .iter()
                .zip(other.data.iter())
                .all(|(a, b)| a.relative_eq(b, epsilon, max_relative))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_output() {
        let output = PredictionOutput::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        assert_eq!(output.num_rows(), 2);
        assert_eq!(output.num_groups(), 2);
        assert_eq!(output.shape(), (2, 2));
    }

    #[test]
    fn zeros() {
        let output = PredictionOutput::zeros(3, 2);
        assert_eq!(output.as_slice(), &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn row_access() {
        let output = PredictionOutput::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
        assert_eq!(output.row(0), &[1.0, 2.0]);
        assert_eq!(output.row(1), &[3.0, 4.0]);
        assert_eq!(output.row(2), &[5.0, 6.0]);
    }

    #[test]
    fn row_mut() {
        let mut output = PredictionOutput::zeros(2, 2);
        output.row_mut(0)[0] = 1.0;
        output.row_mut(1)[1] = 2.0;
        assert_eq!(output.as_slice(), &[1.0, 0.0, 0.0, 2.0]);
    }

    #[test]
    fn rows_iteration() {
        let output = PredictionOutput::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let rows: Vec<_> = output.rows().collect();
        assert_eq!(rows, vec![&[1.0, 2.0][..], &[3.0, 4.0][..]]);
    }

    #[test]
    fn to_nested() {
        let output = PredictionOutput::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let nested = output.to_nested();
        assert_eq!(nested, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    }

    #[test]
    fn into_vec() {
        let output = PredictionOutput::new(vec![1.0, 2.0, 3.0], 3, 1);
        let data = output.into_vec();
        assert_eq!(data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    #[should_panic(expected = "does not match shape")]
    fn wrong_size_panics() {
        PredictionOutput::new(vec![1.0, 2.0, 3.0], 2, 2);
    }

    // =========================================================================
    // Approx trait tests
    // =========================================================================

    #[test]
    fn abs_diff_eq_equal() {
        let a = PredictionOutput::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let b = PredictionOutput::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        assert!(a.abs_diff_eq(&b, 0.0));
    }

    #[test]
    fn abs_diff_eq_within_epsilon() {
        let a = PredictionOutput::new(vec![1.0, 2.0], 2, 1);
        let b = PredictionOutput::new(vec![1.00001, 2.00001], 2, 1);
        assert!(a.abs_diff_eq(&b, 1e-4));
        assert!(!a.abs_diff_eq(&b, 1e-6));
    }

    #[test]
    fn abs_diff_eq_shape_mismatch() {
        let a = PredictionOutput::new(vec![1.0, 2.0], 2, 1);
        let b = PredictionOutput::new(vec![1.0, 2.0], 1, 2);
        assert!(!a.abs_diff_eq(&b, 1.0));
    }

    #[test]
    fn relative_eq_equal() {
        let a = PredictionOutput::new(vec![1.0, 2.0, 3.0], 3, 1);
        let b = PredictionOutput::new(vec![1.0, 2.0, 3.0], 3, 1);
        assert!(a.relative_eq(&b, 0.0, 0.0));
    }

    #[test]
    fn relative_eq_within_tolerance() {
        let a = PredictionOutput::new(vec![100.0, 200.0], 2, 1);
        let b = PredictionOutput::new(vec![100.001, 200.002], 2, 1);
        // Should pass with relative tolerance of 1e-4 (0.01%)
        assert!(a.relative_eq(&b, 0.0, 1e-4));
    }

    #[test]
    fn approx_macro_integration() {
        use approx::{assert_abs_diff_eq, assert_relative_eq};

        let a = PredictionOutput::new(vec![1.0, 2.0, 3.0], 3, 1);
        let b = PredictionOutput::new(vec![1.0, 2.0, 3.0], 3, 1);

        assert_abs_diff_eq!(a, b);
        assert_relative_eq!(a, b);
    }
}

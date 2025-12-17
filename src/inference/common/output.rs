//! Prediction output types.

use approx::{AbsDiffEq, RelativeEq};

/// Prediction output: flat storage with shape metadata.
///
/// Stores predictions in column-major layout for efficient training and
/// vectorized operations. Each column contains all values for one output group.
///
/// # Memory Layout
///
/// ```text
/// data[group * num_rows + row] = prediction for (row, group)
/// ```
///
/// Column-major layout enables:
/// - Efficient base score initialization via `column_mut(g).fill(base_score)`
/// - Sequential writes when accumulating tree outputs for a group
/// - Vectorized operations (softmax, transforms) over columns
///
/// # Example
///
/// ```
/// use booste_rs::inference::common::PredictionOutput;
///
/// // 3 rows, 2 groups (binary classification logits)
/// // Column-major: [g0r0, g0r1, g0r2, g1r0, g1r1, g1r2]
/// let output = PredictionOutput::new(vec![0.1, 0.2, 0.3, -0.1, -0.2, -0.3], 3, 2);
///
/// assert_eq!(output.column(0), &[0.1, 0.2, 0.3]);
/// assert_eq!(output.column(1), &[-0.1, -0.2, -0.3]);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct PredictionOutput {
    /// Flat data in column-major layout.
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

    /// Get a single column (all rows for one output group).
    ///
    /// # Panics
    ///
    /// Panics if `group_idx >= num_groups`.
    #[inline]
    pub fn column(&self, group_idx: usize) -> &[f32] {
        let start = group_idx * self.num_rows;
        &self.data[start..start + self.num_rows]
    }

    /// Get mutable column (all rows for one output group).
    ///
    /// # Panics
    ///
    /// Panics if `group_idx >= num_groups`.
    #[inline]
    pub fn column_mut(&mut self, group_idx: usize) -> &mut [f32] {
        let start = group_idx * self.num_rows;
        &mut self.data[start..start + self.num_rows]
    }

    /// Get prediction value at (row, group).
    #[inline]
    pub fn get(&self, row_idx: usize, group_idx: usize) -> f32 {
        self.data[group_idx * self.num_rows + row_idx]
    }

    /// Set prediction value at (row, group).
    #[inline]
    pub fn set(&mut self, row_idx: usize, group_idx: usize, value: f32) {
        self.data[group_idx * self.num_rows + row_idx] = value;
    }

    /// Add to prediction value at (row, group).
    #[inline]
    pub fn add(&mut self, row_idx: usize, group_idx: usize, value: f32) {
        self.data[group_idx * self.num_rows + row_idx] += value;
    }

    /// Iterate over columns.
    pub fn columns(&self) -> impl Iterator<Item = &[f32]> {
        self.data.chunks_exact(self.num_rows)
    }

    /// Iterate over columns mutably.
    pub fn columns_mut(&mut self) -> impl Iterator<Item = &mut [f32]> {
        self.data.chunks_exact_mut(self.num_rows)
    }

    /// Get raw flat data (column-major layout).
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Get mutable raw flat data (column-major layout).
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Consume and return raw data (column-major layout).
    pub fn into_vec(self) -> Vec<f32> {
        self.data
    }

    /// Convert to `Vec<Vec<f32>>` where each inner vec is a column (group).
    pub fn to_nested(&self) -> Vec<Vec<f32>> {
        self.columns().map(|c| c.to_vec()).collect()
    }

    /// Get row values by copying into a buffer.
    ///
    /// For column-major layout, row access requires strided iteration.
    #[inline]
    pub fn copy_row(&self, row_idx: usize, out: &mut [f32]) {
        debug_assert_eq!(out.len(), self.num_groups);
        for (group, val) in out.iter_mut().enumerate() {
            *val = self.get(row_idx, group);
        }
    }

    /// Get all values for a single row as a new Vec.
    ///
    /// Note: This allocates. For tight loops, prefer `copy_row`.
    pub fn row_vec(&self, row_idx: usize) -> Vec<f32> {
        let mut out = vec![0.0; self.num_groups];
        self.copy_row(row_idx, &mut out);
        out
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
    fn column_access() {
        // Column-major: [g0r0, g0r1, g0r2, g1r0, g1r1, g1r2]
        let output = PredictionOutput::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
        assert_eq!(output.column(0), &[1.0, 2.0, 3.0]);
        assert_eq!(output.column(1), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn get_set() {
        let mut output = PredictionOutput::zeros(2, 2);
        output.set(0, 0, 1.0);
        output.set(1, 1, 2.0);
        // Column-major: [g0r0, g0r1, g1r0, g1r1] = [1.0, 0.0, 0.0, 2.0]
        assert_eq!(output.as_slice(), &[1.0, 0.0, 0.0, 2.0]);
        assert_eq!(output.get(0, 0), 1.0);
        assert_eq!(output.get(1, 1), 2.0);
    }

    #[test]
    fn add() {
        let mut output = PredictionOutput::zeros(2, 2);
        output.add(0, 0, 1.0);
        output.add(0, 0, 0.5);
        assert_eq!(output.get(0, 0), 1.5);
    }

    #[test]
    fn columns_iteration() {
        let output = PredictionOutput::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let cols: Vec<_> = output.columns().collect();
        assert_eq!(cols, vec![&[1.0, 2.0][..], &[3.0, 4.0][..]]);
    }

    #[test]
    fn row_vec_access() {
        // Column-major: [g0r0, g0r1, g0r2, g1r0, g1r1, g1r2]
        let output = PredictionOutput::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
        assert_eq!(output.row_vec(0), vec![1.0, 4.0]);
        assert_eq!(output.row_vec(1), vec![2.0, 5.0]);
        assert_eq!(output.row_vec(2), vec![3.0, 6.0]);
    }

    #[test]
    fn to_nested() {
        // Column-major: columns become the nested vecs
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

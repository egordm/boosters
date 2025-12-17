//! Linear model data structure and prediction.

use rayon::prelude::*;

use crate::data::{DataMatrix, RowView};
use crate::inference::common::PredictionOutput;

/// Linear booster model (weights + bias).
///
/// Stores a weight matrix for linear prediction. The weights are laid out
/// in feature-major, group-minor order with bias in the last row:
///
/// ```text
/// weights[feature * num_groups + group] → coefficient
/// weights[num_features * num_groups + group] → bias
/// ```
///
/// # Example
///
/// ```
/// use booste_rs::inference::gblinear::LinearModel;
///
/// // 3 features, 2 output groups (e.g., binary classification)
/// let weights = vec![
///     0.1, 0.2,  // feature 0: group 0, group 1
///     0.3, 0.4,  // feature 1
///     0.5, 0.6,  // feature 2
///     0.0, 0.0,  // bias
/// ];
/// let model = LinearModel::new(weights.into_boxed_slice(), 3, 2);
///
/// assert_eq!(model.weight(0, 0), 0.1);
/// assert_eq!(model.weight(0, 1), 0.2);
/// assert_eq!(model.bias(0), 0.0);
/// ```
#[derive(Debug, Clone)]
pub struct LinearModel {
    /// Flat weight array: (num_features + 1) × num_groups
    /// Layout: feature-major, group-minor. Last row is bias.
    weights: Box<[f32]>,

    /// Number of input features.
    num_features: usize,

    /// Number of output groups (1 for regression, K for K-class).
    num_groups: usize,
}

impl LinearModel {
    /// Create a new linear model from weights.
    ///
    /// # Arguments
    ///
    /// * `weights` - Flat weight array of size `(num_features + 1) * num_groups`
    /// * `num_features` - Number of input features
    /// * `num_groups` - Number of output groups
    ///
    /// # Panics
    ///
    /// Panics if weights length doesn't match `(num_features + 1) * num_groups`.
    pub fn new(weights: Box<[f32]>, num_features: usize, num_groups: usize) -> Self {
        let expected_len = (num_features + 1) * num_groups;
        assert_eq!(
            weights.len(),
            expected_len,
            "weights length {} doesn't match (num_features + 1) * num_groups = {}",
            weights.len(),
            expected_len
        );

        Self {
            weights,
            num_features,
            num_groups,
        }
    }

    /// Create a zero-initialized linear model.
    pub fn zeros(num_features: usize, num_groups: usize) -> Self {
        let weights = vec![0.0; (num_features + 1) * num_groups].into_boxed_slice();
        Self {
            weights,
            num_features,
            num_groups,
        }
    }

    /// Number of input features.
    #[inline]
    pub fn num_features(&self) -> usize {
        self.num_features
    }

    /// Number of output groups.
    #[inline]
    pub fn num_groups(&self) -> usize {
        self.num_groups
    }

    /// Get weight for a feature and group.
    ///
    /// # Arguments
    ///
    /// * `feature` - Feature index (0..num_features)
    /// * `group` - Output group index (0..num_groups)
    #[inline]
    pub fn weight(&self, feature: usize, group: usize) -> f32 {
        debug_assert!(feature < self.num_features, "feature index out of bounds");
        debug_assert!(group < self.num_groups, "group index out of bounds");
        self.weights[feature * self.num_groups + group]
    }

    /// Get bias for a group.
    #[inline]
    pub fn bias(&self, group: usize) -> f32 {
        debug_assert!(group < self.num_groups, "group index out of bounds");
        self.weights[self.num_features * self.num_groups + group]
    }

    /// Raw access to weights (for feature importance, serialization, etc.).
    #[inline]
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Mutable access to weights (for training).
    #[inline]
    pub fn weights_mut(&mut self) -> &mut [f32] {
        &mut self.weights
    }

    /// Set weight for a feature and group.
    #[inline]
    pub fn set_weight(&mut self, feature: usize, group: usize, value: f32) {
        debug_assert!(feature < self.num_features, "feature index out of bounds");
        debug_assert!(group < self.num_groups, "group index out of bounds");
        self.weights[feature * self.num_groups + group] = value;
    }

    /// Set bias for a group.
    #[inline]
    pub fn set_bias(&mut self, group: usize, value: f32) {
        debug_assert!(group < self.num_groups, "group index out of bounds");
        self.weights[self.num_features * self.num_groups + group] = value;
    }

    /// Add to weight for a feature and group.
    #[inline]
    pub fn add_weight(&mut self, feature: usize, group: usize, delta: f32) {
        debug_assert!(feature < self.num_features, "feature index out of bounds");
        debug_assert!(group < self.num_groups, "group index out of bounds");
        self.weights[feature * self.num_groups + group] += delta;
    }

    /// Add to bias for a group.
    #[inline]
    pub fn add_bias(&mut self, group: usize, delta: f32) {
        debug_assert!(group < self.num_groups, "group index out of bounds");
        self.weights[self.num_features * self.num_groups + group] += delta;
    }

    // =========================================================================
    // Column-Major Prediction (for training)
    // =========================================================================

    /// Predict into a column-major buffer.
    ///
    /// Output layout: `output[group * num_rows + row]`
    ///
    /// This is the preferred method for training where predictions are stored
    /// in column-major layout for efficient gradient computation.
    ///
    /// # Arguments
    ///
    /// * `data` - Feature matrix (must support column access)
    /// * `output` - Pre-allocated buffer of size `num_rows * num_groups`
    pub fn predict_col_major(&self, data: &crate::data::ColMatrix<f32>, output: &mut [f32]) {
        let num_rows = data.num_rows();
        let num_groups = self.num_groups;
        debug_assert_eq!(output.len(), num_rows * num_groups);

        // Initialize with bias (column-major: group-first)
        for group in 0..num_groups {
            output[group * num_rows..(group + 1) * num_rows].fill(self.bias(group));
        }

        // Add weighted features
        for feat_idx in 0..self.num_features {
            for (row_idx, value) in data.column(feat_idx) {
                for group in 0..num_groups {
                    output[group * num_rows + row_idx] += value * self.weight(feat_idx, group);
                }
            }
        }
    }

    // =========================================================================
    // Row-Major Prediction (for inference)
    // =========================================================================

    /// Predict for a single row.
    ///
    /// Returns a vector of length `num_groups`.
    pub fn predict_row(&self, features: &[f32], base_score: &[f32]) -> Vec<f32> {
        debug_assert!(
            features.len() >= self.num_features,
            "not enough features: got {}, need {}",
            features.len(),
            self.num_features
        );

        let mut outputs = Vec::with_capacity(self.num_groups);

        for group in 0..self.num_groups {
            let base = base_score.get(group).copied().unwrap_or(0.0);
            let mut sum = base + self.bias(group);

            for (feat_idx, &value) in features.iter().take(self.num_features).enumerate() {
                sum += value * self.weight(feat_idx, group);
            }

            outputs.push(sum);
        }

        outputs
    }

    /// Predict for a batch of rows.
    ///
    /// Returns a column-major output buffer with shape `(num_rows, num_groups)`.
    pub fn predict<M: DataMatrix<Element = f32>>(
        &self,
        data: &M,
        base_score: &[f32],
    ) -> PredictionOutput {
        let num_rows = data.num_rows();
        let mut output = vec![0.0; num_rows * self.num_groups];

        // Column-major: output[group * num_rows + row] = prediction
        for row_idx in 0..num_rows {
            let row = data.row(row_idx);

            for group in 0..self.num_groups {
                let base = base_score.get(group).copied().unwrap_or(0.0);
                let mut sum = base + self.bias(group);

                for feat_idx in 0..self.num_features {
                    let value = row.get(feat_idx).unwrap_or(0.0);
                    sum += value * self.weight(feat_idx, group);
                }

                output[group * num_rows + row_idx] = sum;
            }
        }

        PredictionOutput::new(output, num_rows, self.num_groups)
    }

    /// Parallel prediction for a batch of rows.
    ///
    /// Uses Rayon to parallelize over rows. Returns column-major output.
    pub fn par_predict<M: DataMatrix<Element = f32> + Sync>(
        &self,
        data: &M,
        base_score: &[f32],
    ) -> PredictionOutput {
        let num_rows = data.num_rows();
        let num_groups = self.num_groups;

        // First collect row-major results per row (each row returns its groups)
        let row_outputs: Vec<Vec<f32>> = (0..num_rows)
            .into_par_iter()
            .map(|row_idx| {
                let row = data.row(row_idx);
                let mut row_output = Vec::with_capacity(num_groups);

                for group in 0..num_groups {
                    let base = base_score.get(group).copied().unwrap_or(0.0);
                    let mut sum = base + self.bias(group);

                    for feat_idx in 0..self.num_features {
                        let value = row.get(feat_idx).unwrap_or(0.0);
                        sum += value * self.weight(feat_idx, group);
                    }

                    row_output.push(sum);
                }

                row_output
            })
            .collect();

        // Convert to column-major layout
        let mut output = vec![0.0; num_rows * num_groups];
        for (row_idx, row_output) in row_outputs.into_iter().enumerate() {
            for (group, value) in row_output.into_iter().enumerate() {
                output[group * num_rows + row_idx] = value;
            }
        }

        PredictionOutput::new(output, num_rows, num_groups)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::RowMatrix;

    #[test]
    fn linear_model_new() {
        // 2 features, 1 group (regression)
        let weights = vec![0.5, 0.3, 0.1].into_boxed_slice(); // w0, w1, bias
        let model = LinearModel::new(weights, 2, 1);

        assert_eq!(model.num_features(), 2);
        assert_eq!(model.num_groups(), 1);
        assert_eq!(model.weight(0, 0), 0.5);
        assert_eq!(model.weight(1, 0), 0.3);
        assert_eq!(model.bias(0), 0.1);
    }

    #[test]
    fn linear_model_multigroup() {
        // 2 features, 2 groups (binary classification)
        let weights = vec![
            0.1, 0.2, // feature 0: group 0, group 1
            0.3, 0.4, // feature 1
            0.5, 0.6, // bias
        ]
        .into_boxed_slice();
        let model = LinearModel::new(weights, 2, 2);

        assert_eq!(model.weight(0, 0), 0.1);
        assert_eq!(model.weight(0, 1), 0.2);
        assert_eq!(model.weight(1, 0), 0.3);
        assert_eq!(model.weight(1, 1), 0.4);
        assert_eq!(model.bias(0), 0.5);
        assert_eq!(model.bias(1), 0.6);
    }

    #[test]
    fn linear_model_zeros() {
        let model = LinearModel::zeros(3, 2);

        assert_eq!(model.num_features(), 3);
        assert_eq!(model.num_groups(), 2);
        assert_eq!(model.weights().len(), 8); // (3+1) * 2
        assert!(model.weights().iter().all(|&w| w == 0.0));
    }

    #[test]
    fn predict_row_regression() {
        // y = 0.5 * x0 + 0.3 * x1 + 0.1
        let weights = vec![0.5, 0.3, 0.1].into_boxed_slice();
        let model = LinearModel::new(weights, 2, 1);

        let features = vec![2.0, 3.0]; // 0.5*2 + 0.3*3 + 0.1 = 1.0 + 0.9 + 0.1 = 2.0
        let output = model.predict_row(&features, &[0.0]);

        assert_eq!(output.len(), 1);
        assert!((output[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn predict_row_with_base_score() {
        let weights = vec![0.5, 0.3, 0.1].into_boxed_slice();
        let model = LinearModel::new(weights, 2, 1);

        let features = vec![2.0, 3.0];
        let output = model.predict_row(&features, &[0.5]); // base_score = 0.5

        // 0.5 + 0.5*2 + 0.3*3 + 0.1 = 2.5
        assert!((output[0] - 2.5).abs() < 1e-6);
    }

    #[test]
    fn predict_batch() {
        let weights = vec![0.5, 0.3, 0.1].into_boxed_slice();
        let model = LinearModel::new(weights, 2, 1);

        let data = RowMatrix::from_vec(
            vec![
                2.0, 3.0, // row 0: 0.5*2 + 0.3*3 + 0.1 = 2.0
                1.0, 1.0, // row 1: 0.5*1 + 0.3*1 + 0.1 = 0.9
            ],
            2,
            2,
        );

        let output = model.predict(&data, &[0.0]);

        assert_eq!(output.shape(), (2, 1));
        assert!((output.get(0, 0) - 2.0).abs() < 1e-6);
        assert!((output.get(1, 0) - 0.9).abs() < 1e-6);
    }

    #[test]
    fn predict_batch_multigroup() {
        // 2 features, 2 groups
        let weights = vec![
            0.1, 0.2, // feature 0
            0.3, 0.4, // feature 1
            0.0, 0.0, // bias
        ]
        .into_boxed_slice();
        let model = LinearModel::new(weights, 2, 2);

        let data = RowMatrix::from_vec(vec![1.0, 1.0], 1, 2);
        let output = model.predict(&data, &[0.0, 0.0]);

        assert_eq!(output.shape(), (1, 2));
        // group 0: 0.1*1 + 0.3*1 = 0.4
        // group 1: 0.2*1 + 0.4*1 = 0.6
        assert!((output.get(0, 0) - 0.4).abs() < 1e-6);
        assert!((output.get(0, 1) - 0.6).abs() < 1e-6);
    }

    #[test]
    fn par_predict_batch() {
        let weights = vec![0.5, 0.3, 0.1].into_boxed_slice();
        let model = LinearModel::new(weights, 2, 1);

        let data = RowMatrix::from_vec(
            vec![
                2.0, 3.0, // row 0
                1.0, 1.0, // row 1
            ],
            2,
            2,
        );

        let output = model.par_predict(&data, &[0.0]);

        assert_eq!(output.shape(), (2, 1));
        assert!((output.get(0, 0) - 2.0).abs() < 1e-6);
        assert!((output.get(1, 0) - 0.9).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "weights length")]
    fn linear_model_wrong_weights_length() {
        let weights = vec![0.5, 0.3].into_boxed_slice(); // Should be 3 for 2 features + bias
        LinearModel::new(weights, 2, 1);
    }
}

//! Linear model data structure.

use ndarray::{s, Array2, ArrayView1, ArrayView2, ArrayViewMut2};

use crate::data::{FeaturesView, SamplesView};
use crate::utils::Parallelism;

/// Linear booster model (weights + bias).
///
/// Stores a weight matrix for linear prediction using ndarray. The weights
/// are stored as an `Array2<f32>` with shape `[n_features + 1, n_groups]`:
///
/// ```text
/// weights[[feature, group]] → coefficient
/// weights[[n_features, group]] → bias (last row)
/// ```
///
/// This layout enables efficient dot-product prediction:
/// `output = features · weights[:-1, :] + weights[-1, :]`
///
/// # Example
///
/// ```
/// use boosters::repr::gblinear::LinearModel;
/// use ndarray::array;
///
/// // 3 features, 2 output groups (e.g., binary classification)
/// let weights = array![
///     [0.1, 0.2],  // feature 0: group 0, group 1
///     [0.3, 0.4],  // feature 1
///     [0.5, 0.6],  // feature 2
///     [0.0, 0.0],  // bias
/// ];
/// let model = LinearModel::new(weights);
///
/// assert_eq!(model.weight(0, 0), 0.1);
/// assert_eq!(model.weight(0, 1), 0.2);
/// assert_eq!(model.bias(0), 0.0);
/// ```
#[derive(Debug, Clone)]
pub struct LinearModel {
    /// Weight matrix: shape `[n_features + 1, n_groups]`
    /// Last row is the bias term.
    weights: Array2<f32>,
}

impl LinearModel {
    /// Create a linear model from an ndarray.
    ///
    /// # Arguments
    ///
    /// * `weights` - Weight matrix with shape `[n_features + 1, n_groups]`
    ///               where the last row contains biases.
    ///
    /// # Panics
    ///
    /// Panics if the array has fewer than 1 row (need at least bias row).
    pub fn new(weights: Array2<f32>) -> Self {
        assert!(
            weights.nrows() >= 1,
            "weights must have at least 1 row (bias)"
        );
        Self { weights }
    }

    /// Create a zero-initialized linear model.
    pub fn zeros(n_features: usize, n_groups: usize) -> Self {
        Self {
            weights: Array2::zeros((n_features + 1, n_groups)),
        }
    }

    /// Number of input features.
    #[inline]
    pub fn n_features(&self) -> usize {
        // Last row is bias, so n_features = nrows - 1
        self.weights.nrows() - 1
    }

    /// Number of output groups.
    #[inline]
    pub fn n_groups(&self) -> usize {
        self.weights.ncols()
    }

    /// Get weight for a feature and group.
    ///
    /// # Arguments
    ///
    /// * `feature` - Feature index (0..n_features)
    /// * `group` - Output group index (0..n_groups)
    #[inline]
    pub fn weight(&self, feature: usize, group: usize) -> f32 {
        self.weights[[feature, group]]
    }

    /// Get bias for a group.
    ///
    /// For bulk access, use [`biases()`](Self::biases) instead.
    #[inline]
    pub fn bias(&self, group: usize) -> f32 {
        self.weights[[self.n_features(), group]]
    }

    /// Get all biases as a view.
    ///
    /// Returns a view of length `n_groups`.
    #[inline]
    pub fn biases(&self) -> ArrayView1<'_, f32> {
        self.weights.row(self.n_features())
    }

    /// Raw access to weights as a flat slice.
    ///
    /// Layout: `[n_features + 1, n_groups]` in row-major order.
    /// Last row contains biases.
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        self.weights.as_slice().expect("weights should be contiguous")
    }

    /// Mutable access to weights as a flat slice.
    #[inline]
    pub fn as_slice_mut(&mut self) -> &mut [f32] {
        self.weights.as_slice_mut().expect("weights should be contiguous")
    }

    /// Get the weight matrix (excluding bias row) as an ArrayView2.
    ///
    /// Returns a view of shape `[n_features, n_groups]` for use in matrix operations.
    #[inline]
    pub fn weight_view(&self) -> ndarray::ArrayView2<'_, f32> {
        self.weights.slice(s![..self.n_features(), ..])
    }

    /// Set weight for a feature and group.
    #[inline]
    pub fn set_weight(&mut self, feature: usize, group: usize, value: f32) {
        self.weights[[feature, group]] = value;
    }

    /// Set bias for a group.
    #[inline]
    pub fn set_bias(&mut self, group: usize, value: f32) {
        let n_features = self.n_features();
        self.weights[[n_features, group]] = value;
    }

    /// Add to weight for a feature and group.
    #[inline]
    pub fn add_weight(&mut self, feature: usize, group: usize, delta: f32) {
        self.weights[[feature, group]] += delta;
    }

    /// Add to bias for a group.
    #[inline]
    pub fn add_bias(&mut self, group: usize, delta: f32) {
        let n_features = self.n_features();
        self.weights[[n_features, group]] += delta;
    }

    // =========================================================================
    // Prediction
    // =========================================================================

    /// Predict for a batch of rows.
    ///
    /// Allocates and returns predictions as `Array2<f32>` with shape `[n_samples, n_groups]`.
    ///
    /// # Arguments
    ///
    /// * `data` - Feature matrix with shape `[n_samples, n_features]` (sample-major)
    /// * `base_score` - Base score to add per group
    pub fn predict(&self, data: SamplesView<'_>, base_score: &[f32]) -> Array2<f32> {
        let mut output = Array2::zeros((data.n_samples(), self.n_groups()));
        self.predict_into(data, base_score, Parallelism::Sequential, output.view_mut());
        output
    }

    /// Predict into a provided output buffer.
    ///
    /// Writes predictions to `output` with shape `[n_samples, n_groups]`.
    ///
    /// # Arguments
    ///
    /// * `data` - Feature matrix with shape `[n_samples, n_features]` (sample-major)
    /// * `base_score` - Base score to add per group
    /// * `parallelism` - Whether to use parallel execution
    /// * `output` - Mutable view with shape `[n_samples, n_groups]` to write predictions into
    pub fn predict_into(
        &self,
        data: SamplesView<'_>,
        base_score: &[f32],
        parallelism: Parallelism,
        mut output: ArrayViewMut2<'_, f32>,
    ) {
        debug_assert_eq!(output.dim(), (data.n_samples(), self.n_groups()));

        // Compute dot product: data [n_samples, n_features] · weights [n_features, n_groups]
        // ndarray doesn't have in-place gemm, so we compute and assign
        let weights = self.weights.slice(s![..self.n_features(), ..]);
        ndarray::linalg::general_mat_mul(1.0, &data.as_array(), &weights, 0.0, &mut output);

        // Add biases: output[row, group] += bias[group] + base_score[group]
        let biases = self.biases();
        let add_bias = |mut row: ndarray::ArrayViewMut1<f32>| {
            for (group, val) in row.iter_mut().enumerate() {
                let base = base_score.get(group).copied().unwrap_or(0.0);
                *val += biases[group] + base;
            }
        };
        parallelism.maybe_par_bridge_for_each(output.rows_mut().into_iter(), add_bias);
    }

    /// Predict into a column-major buffer.
    ///
    /// Output layout: `output[group * n_rows + row]`
    ///
    /// This is the preferred method for training where predictions are stored
    /// in column-major layout for efficient gradient computation.
    pub fn predict_col_major(&self, data: FeaturesView<'_>, output: &mut [f32]) {
        let n_rows = data.n_samples();
        let n_groups = self.n_groups();
        let n_features = self.n_features();
        debug_assert_eq!(output.len(), n_rows * n_groups);

        // Initialize with bias (column-major: group-first)
        for group in 0..n_groups {
            output[group * n_rows..(group + 1) * n_rows].fill(self.bias(group));
        }

        // Add weighted features - iterate over features (rows in FeaturesView)
        for feat_idx in 0..n_features {
            let feature_values = data.feature(feat_idx);
            for (row_idx, &value) in feature_values.iter().enumerate() {
                for group in 0..n_groups {
                    output[group * n_rows + row_idx] += value * self.weight(feat_idx, group);
                }
            }
        }
    }

    /// Predict from feature-major data, returning group-major output.
    ///
    /// # Arguments
    ///
    /// * `features` - Feature matrix with shape `[n_features, n_samples]` (feature-major)
    ///
    /// # Returns
    ///
    /// Array2 with shape `[n_groups, n_samples]`.
    pub fn predict_feature_major(&self, features: ArrayView2<f32>) -> Array2<f32> {
        let n_features_data = features.nrows();
        let n_samples = features.ncols();
        let n_groups = self.n_groups();
        let n_features = self.n_features();

        debug_assert_eq!(
            n_features_data, n_features,
            "features has {} features but model expects {}",
            n_features_data, n_features
        );

        // Output shape: [n_groups, n_samples]
        let mut output = Array2::<f32>::zeros((n_groups, n_samples));

        // Initialize with bias
        for group in 0..n_groups {
            let bias = self.bias(group);
            output.row_mut(group).fill(bias);
        }

        // Add weighted features - iterate over features (rows in feature-major)
        for feat_idx in 0..n_features {
            let feature_row = features.row(feat_idx);
            for (sample_idx, &value) in feature_row.iter().enumerate() {
                for group in 0..n_groups {
                    output[[group, sample_idx]] += value * self.weight(feat_idx, group);
                }
            }
        }

        output
    }

    /// Predict for a single row into a provided buffer.
    ///
    /// Writes `n_groups` predictions to `output`.
    ///
    /// # Arguments
    ///
    /// * `features` - Feature values for one sample (length >= n_features)
    /// * `base_score` - Base score to add per group
    /// * `output` - Buffer to write predictions into (length >= n_groups)
    pub fn predict_row_into(&self, features: &[f32], base_score: &[f32], output: &mut [f32]) {
        let n_features = self.n_features();
        let n_groups = self.n_groups();

        debug_assert!(features.len() >= n_features);
        debug_assert!(output.len() >= n_groups);

        for group in 0..n_groups {
            let base = base_score.get(group).copied().unwrap_or(0.0);
            let mut sum = base + self.bias(group);
            for (feat_idx, &value) in features.iter().take(n_features).enumerate() {
                sum += value * self.weight(feat_idx, group);
            }
            output[group] = sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::SamplesView;
    use ndarray::array;

    #[test]
    fn linear_model_new() {
        // 2 features, 1 group (regression)
        let weights = array![
            [0.5],
            [0.3],
            [0.1], // bias
        ];
        let model = LinearModel::new(weights);

        assert_eq!(model.n_features(), 2);
        assert_eq!(model.n_groups(), 1);
        assert_eq!(model.weight(0, 0), 0.5);
        assert_eq!(model.weight(1, 0), 0.3);
        assert_eq!(model.bias(0), 0.1);
    }

    #[test]
    fn linear_model_multigroup() {
        // 2 features, 2 groups (binary classification)
        let weights = array![
            [0.1, 0.2], // feature 0
            [0.3, 0.4], // feature 1
            [0.5, 0.6], // bias
        ];
        let model = LinearModel::new(weights);

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

        assert_eq!(model.n_features(), 3);
        assert_eq!(model.n_groups(), 2);
        assert_eq!(model.as_slice().len(), 8); // (3+1) * 2
        assert!(model.as_slice().iter().all(|&w| w == 0.0));
    }

    #[test]
    fn linear_model_mutation() {
        let mut model = LinearModel::zeros(2, 1);

        model.set_weight(0, 0, 0.5);
        model.set_weight(1, 0, 0.3);
        model.set_bias(0, 0.1);

        assert_eq!(model.weight(0, 0), 0.5);
        assert_eq!(model.weight(1, 0), 0.3);
        assert_eq!(model.bias(0), 0.1);

        model.add_weight(0, 0, 0.1);
        model.add_bias(0, 0.2);

        assert!((model.weight(0, 0) - 0.6).abs() < 1e-6);
        assert!((model.bias(0) - 0.3).abs() < 1e-6);
    }

    #[test]
    fn biases_view() {
        let weights = array![
            [0.1, 0.2],
            [0.5, 0.6], // bias
        ];
        let model = LinearModel::new(weights);

        let biases = model.biases();
        assert_eq!(biases.len(), 2);
        assert_eq!(biases[0], 0.5);
        assert_eq!(biases[1], 0.6);
    }

    #[test]
    #[should_panic(expected = "at least 1 row")]
    fn linear_model_wrong_weights_shape() {
        let weights = Array2::<f32>::zeros((0, 1)); // Empty - must have at least 1 row
        LinearModel::new(weights);
    }

    // Prediction tests

    #[test]
    fn predict_row_into_regression() {
        // y = 0.5 * x0 + 0.3 * x1 + 0.1
        let weights = array![[0.5], [0.3], [0.1]];
        let model = LinearModel::new(weights);

        let features = vec![2.0, 3.0]; // 0.5*2 + 0.3*3 + 0.1 = 1.0 + 0.9 + 0.1 = 2.0
        let mut output = [0.0f32; 1];
        model.predict_row_into(&features, &[0.0], &mut output);

        assert!((output[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn predict_row_into_with_base_score() {
        let weights = array![[0.5], [0.3], [0.1]];
        let model = LinearModel::new(weights);

        let features = vec![2.0, 3.0];
        let mut output = [0.0f32; 1];
        model.predict_row_into(&features, &[0.5], &mut output); // base_score = 0.5

        // 0.5 + 0.5*2 + 0.3*3 + 0.1 = 2.5
        assert!((output[0] - 2.5).abs() < 1e-6);
    }

    #[test]
    fn predict_batch() {
        let weights = array![[0.5], [0.3], [0.1]];
        let model = LinearModel::new(weights);

        let data = [
            2.0f32, 3.0, // row 0: 0.5*2 + 0.3*3 + 0.1 = 2.0
            1.0, 1.0,    // row 1: 0.5*1 + 0.3*1 + 0.1 = 0.9
        ];
        let view = SamplesView::from_slice(&data, 2, 2).unwrap();

        let output = model.predict(view, &[0.0]);

        assert_eq!(output.dim(), (2, 1));
        assert!((output[[0, 0]] - 2.0).abs() < 1e-6);
        assert!((output[[1, 0]] - 0.9).abs() < 1e-6);
    }

    #[test]
    fn predict_batch_multigroup() {
        // 2 features, 2 groups
        let weights = array![
            [0.1, 0.2], // feature 0
            [0.3, 0.4], // feature 1
            [0.0, 0.0], // bias
        ];
        let model = LinearModel::new(weights);

        let data = [1.0f32, 1.0];
        let view = SamplesView::from_slice(&data, 1, 2).unwrap();
        let output = model.predict(view, &[0.0, 0.0]);

        assert_eq!(output.dim(), (1, 2));
        // group 0: 0.1*1 + 0.3*1 = 0.4
        // group 1: 0.2*1 + 0.4*1 = 0.6
        assert!((output[[0, 0]] - 0.4).abs() < 1e-6);
        assert!((output[[0, 1]] - 0.6).abs() < 1e-6);
    }

    #[test]
    fn predict_with_parallelism() {
        let weights = array![[0.5], [0.3], [0.1]];
        let model = LinearModel::new(weights);

        let data = [
            2.0f32, 3.0, // row 0
            1.0, 1.0,    // row 1
        ];
        let view = SamplesView::from_slice(&data, 2, 2).unwrap();

        // Test parallel prediction
        let mut output = Array2::<f32>::zeros((2, 1));
        model.predict_into(view, &[0.0], Parallelism::Parallel, output.view_mut());

        assert_eq!(output.dim(), (2, 1));
        assert!((output[[0, 0]] - 2.0).abs() < 1e-6);
        assert!((output[[1, 0]] - 0.9).abs() < 1e-6);

        // Sequential should give same results
        let mut seq_output = Array2::<f32>::zeros((2, 1));
        model.predict_into(view, &[0.0], Parallelism::Sequential, seq_output.view_mut());
        assert_eq!(output, seq_output);
    }

    #[test]
    fn predict_feature_major_matches_sample_major() {
        let weights = array![[0.5], [0.3], [0.1]];
        let model = LinearModel::new(weights);

        // Sample-major: [2 samples, 2 features]
        // sample 0: [2.0, 3.0] -> 0.5*2 + 0.3*3 + 0.1 = 2.0
        // sample 1: [1.0, 1.0] -> 0.5*1 + 0.3*1 + 0.1 = 0.9
        let sample_major = array![
            [2.0, 3.0],
            [1.0, 1.0],
        ];

        // Feature-major: [2 features, 2 samples]
        // feature 0: [2.0, 1.0]
        // feature 1: [3.0, 1.0]
        let feature_major = sample_major.t();

        // Predict using feature-major path
        let output = model.predict_feature_major(feature_major);

        // Output should be [n_groups=1, n_samples=2]
        assert_eq!(output.dim(), (1, 2));
        assert!((output[[0, 0]] - 2.0).abs() < 1e-6);
        assert!((output[[0, 1]] - 0.9).abs() < 1e-6);
    }

    #[test]
    fn predict_feature_major_multigroup() {
        // 2 features, 2 groups
        let weights = array![
            [0.1, 0.2], // feature 0
            [0.3, 0.4], // feature 1
            [0.5, 0.6], // bias
        ];
        let model = LinearModel::new(weights);

        // Sample-major: [2 samples, 2 features]
        let sample_major = array![
            [1.0, 2.0], // sample 0
            [3.0, 4.0], // sample 1
        ];
        let feature_major = sample_major.t();

        let output = model.predict_feature_major(feature_major);

        // Output should be [n_groups=2, n_samples=2]
        assert_eq!(output.dim(), (2, 2));

        // sample 0, group 0: 0.1*1 + 0.3*2 + 0.5 = 0.1 + 0.6 + 0.5 = 1.2
        // sample 0, group 1: 0.2*1 + 0.4*2 + 0.6 = 0.2 + 0.8 + 0.6 = 1.6
        // sample 1, group 0: 0.1*3 + 0.3*4 + 0.5 = 0.3 + 1.2 + 0.5 = 2.0
        // sample 1, group 1: 0.2*3 + 0.4*4 + 0.6 = 0.6 + 1.6 + 0.6 = 2.8
        assert!((output[[0, 0]] - 1.2).abs() < 1e-6);
        assert!((output[[1, 0]] - 1.6).abs() < 1e-6);
        assert!((output[[0, 1]] - 2.0).abs() < 1e-6);
        assert!((output[[1, 1]] - 2.8).abs() < 1e-6);
    }
}

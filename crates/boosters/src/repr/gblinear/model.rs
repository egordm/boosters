//! Linear model data structure.

use ndarray::{Array2, ArrayView1, ArrayViewMut1, ArrayViewMut2, s};

use crate::data::Dataset;

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
    ///   where the last row contains biases.
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

    /// Get all weights **and** the bias for a single output group.
    ///
    /// Returns a strided view of length `n_features + 1`:
    /// - `view[0..n_features]` are feature weights
    /// - `view[n_features]` is the bias
    #[inline]
    pub fn weights_and_bias(&self, group: usize) -> ArrayView1<'_, f32> {
        self.weights.column(group)
    }

    /// Mutable variant of [`weights_and_bias`](Self::weights_and_bias).
    #[inline]
    pub fn weights_and_bias_mut(&mut self, group: usize) -> ArrayViewMut1<'_, f32> {
        self.weights.column_mut(group)
    }

    /// Raw access to weights as a flat slice.
    ///
    /// Layout: `[n_features + 1, n_groups]` in row-major order.
    /// Last row contains biases.
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        self.weights
            .as_slice()
            .expect("weights should be contiguous")
    }

    /// Mutable access to weights as a flat slice.
    #[inline]
    pub fn as_slice_mut(&mut self) -> &mut [f32] {
        self.weights
            .as_slice_mut()
            .expect("weights should be contiguous")
    }

    /// Get the weight matrix (excluding bias row) as an ArrayView2.
    ///
    /// Returns a view of shape `[n_features, n_groups]` for use in matrix operations.
    #[inline]
    pub fn weight_view(&self) -> ndarray::ArrayView2<'_, f32> {
        self.weights.slice(s![..self.n_features(), ..])
    }

    /// Set bias for a group.
    #[inline]
    pub fn set_bias(&mut self, group: usize, value: f32) {
        let n_features = self.n_features();
        self.weights[[n_features, group]] = value;
    }

    // =========================================================================
    // Prediction
    // =========================================================================

    /// Predict for a batch of samples.
    ///
    /// Allocates and returns predictions as `Array2<f32>` with shape `[n_groups, n_samples]`.
    ///
    /// # Arguments
    ///
    /// * `dataset` - Dataset containing features (targets are ignored)
    pub fn predict(&self, dataset: &Dataset) -> Array2<f32> {
        let n_samples = dataset.n_samples();
        let n_groups = self.n_groups();

        let mut output = Array2::zeros((n_groups, n_samples));
        self.predict_into(dataset, output.view_mut());
        output
    }

    /// Predict into a provided output buffer.
    ///
    /// Writes predictions to `output` with shape `[n_groups, n_samples]`.
    ///
    /// # Arguments
    ///
    /// * `dataset` - Dataset containing features (targets are ignored)
    /// * `output` - Mutable view with shape `[n_groups, n_samples]` to write predictions into
    pub fn predict_into(&self, dataset: &Dataset, mut output: ArrayViewMut2<'_, f32>) {
        let n_samples = dataset.n_samples();
        let n_groups = self.n_groups();
        let n_features = self.n_features();

        debug_assert_eq!(output.dim(), (n_groups, n_samples));
        debug_assert_eq!(
            dataset.n_features(),
            n_features,
            "dataset has {} features but model expects {}",
            dataset.n_features(),
            n_features
        );

        // Initialize with bias using row views
        let biases = self.biases();
        for (group, bias) in biases.iter().enumerate() {
            output.row_mut(group).fill(*bias);
        }

        // Add weighted features
        // For each feature, get the weights once (row of weight matrix)
        for feat_idx in 0..n_features {
            let feat_weights = self.weights.row(feat_idx);

            dataset.for_each_feature_value(feat_idx, |sample_idx, value| {
                // Treat NaN as missing: contributes 0 to the dot product.
                if value.is_nan() {
                    return;
                }
                for (group, &weight) in feat_weights.iter().enumerate() {
                    output[[group, sample_idx]] += value * weight;
                }
            });
        }
    }

    /// Predict for a single row into a provided buffer.
    ///
    /// Writes `n_groups` predictions to `output`.
    ///
    /// # Arguments
    ///
    /// * `features` - Feature values for one sample (length >= n_features)
    /// * `output` - Buffer to write predictions into (length >= n_groups)
    pub fn predict_row_into(&self, features: &[f32], output: &mut [f32]) {
        let n_features = self.n_features();
        let n_groups = self.n_groups();

        debug_assert!(features.len() >= n_features);
        debug_assert!(output.len() >= n_groups);

        // Initialize with biases
        let biases = self.biases();
        for (group, out) in output.iter_mut().enumerate().take(n_groups) {
            *out = biases[group];
        }

        // Accumulate weighted features
        for (feat_idx, &value) in features.iter().take(n_features).enumerate() {
            // Treat NaN as missing: contributes 0 to the dot product.
            if value.is_nan() {
                continue;
            }
            let feat_weights = self.weights.row(feat_idx);
            for (group, out) in output.iter_mut().enumerate().take(n_groups) {
                *out += value * feat_weights[group];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, array};

    use crate::data::Dataset;

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
    fn linear_model_mutation_via_slice() {
        let mut model = LinearModel::zeros(2, 1);

        // Use weights_and_bias_mut for bulk modification
        let mut wb = model.weights_and_bias_mut(0);
        wb[0] = 0.5; // feature 0
        wb[1] = 0.3; // feature 1
        wb[2] = 0.1; // bias

        assert_eq!(model.weight(0, 0), 0.5);
        assert_eq!(model.weight(1, 0), 0.3);
        assert_eq!(model.bias(0), 0.1);
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
        model.predict_row_into(&features, &mut output);

        assert!((output[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn predict_row_into_with_bias() {
        // Bias is built into the model, not passed as base_score
        let weights = array![[0.5], [0.3], [0.6]]; // bias = 0.6 (was 0.1 + 0.5 base_score)
        let model = LinearModel::new(weights);

        let features = vec![2.0, 3.0];
        let mut output = [0.0f32; 1];
        model.predict_row_into(&features, &mut output);

        // 0.5*2 + 0.3*3 + 0.6 = 2.5
        assert!((output[0] - 2.5).abs() < 1e-6);
    }

    #[test]
    fn predict_batch() {
        let weights = array![[0.5], [0.3], [0.1]];
        let model = LinearModel::new(weights);

        // Feature-major: [n_features=2, n_samples=2]
        // feature 0: [2.0, 1.0] (samples 0 and 1)
        // feature 1: [3.0, 1.0]
        // sample 0: 0.5*2 + 0.3*3 + 0.1 = 2.0
        // sample 1: 0.5*1 + 0.3*1 + 0.1 = 0.9
        let features = array![
            [2.0, 1.0], // feature 0
            [3.0, 1.0], // feature 1
        ];
        let dataset = Dataset::from_array(features.view(), None, None);

        let output = model.predict(&dataset);

        // Output: [n_groups=1, n_samples=2]
        assert_eq!(output.dim(), (1, 2));
        assert!((output[[0, 0]] - 2.0).abs() < 1e-6);
        assert!((output[[0, 1]] - 0.9).abs() < 1e-6);
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

        // Feature-major: [2 features, 1 sample]
        // feature 0: [1.0]
        // feature 1: [1.0]
        let features = array![
            [1.0], // feature 0
            [1.0], // feature 1
        ];
        let dataset = Dataset::from_array(features.view(), None, None);
        let output = model.predict(&dataset);

        // Output: [n_groups=2, n_samples=1]
        assert_eq!(output.dim(), (2, 1));
        // group 0: 0.1*1 + 0.3*1 = 0.4
        // group 1: 0.2*1 + 0.4*1 = 0.6
        assert!((output[[0, 0]] - 0.4).abs() < 1e-6);
        assert!((output[[1, 0]] - 0.6).abs() < 1e-6);
    }

    #[test]
    fn predict_into_buffer() {
        let weights = array![[0.5], [0.3], [0.1]];
        let model = LinearModel::new(weights);

        // Feature-major: [2 features, 2 samples]
        // feature 0: [2.0, 1.0]
        // feature 1: [3.0, 1.0]
        let features = array![
            [2.0, 1.0], // feature 0
            [3.0, 1.0], // feature 1
        ];
        let dataset = Dataset::from_array(features.view(), None, None);

        let mut output = Array2::<f32>::zeros((1, 2)); // [n_groups=1, n_samples=2]
        model.predict_into(&dataset, output.view_mut());

        assert_eq!(output.dim(), (1, 2));
        assert!((output[[0, 0]] - 2.0).abs() < 1e-6);
        assert!((output[[0, 1]] - 0.9).abs() < 1e-6);
    }

    #[test]
    fn predict_multigroup() {
        // 2 features, 2 groups
        let weights = array![
            [0.1, 0.2], // feature 0
            [0.3, 0.4], // feature 1
            [0.5, 0.6], // bias
        ];
        let model = LinearModel::new(weights);

        // Feature-major: [2 features, 2 samples]
        // feature 0: [1.0, 3.0]
        // feature 1: [2.0, 4.0]
        let feature_major = array![
            [1.0, 3.0], // feature 0
            [2.0, 4.0], // feature 1
        ];
        let dataset = Dataset::from_array(feature_major.view(), None, None);

        let output = model.predict(&dataset);

        // Output: [n_groups=2, n_samples=2]
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

    #[test]
    fn predict_row_into_skips_nan_feature() {
        // y = 0.5 * x0 + 0.3 * x1 + 0.1
        // If x1 is NaN, treat it as missing (0 contribution).
        let weights = array![[0.5], [0.3], [0.1]];
        let model = LinearModel::new(weights);

        let mut output = [0.0f32; 1];
        model.predict_row_into(&[2.0, f32::NAN], &mut output);

        // 0.5*2.0 + 0.1 = 1.1
        assert!((output[0] - 1.1).abs() < 1e-6);
        assert!(output[0].is_finite());
    }

    #[test]
    fn predict_batch_skips_nan_feature() {
        // 2 features, 1 group
        let weights = array![[1.0], [2.0], [0.0]];
        let model = LinearModel::new(weights);

        // Feature-major: [2 features, 2 samples]
        // sample 0: x0=1.0, x1=NaN => y = 1.0*1.0 + 2.0*0.0 = 1.0
        // sample 1: x0=1.0, x1=3.0 => y = 1.0 + 6.0 = 7.0
        let features = array![[1.0, 1.0], [f32::NAN, 3.0]];
        let dataset = Dataset::from_array(features.view(), None, None);

        let output = model.predict(&dataset);

        assert_eq!(output.dim(), (1, 2));
        assert!((output[[0, 0]] - 1.0).abs() < 1e-6);
        assert!((output[[0, 1]] - 7.0).abs() < 1e-6);
        assert!(output[[0, 0]].is_finite());
        assert!(output[[0, 1]].is_finite());
    }
}

//! Linear model data structure.

use ndarray::{s, Array2, ArrayView1, ArrayView2, ArrayViewMut2};

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
/// let model = LinearModel::from_array(weights);
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
    /// Create a new linear model from a flat weight slice.
    ///
    /// This is a compatibility constructor for code that uses the old flat layout.
    /// Prefer [`from_array`](Self::from_array) for new code.
    ///
    /// # Arguments
    ///
    /// * `weights` - Flat weight array of size `(num_features + 1) * num_groups`
    ///               in row-major order (feature-major, group-minor)
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

        // Convert flat array to Array2 with shape [n_features + 1, n_groups]
        let weights_vec = weights.into_vec();
        let weights_arr =
            Array2::from_shape_vec((num_features + 1, num_groups), weights_vec)
                .expect("shape mismatch");

        Self { weights: weights_arr }
    }

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
    pub fn from_array(weights: Array2<f32>) -> Self {
        assert!(
            weights.nrows() >= 1,
            "weights must have at least 1 row (bias)"
        );
        Self { weights }
    }

    /// Create a zero-initialized linear model.
    pub fn zeros(num_features: usize, num_groups: usize) -> Self {
        Self {
            weights: Array2::zeros((num_features + 1, num_groups)),
        }
    }

    /// Number of input features.
    #[inline]
    pub fn n_features(&self) -> usize {
        // Last row is bias, so num_features = nrows - 1
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
    /// * `feature` - Feature index (0..num_features)
    /// * `group` - Output group index (0..num_groups)
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
    /// Returns a view of length `num_groups`.
    #[inline]
    pub fn biases(&self) -> ArrayView1<'_, f32> {
        self.weights.row(self.n_features())
    }

    /// Get the weight matrix (excluding bias row).
    ///
    /// Returns a view with shape `[n_features, n_groups]`.
    #[inline]
    pub fn weight_matrix(&self) -> ArrayView2<'_, f32> {
        self.weights.slice(s![..self.n_features(), ..])
    }

    /// Get mutable access to the weight matrix (excluding bias row).
    ///
    /// Returns a mutable view with shape `[n_features, n_groups]`.
    #[inline]
    pub fn weight_matrix_mut(&mut self) -> ArrayViewMut2<'_, f32> {
        let n_features = self.n_features();
        self.weights.slice_mut(s![..n_features, ..])
    }

    /// Get the underlying array (for serialization, etc.).
    #[inline]
    pub fn as_array(&self) -> ArrayView2<'_, f32> {
        self.weights.view()
    }

    /// Get mutable access to the underlying array.
    #[inline]
    pub fn as_array_mut(&mut self) -> ArrayViewMut2<'_, f32> {
        self.weights.view_mut()
    }

    /// Raw access to weights as a flat slice (for compatibility).
    ///
    /// Returns weights in row-major order (feature-major, group-minor).
    #[inline]
    pub fn weights(&self) -> &[f32] {
        self.weights
            .as_slice()
            .expect("weights should be contiguous")
    }

    /// Mutable access to weights as a flat slice (for training compatibility).
    #[inline]
    pub fn weights_mut(&mut self) -> &mut [f32] {
        self.weights
            .as_slice_mut()
            .expect("weights should be contiguous")
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn linear_model_new() {
        // 2 features, 1 group (regression)
        let weights = vec![0.5, 0.3, 0.1].into_boxed_slice(); // w0, w1, bias
        let model = LinearModel::new(weights, 2, 1);

        assert_eq!(model.n_features(), 2);
        assert_eq!(model.n_groups(), 1);
        assert_eq!(model.weight(0, 0), 0.5);
        assert_eq!(model.weight(1, 0), 0.3);
        assert_eq!(model.bias(0), 0.1);
    }

    #[test]
    fn linear_model_from_array() {
        let weights = array![
            [0.5],
            [0.3],
            [0.1], // bias
        ];
        let model = LinearModel::from_array(weights);

        assert_eq!(model.n_features(), 2);
        assert_eq!(model.n_groups(), 1);
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
    fn linear_model_multigroup_from_array() {
        let weights = array![
            [0.1, 0.2], // feature 0
            [0.3, 0.4], // feature 1
            [0.5, 0.6], // bias
        ];
        let model = LinearModel::from_array(weights);

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
        assert_eq!(model.weights().len(), 8); // (3+1) * 2
        assert!(model.weights().iter().all(|&w| w == 0.0));
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
    fn weight_matrix_view() {
        let weights = array![
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6], // bias
        ];
        let model = LinearModel::from_array(weights);

        let weight_mat = model.weight_matrix();
        assert_eq!(weight_mat.dim(), (2, 2));
        assert_eq!(weight_mat[[0, 0]], 0.1);
        assert_eq!(weight_mat[[1, 1]], 0.4);
    }

    #[test]
    fn biases_view() {
        let weights = array![
            [0.1, 0.2],
            [0.5, 0.6], // bias
        ];
        let model = LinearModel::from_array(weights);

        let biases = model.biases();
        assert_eq!(biases.len(), 2);
        assert_eq!(biases[0], 0.5);
        assert_eq!(biases[1], 0.6);
    }

    #[test]
    #[should_panic(expected = "weights length")]
    fn linear_model_wrong_weights_length() {
        let weights = vec![0.5, 0.3].into_boxed_slice(); // Should be 3 for 2 features + bias
        LinearModel::new(weights, 2, 1);
    }
}

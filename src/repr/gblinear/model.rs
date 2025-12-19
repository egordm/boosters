//! Linear model data structure.

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
/// use boosters::repr::gblinear::LinearModel;
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
    #[should_panic(expected = "weights length")]
    fn linear_model_wrong_weights_length() {
        let weights = vec![0.5, 0.3].into_boxed_slice(); // Should be 3 for 2 features + bias
        LinearModel::new(weights, 2, 1);
    }
}

//! SHAP values container.
//!
//! Stores SHAP values for a batch of samples with proper indexing
//! and verification utilities.

use ndarray::{Array3, ArrayView3, ArrayViewMut3, s};

/// Container for SHAP values.
///
/// Wraps a 3D array with shape `[n_samples, n_features + 1, n_outputs]`.
/// The last feature slot (index `n_features`) stores the base value.
///
/// # Data Layout
///
/// - Axis 0: Samples (batch dimension)
/// - Axis 1: Features + base value (n_features slots, then 1 base value slot)
/// - Axis 2: Outputs (1 for regression, n_classes for multiclass)
///
/// # Precision
///
/// SHAP values are stored as f32 for consistency with predictions and memory
/// efficiency. The TreeSHAP algorithm internally uses f64 for numerical stability
/// during computation, but final values are cast to f32 for storage.
#[derive(Clone, Debug)]
pub struct ShapValues(Array3<f32>);

impl ShapValues {
    /// Create ShapValues from an existing array.
    ///
    /// Shape must be `[n_samples, n_features + 1, n_outputs]`.
    pub fn new(arr: Array3<f32>) -> Self {
        Self(arr)
    }

    /// Create a zero-initialized ShapValues container.
    ///
    /// # Arguments
    /// * `n_samples` - Number of samples
    /// * `n_features` - Number of features (not including base value)
    /// * `n_outputs` - Number of outputs (1 for regression, n_classes for multiclass)
    pub fn zeros(n_samples: usize, n_features: usize, n_outputs: usize) -> Self {
        Self(Array3::zeros((n_samples, n_features + 1, n_outputs)))
    }

    /// Number of samples.
    #[inline]
    pub fn n_samples(&self) -> usize {
        self.0.dim().0
    }

    /// Number of features (not including base value).
    #[inline]
    pub fn n_features(&self) -> usize {
        self.0.dim().1 - 1
    }

    /// Number of outputs.
    #[inline]
    pub fn n_outputs(&self) -> usize {
        self.0.dim().2
    }

    /// Get SHAP value for a specific sample, feature, and output.
    #[inline]
    pub fn get(&self, sample: usize, feature: usize, output: usize) -> f32 {
        self.0[[sample, feature, output]]
    }

    /// Set SHAP value for a specific sample, feature, and output.
    ///
    /// Accepts f64 for compatibility with TreeSHAP's internal f64 computations.
    #[inline]
    pub fn set(&mut self, sample: usize, feature: usize, output: usize, value: f64) {
        self.0[[sample, feature, output]] = value as f32;
    }

    /// Add to SHAP value for a specific sample, feature, and output.
    ///
    /// Accepts f64 for compatibility with TreeSHAP's internal f64 computations.
    #[inline]
    pub fn add(&mut self, sample: usize, feature: usize, output: usize, delta: f64) {
        self.0[[sample, feature, output]] += delta as f32;
    }

    /// Get the base value (expected value) for a sample and output.
    ///
    /// Base value is stored at feature index = n_features.
    #[inline]
    pub fn base_value(&self, sample: usize, output: usize) -> f32 {
        self.get(sample, self.n_features(), output)
    }

    /// Set the base value for a sample and output.
    ///
    /// Accepts f64 for compatibility with TreeSHAP's internal f64 computations.
    #[inline]
    pub fn set_base_value(&mut self, sample: usize, output: usize, value: f64) {
        let n_features = self.n_features();
        self.set(sample, n_features, output, value);
    }

    /// Get all SHAP values for a single sample (including base).
    ///
    /// Returns a view of shape `[n_features + 1, n_outputs]`.
    pub fn sample(&self, sample_idx: usize) -> ndarray::ArrayView2<'_, f32> {
        self.0.slice(s![sample_idx, .., ..])
    }

    /// Get feature SHAP values only (excluding base) for a sample and output.
    ///
    /// Returns a Vec since values are not contiguous.
    pub fn feature_shap(&self, sample_idx: usize, output: usize) -> Vec<f32> {
        let n_features = self.n_features();
        (0..n_features)
            .map(|f| self.get(sample_idx, f, output))
            .collect()
    }

    /// Verify that SHAP values satisfy the sum property.
    ///
    /// For each sample: sum(shap_values) + base_value ≈ prediction
    ///
    /// Returns `true` if all samples are within tolerance.
    pub fn verify(&self, predictions: &[f32], tolerance: f32) -> bool {
        let n_samples = self.n_samples();
        let n_features = self.n_features();
        let n_outputs = self.n_outputs();

        if predictions.len() != n_samples * n_outputs {
            return false;
        }

        for sample in 0..n_samples {
            for output in 0..n_outputs {
                let mut sum = self.base_value(sample, output);
                for feature in 0..n_features {
                    sum += self.get(sample, feature, output);
                }

                let pred = predictions[sample * n_outputs + output];
                if (sum - pred).abs() > tolerance {
                    return false;
                }
            }
        }
        true
    }

    /// Get the underlying array view.
    pub fn as_array(&self) -> ArrayView3<'_, f32> {
        self.0.view()
    }

    /// Get mutable access to the underlying array.
    pub fn as_array_mut(&mut self) -> ArrayViewMut3<'_, f32> {
        self.0.view_mut()
    }

    /// Get the raw values slice (if contiguous).
    pub fn as_slice(&self) -> Option<&[f32]> {
        self.0.as_slice()
    }

    /// Get shape as (n_samples, n_features + 1, n_outputs).
    pub fn shape(&self) -> (usize, usize, usize) {
        self.0.dim()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_zeros() {
        let shap = ShapValues::zeros(10, 5, 1);
        assert_eq!(shap.n_samples(), 10);
        assert_eq!(shap.n_features(), 5);
        assert_eq!(shap.n_outputs(), 1);
        assert_eq!(shap.shape(), (10, 6, 1)); // 5 features + 1 base
    }

    #[test]
    fn test_get_set() {
        let mut shap = ShapValues::zeros(2, 3, 1);

        shap.set(0, 0, 0, 1.0);
        shap.set(0, 1, 0, 2.0);
        shap.set(1, 2, 0, 3.0);

        assert_eq!(shap.get(0, 0, 0), 1.0f32);
        assert_eq!(shap.get(0, 1, 0), 2.0f32);
        assert_eq!(shap.get(1, 2, 0), 3.0f32);
        assert_eq!(shap.get(0, 2, 0), 0.0f32); // Default is 0
    }

    #[test]
    fn test_add() {
        let mut shap = ShapValues::zeros(1, 2, 1);

        shap.add(0, 0, 0, 1.5);
        shap.add(0, 0, 0, 2.5);

        assert_eq!(shap.get(0, 0, 0), 4.0f32);
    }

    #[test]
    fn test_base_value() {
        let mut shap = ShapValues::zeros(2, 3, 1);

        shap.set_base_value(0, 0, 0.5);
        shap.set_base_value(1, 0, 0.3);

        assert_eq!(shap.base_value(0, 0), 0.5f32);
        assert_eq!(shap.base_value(1, 0), 0.3f32);
    }

    #[test]
    fn test_sample_view() {
        let mut shap = ShapValues::zeros(2, 2, 1);

        // Sample 0: features [1.0, 2.0], base 3.0
        shap.set(0, 0, 0, 1.0);
        shap.set(0, 1, 0, 2.0);
        shap.set_base_value(0, 0, 3.0);

        let sample = shap.sample(0);
        let expected = array![[1.0f32], [2.0], [3.0]];
        assert_eq!(sample, expected);
    }

    #[test]
    fn test_feature_shap() {
        let mut shap = ShapValues::zeros(1, 3, 1);

        shap.set(0, 0, 0, 1.0);
        shap.set(0, 1, 0, 2.0);
        shap.set(0, 2, 0, 3.0);
        shap.set_base_value(0, 0, 0.5);

        let features = shap.feature_shap(0, 0);
        assert_eq!(features, vec![1.0f32, 2.0, 3.0]);
    }

    #[test]
    fn test_verify_correct() {
        let mut shap = ShapValues::zeros(2, 2, 1);

        // Sample 0: shap = [1.0, 2.0], base = 0.5 → sum = 3.5
        shap.set(0, 0, 0, 1.0);
        shap.set(0, 1, 0, 2.0);
        shap.set_base_value(0, 0, 0.5);

        // Sample 1: shap = [0.5, 0.5], base = 1.0 → sum = 2.0
        shap.set(1, 0, 0, 0.5);
        shap.set(1, 1, 0, 0.5);
        shap.set_base_value(1, 0, 1.0);

        let predictions = vec![3.5f32, 2.0];
        assert!(shap.verify(&predictions, 1e-5));
    }

    #[test]
    fn test_verify_incorrect() {
        let mut shap = ShapValues::zeros(1, 2, 1);

        shap.set(0, 0, 0, 1.0);
        shap.set(0, 1, 0, 2.0);
        shap.set_base_value(0, 0, 0.5);

        // Prediction doesn't match sum (3.5)
        let predictions = vec![5.0f32];
        assert!(!shap.verify(&predictions, 1e-5));
    }

    #[test]
    fn test_multi_output() {
        let mut shap = ShapValues::zeros(1, 2, 2);

        // Output 0
        shap.set(0, 0, 0, 1.0);
        shap.set(0, 1, 0, 2.0);
        shap.set_base_value(0, 0, 0.5);

        // Output 1
        shap.set(0, 0, 1, 0.5);
        shap.set(0, 1, 1, 1.5);
        shap.set_base_value(0, 1, 0.0);

        let predictions = vec![3.5f32, 2.0];
        assert!(shap.verify(&predictions, 1e-5));
    }

    #[test]
    fn test_shape() {
        let shap = ShapValues::zeros(2, 3, 1);
        assert_eq!(shap.shape(), (2, 4, 1)); // 3 features + 1 base
    }
}

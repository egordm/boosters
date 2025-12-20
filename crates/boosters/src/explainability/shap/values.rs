//! SHAP values container.
//!
//! Stores SHAP values for a batch of samples with proper indexing
//! and verification utilities.

/// Container for SHAP values.
///
/// Stores per-sample, per-feature, per-output SHAP contributions.
/// Layout is [samples × (features + 1) × outputs] where +1 is for base value.
#[derive(Clone, Debug)]
pub struct ShapValues {
    /// Flat storage: [sample][feature + base][output]
    values: Vec<f64>,
    /// Number of samples
    n_samples: usize,
    /// Number of features (not including base value)
    n_features: usize,
    /// Number of outputs (1 for regression, n_classes for multiclass)
    n_outputs: usize,
}

impl ShapValues {
    /// Create a new ShapValues container initialized to zeros.
    pub fn new(n_samples: usize, n_features: usize, n_outputs: usize) -> Self {
        // Layout: features first, then base value at the end
        let values = vec![0.0; n_samples * (n_features + 1) * n_outputs];
        Self { values, n_samples, n_features, n_outputs }
    }

    /// Number of samples.
    #[inline]
    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    /// Number of features (not including base value).
    #[inline]
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Number of outputs.
    #[inline]
    pub fn n_outputs(&self) -> usize {
        self.n_outputs
    }

    /// Get the index into the flat array.
    #[inline]
    fn index(&self, sample: usize, feature: usize, output: usize) -> usize {
        sample * (self.n_features + 1) * self.n_outputs
            + feature * self.n_outputs
            + output
    }

    /// Get SHAP value for a specific sample, feature, and output.
    #[inline]
    pub fn get(&self, sample: usize, feature: usize, output: usize) -> f64 {
        self.values[self.index(sample, feature, output)]
    }

    /// Set SHAP value for a specific sample, feature, and output.
    #[inline]
    pub fn set(&mut self, sample: usize, feature: usize, output: usize, value: f64) {
        let idx = self.index(sample, feature, output);
        self.values[idx] = value;
    }

    /// Add to SHAP value for a specific sample, feature, and output.
    #[inline]
    pub fn add(&mut self, sample: usize, feature: usize, output: usize, delta: f64) {
        let idx = self.index(sample, feature, output);
        self.values[idx] += delta;
    }

    /// Get the base value (expected value) for a sample and output.
    ///
    /// Base value is stored at feature index = n_features.
    #[inline]
    pub fn base_value(&self, sample: usize, output: usize) -> f64 {
        self.get(sample, self.n_features, output)
    }

    /// Set the base value for a sample and output.
    #[inline]
    pub fn set_base_value(&mut self, sample: usize, output: usize, value: f64) {
        self.set(sample, self.n_features, output, value);
    }

    /// Get all SHAP values for a single sample (including base).
    ///
    /// Returns a slice of length (n_features + 1) * n_outputs.
    pub fn sample(&self, sample_idx: usize) -> &[f64] {
        let start = sample_idx * (self.n_features + 1) * self.n_outputs;
        let end = start + (self.n_features + 1) * self.n_outputs;
        &self.values[start..end]
    }

    /// Get feature SHAP values only (excluding base) for a sample and output.
    ///
    /// Returns a Vec since values are not contiguous.
    pub fn feature_shap(&self, sample_idx: usize, output: usize) -> Vec<f64> {
        (0..self.n_features)
            .map(|f| self.get(sample_idx, f, output))
            .collect()
    }

    /// Verify that SHAP values satisfy the sum property.
    ///
    /// For each sample: sum(shap_values) + base_value ≈ prediction
    ///
    /// Returns `true` if all samples are within tolerance.
    pub fn verify(&self, predictions: &[f64], tolerance: f64) -> bool {
        if predictions.len() != self.n_samples * self.n_outputs {
            return false;
        }

        for sample in 0..self.n_samples {
            for output in 0..self.n_outputs {
                let mut sum = self.base_value(sample, output);
                for feature in 0..self.n_features {
                    sum += self.get(sample, feature, output);
                }

                let pred = predictions[sample * self.n_outputs + output];
                if (sum - pred).abs() > tolerance {
                    return false;
                }
            }
        }
        true
    }

    /// Get the raw values slice.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Convert to a 3D representation for Python/NumPy.
    ///
    /// Returns (data, shape) where shape is (n_samples, n_features + 1, n_outputs).
    pub fn to_3d(&self) -> (&[f64], (usize, usize, usize)) {
        (&self.values, (self.n_samples, self.n_features + 1, self.n_outputs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let shap = ShapValues::new(10, 5, 1);
        assert_eq!(shap.n_samples(), 10);
        assert_eq!(shap.n_features(), 5);
        assert_eq!(shap.n_outputs(), 1);
        assert_eq!(shap.values().len(), 10 * 6 * 1);
    }

    #[test]
    fn test_get_set() {
        let mut shap = ShapValues::new(2, 3, 1);
        
        shap.set(0, 0, 0, 1.0);
        shap.set(0, 1, 0, 2.0);
        shap.set(1, 2, 0, 3.0);
        
        assert_eq!(shap.get(0, 0, 0), 1.0);
        assert_eq!(shap.get(0, 1, 0), 2.0);
        assert_eq!(shap.get(1, 2, 0), 3.0);
        assert_eq!(shap.get(0, 2, 0), 0.0);  // Default is 0
    }

    #[test]
    fn test_add() {
        let mut shap = ShapValues::new(1, 2, 1);
        
        shap.add(0, 0, 0, 1.5);
        shap.add(0, 0, 0, 2.5);
        
        assert_eq!(shap.get(0, 0, 0), 4.0);
    }

    #[test]
    fn test_base_value() {
        let mut shap = ShapValues::new(2, 3, 1);
        
        shap.set_base_value(0, 0, 0.5);
        shap.set_base_value(1, 0, 0.3);
        
        assert_eq!(shap.base_value(0, 0), 0.5);
        assert_eq!(shap.base_value(1, 0), 0.3);
    }

    #[test]
    fn test_sample_slice() {
        let mut shap = ShapValues::new(2, 2, 1);
        
        // Sample 0: features [1.0, 2.0], base 3.0
        shap.set(0, 0, 0, 1.0);
        shap.set(0, 1, 0, 2.0);
        shap.set_base_value(0, 0, 3.0);
        
        let sample = shap.sample(0);
        assert_eq!(sample, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_feature_shap() {
        let mut shap = ShapValues::new(1, 3, 1);
        
        shap.set(0, 0, 0, 1.0);
        shap.set(0, 1, 0, 2.0);
        shap.set(0, 2, 0, 3.0);
        shap.set_base_value(0, 0, 0.5);
        
        let features = shap.feature_shap(0, 0);
        assert_eq!(features, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_verify_correct() {
        let mut shap = ShapValues::new(2, 2, 1);
        
        // Sample 0: shap = [1.0, 2.0], base = 0.5 → sum = 3.5
        shap.set(0, 0, 0, 1.0);
        shap.set(0, 1, 0, 2.0);
        shap.set_base_value(0, 0, 0.5);
        
        // Sample 1: shap = [0.5, 0.5], base = 1.0 → sum = 2.0
        shap.set(1, 0, 0, 0.5);
        shap.set(1, 1, 0, 0.5);
        shap.set_base_value(1, 0, 1.0);
        
        let predictions = vec![3.5, 2.0];
        assert!(shap.verify(&predictions, 1e-10));
    }

    #[test]
    fn test_verify_incorrect() {
        let mut shap = ShapValues::new(1, 2, 1);
        
        shap.set(0, 0, 0, 1.0);
        shap.set(0, 1, 0, 2.0);
        shap.set_base_value(0, 0, 0.5);
        
        // Prediction doesn't match sum (3.5)
        let predictions = vec![5.0];
        assert!(!shap.verify(&predictions, 1e-10));
    }

    #[test]
    fn test_multi_output() {
        let mut shap = ShapValues::new(1, 2, 2);
        
        // Output 0
        shap.set(0, 0, 0, 1.0);
        shap.set(0, 1, 0, 2.0);
        shap.set_base_value(0, 0, 0.5);
        
        // Output 1
        shap.set(0, 0, 1, 0.5);
        shap.set(0, 1, 1, 1.5);
        shap.set_base_value(0, 1, 0.0);
        
        let predictions = vec![3.5, 2.0];
        assert!(shap.verify(&predictions, 1e-10));
    }

    #[test]
    fn test_to_3d() {
        let shap = ShapValues::new(2, 3, 1);
        let (data, shape) = shap.to_3d();
        
        assert_eq!(shape, (2, 4, 1));  // 3 features + 1 base
        assert_eq!(data.len(), 2 * 4 * 1);
    }
}

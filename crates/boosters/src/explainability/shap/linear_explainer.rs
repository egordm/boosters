//! Linear SHAP explainer for linear models.
//!
//! SHAP values for linear models have a closed-form solution:
//! shap[i] = weight[i] * (x[i] - mean[i])

use crate::data::Dataset;
use crate::explainability::ExplainError;
use crate::explainability::shap::ShapValues;
use crate::repr::gblinear::LinearModel;

/// Linear SHAP explainer for linear models.
///
/// Computes exact SHAP values in closed form.
pub struct LinearExplainer<'a> {
    /// Reference to the linear model
    model: &'a LinearModel,
    /// Feature means for background
    feature_means: Vec<f64>,
}

impl<'a> LinearExplainer<'a> {
    /// Create a new LinearExplainer with provided feature means.
    ///
    /// # Arguments
    /// * `model` - Reference to the linear model
    /// * `feature_means` - Mean value for each feature (background distribution)
    ///
    /// # Errors
    /// Returns error if feature_means length doesn't match model features.
    pub fn new(model: &'a LinearModel, feature_means: Vec<f64>) -> Result<Self, ExplainError> {
        if feature_means.len() != model.n_features() {
            return Err(ExplainError::MissingNodeStats(
                "feature_means length must match number of features",
            ));
        }
        Ok(Self {
            model,
            feature_means,
        })
    }

    /// Create a LinearExplainer using zeros as feature means.
    ///
    /// This is useful when the data is already centered.
    pub fn with_zero_means(model: &'a LinearModel) -> Self {
        let feature_means = vec![0.0; model.n_features()];
        Self {
            model,
            feature_means,
        }
    }

    /// Get the expected value (base value).
    ///
    /// For linear models: `E[f(x)] = sum(w[i] * mean[i]) + bias`
    pub fn base_value(&self, output: usize) -> f64 {
        let weights = self.model.as_slice();
        let n_features = self.model.n_features();
        let n_groups = self.model.n_groups();

        let mut base = self.model.bias(output) as f64;

        for feature in 0..n_features {
            let weight_idx = feature * n_groups + output;
            base += weights[weight_idx] as f64 * self.feature_means[feature];
        }

        base
    }

    /// Compute SHAP values for a batch of samples.
    ///
    /// # Arguments
    /// * `dataset` - Dataset containing features (targets are ignored)
    ///
    /// # Returns
    /// ShapValues container with shape `[n_samples, n_features + 1, n_outputs]`.
    pub fn shap_values(&self, dataset: &Dataset) -> ShapValues {
        let n_samples = dataset.n_samples();
        let n_features = self.model.n_features();
        let n_outputs = self.model.n_groups();
        let mut shap = ShapValues::zeros(n_samples, n_features, n_outputs);

        let weights = self.model.as_slice();

        // Iterate feature-major: for each feature, compute SHAP for all samples
        // This is cache-friendly as feature values are contiguous
        for feature in 0..n_features {
            let mean = self.feature_means[feature];

            for output in 0..n_outputs {
                let weight_idx = feature * n_outputs + output;
                let weight = weights[weight_idx] as f64;

                dataset.for_each_feature_value(feature, |sample_idx, value| {
                    let x = value as f64;
                    let contribution = weight * (x - mean);
                    shap.set(sample_idx, feature, output, contribution);
                });
            }
        }

        // Set base values for all samples
        for sample_idx in 0..n_samples {
            for output in 0..n_outputs {
                shap.set_base_value(sample_idx, output, self.base_value(output));
            }
        }

        shap
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    fn make_simple_model() -> LinearModel {
        // 2 features, 1 output: y = 2*x0 + 3*x1 + 0.5 (bias)
        LinearModel::new(array![[2.0f32], [3.0], [0.5]])
    }

    #[test]
    fn test_explainer_creation() {
        let model = make_simple_model();
        let means = vec![1.0, 2.0];
        let explainer = LinearExplainer::new(&model, means);
        assert!(explainer.is_ok());
    }

    #[test]
    fn test_wrong_means_length() {
        let model = make_simple_model();
        let means = vec![1.0]; // Wrong length
        let explainer = LinearExplainer::new(&model, means);
        assert!(explainer.is_err());
    }

    #[test]
    fn test_zero_means() {
        let model = make_simple_model();
        let explainer = LinearExplainer::with_zero_means(&model);
        assert_eq!(explainer.feature_means, vec![0.0, 0.0]);
    }

    #[test]
    fn test_base_value() {
        let model = make_simple_model();
        let means = vec![1.0, 2.0];
        let explainer = LinearExplainer::new(&model, means).unwrap();

        // base = 2*1.0 + 3*2.0 + 0.5 = 2 + 6 + 0.5 = 8.5
        let base = explainer.base_value(0);
        assert!((base - 8.5).abs() < 1e-10);
    }

    #[test]
    fn test_shap_values() {
        let model = make_simple_model();
        let means = vec![1.0, 2.0];
        let explainer = LinearExplainer::new(&model, means).unwrap();

        // Sample: x = [3.0, 4.0] - feature-major layout [n_features=2, n_samples=1]
        let data = array![[3.0f32], [4.0f32]];
        let dataset = Dataset::from_array(data.view(), None, None);
        let shap = explainer.shap_values(&dataset);

        // shap[0] = 2 * (3 - 1) = 4
        // shap[1] = 3 * (4 - 2) = 6
        assert!((shap.get(0, 0, 0) - 4.0f32).abs() < 1e-5);
        assert!((shap.get(0, 1, 0) - 6.0f32).abs() < 1e-5);
    }

    #[test]
    fn test_shap_sums_to_prediction() {
        let model = make_simple_model();
        let means = vec![1.0, 2.0];
        let explainer = LinearExplainer::new(&model, means).unwrap();

        // Sample: x = [3.0, 4.0]
        // Prediction: 2*3 + 3*4 + 0.5 = 18.5
        let data = array![[3.0f32], [4.0f32]];
        let dataset = Dataset::from_array(data.view(), None, None);
        let shap = explainer.shap_values(&dataset);

        let sum: f32 = (0..2).map(|f| shap.get(0, f, 0)).sum();
        let base = shap.base_value(0, 0);
        let from_shap = sum + base;

        // Verify: shap[0] + shap[1] + base = 4 + 6 + 8.5 = 18.5
        let prediction = 18.5f32;
        assert!(
            (from_shap - prediction).abs() < 1e-5,
            "SHAP sum {} + base {} = {} should equal prediction {}",
            sum,
            base,
            from_shap,
            prediction
        );
    }

    #[test]
    fn test_verify_property() {
        let model = make_simple_model();
        let means = vec![1.0, 2.0];
        let explainer = LinearExplainer::new(&model, means).unwrap();

        let data_row = [3.0f32, 4.0f32];
        let data = array![[3.0f32], [4.0f32]];
        let dataset = Dataset::from_array(data.view(), None, None);
        let shap = explainer.shap_values(&dataset);

        // The prediction from model (bias is baked in)
        let mut output = [0.0f32; 1];
        model.predict_row_into(&data_row, &mut output);
        let prediction = output[0];

        // Verify the sum property
        let predictions = vec![prediction];
        assert!(shap.verify(&predictions, 1e-5));
    }
}

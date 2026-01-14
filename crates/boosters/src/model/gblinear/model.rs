//! GBLinear model implementation.
//!
//! High-level wrapper around [`LinearModel`] with training and prediction.
//! Access components via [`linear()`](GBLinearModel::linear), [`meta()`](GBLinearModel::meta),
//! and [`output_transform()`](GBLinearModel::output_transform).

use crate::Parallelism;
use crate::data::Dataset;
use crate::explainability::{ExplainError, LinearExplainer, ShapValues};
use crate::model::OutputTransform;
use crate::model::meta::ModelMeta;
use crate::repr::gblinear::LinearModel;
use crate::training::gblinear::{GBLinearTrainError, GBLinearTrainer};

use ndarray::Array2;

use super::GBLinearConfig;

/// High-level GBLinear model with training, prediction, and explainability.
///
/// Access components via [`linear()`](Self::linear), [`meta()`](Self::meta),
/// and [`output_transform()`](Self::output_transform).
#[derive(Clone)]
pub struct GBLinearModel {
    /// The underlying linear model.
    model: LinearModel,
    /// Model metadata.
    meta: ModelMeta,
    /// Output transform for prediction (derived from objective).
    ///
    /// This is persisted with the model and used for inference-time
    /// transformation (e.g., sigmoid for binary classification).
    output_transform: OutputTransform,
}

impl GBLinearModel {
    /// Train a new GBLinear model.
    ///
    /// # Arguments
    ///
    /// * `dataset` - Training dataset (features, targets, optional weights)
    /// * `val_set` - Optional validation dataset for early stopping and monitoring
    /// * `config` - Training configuration
    /// * `n_threads` - Thread count: 0 = auto, 1 = sequential, >1 = exact count
    pub fn train(
        dataset: &Dataset,
        val_set: Option<&Dataset>,
        config: GBLinearConfig,
        n_threads: usize,
    ) -> Result<Self, GBLinearTrainError> {
        crate::run_with_threads(n_threads, |parallelism| {
            Self::train_inner(dataset, val_set, config, parallelism)
        })
    }

    /// Internal training implementation (no thread pool management).
    ///
    /// This method assumes the caller has already set up any necessary thread pool.
    /// Use `train()` for the public API that handles threading automatically.
    fn train_inner(
        dataset: &Dataset,
        val_set: Option<&Dataset>,
        config: GBLinearConfig,
        _parallelism: Parallelism, // Reserved for future use
    ) -> Result<Self, GBLinearTrainError> {
        let n_features = dataset.n_features();

        // GBLinear uses Dataset directly - no binning needed!
        // Extract targets and weights from Dataset
        let targets = dataset
            .targets()
            .expect("dataset must have targets for training");
        let weights = dataset.weights();

        // Convert config to trainer params, then move out the objective/metric.
        let params = config.to_trainer_params();
        let GBLinearConfig {
            objective, metric, ..
        } = config;

        let n_outputs = objective.n_outputs();

        let output_transform = objective.output_transform();

        // Create trainer with objective and metric from config
        let trainer = GBLinearTrainer::new(objective, metric, params);
        let linear_model = trainer.train(dataset, targets, weights, val_set)?;

        // Extract feature names from dataset schema if available
        let feature_names = dataset.schema().feature_names();

        let meta = ModelMeta {
            n_features,
            n_groups: n_outputs,
            feature_names,
            ..Default::default()
        };

        Ok(Self::from_parts(linear_model, meta, output_transform))
    }

    /// Create a model from all its parts.
    ///
    /// Used when loading from persistence, or after training.
    pub fn from_parts(
        model: LinearModel,
        meta: ModelMeta,
        output_transform: OutputTransform,
    ) -> Self {
        Self {
            model,
            meta,
            output_transform,
        }
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    /// Get reference to the underlying linear model.
    pub fn linear(&self) -> &LinearModel {
        &self.model
    }

    /// Get reference to model metadata.
    pub fn meta(&self) -> &ModelMeta {
        &self.meta
    }

    /// Get the inference-time output transform.
    pub fn output_transform(&self) -> OutputTransform {
        self.output_transform
    }

    // =========================================================================
    // Prediction
    // =========================================================================

    /// Predict from feature-major data, returning transformed predictions.
    ///
    /// This is the **preferred** prediction method. It accepts features in
    /// feature-major layout `[n_features, n_samples]` and uses efficient
    /// per-feature iteration.
    ///
    /// Returns probabilities for classification (sigmoid/softmax) or raw values
    /// for regression.
    ///
    /// # Arguments
    ///
    /// * `dataset` - Dataset containing features (targets are ignored)
    ///
    /// # Returns
    ///
    /// Array2 with shape `[n_groups, n_samples]`. Access group predictions via `.row(group_idx)`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let preds = model.predict(dataset);
    /// ```
    pub fn predict(&self, dataset: &Dataset) -> Array2<f32> {
        // Use efficient feature-major prediction
        let mut output = self.predict_raw(dataset);

        // Apply output transform (sigmoid, softmax, or identity)
        self.output_transform.transform_inplace(output.view_mut());

        output
    }

    /// Predict from feature-major data, returning raw margin scores (no transform).
    ///
    /// # Arguments
    ///
    /// * `dataset` - Dataset containing features (targets are ignored)
    ///
    /// # Returns
    ///
    /// Array2 with shape `[n_groups, n_samples]`.
    pub fn predict_raw(&self, dataset: &Dataset) -> Array2<f32> {
        let n_samples = dataset.n_samples();
        let n_groups = self.meta.n_groups;

        if n_samples == 0 {
            return Array2::zeros((n_groups, 0));
        }

        self.model.predict(dataset)
    }

    // =========================================================================
    // Explainability
    // =========================================================================

    /// Compute SHAP values for a batch of samples.
    ///
    /// Linear SHAP: `shap[i] = w[i] * (x[i] - mean[i])`.
    /// Pass `None` for `feature_means` to assume centered data (zero means).
    ///
    /// # Arguments
    /// * `data` - Dataset containing features (targets are ignored)
    /// * `feature_means` - Optional feature means for background distribution
    ///
    /// # Returns
    /// ShapValues container with shape `[n_samples, n_features + 1, n_outputs]`.
    pub fn shap_values(
        &self,
        data: &Dataset,
        feature_means: Option<Vec<f64>>,
    ) -> Result<ShapValues, ExplainError> {
        let means = feature_means.unwrap_or_else(|| vec![0.0; self.meta.n_features]);
        let explainer = LinearExplainer::new(&self.model, means)?;

        // LinearExplainer takes &Dataset directly
        Ok(explainer.shap_values(data))
    }
}

impl std::fmt::Debug for GBLinearModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GBLinearModel")
            .field("n_features", &self.meta.n_features)
            .field("n_groups", &self.meta.n_groups)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr2, array};

    fn make_simple_model() -> LinearModel {
        // y = 0.5*x0 + 0.3*x1 + 0.1
        LinearModel::new(array![[0.5], [0.3], [0.1]])
    }

    #[test]
    fn from_parts() {
        let linear = make_simple_model();
        let meta = ModelMeta::for_regression(2);
        let model = GBLinearModel::from_parts(linear, meta, OutputTransform::Identity);

        assert_eq!(model.linear().n_features(), 2);
        assert_eq!(model.linear().n_groups(), 1);
        assert_eq!(model.output_transform(), OutputTransform::Identity);
    }

    #[test]
    fn from_parts_stores_transform() {
        let linear = make_simple_model();
        let meta = ModelMeta::for_regression(2);
        let model = GBLinearModel::from_parts(linear, meta, OutputTransform::Sigmoid);

        assert_eq!(model.output_transform(), OutputTransform::Sigmoid);
    }

    #[test]
    fn predict_basic() {
        let linear = make_simple_model();
        let meta = ModelMeta::for_regression(2);
        let model = GBLinearModel::from_parts(linear, meta, OutputTransform::Identity);

        // y = 0.5*1.0 + 0.3*2.0 + 0.1 = 0.5 + 0.6 + 0.1 = 1.2
        // Feature-major: [n_features=2, n_samples=1]
        let features_fm = arr2(&[[1.0], [2.0]]);
        let dataset = Dataset::from_array(features_fm.view(), None, None);
        let preds = model.predict_raw(&dataset);
        assert!((preds[[0, 0]] - 1.2).abs() < 1e-6);
    }

    #[test]
    fn predict_batch_rows() {
        let linear = make_simple_model();
        let meta = ModelMeta::for_regression(2);
        let model = GBLinearModel::from_parts(linear, meta, OutputTransform::Identity);

        // Feature-major: [n_features=2, n_samples=2]
        // feature 0: [1.0, 0.0]
        // feature 1: [2.0, 0.0]
        let features_fm = arr2(&[
            [1.0, 0.0], // feature 0 values for samples 0, 1
            [2.0, 0.0], // feature 1 values for samples 0, 1
        ]);
        let dataset = Dataset::from_array(features_fm.view(), None, None);
        let preds = model.predict_raw(&dataset);

        // Shape is [n_groups, n_samples] = [1, 2]
        // sample 0: 0.5*1.0 + 0.3*2.0 + 0.1 = 1.2
        // sample 1: 0.5*0.0 + 0.3*0.0 + 0.1 = 0.1
        assert!((preds[[0, 0]] - 1.2).abs() < 1e-6);
        assert!((preds[[0, 1]] - 0.1).abs() < 1e-6);
    }

    #[test]
    fn weights_and_bias_via_accessor() {
        let linear = make_simple_model();
        let meta = ModelMeta::for_regression(2);
        let model = GBLinearModel::from_parts(linear, meta, OutputTransform::Identity);

        assert_eq!(model.linear().weight(0, 0), 0.5);
        assert_eq!(model.linear().weight(1, 0), 0.3);
        assert_eq!(model.linear().bias(0), 0.1);
    }

    #[test]
    fn shap_values() {
        let linear = make_simple_model();
        let meta = ModelMeta::for_regression(2);
        let model = GBLinearModel::from_parts(linear, meta, OutputTransform::Identity);

        // Test with means - feature-major layout [n_features=2, n_samples=1]
        let features = arr2(&[[1.0], [2.0]]);
        let dataset = Dataset::from_array(features.view(), None, None);
        let means = vec![0.5, 1.0]; // Centered around different values
        let shap = model.shap_values(&dataset, Some(means)).unwrap();

        assert_eq!(shap.n_samples(), 1);
        assert_eq!(shap.n_features(), 2);

        // SHAP[0] = 0.5 * (1.0 - 0.5) = 0.25
        // SHAP[1] = 0.3 * (2.0 - 1.0) = 0.30
        // base = 0.5*0.5 + 0.3*1.0 + 0.1 = 0.25 + 0.3 + 0.1 = 0.65
        // sum = 0.25 + 0.30 + 0.65 = 1.2 (equals prediction)
        assert!(shap.verify(&[1.2], 1e-5));
    }

    #[test]
    fn shap_values_zero_means() {
        let linear = make_simple_model();
        let meta = ModelMeta::for_regression(2);
        let model = GBLinearModel::from_parts(linear, meta, OutputTransform::Identity);

        // Feature-major layout [n_features=2, n_samples=1]
        let features = arr2(&[[1.0], [2.0]]);
        let dataset = Dataset::from_array(features.view(), None, None);
        // Use None for zero means (centered data assumption)
        let shap = model.shap_values(&dataset, None).unwrap();

        assert_eq!(shap.n_samples(), 1);
        assert_eq!(shap.n_features(), 2);

        // With zero means: SHAP = weights * features
        // SHAP[0] = 0.5 * 1.0 = 0.5
        // SHAP[1] = 0.3 * 2.0 = 0.6
        // base = bias = 0.1
        // sum = 0.5 + 0.6 + 0.1 = 1.2
        assert!(shap.verify(&[1.2], 1e-5));
    }

    #[test]
    fn predict_from_dataset() {
        let linear = make_simple_model();
        let meta = ModelMeta::for_regression(2);
        let model = GBLinearModel::from_parts(linear, meta, OutputTransform::Identity);

        // Feature-major layout: [n_features=2, n_samples=2]
        // feature 0: [1.0, 0.0]
        // feature 1: [2.0, 0.0]
        let feature_major = array![[1.0, 0.0], [2.0, 0.0]];
        let dataset = Dataset::from_array(feature_major.view(), None, None);

        // Use predict method taking Dataset
        let preds = model.predict_raw(&dataset);

        // Values should be:
        // sample 0: 0.5*1.0 + 0.3*2.0 + 0.1 = 1.2
        // sample 1: 0.5*0.0 + 0.3*0.0 + 0.1 = 0.1
        assert_eq!(preds.dim(), (1, 2));
        assert!((preds[[0, 0]] - 1.2).abs() < 1e-6);
        assert!((preds[[0, 1]] - 0.1).abs() < 1e-6);
    }
}

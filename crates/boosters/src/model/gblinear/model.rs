//! GBLinear model implementation.
//!
//! High-level wrapper around [`LinearModel`] with training and prediction.
//! Access components via [`linear()`](GBLinearModel::linear), [`meta()`](GBLinearModel::meta),
//! and [`config()`](GBLinearModel::config).

use crate::dataset::Dataset;
use crate::explainability::{ExplainError, LinearExplainer, ShapValues};
use crate::model::meta::ModelMeta;
use crate::repr::gblinear::LinearModel;
use crate::training::gblinear::GBLinearTrainer;
use crate::training::{EvalSet, Metric, ObjectiveFn};

use ndarray::Array2;

use super::GBLinearConfig;

/// High-level GBLinear model with training, prediction, and explainability.
///
/// Access components via [`linear()`](Self::linear), [`meta()`](Self::meta),
/// and [`config()`](Self::config).
pub struct GBLinearModel {
    /// The underlying linear model.
    model: LinearModel,
    /// Model metadata.
    meta: ModelMeta,
    /// Training configuration (if available).
    ///
    /// This is `Some` when trained with the new API or loaded from a format
    /// that includes config. May be `None` for models loaded from legacy
    /// formats or created with `from_linear_model()`.
    config: GBLinearConfig,
}

impl GBLinearModel {
    /// Train a new GBLinear model.
    ///
    /// # Arguments
    ///
    /// * `dataset` - Training dataset (features, targets, optional weights)
    /// * `eval_sets` - Evaluation sets for monitoring (`&[]` if not needed)
    /// * `config` - Training configuration
    /// * `n_threads` - Thread count: 0 = auto, 1 = sequential, >1 = exact count
    pub fn train(
        dataset: &Dataset,
        eval_sets: &[EvalSet<'_>],
        config: GBLinearConfig,
        n_threads: usize,
    ) -> Option<Self> {
        crate::run_with_threads(n_threads, |_parallelism| {
            Self::train_inner(dataset, eval_sets, config)
        })
    }

    /// Internal training implementation (no thread pool management).
    ///
    /// This method assumes the caller has already set up any necessary thread pool.
    /// Use `train()` for the public API that handles threading automatically.
    fn train_inner(
        dataset: &Dataset,
        eval_sets: &[EvalSet<'_>],
        config: GBLinearConfig,
    ) -> Option<Self> {
        let n_features = dataset.n_features();
        let n_outputs = config.objective.n_outputs();
        
        // Get task kind from objective (not inferred from n_outputs)
        // This correctly handles multi-output regression (e.g., multi-quantile)
        let task = config.objective.task_kind();
        
        // Convert config to trainer params
        let params = config.to_trainer_params();
        
        // Convert Option<Metric> to Metric (None -> Metric::None)
        let metric = config.metric.clone().unwrap_or(Metric::none());
        
        // Create trainer with objective and metric from config
        let trainer = GBLinearTrainer::new(
            config.objective.clone(),
            metric,
            params,
        );
        let linear_model = trainer.train(dataset, eval_sets)?;

        let meta = ModelMeta {
            n_features,
            n_groups: n_outputs,
            task,
            ..Default::default()
        };

        Some(Self { model: linear_model, meta, config })
    }

    /// Create a model from a linear model and metadata.
    ///
    /// Use this when loading models from formats that don't include config,
    /// or for quick testing. For training new models, prefer [`GBLinearModel::train`].
    pub fn from_linear_model(
        model: LinearModel,
        meta: ModelMeta,
    ) -> Self {
        Self { model, meta, config: GBLinearConfig::default() }
    }

    /// Create a model from all its parts.
    ///
    /// Used when loading from a format that includes config, or after training
    /// with the new config-based API.
    pub fn from_parts(
        model: LinearModel,
        meta: ModelMeta,
        config: GBLinearConfig,
    ) -> Self {
        Self { model, meta, config }
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

    /// Get reference to training configuration.
    pub fn config(&self) -> &GBLinearConfig {
        &self.config
    }

    /// Set feature names.
    ///
    /// This mutates the metadata. For new models, prefer setting feature names
    /// during training.
    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        self.meta.feature_names = Some(names);
        self
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
    /// * `features` - Feature-major data `[n_features, n_samples]`
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
    pub fn predict(&self, dataset: Dataset) -> Array2<f32> {
        // Use efficient feature-major prediction
        let mut output = self.predict_raw(dataset);

        // Apply transformation if we have config with objective
        self.config.objective.transform_predictions(output.view_mut());

        output
    }

    /// Predict from feature-major data, returning raw margin scores (no transform).
    ///
    /// # Arguments
    ///
    /// * `features` - Feature-major data `[n_features, n_samples]`
    ///
    /// # Returns
    ///
    /// Array2 with shape `[n_groups, n_samples]`.
    pub fn predict_raw(&self, dataset: Dataset) -> Array2<f32> {
        let n_samples = dataset.n_samples();
        let n_groups = self.meta.n_groups;

        if n_samples == 0 {
            return Array2::zeros((n_groups, 0));
        }

        self.model.predict(dataset.features())
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
        use crate::data::{transpose_to_c_order, SamplesView};

        let means = feature_means.unwrap_or_else(|| vec![0.0; self.meta.n_features]);
        let explainer = LinearExplainer::new(&self.model, means)?;
        
        // Transpose feature-major [n_features, n_samples] to sample-major [n_samples, n_features]
        let features = data.features();
        let samples_array = transpose_to_c_order(features.view());
        let samples_view = SamplesView::from_array(samples_array.view());
        
        Ok(explainer.shap_values(samples_view))
    }
}

impl std::fmt::Debug for GBLinearModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GBLinearModel")
            .field("n_features", &self.meta.n_features)
            .field("n_groups", &self.meta.n_groups)
            .field("n_rounds", &self.config.n_rounds)
            .field("task", &self.meta.task)
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
    fn from_linear_model() {
        let linear = make_simple_model();
        let meta = ModelMeta::for_regression(2);
        let model = GBLinearModel::from_linear_model(linear, meta);

        assert_eq!(model.linear().n_features(), 2);
        assert_eq!(model.linear().n_groups(), 1);
        // from_linear_model uses default config
        assert_eq!(model.config().n_rounds, GBLinearConfig::default().n_rounds);
    }

    #[test]
    fn from_parts_with_config() {
        let linear = make_simple_model();
        let meta = ModelMeta::for_regression(2);
        let config = GBLinearConfig::builder().n_rounds(200).build().unwrap();
        let model = GBLinearModel::from_parts(linear, meta, config);

        assert_eq!(model.config().n_rounds, 200);
    }

    #[test]
    fn predict_basic() {
        let linear = make_simple_model();
        let meta = ModelMeta::for_regression(2);
        let model = GBLinearModel::from_linear_model(linear, meta);

        // y = 0.5*1.0 + 0.3*2.0 + 0.1 = 0.5 + 0.6 + 0.1 = 1.2
        // Feature-major: [n_features=2, n_samples=1]
        let features_fm = arr2(&[[1.0], [2.0]]);
        let dataset = Dataset::new(features_fm.view(), None, None);
        let preds = model.predict_raw(dataset);
        assert!((preds[[0, 0]] - 1.2).abs() < 1e-6);
    }

    #[test]
    fn predict_batch_rows() {
        let linear = make_simple_model();
        let meta = ModelMeta::for_regression(2);
        let model = GBLinearModel::from_linear_model(linear, meta);

        // Feature-major: [n_features=2, n_samples=2]
        // feature 0: [1.0, 0.0]
        // feature 1: [2.0, 0.0]
        let features_fm = arr2(&[
            [1.0, 0.0], // feature 0 values for samples 0, 1
            [2.0, 0.0], // feature 1 values for samples 0, 1
        ]);
        let dataset = Dataset::new(features_fm.view(), None, None);
        let preds = model.predict_raw(dataset);

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
        let model = GBLinearModel::from_linear_model(linear, meta);

        assert_eq!(model.linear().weight(0, 0), 0.5);
        assert_eq!(model.linear().weight(1, 0), 0.3);
        assert_eq!(model.linear().bias(0), 0.1);
    }

    #[test]
    fn shap_values() {
        let linear = make_simple_model();
        let meta = ModelMeta::for_regression(2);
        let model = GBLinearModel::from_linear_model(linear, meta);

        // Test with means - feature-major layout [n_features=2, n_samples=1]
        let features = arr2(&[[1.0], [2.0]]);
        let dataset = Dataset::new(features.view(), None, None);
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
        let model = GBLinearModel::from_linear_model(linear, meta);

        // Feature-major layout [n_features=2, n_samples=1]
        let features = arr2(&[[1.0], [2.0]]);
        let dataset = Dataset::new(features.view(), None, None);
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
        let model = GBLinearModel::from_linear_model(linear, meta);

        // Feature-major layout: [n_features=2, n_samples=2]
        // feature 0: [1.0, 0.0]
        // feature 1: [2.0, 0.0]
        let feature_major = array![[1.0, 0.0], [2.0, 0.0]];
        let dataset = Dataset::new(feature_major.view(), None, None);

        // Use predict method taking Dataset
        let preds = model.predict_raw(dataset);

        // Values should be:
        // sample 0: 0.5*1.0 + 0.3*2.0 + 0.1 = 1.2
        // sample 1: 0.5*0.0 + 0.3*0.0 + 0.1 = 0.1
        assert_eq!(preds.dim(), (1, 2));
        assert!((preds[[0, 0]] - 1.2).abs() < 1e-6);
        assert!((preds[[0, 1]] - 0.1).abs() < 1e-6);
    }
}
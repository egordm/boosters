//! GBLinear model implementation.
//!
//! High-level wrapper around [`LinearModel`] with training and prediction.
//! Access components via [`linear()`](GBLinearModel::linear), [`meta()`](GBLinearModel::meta),
//! and [`objective()`](GBLinearModel::objective).

use crate::Parallelism;
use crate::data::Dataset;
use crate::explainability::{ExplainError, LinearExplainer, ShapValues};
use crate::model::meta::ModelMeta;
use crate::model::OutputTransform;
use crate::repr::gblinear::LinearModel;
use crate::training::gblinear::GBLinearTrainer;
use crate::training::ObjectiveFn;

use ndarray::Array2;

use super::GBLinearConfig;

/// High-level GBLinear model with training, prediction, and explainability.
///
/// Access components via [`linear()`](Self::linear), [`meta()`](Self::meta),
/// and [`objective()`](Self::objective).
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
    /// Objective function used for training.
    ///
    /// Kept for compatibility and training-related queries.
    /// For prediction transform, use `output_transform` instead.
    objective: crate::training::Objective,
}

impl GBLinearModel {
    fn default_objective(task: crate::model::TaskKind) -> crate::training::Objective {
        match task {
            crate::model::TaskKind::Regression => crate::training::Objective::squared(),
            crate::model::TaskKind::BinaryClassification => {
                crate::training::Objective::logistic()
            }
            crate::model::TaskKind::MulticlassClassification { n_classes } => {
                crate::training::Objective::softmax(n_classes)
            }
            crate::model::TaskKind::Ranking => crate::training::Objective::squared(),
        }
    }

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
    ) -> Option<Self> {
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
    ) -> Option<Self> {
        let n_features = dataset.n_features();
        let n_outputs = config.objective.n_outputs();

        // Get task kind from objective (not inferred from n_outputs)
        // This correctly handles multi-output regression (e.g., multi-quantile)
        let task = config.objective.task_kind();

        // GBLinear uses Dataset directly - no binning needed!
        // Extract targets and weights from Dataset
        let targets = dataset
            .targets()
            .expect("dataset must have targets for training");
        let weights = dataset.weights();

        // Convert config to trainer params
        let params = config.to_trainer_params();

        // Use provided metric or derive default from objective
        let metric = config
            .metric
            .clone()
            .unwrap_or_else(|| crate::training::default_metric_for_objective(&config.objective));

        // Create trainer with objective and metric from config
        let trainer = GBLinearTrainer::new(config.objective.clone(), metric, params);
        let linear_model = trainer.train(dataset, targets, weights, val_set)?;

        let meta = ModelMeta {
            n_features,
            n_groups: n_outputs,
            task,
            ..Default::default()
        };

        Some(Self::from_parts(linear_model, meta, config.objective))
    }

    /// Create a model from a linear model and metadata.
    ///
    /// Use this when loading models from formats that don't include config,
    /// or for quick testing. For training new models, prefer [`GBLinearModel::train`].
    pub fn from_linear_model(model: LinearModel, meta: ModelMeta) -> Self {
        let objective = Self::default_objective(meta.task);
        let output_transform = objective.output_transform();
        Self {
            model,
            meta,
            objective,
            output_transform,
        }
    }

    /// Create a model from all its parts.
    ///
    /// Used when loading from a format that includes an explicit objective,
    /// or after training.
    pub fn from_parts(
        model: LinearModel,
        meta: ModelMeta,
        objective: crate::training::Objective,
    ) -> Self {
        let output_transform = objective.output_transform();
        Self {
            model,
            meta,
            objective,
            output_transform,
        }
    }

    /// Create a model from all parts, including an explicit output transform.
    ///
    /// Used when loading models from persistence where the transform is stored
    /// separately. For training, prefer [`GBLinearModel::train`].
    pub fn from_parts_with_transform(
        model: LinearModel,
        meta: ModelMeta,
        objective: crate::training::Objective,
        output_transform: OutputTransform,
    ) -> Self {
        Self {
            model,
            meta,
            objective,
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

    /// Get reference to the objective function.
    pub fn objective(&self) -> &crate::training::Objective {
        &self.objective
    }

    /// Get reference to the output transform.
    pub fn output_transform(&self) -> &OutputTransform {
        &self.output_transform
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
        let n_outputs = self.meta.n_groups;
        self.output_transform.transform_inplace(
            output.as_slice_mut().expect("output must be contiguous"),
            n_outputs,
        );

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
        // from_linear_model selects a default objective from task kind
        assert_eq!(model.objective().name(), "squared");
    }

    #[test]
    fn from_parts_with_config() {
        let linear = make_simple_model();
        let meta = ModelMeta::for_regression(2);
        let objective = crate::training::Objective::logistic();
        let model = GBLinearModel::from_parts(linear, meta, objective);

        assert_eq!(model.objective().name(), "logistic");
    }

    #[test]
    fn predict_basic() {
        let linear = make_simple_model();
        let meta = ModelMeta::for_regression(2);
        let model = GBLinearModel::from_linear_model(linear, meta);

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
        let model = GBLinearModel::from_linear_model(linear, meta);

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
        let model = GBLinearModel::from_linear_model(linear, meta);

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
        let model = GBLinearModel::from_linear_model(linear, meta);

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

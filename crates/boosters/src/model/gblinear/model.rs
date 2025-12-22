//! GBLinear model implementation.
//!
//! High-level wrapper around LinearModel with training, prediction, and serialization.
//!
//! # Access Pattern
//!
//! Use accessors to access model components:
//! - `model.linear()` - underlying linear model (weights, biases)
//! - `model.meta()` - model metadata (n_features, n_groups, task)
//! - `model.config()` - training configuration (if available)
//!
//! # Prediction
//!
//! - [`predict()`](GBLinearModel::predict) - Returns probabilities for classification,
//!   raw values for regression
//! - [`predict_raw()`](GBLinearModel::predict_raw) - Returns raw margin scores

use crate::data::{ColMajor, ColMatrix, DataMatrix, Dataset, DenseMatrix};
use crate::model::meta::ModelMeta;
use crate::repr::gblinear::LinearModel;
use crate::training::gblinear::GBLinearTrainer;
use crate::training::{Metric, ObjectiveFn};

use super::GBLinearConfig;

/// High-level GBLinear model.
///
/// Combines training, prediction, and serialization into a unified interface.
///
/// # Access Pattern
///
/// Use accessors to access model components:
/// - [`linear()`](Self::linear) - underlying linear model
/// - [`meta()`](Self::meta) - model metadata (n_features, n_groups, task)
/// - [`config()`](Self::config) - training configuration (if available)
///
/// # Example
///
/// ```ignore
/// use boosters::model::GBLinearModel;
/// use boosters::model::gblinear::GBLinearConfig;
/// use boosters::data::RowMatrix;
///
/// // Train
/// let config = GBLinearConfig::builder().n_rounds(200).build().unwrap();
/// let model = GBLinearModel::train(&dataset, config, 0)?; // auto threads
///
/// // Access components
/// let n_features = model.meta().n_features;
/// let n_groups = model.meta().n_groups;
/// let lr = model.config().map(|c| c.learning_rate);
///
/// // Predict
/// let features = RowMatrix::from_vec(vec![...], n_rows, n_features);
/// let predictions = model.predict(&features);
/// ```
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
    /// * `config` - Training configuration (objective, metric, hyperparameters)
    /// * `n_threads` - Thread count: 0 = auto, 1 = sequential, >1 = exact count
    ///
    /// # Returns
    ///
    /// Trained model, or `None` if training fails.
    ///
    /// # Threading
    ///
    /// - `0` = Use all available CPU cores (auto-detect)
    /// - `1` = Sequential execution (no parallelism)
    /// - `n > 1` = Use exactly `n` threads
    ///
    /// # Example
    ///
    /// ```ignore
    /// use boosters::model::gblinear::{GBLinearConfig, GBLinearModel};
    /// use boosters::training::{Objective, Metric};
    /// use boosters::data::Dataset;
    ///
    /// let config = GBLinearConfig::builder()
    ///     .objective(Objective::logistic())
    ///     .n_rounds(100)
    ///     .build()
    ///     .unwrap();
    ///
    /// // Sequential training
    /// let model = GBLinearModel::train(&dataset, config.clone(), 1)?;
    ///
    /// // Auto-detect threads
    /// let model = GBLinearModel::train(&dataset, config.clone(), 0)?;
    ///
    /// // Parallel training with 4 threads
    /// let model = GBLinearModel::train(&dataset, config, 4)?;
    /// ```
    pub fn train(dataset: &Dataset, config: GBLinearConfig, n_threads: usize) -> Option<Self> {
        crate::run_with_threads(n_threads, |_parallelism| Self::train_inner(dataset, config))
    }

    /// Internal training implementation (no thread pool management).
    ///
    /// This method assumes the caller has already set up any necessary thread pool.
    /// Use `train()` for the public API that handles threading automatically.
    fn train_inner(dataset: &Dataset, config: GBLinearConfig) -> Option<Self> {
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
        let linear_model = trainer.train(dataset, &[])?;

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
    ///
    /// Use this to access linear model details:
    /// - `model.linear().n_features()` - number of features
    /// - `model.linear().n_groups()` - number of output groups
    /// - `model.linear().weight(f, g)` - weight for feature f, group g
    /// - `model.linear().bias(g)` - bias for group g
    pub fn linear(&self) -> &LinearModel {
        &self.model
    }

    /// Get reference to model metadata.
    ///
    /// Use this to access metadata:
    /// - `model.meta().n_features` - number of input features
    /// - `model.meta().n_groups` - number of output groups
    /// - `model.meta().task` - task type (regression, classification)
    /// - `model.meta().feature_names` - feature names (if set)
    pub fn meta(&self) -> &ModelMeta {
        &self.meta
    }

    /// Get reference to training configuration (if available).
    ///
    /// Returns `Some` if the model was trained with the new config-based API
    /// or loaded from a format that includes config. Returns `None` for models
    /// loaded from legacy formats or created with `from_linear_model()`.
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

    /// Predict for multiple rows, returning transformed predictions.
    ///
    /// Returns probabilities for classification objectives (sigmoid for binary,
    /// softmax for multiclass) and raw values for regression objectives.
    ///
    /// Linear model prediction is fast enough to not benefit from parallelism.
    ///
    /// # Arguments
    ///
    /// * `features` - Feature matrix (any layout implementing [`DataMatrix`])
    ///
    /// # Returns
    ///
    /// Column-major matrix of predictions (n_rows × n_groups).
    ///
    /// # Example
    ///
    /// ```ignore
    /// use boosters::data::RowMatrix;
    ///
    /// let features = RowMatrix::from_vec(vec![0.5, 1.0, 0.3, 2.0], 2, 2);
    /// let predictions = model.predict(&features);
    /// let probs = predictions.col_slice(0);
    /// ```
    pub fn predict<M: DataMatrix<Element = f32>>(&self, features: &M) -> ColMatrix<f32> {
        let n_rows = features.num_rows();
        let n_groups = self.meta.n_groups;

        // Compute raw predictions
        let mut raw = self.compute_predictions_raw(features);

        // Apply transformation if we have config with objective
        let view = ndarray::ArrayViewMut2::from_shape((n_groups, n_rows), &mut raw)
            .expect("prediction buffer shape mismatch");
        self.config.objective.transform_predictions(view);

        DenseMatrix::<f32, ColMajor>::from_vec(raw, n_rows, n_groups)
    }

    /// Predict for multiple rows, returning raw margin scores.
    ///
    /// Returns raw margin scores before any transformation (no sigmoid/softmax).
    ///
    /// # Arguments
    ///
    /// * `features` - Feature matrix (any layout implementing [`DataMatrix`])
    ///
    /// # Returns
    ///
    /// Column-major matrix of raw predictions (n_rows × n_groups).
    pub fn predict_raw<M: DataMatrix<Element = f32>>(&self, features: &M) -> ColMatrix<f32> {
        let n_rows = features.num_rows();
        let n_groups = self.meta.n_groups;

        let raw = self.compute_predictions_raw(features);
        DenseMatrix::<f32, ColMajor>::from_vec(raw, n_rows, n_groups)
    }

    /// Internal: Compute raw predictions in column-major layout.
    ///
    /// Output layout: column-major [n_groups × n_rows], so predictions[g * n_rows + row]
    fn compute_predictions_raw<M: DataMatrix<Element = f32>>(&self, features: &M) -> Vec<f32> {
        let n_rows = features.num_rows();
        let n_features = self.meta.n_features;
        let n_groups = self.meta.n_groups;

        // Column-major output: [n_groups × n_rows]
        let mut output = vec![0.0f32; n_rows * n_groups];
        let mut row_buf = vec![0.0f32; n_features];

        for row_idx in 0..n_rows {
            features.copy_row(row_idx, &mut row_buf);

            for g in 0..n_groups {
                let mut sum = self.model.bias(g);
                for (f, &x) in row_buf.iter().enumerate() {
                    if f < n_features {
                        sum += self.model.weight(f, g) * x;
                    }
                }
                // Column-major: predictions for group g are contiguous
                output[g * n_rows + row_idx] = sum;
            }
        }

        output
    }

    // =========================================================================
    // Explainability
    // =========================================================================

    /// Compute SHAP values for a batch of samples.
    ///
    /// Linear SHAP has a closed-form solution: `shap[i] = w[i] * (x[i] - mean[i])`
    ///
    /// # Arguments
    /// * `features` - Feature matrix, row-major [n_samples × n_features]
    /// * `n_samples` - Number of samples
    /// * `feature_means` - Mean value for each feature (background distribution).
    ///   If `None`, assumes features are centered (zero means).
    ///   For accurate base values, pass training data means.
    ///
    /// # Example
    /// ```ignore
    /// // Option 1: Use centered data assumption (no means needed)
    /// let shap = model.shap_values(&features, n_samples, None)?;
    ///
    /// // Option 2: Use actual feature means for accurate base values
    /// let means = compute_feature_means(&training_data);
    /// let shap = model.shap_values(&features, n_samples, Some(means))?;
    /// // sum(shap) + base_value = prediction
    /// ```
    pub fn shap_values(
        &self,
        features: &[f32],
        n_samples: usize,
        feature_means: Option<Vec<f64>>,
    ) -> Result<crate::explainability::ShapValues, crate::explainability::ExplainError> {
        let means = feature_means.unwrap_or_else(|| vec![0.0; self.meta.n_features]);
        let explainer = crate::explainability::LinearExplainer::new(&self.model, means)?;
        Ok(explainer.shap_values(features, n_samples))
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
    use crate::data::RowMatrix;

    fn make_simple_model() -> LinearModel {
        // y = 0.5*x0 + 0.3*x1 + 0.1
        let weights = vec![0.5, 0.3, 0.1].into_boxed_slice();
        LinearModel::new(weights, 2, 1)
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
        let features = RowMatrix::from_vec(vec![1.0, 2.0], 1, 2);
        let preds = model.predict_raw(&features);
        assert!((preds.col_slice(0)[0] - 1.2).abs() < 1e-6);
    }

    #[test]
    fn predict_batch_rows() {
        let linear = make_simple_model();
        let meta = ModelMeta::for_regression(2);
        let model = GBLinearModel::from_linear_model(linear, meta);

        let features = RowMatrix::from_vec(vec![
            1.0, 2.0, // row 0: 0.5 + 0.6 + 0.1 = 1.2
            0.0, 0.0, // row 1: 0 + 0 + 0.1 = 0.1
        ], 2, 2);
        let preds = model.predict_raw(&features);
        let col = preds.col_slice(0);

        assert!((col[0] - 1.2).abs() < 1e-6);
        assert!((col[1] - 0.1).abs() < 1e-6);
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

        // Test with means
        let features = vec![1.0, 2.0];
        let means = vec![0.5, 1.0]; // Centered around different values
        let shap = model.shap_values(&features, 1, Some(means)).unwrap();

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

        let features = vec![1.0, 2.0];
        // Use None for zero means (centered data assumption)
        let shap = model.shap_values(&features, 1, None).unwrap();

        assert_eq!(shap.n_samples(), 1);
        assert_eq!(shap.n_features(), 2);

        // With zero means: SHAP = weights * features
        // SHAP[0] = 0.5 * 1.0 = 0.5
        // SHAP[1] = 0.3 * 2.0 = 0.6
        // base = bias = 0.1
        // sum = 0.5 + 0.6 + 0.1 = 1.2
        assert!(shap.verify(&[1.2], 1e-5));
    }
}
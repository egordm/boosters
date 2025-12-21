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
//! - [`predict_row()`](GBLinearModel::predict_row) - Single row prediction

use crate::data::{ColMatrix, DataMatrix, DenseMatrix, ColMajor};
use crate::data::Dataset;
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
///
/// // Train
/// let config = GBLinearConfig::builder().n_rounds(200).build().unwrap();
/// let model = GBLinearModel::train(&dataset, &labels, &[], config)?;
///
/// // Access components
/// let n_features = model.meta().n_features;
/// let n_groups = model.meta().n_groups;
/// let lr = model.config().map(|c| c.learning_rate);
///
/// // Predict
/// let predictions = model.predict_batch(&features, n_rows);
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
    config: Option<GBLinearConfig>,
}

impl GBLinearModel {
    /// Train a new GBLinear model.
    ///
    /// # Arguments
    ///
    /// * `dataset` - Training dataset (features, targets, optional weights)
    /// * `config` - Training configuration (objective, metric, hyperparameters)
    ///
    /// # Returns
    ///
    /// Trained model, or `None` if training fails.
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
    ///     .metric(Metric::auc())
    ///     .n_rounds(100)
    ///     .learning_rate(0.5)
    ///     .build()
    ///     .unwrap();
    ///
    /// let model = GBLinearModel::train(&dataset, config)?;
    /// ```
    pub fn train(dataset: &Dataset, config: GBLinearConfig) -> Option<Self> {
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

        Some(Self { model: linear_model, meta, config: Some(config) })
    }

    /// Create a model from an existing LinearModel.
    ///
    /// Config will be `None` since the training parameters are unknown.
    pub fn from_linear_model(model: LinearModel, meta: ModelMeta) -> Self {
        Self { model, meta, config: None }
    }

    /// Create a model from all its parts.
    ///
    /// Used when loading from a format that includes config, or after training
    /// with the new config-based API.
    pub fn from_parts(
        model: LinearModel,
        meta: ModelMeta,
        config: Option<GBLinearConfig>,
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
    pub fn config(&self) -> Option<&GBLinearConfig> {
        self.config.as_ref()
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

    /// Predict for a single row, returning raw margins.
    ///
    /// Returns raw margin scores before any transform (no sigmoid/softmax).
    ///
    /// **Note**: For batch prediction, prefer [`predict()`](Self::predict) or
    /// [`predict_raw()`](Self::predict_raw) which are more efficient.
    pub fn predict_row(&self, features: &[f32]) -> Vec<f32> {
        let n_groups = self.meta.n_groups;
        let n_features = self.meta.n_features;
        let mut output = vec![0.0f32; n_groups];

        for (g, out) in output.iter_mut().enumerate() {
            let mut sum = self.model.bias(g);
            for (f, &x) in features.iter().enumerate() {
                if f < n_features {
                    sum += self.model.weight(f, g) * x;
                }
            }
            *out = sum;
        }

        output
    }

    /// Predict for multiple rows, returning transformed predictions.
    ///
    /// Returns probabilities for classification objectives (sigmoid for binary,
    /// softmax for multiclass) and raw values for regression objectives.
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
        
        // Get raw predictions
        let mut raw = self.predict_matrix(features);
        
        // Apply transformation if we have config with objective
        if let Some(config) = &self.config {
            config.objective.transform_predictions(&mut raw, n_rows, n_groups);
        }
        
        // Convert to column-major matrix
        let mut col_data = vec![0.0f32; n_rows * n_groups];
        for row in 0..n_rows {
            for group in 0..n_groups {
                col_data[group * n_rows + row] = raw[row * n_groups + group];
            }
        }
        
        DenseMatrix::<f32, ColMajor>::from_vec(col_data, n_rows, n_groups)
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
        
        let raw = self.predict_matrix(features);
        
        // Convert to column-major matrix
        let mut col_data = vec![0.0f32; n_rows * n_groups];
        for row in 0..n_rows {
            for group in 0..n_groups {
                col_data[group * n_rows + row] = raw[row * n_groups + group];
            }
        }
        
        DenseMatrix::<f32, ColMajor>::from_vec(col_data, n_rows, n_groups)
    }

    /// Predict for multiple rows using slice input.
    ///
    /// **Legacy API**: For most use cases, prefer [`predict()`](Self::predict)
    /// or [`predict_raw()`](Self::predict_raw) which accept structured matrix input.
    ///
    /// # Arguments
    ///
    /// * `features` - Feature slice, row-major (n_rows × n_features)
    /// * `n_rows` - Number of rows
    ///
    /// # Returns
    ///
    /// Raw predictions as flat slice, length = n_rows × n_groups
    pub fn predict_batch(&self, features: &[f32], n_rows: usize) -> Vec<f32> {
        self.predict_raw_slice(features, n_rows)
    }

    /// Internal prediction from a DataMatrix.
    fn predict_matrix<M: DataMatrix<Element = f32>>(&self, features: &M) -> Vec<f32> {
        let n_rows = features.num_rows();
        let n_features = self.meta.n_features;
        let n_groups = self.meta.n_groups;
        let mut output = vec![0.0f32; n_rows * n_groups];
        let mut row_buf = vec![0.0f32; n_features];

        for row_idx in 0..n_rows {
            features.copy_row(row_idx, &mut row_buf);
            let preds = self.predict_row(&row_buf);
            let offset = row_idx * n_groups;
            output[offset..offset + n_groups].copy_from_slice(&preds);
        }

        output
    }

    /// Internal prediction that returns raw margins as Vec (slice-based).
    fn predict_raw_slice(&self, features: &[f32], n_rows: usize) -> Vec<f32> {
        let n_features = self.meta.n_features;
        let n_groups = self.meta.n_groups;
        let mut output = vec![0.0f32; n_rows * n_groups];

        for (row_idx, row) in features.chunks(n_features).enumerate() {
            let preds = self.predict_row(row);
            let offset = row_idx * n_groups;
            output[offset..offset + n_groups].copy_from_slice(&preds);
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
            .field("n_rounds", &self.config.as_ref().map(|c| c.n_rounds))
            .field("task", &self.meta.task)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(model.config().is_none());
    }

    #[test]
    fn from_parts_with_config() {
        let linear = make_simple_model();
        let meta = ModelMeta::for_regression(2);
        let config = GBLinearConfig::builder().n_rounds(200).build().unwrap();
        let model = GBLinearModel::from_parts(linear, meta, Some(config));

        assert!(model.config().is_some());
        assert_eq!(model.config().unwrap().n_rounds, 200);
    }

    #[test]
    fn predict_row() {
        let linear = make_simple_model();
        let meta = ModelMeta::for_regression(2);
        let model = GBLinearModel::from_linear_model(linear, meta);

        // y = 0.5*1.0 + 0.3*2.0 + 0.1 = 0.5 + 0.6 + 0.1 = 1.2
        let pred = model.predict_row(&[1.0, 2.0]);
        assert!((pred[0] - 1.2).abs() < 1e-6);
    }

    #[test]
    fn predict_batch() {
        let linear = make_simple_model();
        let meta = ModelMeta::for_regression(2);
        let model = GBLinearModel::from_linear_model(linear, meta);

        let features = vec![
            1.0, 2.0, // row 0: 0.5 + 0.6 + 0.1 = 1.2
            0.0, 0.0, // row 1: 0 + 0 + 0.1 = 0.1
        ];
        let preds = model.predict_batch(&features, 2);

        assert!((preds[0] - 1.2).abs() < 1e-6);
        assert!((preds[1] - 0.1).abs() < 1e-6);
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
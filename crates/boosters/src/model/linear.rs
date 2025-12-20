//! GBLinear model implementation.
//!
//! High-level wrapper around LinearModel with training, prediction, and serialization.

#[cfg(feature = "storage")]
use std::path::Path;

use crate::model::meta::{ModelMeta, TaskKind};
use crate::repr::gblinear::LinearModel;

#[cfg(feature = "storage")]
use crate::io::native::{DeserializeError, SerializeError};

/// High-level GBLinear model.
///
/// Combines training, prediction, and serialization into a unified interface.
///
/// # Example
///
/// ```ignore
/// use boosters::model::GBLinearModel;
///
/// // Train
/// let model = GBLinearModel::train(&dataset, &labels, params)?;
///
/// // Predict
/// let predictions = model.predict(&features);
///
/// // Save/Load
/// model.save("model.bstr")?;
/// let loaded = GBLinearModel::load("model.bstr")?;
/// ```
pub struct GBLinearModel {
    /// The underlying linear model.
    model: LinearModel,
    /// Model metadata.
    meta: ModelMeta,
    /// Number of training rounds.
    n_rounds: u32,
}

impl GBLinearModel {
    /// Create a model from an existing LinearModel.
    pub fn from_linear_model(model: LinearModel, meta: ModelMeta, n_rounds: u32) -> Self {
        Self { model, meta, n_rounds }
    }

    /// Get reference to the underlying linear model.
    pub fn linear_model(&self) -> &LinearModel {
        &self.model
    }

    /// Get reference to model metadata.
    pub fn meta(&self) -> &ModelMeta {
        &self.meta
    }

    /// Number of training rounds.
    pub fn n_rounds(&self) -> u32 {
        self.n_rounds
    }

    /// Number of input features.
    pub fn n_features(&self) -> usize {
        self.model.num_features()
    }

    /// Number of output groups.
    pub fn n_groups(&self) -> usize {
        self.model.num_groups()
    }

    /// Task type.
    pub fn task(&self) -> TaskKind {
        self.meta.task
    }

    /// Feature names (if set).
    pub fn feature_names(&self) -> Option<&[String]> {
        self.meta.feature_names.as_deref()
    }

    /// Get weight for a feature in a group.
    pub fn weight(&self, feature: usize, group: usize) -> f32 {
        self.model.weight(feature, group)
    }

    /// Get bias for a group.
    ///
    /// For bulk access, use [`biases()`](Self::biases) instead.
    pub fn bias(&self, group: usize) -> f32 {
        self.model.bias(group)
    }

    /// Get all biases as a slice.
    ///
    /// Returns a slice of length `n_groups`.
    pub fn biases(&self) -> &[f32] {
        self.model.biases()
    }

    /// Get all weights as a flat array.
    ///
    /// Layout: `weights[feature * n_groups + group]`
    pub fn weights(&self) -> &[f32] {
        self.model.weights()
    }

    /// Set feature names.
    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        self.meta.feature_names = Some(names);
        self
    }

    // =========================================================================
    // Prediction
    // =========================================================================

    /// Predict for a single row.
    ///
    /// Returns raw margin scores (before any transform).
    pub fn predict_row(&self, features: &[f32]) -> Vec<f32> {
        let n_groups = self.n_groups();
        let mut output = vec![0.0f32; n_groups];

        for g in 0..n_groups {
            let mut sum = self.bias(g);
            for (f, &x) in features.iter().enumerate() {
                if f < self.n_features() {
                    sum += self.weight(f, g) * x;
                }
            }
            output[g] = sum;
        }

        output
    }

    /// Predict for multiple rows.
    ///
    /// # Arguments
    ///
    /// * `features` - Feature matrix, row-major (n_rows × n_features)
    /// * `n_rows` - Number of rows
    ///
    /// # Returns
    ///
    /// Predictions, length = n_rows × n_groups
    pub fn predict_batch(&self, features: &[f32], n_rows: usize) -> Vec<f32> {
        let n_features = self.n_features();
        let n_groups = self.n_groups();
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
    /// Linear SHAP has a closed-form solution: shap[i] = w[i] * (x[i] - mean[i])
    ///
    /// # Arguments
    /// * `features` - Feature matrix, row-major [n_samples × n_features]
    /// * `n_samples` - Number of samples
    /// * `feature_means` - Mean value for each feature (background distribution).
    ///                     If `None`, assumes features are centered (zero means).
    ///                     For accurate base values, pass training data means.
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
        let means = feature_means.unwrap_or_else(|| vec![0.0; self.n_features()]);
        let explainer = crate::explainability::LinearExplainer::new(&self.model, means)?;
        Ok(explainer.shap_values(features, n_samples))
    }

    // =========================================================================
    // Serialization (requires 'storage' feature)
    // =========================================================================

    /// Save the model to a file.
    #[cfg(feature = "storage")]
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), SerializeError> {
        self.model.save(path, self.n_rounds)
    }

    /// Load a model from a file.
    #[cfg(feature = "storage")]
    pub fn load(path: impl AsRef<Path>) -> Result<Self, DeserializeError> {
        let model = LinearModel::load(path)?;

        let meta = ModelMeta {
            n_features: model.num_features(),
            n_groups: model.num_groups(),
            task: if model.num_groups() == 1 {
                TaskKind::Regression
            } else {
                TaskKind::MulticlassClassification {
                    n_classes: model.num_groups(),
                }
            },
            ..Default::default()
        };

        Ok(Self {
            model,
            meta,
            n_rounds: 0, // Not stored in current format
        })
    }

    /// Serialize the model to bytes.
    #[cfg(feature = "storage")]
    pub fn to_bytes(&self) -> Result<Vec<u8>, SerializeError> {
        self.model.to_bytes(self.n_rounds)
    }

    /// Deserialize a model from bytes.
    #[cfg(feature = "storage")]
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, DeserializeError> {
        let model = LinearModel::from_bytes(bytes)?;

        let meta = ModelMeta {
            n_features: model.num_features(),
            n_groups: model.num_groups(),
            task: if model.num_groups() == 1 {
                TaskKind::Regression
            } else {
                TaskKind::MulticlassClassification {
                    n_classes: model.num_groups(),
                }
            },
            ..Default::default()
        };

        Ok(Self {
            model,
            meta,
            n_rounds: 0,
        })
    }
}

impl std::fmt::Debug for GBLinearModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GBLinearModel")
            .field("n_features", &self.n_features())
            .field("n_groups", &self.n_groups())
            .field("n_rounds", &self.n_rounds())
            .field("task", &self.task())
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
        let model = GBLinearModel::from_linear_model(linear, meta, 100);

        assert_eq!(model.n_features(), 2);
        assert_eq!(model.n_groups(), 1);
        assert_eq!(model.n_rounds(), 100);
    }

    #[test]
    fn predict_row() {
        let linear = make_simple_model();
        let meta = ModelMeta::for_regression(2);
        let model = GBLinearModel::from_linear_model(linear, meta, 100);

        // y = 0.5*1.0 + 0.3*2.0 + 0.1 = 0.5 + 0.6 + 0.1 = 1.2
        let pred = model.predict_row(&[1.0, 2.0]);
        assert!((pred[0] - 1.2).abs() < 1e-6);
    }

    #[test]
    fn predict_batch() {
        let linear = make_simple_model();
        let meta = ModelMeta::for_regression(2);
        let model = GBLinearModel::from_linear_model(linear, meta, 100);

        let features = vec![
            1.0, 2.0, // row 0: 0.5 + 0.6 + 0.1 = 1.2
            0.0, 0.0, // row 1: 0 + 0 + 0.1 = 0.1
        ];
        let preds = model.predict_batch(&features, 2);

        assert!((preds[0] - 1.2).abs() < 1e-6);
        assert!((preds[1] - 0.1).abs() < 1e-6);
    }

    #[test]
    fn weights_and_bias() {
        let linear = make_simple_model();
        let meta = ModelMeta::for_regression(2);
        let model = GBLinearModel::from_linear_model(linear, meta, 100);

        assert_eq!(model.weight(0, 0), 0.5);
        assert_eq!(model.weight(1, 0), 0.3);
        assert_eq!(model.bias(0), 0.1);
    }

    #[cfg(feature = "storage")]
    #[test]
    fn save_load_roundtrip() {
        let linear = make_simple_model();
        let meta = ModelMeta::for_regression(2);
        let model = GBLinearModel::from_linear_model(linear, meta, 100);

        let path = std::env::temp_dir().join("boosters_gblinear_model_test.bstr");

        model.save(&path).unwrap();
        let loaded = GBLinearModel::load(&path).unwrap();

        std::fs::remove_file(&path).ok();

        assert_eq!(model.n_features(), loaded.n_features());
        assert_eq!(model.predict_row(&[1.0, 2.0]), loaded.predict_row(&[1.0, 2.0]));
    }

    #[cfg(feature = "storage")]
    #[test]
    fn bytes_roundtrip() {
        let linear = make_simple_model();
        let meta = ModelMeta::for_regression(2);
        let model = GBLinearModel::from_linear_model(linear, meta, 100);

        let bytes = model.to_bytes().unwrap();
        let loaded = GBLinearModel::from_bytes(&bytes).unwrap();

        assert_eq!(model.n_features(), loaded.n_features());
    }

    #[test]
    fn shap_values() {
        let linear = make_simple_model();
        let meta = ModelMeta::for_regression(2);
        let model = GBLinearModel::from_linear_model(linear, meta, 100);

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
        let model = GBLinearModel::from_linear_model(linear, meta, 100);

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
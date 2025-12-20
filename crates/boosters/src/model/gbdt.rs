//! GBDT model implementation.
//!
//! High-level wrapper around Forest with training, prediction, and serialization.

#[cfg(feature = "storage")]
use std::path::Path;

use crate::data::binned::BinnedDataset;
use crate::inference::gbdt::UnrolledPredictor6;
use crate::model::meta::{ModelMeta, TaskKind};
use crate::repr::gbdt::{Forest, ScalarLeaf};
use crate::training::gbdt::{GBDTParams, GBDTTrainer};
use crate::training::{Metric, ObjectiveFn};

#[cfg(feature = "storage")]
use crate::repr::gbdt::tree::TreeView;
#[cfg(feature = "storage")]
use crate::io::native::{DeserializeError, SerializeError};

/// High-level GBDT model.
///
/// Combines training, prediction, and serialization into a unified interface.
/// Uses an optimized prediction layout for fast inference.
///
/// # Example
///
/// ```ignore
/// use boosters::model::GBDTModel;
/// use boosters::training::{GBDTParams, SquaredLoss, Rmse};
///
/// // Train
/// let params = GBDTParams { n_trees: 50, ..Default::default() };
/// let model = GBDTModel::train(&dataset, &labels, SquaredLoss, Rmse, params);
///
/// // Predict
/// let predictions = model.predict_batch(&features);
///
/// // Save/Load
/// model.save("model.bstr")?;
/// let loaded = GBDTModel::load("model.bstr")?;
/// ```
pub struct GBDTModel {
    /// The underlying forest.
    forest: Forest<ScalarLeaf>,
    /// Model metadata.
    meta: ModelMeta,
}

impl GBDTModel {
    /// Train a new GBDT model.
    ///
    /// # Arguments
    ///
    /// * `dataset` - Binned training dataset
    /// * `targets` - Target values (length = n_rows × n_outputs)
    /// * `weights` - Optional sample weights (empty slice for uniform)
    /// * `objective` - Objective function for training
    /// * `metric` - Evaluation metric
    /// * `params` - Training parameters
    ///
    /// # Returns
    ///
    /// Trained model, or `None` if training fails.
    pub fn train<O, M>(
        dataset: &BinnedDataset,
        targets: &[f32],
        weights: &[f32],
        objective: O,
        metric: M,
        params: GBDTParams,
    ) -> Option<Self>
    where
        O: ObjectiveFn,
        M: Metric,
    {
        let n_features = dataset.n_features();
        let n_outputs = objective.n_outputs();

        let trainer = GBDTTrainer::new(objective, metric, params);
        let forest = trainer.train(dataset, targets, weights, &[])?;

        // Infer task kind from n_outputs
        let task = match n_outputs {
            1 => TaskKind::Regression, // Could also be binary classification
            n => TaskKind::MulticlassClassification { n_classes: n },
        };

        let meta = ModelMeta {
            n_features,
            n_groups: n_outputs,
            task,
            base_scores: forest.base_score().to_vec(),
            ..Default::default()
        };

        Some(Self { forest, meta })
    }

    /// Create a model from an existing forest.
    ///
    /// Useful for wrapping forests loaded from other sources.
    pub fn from_forest(forest: Forest<ScalarLeaf>, meta: ModelMeta) -> Self {
        Self { forest, meta }
    }

    /// Get reference to the underlying forest.
    pub fn forest(&self) -> &Forest<ScalarLeaf> {
        &self.forest
    }

    /// Get reference to model metadata.
    pub fn meta(&self) -> &ModelMeta {
        &self.meta
    }

    /// Number of trees in the forest.
    pub fn n_trees(&self) -> usize {
        self.forest.n_trees()
    }

    /// Number of input features.
    pub fn n_features(&self) -> usize {
        self.meta.n_features
    }

    /// Number of output groups.
    pub fn n_groups(&self) -> usize {
        self.meta.n_groups
    }

    /// Task type.
    pub fn task(&self) -> TaskKind {
        self.meta.task
    }

    /// Feature names (if set).
    pub fn feature_names(&self) -> Option<&[String]> {
        self.meta.feature_names.as_deref()
    }

    /// Best iteration (from early stopping).
    pub fn best_iteration(&self) -> Option<usize> {
        self.meta.best_iteration
    }

    /// Base scores (one per group).
    pub fn base_scores(&self) -> &[f32] {
        &self.meta.base_scores
    }

    /// Set feature names.
    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        self.meta.feature_names = Some(names);
        self
    }

    /// Set best iteration.
    pub fn with_best_iteration(mut self, iter: usize) -> Self {
        self.meta.best_iteration = Some(iter);
        self
    }

    // =========================================================================
    // Prediction
    // =========================================================================

    /// Predict for a single row.
    ///
    /// Returns raw margin scores (before any transform).
    pub fn predict_row(&self, features: &[f32]) -> Vec<f32> {
        self.forest.predict_row(features)
    }

    /// Predict for multiple rows using optimized predictor.
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
        // Create the optimized predictor (borrows forest)
        let predictor = UnrolledPredictor6::new(&self.forest);
        let n_features = self.n_features();
        let n_groups = self.n_groups();

        let mut output = vec![0.0f32; n_rows * n_groups];

        for (row_idx, row) in features.chunks(n_features).enumerate() {
            let preds = predictor.predict_row(row);
            let offset = row_idx * n_groups;
            output[offset..offset + n_groups].copy_from_slice(&preds);
        }

        output
    }

    // =========================================================================
    // Feature Importance
    // =========================================================================

    /// Compute feature importance by split count.
    ///
    /// Compute feature importance with a specific importance type.
    ///
    /// Returns a [`FeatureImportance`] object with the importance scores.
    /// Use `.values()` for raw scores, `.normalized()` for normalized scores,
    /// or `.top_k(n)` for the top n features.
    ///
    /// # Arguments
    /// * `importance_type` - Type of importance (Split, Gain, Cover, etc.)
    ///
    /// # Errors
    /// Returns `ExplainError::MissingNodeStats` if gain/cover importance
    /// is requested but the model doesn't have node statistics.
    ///
    /// # Example
    /// ```ignore
    /// use boosters::explainability::ImportanceType;
    ///
    /// let importance = model.feature_importance(ImportanceType::Split)?;
    /// println!("Raw: {:?}", importance.values());
    /// println!("Normalized: {:?}", importance.normalized());
    /// println!("Top 5: {:?}", importance.top_k(5));
    /// ```
    pub fn feature_importance(
        &self,
        importance_type: crate::explainability::ImportanceType,
    ) -> Result<crate::explainability::FeatureImportance, crate::explainability::ExplainError> {
        crate::explainability::compute_forest_importance(
            &self.forest,
            self.n_features(),
            importance_type,
            self.meta.feature_names.clone(),
        )
    }

    /// Compute SHAP values for a batch of samples.
    ///
    /// Returns per-sample, per-feature SHAP contributions that explain
    /// how each feature contributes to the prediction.
    ///
    /// # Arguments
    /// * `features` - Feature matrix, row-major [n_samples × n_features]
    /// * `n_samples` - Number of samples
    ///
    /// # Errors
    /// Returns `ExplainError::MissingNodeStats` if the model doesn't have
    /// cover statistics (required for TreeSHAP).
    ///
    /// # Example
    /// ```ignore
    /// let shap = model.shap_values(&features, n_samples)?;
    /// let sum: f64 = (0..n_features).map(|f| shap.get(0, f, 0)).sum();
    /// // sum + shap.base_value(0, 0) ≈ prediction
    /// ```
    pub fn shap_values(
        &self,
        features: &[f32],
        n_samples: usize,
    ) -> Result<crate::explainability::ShapValues, crate::explainability::ExplainError> {
        let explainer = crate::explainability::TreeExplainer::new(&self.forest)?;
        Ok(explainer.shap_values(features, n_samples, self.n_features()))
    }

    // =========================================================================
    // Serialization (requires 'storage' feature)
    // =========================================================================

    /// Infer the number of features from the maximum split index in trees.
    #[cfg(feature = "storage")]
    fn infer_n_features(forest: &Forest<ScalarLeaf>) -> usize {
        let mut max_feature = 0u32;
        for tree in forest.trees() {
            for node_idx in 0..tree.n_nodes() as u32 {
                if !tree.is_leaf(node_idx) {
                    max_feature = max_feature.max(tree.split_index(node_idx));
                }
            }
        }
        // +1 because features are 0-indexed
        (max_feature + 1) as usize
    }

    /// Save the model to a file.
    #[cfg(feature = "storage")]
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), SerializeError> {
        let bytes = self.to_bytes()?;
        std::fs::write(path, bytes)?;
        Ok(())
    }

    /// Load a model from a file.
    #[cfg(feature = "storage")]
    pub fn load(path: impl AsRef<Path>) -> Result<Self, DeserializeError> {
        let bytes = std::fs::read(path)?;
        Self::from_bytes(&bytes)
    }

    /// Serialize the model to bytes.
    ///
    /// This properly includes `n_features` and `feature_names` in the serialized payload,
    /// ensuring they survive roundtrip even if not all features are used in splits.
    #[cfg(feature = "storage")]
    pub fn to_bytes(&self) -> Result<Vec<u8>, SerializeError> {
        use crate::io::native::{ModelType, NativeCodec};
        use crate::io::payload::Payload;

        // Update forest base scores from meta
        let mut forest = self.forest.clone();
        forest = forest.with_base_score(self.meta.base_scores.clone());

        // Create payload with explicit metadata from model
        let payload = Payload::from_forest_with_meta(
            &forest,
            self.meta.n_features as u32,
            self.meta.feature_names.clone(),
        );
        let codec = NativeCodec::new();
        codec.serialize(
            ModelType::Gbdt,
            self.meta.n_features as u32,
            self.meta.n_groups as u32,
            &payload,
        )
    }

    /// Deserialize a model from bytes.
    ///
    /// This uses `n_features` and `feature_names` from the serialized payload
    /// rather than inferring from trees, ensuring they match what was originally saved.
    #[cfg(feature = "storage")]
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, DeserializeError> {
        use crate::io::native::{ModelType, NativeCodec};
        use crate::io::payload::Payload;

        let codec = NativeCodec::new();
        let (header, payload): (_, Payload) = codec.deserialize(bytes)?;
        
        if header.model_type != ModelType::Gbdt {
            return Err(DeserializeError::TypeMismatch {
                expected: ModelType::Gbdt,
                actual: header.model_type,
            });
        }

        // Extract feature names before consuming payload
        let feature_names = payload.feature_names().cloned();
        
        let forest = payload.into_forest()?;

        // Use n_features from header if available, otherwise infer
        let n_features = if header.num_features > 0 {
            header.num_features as usize
        } else {
            Self::infer_n_features(&forest)
        };

        // Reconstruct metadata from forest, header, and payload
        let meta = ModelMeta {
            n_features,
            n_groups: forest.n_groups() as usize,
            task: if forest.n_groups() == 1 {
                TaskKind::Regression
            } else {
                TaskKind::MulticlassClassification {
                    n_classes: forest.n_groups() as usize,
                }
            },
            base_scores: forest.base_score().to_vec(),
            feature_names,
            ..Default::default()
        };

        Ok(Self { forest, meta })
    }
}

impl std::fmt::Debug for GBDTModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GBDTModel")
            .field("n_trees", &self.n_trees())
            .field("n_features", &self.n_features())
            .field("n_groups", &self.n_groups())
            .field("task", &self.task())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar_tree;

    fn make_simple_forest() -> Forest<ScalarLeaf> {
        let tree = scalar_tree! {
            0 => num(0, 0.5, L) -> 1, 2,
            1 => leaf(1.0),
            2 => num(1, 0.3, R) -> 3, 4,
            3 => leaf(2.0),
            4 => leaf(3.0),
        };

        let mut forest = Forest::for_regression().with_base_score(vec![0.0]);
        forest.push_tree(tree, 0);
        forest
    }

    #[test]
    fn from_forest() {
        let forest = make_simple_forest();
        let meta = ModelMeta::for_regression(2);
        let model = GBDTModel::from_forest(forest, meta);

        assert_eq!(model.n_trees(), 1);
        assert_eq!(model.n_features(), 2);
        assert_eq!(model.n_groups(), 1);
    }

    #[test]
    fn predict_row() {
        let forest = make_simple_forest();
        let meta = ModelMeta::for_regression(2);
        let model = GBDTModel::from_forest(forest, meta);

        // x0 < 0.5 → leaf 1.0
        assert_eq!(model.predict_row(&[0.3, 0.5]), vec![1.0]);
        // x0 >= 0.5, x1 >= 0.3 → leaf 3.0
        assert_eq!(model.predict_row(&[0.7, 0.5]), vec![3.0]);
    }

    #[test]
    fn predict_batch() {
        let forest = make_simple_forest();
        let meta = ModelMeta::for_regression(2);
        let model = GBDTModel::from_forest(forest, meta);

        let features = vec![
            0.3, 0.5, // row 0
            0.7, 0.5, // row 1
        ];
        let preds = model.predict_batch(&features, 2);

        assert_eq!(preds, vec![1.0, 3.0]);
    }

    #[test]
    fn feature_importance() {
        use crate::explainability::ImportanceType;

        let forest = make_simple_forest();
        let meta = ModelMeta::for_regression(2);
        let model = GBDTModel::from_forest(forest, meta);

        let importance = model.feature_importance(ImportanceType::Split).unwrap();
        assert_eq!(importance.values(), &[1.0, 1.0]); // feature 0 and 1 each used once
    }

    #[test]
    fn feature_importance_with_stats() {
        use crate::explainability::{ExplainError, ImportanceType};

        // Forest with stats
        let tree = scalar_tree! {
            0 => num(0, 0.5, L) -> 1, 2,
            1 => leaf(1.0),
            2 => num(1, 0.3, R) -> 3, 4,
            3 => leaf(2.0),
            4 => leaf(3.0),
        };
        let tree = tree
            .with_gains(vec![10.0, 0.0, 5.0, 0.0, 0.0])
            .with_covers(vec![100.0, 40.0, 60.0, 30.0, 30.0]);

        let mut forest = crate::repr::gbdt::Forest::for_regression().with_base_score(vec![0.0]);
        forest.push_tree(tree, 0);

        let meta = ModelMeta::for_regression(2);
        let model = GBDTModel::from_forest(forest, meta);

        // Split importance always works
        let split_imp = model.feature_importance(ImportanceType::Split).unwrap();
        assert_eq!(split_imp.values(), &[1.0, 1.0]);

        // Gain importance works with stats
        let gain_imp = model.feature_importance(ImportanceType::Gain).unwrap();
        assert_eq!(gain_imp.values(), &[10.0, 5.0]);

        // Forest without stats
        let tree2 = scalar_tree! {
            0 => num(0, 0.5, L) -> 1, 2,
            1 => leaf(1.0),
            2 => leaf(2.0),
        };
        let mut forest2 = crate::repr::gbdt::Forest::for_regression().with_base_score(vec![0.0]);
        forest2.push_tree(tree2, 0);
        let model2 = GBDTModel::from_forest(forest2, ModelMeta::for_regression(2));

        // Gain fails without stats
        assert!(matches!(
            model2.feature_importance(ImportanceType::Gain),
            Err(ExplainError::MissingNodeStats(_))
        ));
    }

    #[test]
    fn feature_names() {
        let forest = make_simple_forest();
        let meta = ModelMeta::for_regression(2);
        let model =
            GBDTModel::from_forest(forest, meta).with_feature_names(vec!["a".into(), "b".into()]);

        assert_eq!(model.feature_names(), Some(&["a".to_string(), "b".to_string()][..]));
    }

    #[cfg(feature = "storage")]
    #[test]
    fn save_load_roundtrip() {
        let forest = make_simple_forest();
        // Use base_scores matching the forest (0.0)
        let meta = ModelMeta::for_regression(2).with_base_scores(vec![0.0]);
        let model = GBDTModel::from_forest(forest, meta);

        let path = std::env::temp_dir().join("boosters_gbdt_model_test.bstr");

        model.save(&path).unwrap();
        let loaded = GBDTModel::load(&path).unwrap();

        std::fs::remove_file(&path).ok();

        assert_eq!(model.n_trees(), loaded.n_trees());
        assert_eq!(model.predict_row(&[0.3, 0.5]), loaded.predict_row(&[0.3, 0.5]));

        // Batch prediction
        let features = vec![0.3, 0.5, 0.7, 0.5];
        let preds_orig = model.forest.predict_row(&features[0..2]);
        let preds_loaded = loaded.predict_batch(&features, 2);
        assert_eq!(preds_orig[0], preds_loaded[0]);
    }

    #[cfg(feature = "storage")]
    #[test]
    fn bytes_roundtrip() {
        let forest = make_simple_forest();
        let meta = ModelMeta::for_regression(2);
        let model = GBDTModel::from_forest(forest, meta);

        let bytes = model.to_bytes().unwrap();
        let loaded = GBDTModel::from_bytes(&bytes).unwrap();

        assert_eq!(model.n_trees(), loaded.n_trees());
    }

    #[test]
    fn shap_values() {
        use crate::explainability::ExplainError;

        // Forest with covers
        let tree = scalar_tree! {
            0 => num(0, 0.5, L) -> 1, 2,
            1 => leaf(-1.0),
            2 => leaf(1.0),
        };
        let tree = tree.with_covers(vec![100.0, 50.0, 50.0]);

        let mut forest = crate::repr::gbdt::Forest::for_regression().with_base_score(vec![0.0]);
        forest.push_tree(tree, 0);

        let meta = ModelMeta::for_regression(2);
        let model = GBDTModel::from_forest(forest, meta);

        // Compute SHAP values
        let features = vec![0.3, 0.7]; // goes left
        let shap = model.shap_values(&features, 1);
        assert!(shap.is_ok());

        let shap = shap.unwrap();
        assert_eq!(shap.n_samples(), 1);
        assert_eq!(shap.n_features(), 2);

        // Forest without covers fails
        let tree2 = scalar_tree! {
            0 => num(0, 0.5, L) -> 1, 2,
            1 => leaf(-1.0),
            2 => leaf(1.0),
        };
        let mut forest2 = crate::repr::gbdt::Forest::for_regression().with_base_score(vec![0.0]);
        forest2.push_tree(tree2, 0);
        let model2 = GBDTModel::from_forest(forest2, ModelMeta::for_regression(2));

        assert!(matches!(
            model2.shap_values(&features, 1),
            Err(ExplainError::MissingNodeStats(_))
        ));
    }
}

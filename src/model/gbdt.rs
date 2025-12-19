//! GBDT model implementation.
//!
//! High-level wrapper around Forest with training, prediction, and serialization.

use std::path::Path;

use crate::data::binned::BinnedDataset;
use crate::inference::gbdt::UnrolledPredictor6;
use crate::model::meta::{ModelMeta, TaskKind};
use crate::repr::gbdt::tree::TreeView;
use crate::repr::gbdt::{Forest, ScalarLeaf};
use crate::training::gbdt::{GBDTParams, GBDTTrainer};
use crate::training::{Metric, Objective};

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
        objective: O,
        metric: M,
        params: GBDTParams,
    ) -> Option<Self>
    where
        O: Objective,
        M: Metric,
    {
        Self::train_with_weights(dataset, targets, &[], objective, metric, params)
    }

    /// Train with sample weights.
    pub fn train_with_weights<O, M>(
        dataset: &BinnedDataset,
        targets: &[f32],
        weights: &[f32],
        objective: O,
        metric: M,
        params: GBDTParams,
    ) -> Option<Self>
    where
        O: Objective,
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
    /// Returns a vector of importance scores, one per feature.
    /// Importance is the number of times each feature is used in splits.
    pub fn feature_importance(&self) -> Vec<u32> {
        let mut counts = vec![0u32; self.n_features()];

        for tree in self.forest.trees() {
            for node_idx in 0..tree.n_nodes() as u32 {
                if !tree.is_leaf(node_idx) {
                    let feature = tree.split_index(node_idx) as usize;
                    if feature < counts.len() {
                        counts[feature] += 1;
                    }
                }
            }
        }

        counts
    }

    /// Compute feature importance with normalized scores.
    ///
    /// Returns importance scores normalized to sum to 1.0.
    pub fn feature_importance_normalized(&self) -> Vec<f32> {
        let counts = self.feature_importance();
        let total: u32 = counts.iter().sum();

        if total == 0 {
            vec![0.0; counts.len()]
        } else {
            counts.iter().map(|&c| c as f32 / total as f32).collect()
        }
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
        // Update forest base scores from meta
        let mut forest = self.forest.clone();
        forest = forest.with_base_score(self.meta.base_scores.clone());
        forest.save(path)
    }

    /// Load a model from a file.
    #[cfg(feature = "storage")]
    pub fn load(path: impl AsRef<Path>) -> Result<Self, DeserializeError> {
        let forest = Forest::load(path)?;

        // Infer n_features from maximum split index in trees
        let n_features = Self::infer_n_features(&forest);

        // Reconstruct metadata from forest
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
            ..Default::default()
        };

        Ok(Self { forest, meta })
    }

    /// Serialize the model to bytes.
    #[cfg(feature = "storage")]
    pub fn to_bytes(&self) -> Result<Vec<u8>, SerializeError> {
        let mut forest = self.forest.clone();
        forest = forest.with_base_score(self.meta.base_scores.clone());
        forest.to_bytes()
    }

    /// Deserialize a model from bytes.
    #[cfg(feature = "storage")]
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, DeserializeError> {
        let forest = Forest::from_bytes(bytes)?;

        // Infer n_features from maximum split index in trees
        let n_features = Self::infer_n_features(&forest);

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
        let forest = make_simple_forest();
        let meta = ModelMeta::for_regression(2);
        let model = GBDTModel::from_forest(forest, meta);

        let importance = model.feature_importance();
        assert_eq!(importance, vec![1, 1]); // feature 0 and 1 each used once
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
}

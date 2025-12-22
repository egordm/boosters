//! GBDT model implementation.
//!
//! High-level wrapper around [`Forest`] with training and prediction.
//! Access components via [`forest()`](GBDTModel::forest), [`meta()`](GBDTModel::meta),
//! and [`config()`](GBDTModel::config).

use crate::data::binned::BinnedDataset;
use crate::inference::gbdt::UnrolledPredictor6;
use crate::model::meta::ModelMeta;
use crate::repr::gbdt::{Forest, ScalarLeaf};
use crate::training::gbdt::GBDTTrainer;
use crate::training::{Metric, ObjectiveFn};
use crate::utils::{Parallelism, run_with_threads};

use ndarray::{Array2, ArrayView1, ArrayView2};

use super::GBDTConfig;

/// High-level GBDT model with training, prediction, and explainability.
///
/// Access components via [`forest()`](Self::forest), [`meta()`](Self::meta),
/// and [`config()`](Self::config).
pub struct GBDTModel {
    /// The underlying forest.
    forest: Forest<ScalarLeaf>,
    /// Model metadata.
    meta: ModelMeta,
    /// Training configuration (if available).
    /// 
    /// This is `Some` when trained with the new API or loaded from a format
    /// that includes config. May be `None` for models loaded from legacy
    /// formats or created with `from_forest()`.
    config: GBDTConfig,
}

impl GBDTModel {
    /// Create a model from a forest and metadata.
    ///
    /// Use this when loading models from formats that don't include config,
    /// or for quick testing. For training new models, prefer [`GBDTModel::train`].
    pub fn from_forest(
        forest: Forest<ScalarLeaf>,
        meta: ModelMeta,
    ) -> Self {
        Self { forest, meta, config: GBDTConfig::default() }
    }

    /// Create a model from all its parts.
    ///
    /// Used when loading from a format that includes config, or after training
    /// with the new config-based API.
    pub fn from_parts(
        forest: Forest<ScalarLeaf>,
        meta: ModelMeta,
        config: GBDTConfig,
    ) -> Self {
        Self { forest, meta, config }
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    /// Get reference to the underlying forest.
    pub fn forest(&self) -> &Forest<ScalarLeaf> {
        &self.forest
    }

    /// Get reference to model metadata.
    pub fn meta(&self) -> &ModelMeta {
        &self.meta
    }

    /// Get reference to training configuration.
    pub fn config(&self) -> &GBDTConfig {
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

    /// Set best iteration (from early stopping).
    pub fn with_best_iteration(mut self, iter: usize) -> Self {
        self.meta.best_iteration = Some(iter);
        self
    }

    /// Train a new GBDT model.
    ///
    /// # Arguments
    ///
    /// * `dataset` - Binned training dataset
    /// * `targets` - Target values (length = n_rows × n_outputs)
    /// * `weights` - Optional sample weights (None for uniform)
    /// * `config` - Training configuration
    /// * `n_threads` - Thread count: 0 = auto, 1 = sequential, >1 = exact count
    pub fn train(
        dataset: &BinnedDataset,
        targets: ArrayView1<f32>,
        weights: Option<ArrayView1<f32>>,
        config: GBDTConfig,
        n_threads: usize,
    ) -> Option<Self> {
        crate::run_with_threads(n_threads, |parallelism| {
            Self::train_inner(dataset, targets, weights, config, parallelism)
        })
    }

    /// Internal training implementation (no thread pool management).
    ///
    /// This method assumes the caller has already set up any necessary thread pool.
    /// Use `train()` for the public API that handles threading automatically.
    fn train_inner(
        dataset: &BinnedDataset,
        targets: ArrayView1<f32>,
        weights: Option<ArrayView1<f32>>,
        config: GBDTConfig,
        parallelism: Parallelism,
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
        let trainer = GBDTTrainer::new(
            config.objective.clone(),
            metric,
            params,
        );
        
        // Components receive parallelism flag; thread pool is already set up
        let forest = trainer.train(dataset, targets, weights, &[], parallelism)?;

        let meta = ModelMeta {
            n_features,
            n_groups: n_outputs,
            task,
            base_scores: forest.base_score().to_vec(),
            ..Default::default()
        };

        Some(Self { forest, meta, config })
    }


    // =========================================================================
    // Prediction
    // =========================================================================

    /// Predict for multiple rows, returning transformed predictions.
    ///
    /// Returns probabilities for classification (sigmoid/softmax) or raw values
    /// for regression. Uses automatic parallelization for large batches.
    ///
    /// # Arguments
    ///
    /// * `features` - Feature matrix with shape `[n_samples, n_features]` (sample-major)
    /// * `n_threads` - Thread count: 0 = auto, 1 = sequential, >1 = exact count
    ///
    /// # Returns
    ///
    /// Array2 with shape `[n_groups, n_samples]`. Access group predictions via `.row(group_idx)`.
    pub fn predict(
        &self,
        features: ArrayView2<f32>,
        n_threads: usize,
    ) -> Array2<f32> {
        let n_rows = features.nrows();
        let n_groups = self.meta.n_groups;

        if n_rows == 0 {
            return Array2::zeros((n_groups, 0));
        }

        // Allocate output [n_groups, n_samples]
        let mut output_array = Array2::<f32>::zeros((n_groups, n_rows));

        // Run prediction with thread pool management
        run_with_threads(n_threads, |parallelism| {
            let predictor = UnrolledPredictor6::new(&self.forest);
            predictor.predict_into(features, None, parallelism, output_array.view_mut());
        });

        // Apply transformation if we have config with objective
        // transform_predictions expects [n_outputs, n_rows] which matches our array shape
        self.config.objective.transform_predictions(output_array.view_mut());

        output_array
    }

    /// Predict for multiple rows, returning raw margin scores (no transform).
    ///
    /// # Arguments
    ///
    /// * `features` - Feature matrix with shape `[n_samples, n_features]` (sample-major)
    /// * `n_threads` - Thread count: 0 = auto, 1 = sequential, >1 = exact count
    ///
    /// # Returns
    ///
    /// Array2 with shape `[n_groups, n_samples]`. Access group predictions via `.row(group_idx)`.
    pub fn predict_raw(
        &self,
        features: ArrayView2<f32>,
        n_threads: usize,
    ) -> Array2<f32> {
        let n_rows = features.nrows();
        let n_groups = self.meta.n_groups;

        if n_rows == 0 {
            return Array2::zeros((n_groups, 0));
        }

        // Allocate output [n_groups, n_samples]
        let mut output_array = Array2::<f32>::zeros((n_groups, n_rows));

        // Run prediction with thread pool management
        run_with_threads(n_threads, |parallelism| {
            let predictor = UnrolledPredictor6::new(&self.forest);
            predictor.predict_into(features, None, parallelism, output_array.view_mut());
        });

        output_array
    }

    // =========================================================================
    // Feature Importance
    // =========================================================================

    /// Compute feature importance.
    ///
    /// Returns `FeatureImportance` with `.values()`, `.normalized()`, and `.top_k(n)` methods.
    /// Gain/Cover types require node statistics (returns `ExplainError::MissingNodeStats` if missing).
    pub fn feature_importance(
        &self,
        importance_type: crate::explainability::ImportanceType,
    ) -> Result<crate::explainability::FeatureImportance, crate::explainability::ExplainError> {
        crate::explainability::compute_forest_importance(
            &self.forest,
            self.meta.n_features,
            importance_type,
            self.meta.feature_names.clone(),
        )
    }

    /// Compute SHAP values for a batch of samples.
    ///
    /// Requires cover statistics (returns `ExplainError::MissingNodeStats` if missing).
    pub fn shap_values(
        &self,
        features: &[f32],
        n_samples: usize,
    ) -> Result<crate::explainability::ShapValues, crate::explainability::ExplainError> {
        let explainer = crate::explainability::TreeExplainer::new(&self.forest)?;
        Ok(explainer.shap_values(features, n_samples, self.meta.n_features))
    }
}

impl std::fmt::Debug for GBDTModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GBDTModel")
            .field("n_trees", &self.forest.n_trees())
            .field("n_features", &self.meta.n_features)
            .field("n_groups", &self.meta.n_groups)
            .field("task", &self.meta.task)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar_tree;
    use ndarray::arr2;

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

        assert_eq!(model.forest().n_trees(), 1);
        assert_eq!(model.meta().n_features, 2);
        assert_eq!(model.meta().n_groups, 1);
        // from_forest uses default config
        assert_eq!(model.config().n_trees, GBDTConfig::default().n_trees);
    }

    #[test]
    fn from_parts_with_config() {
        let forest = make_simple_forest();
        let meta = ModelMeta::for_regression(2);
        let config = GBDTConfig::builder().n_trees(100).build().unwrap();
        let model = GBDTModel::from_parts(forest, meta, config);

        assert_eq!(model.config().n_trees, 100);
    }

    #[test]
    fn predict_basic() {
        let forest = make_simple_forest();
        let meta = ModelMeta::for_regression(2);
        let model = GBDTModel::from_forest(forest, meta);

        // x0 < 0.5 → leaf 1.0
        let features1 = arr2(&[[0.3, 0.5]]);
        let preds1 = model.predict(features1.view(), 1);
        assert_eq!(preds1.row(0).as_slice().unwrap(), &[1.0]);

        // x0 >= 0.5, x1 >= 0.3 → leaf 3.0
        let features2 = arr2(&[[0.7, 0.5]]);
        let preds2 = model.predict(features2.view(), 1);
        assert_eq!(preds2.row(0).as_slice().unwrap(), &[3.0]);
    }

    #[test]
    fn predict_batch_rows() {
        let forest = make_simple_forest();
        let meta = ModelMeta::for_regression(2);
        let model = GBDTModel::from_forest(forest, meta);

        let features = arr2(&[
            [0.3, 0.5], // row 0
            [0.7, 0.5], // row 1
        ]);
        let preds = model.predict(features.view(), 1);

        // Shape is [n_groups, n_samples] = [1, 2]
        assert_eq!(preds.row(0).as_slice().unwrap(), &[1.0, 3.0]);
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

        assert_eq!(
            model.meta().feature_names.as_deref(),
            Some(&["a".to_string(), "b".to_string()][..])
        );
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

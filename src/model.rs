//! High-level model wrapper for inference.
//!
//! The [`Model`] struct provides the main public API for prediction.
//! It wraps a booster (tree ensemble), metadata, and objective function.
//!
//! # Example
//!
//! ```ignore
//! use booste_rs::model::Model;
//! use booste_rs::data::DenseMatrix;
//!
//! let model = Model::load("model.json")?;
//! let features = DenseMatrix::from_vec(data, num_rows, num_features);
//! let predictions = model.predict(&features);
//! ```
//!
//! See RFC-0007 for design rationale.

use crate::data::DataMatrix;
use crate::forest::SoAForest;
use crate::objective::Objective;
use crate::predict::{PredictionOutput, Predictor};
use crate::trees::ScalarLeaf;

// =============================================================================
// Model
// =============================================================================

/// A trained gradient boosting model (inference-ready).
///
/// This is the high-level wrapper for prediction. It contains:
/// - A booster (tree ensemble or linear model)
/// - Model metadata (num_features, num_groups, base_score)
/// - Feature information (names, types)
/// - Objective function (for output transformation)
#[derive(Debug, Clone)]
pub struct Model {
    /// The ensemble (trees, dart, or linear).
    pub booster: Booster,

    /// Model metadata.
    pub meta: ModelMeta,

    /// Feature information.
    pub features: FeatureInfo,

    /// Objective function (for output transformation).
    pub objective: Objective,
}

impl Model {
    /// Create a new model.
    pub fn new(
        booster: Booster,
        meta: ModelMeta,
        features: FeatureInfo,
        objective: Objective,
    ) -> Self {
        Self {
            booster,
            meta,
            features,
            objective,
        }
    }

    /// Number of features expected by the model.
    #[inline]
    pub fn num_features(&self) -> usize {
        self.meta.num_features as usize
    }

    /// Number of output groups (1 for regression, K for K-class).
    #[inline]
    pub fn num_groups(&self) -> usize {
        self.meta.num_groups as usize
    }

    /// Predict raw scores (before objective transformation).
    ///
    /// Returns margin scores for classification, raw output for regression.
    pub fn predict_raw<M: DataMatrix<Element = f32>>(&self, features: &M) -> PredictionOutput {
        match &self.booster {
            Booster::Tree(forest) => {
                let predictor = Predictor::new(forest);
                predictor.predict(features)
            }
            Booster::Dart { forest, weights } => {
                // DART: apply tree weights during prediction
                let predictor = Predictor::new(forest);
                let mut output = predictor.predict(features);

                // Weight the tree contributions
                // Note: This is a simplified approach; proper DART weighting
                // requires per-tree accumulation. For now, we scale the entire output.
                let weight_sum: f32 = weights.iter().sum();
                if weight_sum > 0.0 {
                    let scale = weights.len() as f32 / weight_sum;
                    for val in output.as_mut_slice() {
                        *val *= scale;
                    }
                }

                output
            }
        }
    }

    /// Predict with objective transformation applied.
    ///
    /// For classification, returns probabilities.
    /// For regression, returns predictions (usually same as raw).
    pub fn predict<M: DataMatrix<Element = f32>>(&self, features: &M) -> PredictionOutput {
        let mut output = self.predict_raw(features);
        self.objective.transform(&mut output);
        output
    }
}

// =============================================================================
// Booster
// =============================================================================

/// The ensemble type.
#[derive(Debug, Clone)]
pub enum Booster {
    /// Standard tree ensemble (gbtree).
    Tree(SoAForest<ScalarLeaf>),

    /// DART: trees with per-tree weights applied during inference.
    Dart {
        forest: SoAForest<ScalarLeaf>,
        weights: Box<[f32]>,
    },
}

impl Booster {
    /// Get the underlying forest (for both Tree and Dart).
    pub fn forest(&self) -> &SoAForest<ScalarLeaf> {
        match self {
            Booster::Tree(f) => f,
            Booster::Dart { forest, .. } => forest,
        }
    }

    /// Number of trees in the booster.
    pub fn num_trees(&self) -> usize {
        self.forest().num_trees()
    }
}

// =============================================================================
// Model Metadata
// =============================================================================

/// Model metadata.
#[derive(Debug, Clone)]
pub struct ModelMeta {
    /// Number of input features.
    pub num_features: u32,

    /// Number of output groups (1 for regression, K for K-class).
    pub num_groups: u32,

    /// Base score per group (added to raw predictions).
    pub base_score: Vec<f32>,

    /// Where the model came from.
    pub source: ModelSource,
}

impl ModelMeta {
    /// Create metadata for a regression model.
    pub fn regression(num_features: u32) -> Self {
        Self {
            num_features,
            num_groups: 1,
            base_score: vec![0.5],
            source: ModelSource::Native { version: 1 },
        }
    }

    /// Create metadata for a classification model.
    pub fn classification(num_features: u32, num_groups: u32) -> Self {
        Self {
            num_features,
            num_groups,
            base_score: vec![0.5; num_groups as usize],
            source: ModelSource::Native { version: 1 },
        }
    }
}

/// Where the model came from (for diagnostics).
#[derive(Debug, Clone, PartialEq)]
pub enum ModelSource {
    /// Native booste-rs model.
    Native { version: u32 },

    /// Loaded from XGBoost JSON format.
    XGBoostJson { version: [u32; 3] },

    /// Unknown source.
    Unknown,
}

// =============================================================================
// Feature Information
// =============================================================================

/// Feature metadata for all features.
#[derive(Debug, Clone, Default)]
pub struct FeatureInfo {
    /// Per-feature information.
    pub features: Vec<Feature>,
}

impl FeatureInfo {
    /// Create empty feature info.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create feature info with N unnamed numeric features.
    pub fn numeric(num_features: usize) -> Self {
        Self {
            features: (0..num_features)
                .map(|i| Feature {
                    name: None,
                    feature_type: FeatureType::Numeric,
                    index: i as u32,
                })
                .collect(),
        }
    }

    /// Number of features.
    pub fn len(&self) -> usize {
        self.features.len()
    }

    /// Whether there are no features.
    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }
}

/// Information about a single feature.
#[derive(Debug, Clone)]
pub struct Feature {
    /// Optional feature name.
    pub name: Option<String>,

    /// Feature type (numeric or categorical).
    pub feature_type: FeatureType,

    /// Feature index (column position).
    pub index: u32,
}

/// Type of a feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FeatureType {
    /// Numeric feature (continuous or integer).
    #[default]
    Numeric,

    /// Categorical feature with given number of categories.
    Categorical { num_categories: u32 },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::DenseMatrix;
    use crate::trees::TreeBuilder;

    /// Build a simple tree: feat0 < threshold → left_val, else right_val
    fn build_simple_tree(
        left_val: f32,
        right_val: f32,
        threshold: f32,
    ) -> crate::trees::SoATreeStorage<ScalarLeaf> {
        let mut builder = TreeBuilder::new();
        builder.add_split(0, threshold, true, 1, 2);
        builder.add_leaf(ScalarLeaf(left_val));
        builder.add_leaf(ScalarLeaf(right_val));
        builder.build()
    }

    fn build_regression_model() -> Model {
        let mut forest = SoAForest::for_regression();
        forest.push_tree(build_simple_tree(1.0, 2.0, 0.5), 0);

        Model::new(
            Booster::Tree(forest),
            ModelMeta::regression(1),
            FeatureInfo::numeric(1),
            Objective::SquaredError,
        )
    }

    fn build_binary_model() -> Model {
        let mut forest = SoAForest::for_regression().with_base_score(vec![0.0]);
        // Trees output logits
        forest.push_tree(build_simple_tree(-2.0, 2.0, 0.5), 0);

        Model::new(
            Booster::Tree(forest),
            ModelMeta::regression(1),
            FeatureInfo::numeric(1),
            Objective::BinaryLogistic,
        )
    }

    #[test]
    fn model_predict_raw() {
        let model = build_regression_model();
        let features = DenseMatrix::from_vec(vec![0.3, 0.7], 2, 1);

        let output = model.predict_raw(&features);

        assert_eq!(output.shape(), (2, 1));
        assert_eq!(output.row(0), &[1.0]); // 0.3 < 0.5 → left
        assert_eq!(output.row(1), &[2.0]); // 0.7 >= 0.5 → right
    }

    #[test]
    fn model_predict_with_transform() {
        let model = build_binary_model();
        let features = DenseMatrix::from_vec(vec![0.3, 0.7], 2, 1);

        let output = model.predict(&features);

        assert_eq!(output.shape(), (2, 1));
        // -2.0 → sigmoid ≈ 0.119
        assert!((output.row(0)[0] - 0.119).abs() < 0.01);
        // 2.0 → sigmoid ≈ 0.881
        assert!((output.row(1)[0] - 0.881).abs() < 0.01);
    }

    #[test]
    fn booster_forest_access() {
        let mut forest = SoAForest::for_regression();
        forest.push_tree(build_simple_tree(1.0, 2.0, 0.5), 0);

        let booster = Booster::Tree(forest);
        assert_eq!(booster.num_trees(), 1);
    }

    #[test]
    fn feature_info_numeric() {
        let info = FeatureInfo::numeric(5);
        assert_eq!(info.len(), 5);
        assert!(!info.is_empty());
        assert_eq!(info.features[0].feature_type, FeatureType::Numeric);
        assert_eq!(info.features[0].index, 0);
    }

    #[test]
    fn model_meta_regression() {
        let meta = ModelMeta::regression(10);
        assert_eq!(meta.num_features, 10);
        assert_eq!(meta.num_groups, 1);
    }
}

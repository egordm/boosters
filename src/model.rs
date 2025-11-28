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

// =============================================================================
// Objective
// =============================================================================

/// Objective function for output transformation.
///
/// Determines how raw model output is transformed for final predictions.
/// For example, binary classification uses sigmoid to convert logits to probabilities.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Objective {
    // --- Regression ---
    /// Squared error loss (no transformation).
    SquaredError,

    /// Absolute error loss (no transformation).
    AbsoluteError,

    /// Tweedie regression.
    Tweedie { variance_power: f32 },

    /// Gamma regression (exp transform).
    Gamma,

    /// Poisson regression (exp transform).
    Poisson,

    // --- Binary Classification ---
    /// Binary logistic (sigmoid transform).
    BinaryLogistic,

    /// Binary logit raw (no transformation, return logits).
    BinaryLogitRaw,

    // --- Multiclass Classification ---
    /// Multiclass softmax (returns class index, not probabilities).
    MultiSoftmax { num_class: u32 },

    /// Multiclass softprob (softmax transform, returns probabilities).
    MultiSoftprob { num_class: u32 },

    // --- Ranking ---
    /// Pairwise ranking.
    RankPairwise,

    /// NDCG ranking.
    RankNdcg,

    /// MAP ranking.
    RankMap,

    // --- Survival ---
    /// Cox proportional hazards (exp transform).
    SurvivalCox,

    // --- Other ---
    /// Custom/unknown objective (no transformation).
    Custom,
}

impl Default for Objective {
    fn default() -> Self {
        Objective::SquaredError
    }
}

impl Objective {
    /// Apply the objective transformation to predictions in-place.
    pub fn transform(&self, output: &mut PredictionOutput) {
        match self {
            // No transformation
            Objective::SquaredError
            | Objective::AbsoluteError
            | Objective::BinaryLogitRaw
            | Objective::RankPairwise
            | Objective::RankNdcg
            | Objective::RankMap
            | Objective::Custom => {}

            // Sigmoid for binary classification
            Objective::BinaryLogistic => {
                for val in output.as_mut_slice() {
                    *val = sigmoid(*val);
                }
            }

            // Exp transform
            Objective::Gamma | Objective::Poisson | Objective::SurvivalCox => {
                for val in output.as_mut_slice() {
                    *val = val.exp();
                }
            }

            // Tweedie: exp transform
            Objective::Tweedie { .. } => {
                for val in output.as_mut_slice() {
                    *val = val.exp();
                }
            }

            // Softmax for multiclass probabilities
            Objective::MultiSoftprob { .. } => {
                let num_groups = output.num_groups();
                for row_idx in 0..output.num_rows() {
                    let row = output.row_mut(row_idx);
                    softmax_inplace(row);
                    // Ensure we processed correct number of groups
                    debug_assert_eq!(row.len(), num_groups);
                }
            }

            // Softmax then argmax for class prediction
            Objective::MultiSoftmax { .. } => {
                for row_idx in 0..output.num_rows() {
                    let row = output.row_mut(row_idx);
                    let class_idx = argmax(row);
                    // Store class index in first position
                    row[0] = class_idx as f32;
                    // Zero out the rest (or could resize output, but simpler to keep shape)
                    for val in row.iter_mut().skip(1) {
                        *val = 0.0;
                    }
                }
            }
        }
    }

    /// Whether this objective produces probabilities.
    pub fn produces_probabilities(&self) -> bool {
        matches!(
            self,
            Objective::BinaryLogistic | Objective::MultiSoftprob { .. }
        )
    }

    /// Whether this objective produces class indices.
    pub fn produces_class_indices(&self) -> bool {
        matches!(self, Objective::MultiSoftmax { .. })
    }
}

// =============================================================================
// Helper functions
// =============================================================================

/// Sigmoid function: 1 / (1 + exp(-x))
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Softmax in-place over a slice.
fn softmax_inplace(values: &mut [f32]) {
    if values.is_empty() {
        return;
    }

    // Find max for numerical stability
    let max_val = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    // Compute exp(x - max) and sum
    let mut sum = 0.0f32;
    for val in values.iter_mut() {
        *val = (*val - max_val).exp();
        sum += *val;
    }

    // Normalize
    if sum > 0.0 {
        for val in values.iter_mut() {
            *val /= sum;
        }
    }
}

/// Argmax: index of maximum value.
fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
        .unwrap_or(0)
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
    fn sigmoid_function() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!((sigmoid(2.0) - 0.8807971).abs() < 1e-5);
        assert!((sigmoid(-2.0) - 0.1192029).abs() < 1e-5);
    }

    #[test]
    fn softmax_function() {
        let mut values = vec![1.0, 2.0, 3.0];
        softmax_inplace(&mut values);

        // Should sum to 1
        let sum: f32 = values.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Largest input should have largest probability
        assert!(values[2] > values[1]);
        assert!(values[1] > values[0]);
    }

    #[test]
    fn argmax_function() {
        assert_eq!(argmax(&[1.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[3.0, 1.0, 2.0]), 0);
        assert_eq!(argmax(&[1.0, 2.0, 3.0]), 2);
    }

    #[test]
    fn objective_no_transform() {
        let mut output = PredictionOutput::new(vec![1.0, 2.0], 2, 1);
        Objective::SquaredError.transform(&mut output);
        assert_eq!(output.as_slice(), &[1.0, 2.0]);
    }

    #[test]
    fn objective_sigmoid() {
        let mut output = PredictionOutput::new(vec![0.0, 2.0], 2, 1);
        Objective::BinaryLogistic.transform(&mut output);

        assert!((output.row(0)[0] - 0.5).abs() < 1e-6);
        assert!((output.row(1)[0] - 0.8807971).abs() < 1e-5);
    }

    #[test]
    fn objective_softmax() {
        // 2 rows, 3 classes
        let mut output = PredictionOutput::new(vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0], 2, 3);
        Objective::MultiSoftprob { num_class: 3 }.transform(&mut output);

        // First row: should be proper probabilities
        let row0 = output.row(0);
        let sum: f32 = row0.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Second row: uniform (all zeros → equal probs)
        let row1 = output.row(1);
        let sum: f32 = row1.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        // All equal
        assert!((row1[0] - row1[1]).abs() < 1e-6);
        assert!((row1[1] - row1[2]).abs() < 1e-6);
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

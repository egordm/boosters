//! Schema types for model serialization.
//!
//! These types provide a stable serialization format independent of runtime types.
//! Schema types are separate from runtime types for:
//! - Forward/backward compatibility (schema can evolve independently)
//! - Validation during deserialization
//! - Clear migration paths between schema versions
//!
//! All schema types use `BTreeMap` for deterministic JSON output.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Task type for model output interpretation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskKindSchema {
    /// Regression task.
    Regression,
    /// Binary classification task.
    BinaryClassification,
    /// Multiclass classification task.
    MulticlassClassification,
    /// Ranking task.
    Ranking,
}

/// Feature type for metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureTypeSchema {
    /// Numeric feature.
    Numeric,
    /// Categorical feature.
    Categorical,
}

/// Model metadata schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetaSchema {
    /// Task type.
    pub task: TaskKindSchema,
    /// Number of features.
    pub num_features: usize,
    /// Number of classes (for multiclass).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_classes: Option<usize>,
    /// Feature names (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub feature_names: Option<Vec<String>>,
    /// Feature types (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub feature_types: Option<Vec<FeatureTypeSchema>>,
}

/// Leaf values schema (supports scalar and multi-output).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum LeafValuesSchema {
    /// Scalar leaves (one f64 per leaf).
    Scalar { values: Vec<f64> },
    /// Vector leaves (multiple f64 per leaf).
    Vector { values: Vec<Vec<f64>> },
}

/// Category mapping schema for categorical splits.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CategoriesSchema {
    /// Node indices that have category sets.
    pub node_indices: Vec<u32>,
    /// Category sets (one per node in node_indices).
    pub category_sets: Vec<Vec<u32>>,
}

/// Linear coefficients schema (for linear-in-leaves).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LinearCoefficientsSchema {
    /// Node indices that have linear coefficients.
    pub node_indices: Vec<u32>,
    /// Coefficient arrays (one per node in node_indices).
    pub coefficients: Vec<Vec<f64>>,
}

/// Tree schema (SoA layout).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeSchema {
    /// Number of nodes (internal + leaves).
    pub num_nodes: u32,
    /// Split feature index for each internal node (usize as u32).
    pub split_indices: Vec<u32>,
    /// Split threshold for each internal node.
    pub thresholds: Vec<f64>,
    /// Left child index for each internal node (0 = missing indicator).
    pub children_left: Vec<u32>,
    /// Right child index for each internal node (0 = missing indicator).
    pub children_right: Vec<u32>,
    /// Default direction (true = left) for each internal node.
    pub default_left: Vec<bool>,
    /// Leaf values.
    pub leaf_values: LeafValuesSchema,
    /// Optional category mappings for categorical splits.
    #[serde(default, skip_serializing_if = "is_categories_empty")]
    pub categories: CategoriesSchema,
    /// Optional linear coefficients for linear-in-leaves.
    #[serde(default, skip_serializing_if = "is_linear_coefficients_empty")]
    pub linear_coefficients: LinearCoefficientsSchema,
    /// Optional split gains for each internal node.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gains: Option<Vec<f64>>,
    /// Optional sample covers for each node.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub covers: Option<Vec<f64>>,
}

fn is_categories_empty(c: &CategoriesSchema) -> bool {
    c.node_indices.is_empty()
}

fn is_linear_coefficients_empty(c: &LinearCoefficientsSchema) -> bool {
    c.node_indices.is_empty()
}

/// Forest schema (collection of trees).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForestSchema {
    /// Trees in iteration order.
    pub trees: Vec<TreeSchema>,
    /// Tree group boundaries (for multi-output).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tree_groups: Option<Vec<usize>>,
    /// Number of output groups.
    pub n_groups: usize,
    /// Base score(s).
    pub base_score: Vec<f64>,
}

/// GBDT training configuration schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GBDTConfigSchema {
    /// Objective function (built-in only).
    pub objective: ObjectiveSchema,
    /// Evaluation metric.
    ///
    /// `None` means "no metric".
    pub metric: Option<MetricSchema>,

    /// Number of boosting rounds (trees).
    pub n_trees: u32,
    /// Learning rate.
    pub learning_rate: f64,

    /// Tree growth strategy.
    pub growth_strategy: GrowthStrategySchema,
    /// Maximum categories for one-hot encoding.
    pub max_onehot_cats: u32,

    /// L2 regularization term on leaf weights.
    pub lambda: f64,
    /// L1 regularization term on leaf weights.
    pub alpha: f64,
    /// Minimum sum of hessians required in a leaf.
    pub min_child_weight: f64,
    /// Minimum gain required to make a split.
    pub min_gain: f64,
    /// Minimum samples in a leaf.
    pub min_samples_leaf: u32,

    /// Row subsampling ratio.
    pub subsample: f64,
    /// Column subsampling ratio per tree.
    pub colsample_bytree: f64,
    /// Column subsampling ratio per level.
    pub colsample_bylevel: f64,

    /// Binning configuration.
    pub binning: BinningConfigSchema,
    /// Linear leaf configuration.
    pub linear_leaves: Option<LinearLeafConfigSchema>,
    /// Early stopping rounds.
    pub early_stopping_rounds: Option<u32>,

    /// Histogram cache size.
    pub cache_size: usize,
    /// Random seed.
    pub seed: u64,
    /// Verbosity level.
    pub verbosity: VerbositySchema,

    /// Additional parameters (key → value).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extra: BTreeMap<String, serde_json::Value>,
}

/// Objective schema (stable serialization format).
///
/// Note: this is intentionally separate from `training::Objective` to avoid
/// adding persistence-only types/representations into the training module.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ObjectiveSchema {
    SquaredLoss,
    AbsoluteLoss,
    LogisticLoss,
    HingeLoss,
    SoftmaxLoss { n_classes: usize },
    PinballLoss { alphas: Vec<f64> },
    PseudoHuberLoss { delta: f64 },
    PoissonLoss,
    Custom { name: String },
}

/// Metric schema (stable serialization format).
///
/// Note: this is intentionally separate from `training::Metric` to avoid
/// adding persistence-only types/representations into the training module.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MetricSchema {
    None,
    Rmse,
    Mae,
    Mape,
    LogLoss,
    Accuracy { threshold: f64 },
    MarginAccuracy,
    Auc,
    MulticlassLogLoss,
    MulticlassAccuracy,
    Quantile { alphas: Vec<f64> },
    Huber { delta: f64 },
    PoissonDeviance,
    Custom { name: String },
}

/// Tree growth strategy schema.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum GrowthStrategySchema {
    DepthWise { max_depth: u32 },
    LeafWise { max_leaves: u32 },
}

/// Binning configuration schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinningConfigSchema {
    pub max_bins: u32,
    pub sparsity_threshold: f64,
    pub enable_bundling: bool,
    pub max_categorical_cardinality: u32,
    pub sample_cnt: usize,
}

/// Linear leaf configuration schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearLeafConfigSchema {
    pub lambda: f64,
    pub alpha: f64,
    pub max_iterations: u32,
    pub tolerance: f64,
    pub min_samples: usize,
    pub coefficient_threshold: f64,
    pub max_features: usize,
}

/// Verbosity schema.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerbositySchema {
    Silent,
    Warning,
    Info,
    Debug,
}

/// Full GBDT model schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GBDTModelSchema {
    /// Model metadata.
    pub meta: ModelMetaSchema,
    /// Tree forest.
    pub forest: ForestSchema,
    /// Training configuration.
    pub config: GBDTConfigSchema,
}

impl GBDTModelSchema {
    /// Model type string.
    pub const MODEL_TYPE: &'static str = "gbdt";
}

/// GBLinear weight schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearWeightsSchema {
    /// Weight values in [group × feature] order (row-major).
    pub values: Vec<f64>,
    /// Number of features.
    pub num_features: usize,
    /// Number of output groups.
    pub num_groups: usize,
}

/// GBLinear training configuration schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GBLinearConfigSchema {
    /// Objective function (built-in only).
    pub objective: ObjectiveSchema,
    /// Evaluation metric (optional).
    pub metric: Option<MetricSchema>,

    /// Number of boosting rounds.
    pub n_rounds: u32,
    /// Learning rate.
    pub learning_rate: f64,

    /// L1 regularization.
    pub alpha: f64,
    /// L2 regularization.
    pub lambda: f64,

    /// Coordinate descent update strategy.
    pub update_strategy: UpdateStrategySchema,
    /// Feature selection strategy.
    pub feature_selector: FeatureSelectorSchema,

    /// Maximum per-coordinate Newton step (0 disables).
    pub max_delta_step: f64,
    /// Early stopping rounds.
    pub early_stopping_rounds: Option<u32>,
    /// Random seed.
    pub seed: u64,
    /// Verbosity.
    pub verbosity: VerbositySchema,

    /// Additional parameters.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extra: BTreeMap<String, serde_json::Value>,
}

/// Update strategy schema (GBLinear).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UpdateStrategySchema {
    Shotgun,
    Sequential,
}

/// Feature selector schema (GBLinear).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum FeatureSelectorSchema {
    Cyclic,
    Shuffle,
    Random,
    Greedy { top_k: usize },
    Thrifty { top_k: usize },
}

/// Full GBLinear model schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GBLinearModelSchema {
    /// Model metadata.
    pub meta: ModelMetaSchema,
    /// Linear weights.
    pub weights: LinearWeightsSchema,
    /// Base score(s).
    pub base_score: Vec<f64>,
    /// Training configuration.
    pub config: GBLinearConfigSchema,
}

impl GBLinearModelSchema {
    /// Model type string.
    pub const MODEL_TYPE: &'static str = "gblinear";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn task_kind_serde() {
        let task = TaskKindSchema::BinaryClassification;
        let json = serde_json::to_string(&task).unwrap();
        assert_eq!(json, r#""binary_classification""#);

        let parsed: TaskKindSchema = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, task);
    }

    #[test]
    fn leaf_values_tagged() {
        let scalar = LeafValuesSchema::Scalar {
            values: vec![1.0, 2.0, 3.0],
        };
        let json = serde_json::to_string(&scalar).unwrap();
        assert!(json.contains(r#""type":"scalar""#));

        let vector = LeafValuesSchema::Vector {
            values: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
        };
        let json = serde_json::to_string(&vector).unwrap();
        assert!(json.contains(r#""type":"vector""#));
    }

    #[test]
    fn model_meta_optional_fields() {
        let meta = ModelMetaSchema {
            task: TaskKindSchema::Regression,
            num_features: 10,
            num_classes: None,
            feature_names: None,
            feature_types: None,
        };

        let json = serde_json::to_string(&meta).unwrap();
        assert!(!json.contains("num_classes"));
        assert!(!json.contains("feature_names"));
        assert!(!json.contains("feature_types"));
    }

    #[test]
    fn gbdt_config_extra_deterministic() {
        let mut extra = BTreeMap::new();
        extra.insert("z_param".to_string(), serde_json::json!(1));
        extra.insert("a_param".to_string(), serde_json::json!(2));
        extra.insert("m_param".to_string(), serde_json::json!(3));

        let config = GBDTConfigSchema {
            objective: ObjectiveSchema::SquaredLoss,
            metric: None,
            n_trees: 100,
            learning_rate: 0.1,
            growth_strategy: GrowthStrategySchema::DepthWise { max_depth: 6 },
            max_onehot_cats: 4,
            lambda: 1.0,
            alpha: 0.0,
            min_child_weight: 1.0,
            min_gain: 0.0,
            min_samples_leaf: 1,
            subsample: 1.0,
            colsample_bytree: 1.0,
            colsample_bylevel: 1.0,
            binning: BinningConfigSchema {
                max_bins: 256,
                sparsity_threshold: 0.9,
                enable_bundling: true,
                max_categorical_cardinality: 0,
                sample_cnt: 200_000,
            },
            linear_leaves: None,
            early_stopping_rounds: None,
            cache_size: 8,
            seed: 42,
            verbosity: VerbositySchema::Silent,
            extra,
        };

        let json = serde_json::to_string(&config).unwrap();
        // BTreeMap ensures a_param < m_param < z_param in output
        let a_pos = json.find("a_param").unwrap();
        let m_pos = json.find("m_param").unwrap();
        let z_pos = json.find("z_param").unwrap();
        assert!(a_pos < m_pos && m_pos < z_pos);
    }

    #[test]
    fn categories_skip_when_empty() {
        let tree = TreeSchema {
            num_nodes: 1,
            split_indices: vec![],
            thresholds: vec![],
            children_left: vec![],
            children_right: vec![],
            default_left: vec![],
            leaf_values: LeafValuesSchema::Scalar { values: vec![1.0] },
            categories: CategoriesSchema::default(),
            linear_coefficients: LinearCoefficientsSchema::default(),
            gains: None,
            covers: None,
        };

        let json = serde_json::to_string(&tree).unwrap();
        assert!(!json.contains("categories"));
        assert!(!json.contains("linear_coefficients"));
    }
}

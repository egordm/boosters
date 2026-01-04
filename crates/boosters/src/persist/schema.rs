//! Schema types for model serialization.
//!
//! These types provide a stable serialization format independent of runtime types.
//! Schema types are separate from runtime types for:
//! - Forward/backward compatibility (schema can evolve independently)
//! - Validation during deserialization
//! - Clear migration paths between schema versions
//!
//! All schema types use `BTreeMap` for deterministic JSON output.

use serde::{Deserialize, Serialize};

use crate::model::OutputTransform;

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
    /// Number of features.
    pub num_features: usize,
    /// Number of output groups.
    #[serde(default = "default_num_groups")]
    pub num_groups: usize,
    /// Feature names (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub feature_names: Option<Vec<String>>,
    /// Feature types (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub feature_types: Option<Vec<FeatureTypeSchema>>,
    /// Objective name (for debugging/reproducibility).
    ///
    /// Added in schema v3. Not used for inference.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub objective_name: Option<String>,
}

fn default_num_groups() -> usize {
    1
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

/// Full GBDT model schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GBDTModelSchema {
    /// Model metadata.
    pub meta: ModelMetaSchema,
    /// Tree forest.
    pub forest: ForestSchema,
    /// Output transform for inference.
    pub output_transform: OutputTransform,
}

impl GBDTModelSchema {
    /// Model type string.
    pub const MODEL_TYPE: &'static str = "gbdt";
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
    /// Output transform for inference.
    pub output_transform: OutputTransform,
}

impl GBLinearModelSchema {
    /// Model type string.
    pub const MODEL_TYPE: &'static str = "gblinear";
}

#[cfg(test)]
mod tests {
    use super::*;

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
            num_features: 10,
            num_groups: 1,
            feature_names: None,
            feature_types: None,
            objective_name: None,
        };

        let json = serde_json::to_string(&meta).unwrap();
        assert!(!json.contains("feature_names"));
        assert!(!json.contains("feature_types"));
        assert!(!json.contains("objective_name"));
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

//! Payload structures for native storage format.
//!
//! These structs are specifically designed for serialization with Postcard.
//! They mirror the runtime types but are optimized for compact binary storage.

use serde::{Deserialize, Serialize};

// ============================================================================
// Top-Level Payload
// ============================================================================

/// Version-tagged payload enum for forward compatibility.
///
/// New format versions add new variants rather than modifying existing ones.
/// Older readers can detect unsupported versions by the enum discriminant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Payload {
    /// Version 1 payload format.
    V1(PayloadV1),
}

/// Version 1 payload structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PayloadV1 {
    /// Model metadata.
    pub metadata: ModelMetadata,
    /// Model-specific payload.
    pub model: ModelPayload,
}

// ============================================================================
// Metadata
// ============================================================================

/// Metadata common to all model types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Number of input features.
    pub num_features: u32,
    /// Number of output groups.
    pub num_groups: u32,
    /// Base score for each output group.
    pub base_scores: Vec<f32>,
    /// Objective function name (e.g., "reg:squarederror").
    pub objective: Option<String>,
    /// Feature names (optional).
    pub feature_names: Option<Vec<String>>,
    /// Additional key-value attributes.
    pub attributes: Vec<(String, String)>,
}

impl Default for ModelMetadata {
    fn default() -> Self {
        Self {
            num_features: 0,
            num_groups: 1,
            base_scores: vec![0.0],
            objective: None,
            feature_names: None,
            attributes: Vec::new(),
        }
    }
}

// ============================================================================
// Model Payloads
// ============================================================================

/// Model-specific payload variant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelPayload {
    /// Gradient-boosted decision tree payload.
    Gbdt(GbdtPayload),
    /// Gradient-boosted linear model payload.
    GbLinear(GbLinearPayload),
}

/// GBDT (tree ensemble) payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GbdtPayload {
    /// Forest of trees.
    pub forest: ForestPayload,
    /// Categorical split data (optional).
    pub categories: Option<CategoriesPayload>,
    /// Linear leaf coefficients (optional).
    pub linear_leaves: Option<LinearLeavesPayload>,
}

/// Forest of decision trees.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForestPayload {
    /// Number of trees.
    pub num_trees: u32,
    /// Group assignment for each tree.
    pub tree_groups: Vec<u32>,
    /// Individual tree payloads.
    pub trees: Vec<TreePayload>,
}

/// Single decision tree payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreePayload {
    /// Number of nodes.
    pub num_nodes: u32,
    /// Split feature indices (one per node).
    pub split_features: Vec<u32>,
    /// Split thresholds (one per node, 0.0 for categorical/leaf).
    pub thresholds: Vec<f32>,
    /// Left child indices (one per node, 0 for leaves).
    pub left_children: Vec<u32>,
    /// Right child indices (one per node, 0 for leaves).
    pub right_children: Vec<u32>,
    /// Default direction for missing values (one per node).
    pub default_left: Vec<bool>,
    /// Whether each node is a leaf.
    pub is_leaf: Vec<bool>,
    /// Leaf values (one per node, 0.0 for internal nodes).
    pub leaf_values: Vec<f32>,
    /// Split types (0=numeric, 1=categorical).
    pub split_types: Vec<u8>,
    /// Optional: gain at each split (for explainability).
    pub gains: Option<Vec<f32>>,
    /// Optional: cover/hessian sum at each node (for explainability).
    pub covers: Option<Vec<f32>>,
}

/// Categorical split storage payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoriesPayload {
    /// Tree index for each category segment.
    pub tree_indices: Vec<u32>,
    /// Node index within tree for each segment.
    pub node_indices: Vec<u32>,
    /// Start offset in bitsets for each segment.
    pub starts: Vec<u32>,
    /// Size (in u32 words) for each segment.
    pub sizes: Vec<u32>,
    /// Packed bitset data for all categorical splits.
    pub bitsets: Vec<u32>,
}

/// Linear leaf coefficients payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearLeavesPayload {
    /// Tree index for each linear leaf.
    pub tree_indices: Vec<u32>,
    /// Node index within tree for each linear leaf.
    pub node_indices: Vec<u32>,
    /// Intercept for each linear leaf.
    pub intercepts: Vec<f32>,
    /// Start offset in coefficients arrays.
    pub starts: Vec<u32>,
    /// Number of coefficients for each linear leaf.
    pub sizes: Vec<u32>,
    /// Feature indices for all linear terms (packed).
    pub feature_indices: Vec<u32>,
    /// Coefficient values for all linear terms (packed).
    pub coefficients: Vec<f32>,
}

/// GBLinear (linear booster) payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GbLinearPayload {
    /// Flat weight array: (num_features + 1) * num_groups
    /// Layout: feature-major, group-minor. Last row is bias.
    pub weights: Vec<f32>,
    /// Number of boosting rounds (for metadata).
    pub num_boosted_rounds: u32,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn payload_serializes() {
        let payload = Payload::V1(PayloadV1 {
            metadata: ModelMetadata {
                num_features: 10,
                num_groups: 1,
                base_scores: vec![0.5],
                objective: Some("reg:squarederror".to_string()),
                feature_names: None,
                attributes: vec![],
            },
            model: ModelPayload::GbLinear(GbLinearPayload {
                weights: vec![0.1, 0.2, 0.3],
                num_boosted_rounds: 100,
            }),
        });

        // Should serialize without panic
        let bytes = postcard::to_allocvec(&payload).unwrap();
        assert!(!bytes.is_empty());

        // Should deserialize back
        let decoded: Payload = postcard::from_bytes(&bytes).unwrap();
        match decoded {
            Payload::V1(v1) => {
                assert_eq!(v1.metadata.num_features, 10);
                match v1.model {
                    ModelPayload::GbLinear(linear) => {
                        assert_eq!(linear.weights, vec![0.1, 0.2, 0.3]);
                    }
                    _ => panic!("wrong model type"),
                }
            }
        }
    }

    #[test]
    fn tree_payload_roundtrip() {
        let tree = TreePayload {
            num_nodes: 3,
            split_features: vec![0, 0, 0],
            thresholds: vec![0.5, 0.0, 0.0],
            left_children: vec![1, 0, 0],
            right_children: vec![2, 0, 0],
            default_left: vec![true, false, false],
            is_leaf: vec![false, true, true],
            leaf_values: vec![0.0, 1.0, 2.0],
            split_types: vec![0, 0, 0],
            gains: None,
            covers: None,
        };

        let bytes = postcard::to_allocvec(&tree).unwrap();
        let decoded: TreePayload = postcard::from_bytes(&bytes).unwrap();

        assert_eq!(decoded.num_nodes, 3);
        assert_eq!(decoded.leaf_values, vec![0.0, 1.0, 2.0]);
    }

    #[test]
    fn gbdt_payload_with_categories() {
        let payload = GbdtPayload {
            forest: ForestPayload {
                num_trees: 1,
                tree_groups: vec![0],
                trees: vec![TreePayload {
                    num_nodes: 3,
                    split_features: vec![0, 0, 0],
                    thresholds: vec![0.0, 0.0, 0.0],
                    left_children: vec![1, 0, 0],
                    right_children: vec![2, 0, 0],
                    default_left: vec![true, false, false],
                    is_leaf: vec![false, true, true],
                    leaf_values: vec![0.0, -1.0, 1.0],
                    split_types: vec![1, 0, 0], // Categorical root
                    gains: None,
                    covers: None,
                }],
            },
            categories: Some(CategoriesPayload {
                tree_indices: vec![0],
                node_indices: vec![0],
                starts: vec![0],
                sizes: vec![1],
                bitsets: vec![0b1010], // Categories 1,3 go right
            }),
            linear_leaves: None,
        };

        let bytes = postcard::to_allocvec(&payload).unwrap();
        let decoded: GbdtPayload = postcard::from_bytes(&bytes).unwrap();

        assert!(decoded.categories.is_some());
        assert_eq!(decoded.categories.unwrap().bitsets, vec![0b1010]);
    }
}

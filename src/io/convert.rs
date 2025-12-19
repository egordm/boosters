//! Conversion between runtime types and payload types.
//!
//! This module provides bidirectional conversion between the runtime
//! model types (`Forest`, `LinearModel`) and serialization payloads.
//!
//! # High-Level API
//!
//! For most use cases, use the convenience methods on `Forest` and `LinearModel`:
//!
//! ```ignore
//! use boosters::repr::gbdt::Forest;
//!
//! // Save to file
//! forest.save("model.bstr")?;
//!
//! // Load from file
//! let forest = Forest::load("model.bstr")?;
//!
//! // Or work with bytes directly
//! let bytes = forest.to_bytes()?;
//! let restored = Forest::from_bytes(&bytes)?;
//! ```

use std::path::Path;

use crate::io::native::{DeserializeError, ModelType, NativeCodec, SerializeError};
use crate::io::payload::{
    CategoriesPayload, ForestPayload, GbLinearPayload, GbdtPayload, LinearLeavesPayload,
    ModelMetadata, ModelPayload, Payload, PayloadV1, TreePayload,
};
use crate::repr::gbdt::categories::CategoriesStorage;
use crate::repr::gbdt::coefficients::{LeafCoefficients, LeafCoefficientsBuilder};
use crate::repr::gbdt::node::SplitType;
use crate::repr::gbdt::tree::TreeView;
use crate::repr::gbdt::{Forest, ScalarLeaf, Tree};
use crate::repr::gblinear::LinearModel;

// ============================================================================
// Forest Serialization API
// ============================================================================

impl Forest<ScalarLeaf> {
    /// Save the forest to a file in native `.bstr` format.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the output file
    ///
    /// # Example
    ///
    /// ```ignore
    /// forest.save("model.bstr")?;
    /// ```
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), SerializeError> {
        let bytes = self.to_bytes()?;
        std::fs::write(path, bytes)?;
        Ok(())
    }

    /// Load a forest from a file in native `.bstr` format.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the model file
    ///
    /// # Example
    ///
    /// ```ignore
    /// let forest = Forest::load("model.bstr")?;
    /// ```
    pub fn load(path: impl AsRef<Path>) -> Result<Self, DeserializeError> {
        let bytes = std::fs::read(path)?;
        Self::from_bytes(&bytes)
    }

    /// Serialize the forest to bytes.
    ///
    /// Returns the complete serialized model including header.
    pub fn to_bytes(&self) -> Result<Vec<u8>, SerializeError> {
        let payload = Payload::from_forest(self);
        let codec = NativeCodec::new();
        codec.serialize(
            ModelType::Gbdt,
            payload.num_features(),
            payload.num_groups(),
            &payload,
        )
    }

    /// Deserialize a forest from bytes.
    ///
    /// The bytes must include the complete model with header.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, DeserializeError> {
        let codec = NativeCodec::new();
        let (header, payload): (_, Payload) = codec.deserialize(bytes)?;
        if header.model_type != ModelType::Gbdt {
            return Err(DeserializeError::TypeMismatch {
                expected: ModelType::Gbdt,
                actual: header.model_type,
            });
        }
        payload.into_forest()
    }
}

// ============================================================================
// LinearModel Serialization API
// ============================================================================

impl LinearModel {
    /// Save the model to a file in native `.bstr` format.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the output file
    /// * `num_rounds` - Number of training rounds (for metadata)
    ///
    /// # Example
    ///
    /// ```ignore
    /// model.save("linear.bstr", 100)?;
    /// ```
    pub fn save(&self, path: impl AsRef<Path>, num_rounds: u32) -> Result<(), SerializeError> {
        let bytes = self.to_bytes(num_rounds)?;
        std::fs::write(path, bytes)?;
        Ok(())
    }

    /// Load a model from a file in native `.bstr` format.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the model file
    ///
    /// # Example
    ///
    /// ```ignore
    /// let model = LinearModel::load("linear.bstr")?;
    /// ```
    pub fn load(path: impl AsRef<Path>) -> Result<Self, DeserializeError> {
        let bytes = std::fs::read(path)?;
        Self::from_bytes(&bytes)
    }

    /// Serialize the model to bytes.
    ///
    /// # Arguments
    ///
    /// * `num_rounds` - Number of training rounds (for metadata)
    pub fn to_bytes(&self, num_rounds: u32) -> Result<Vec<u8>, SerializeError> {
        let payload = Payload::from_linear_model(self, num_rounds);
        let codec = NativeCodec::new();
        codec.serialize(
            ModelType::GbLinear,
            payload.num_features(),
            payload.num_groups(),
            &payload,
        )
    }

    /// Deserialize a model from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, DeserializeError> {
        let codec = NativeCodec::new();
        let (header, payload): (_, Payload) = codec.deserialize(bytes)?;
        if header.model_type != ModelType::GbLinear {
            return Err(DeserializeError::TypeMismatch {
                expected: ModelType::GbLinear,
                actual: header.model_type,
            });
        }
        payload.into_linear_model()
    }
}

// ============================================================================
// Forest -> Payload Conversion
// ============================================================================

impl Payload {
    /// Create a payload from a Forest.
    ///
    /// Serializes the forest into the V1 payload format.
    pub fn from_forest(forest: &Forest<ScalarLeaf>) -> Self {
        let n_groups = forest.n_groups();

        // Build forest payload
        let trees: Vec<TreePayload> = forest.trees().map(tree_to_payload).collect();

        let forest_payload = ForestPayload {
            num_trees: forest.n_trees() as u32,
            tree_groups: (0..forest.n_trees())
                .map(|i| forest.tree_group(i))
                .collect(),
            trees,
        };

        // Collect categories across all trees
        let categories = collect_categories(forest);

        // Collect linear leaves across all trees
        let linear_leaves = collect_linear_leaves(forest);

        let gbdt_payload = GbdtPayload {
            forest: forest_payload,
            categories,
            linear_leaves,
        };

        let metadata = ModelMetadata {
            num_features: 0, // TODO: get from forest if we track it
            num_groups: n_groups,
            base_scores: forest.base_score().to_vec(),
            objective: None,
            feature_names: None,
            attributes: Vec::new(),
        };

        Payload::V1(PayloadV1 {
            metadata,
            model: ModelPayload::Gbdt(gbdt_payload),
        })
    }

    /// Create a payload from a LinearModel.
    pub fn from_linear_model(model: &LinearModel, num_boosted_rounds: u32) -> Self {
        let linear_payload = GbLinearPayload {
            weights: model.weights().to_vec(),
            num_boosted_rounds,
        };

        let metadata = ModelMetadata {
            num_features: model.num_features() as u32,
            num_groups: model.num_groups() as u32,
            base_scores: vec![0.0; model.num_groups()], // Linear model has bias in weights
            objective: None,
            feature_names: None,
            attributes: Vec::new(),
        };

        Payload::V1(PayloadV1 {
            metadata,
            model: ModelPayload::GbLinear(linear_payload),
        })
    }

    /// Get the model type from this payload.
    pub fn model_type(&self) -> ModelType {
        match self {
            Payload::V1(v1) => match &v1.model {
                ModelPayload::Gbdt(_) => ModelType::Gbdt,
                ModelPayload::GbLinear(_) => ModelType::GbLinear,
            },
        }
    }

    /// Get the number of features from this payload.
    pub fn num_features(&self) -> u32 {
        match self {
            Payload::V1(v1) => v1.metadata.num_features,
        }
    }

    /// Get the number of groups from this payload.
    pub fn num_groups(&self) -> u32 {
        match self {
            Payload::V1(v1) => v1.metadata.num_groups,
        }
    }

    /// Extract a Forest from this payload.
    ///
    /// Returns an error if the payload doesn't contain a GBDT model.
    pub fn into_forest(self) -> Result<Forest<ScalarLeaf>, DeserializeError> {
        match self {
            Payload::V1(v1) => match v1.model {
                ModelPayload::Gbdt(gbdt) => {
                    payload_to_forest(gbdt, v1.metadata)
                }
                ModelPayload::GbLinear(_) => Err(DeserializeError::TypeMismatch {
                    expected: ModelType::Gbdt,
                    actual: ModelType::GbLinear,
                }),
            },
        }
    }

    /// Extract a LinearModel from this payload.
    ///
    /// Returns an error if the payload doesn't contain a GBLinear model.
    pub fn into_linear_model(self) -> Result<LinearModel, DeserializeError> {
        match self {
            Payload::V1(v1) => match v1.model {
                ModelPayload::GbLinear(linear) => {
                    Ok(LinearModel::new(
                        linear.weights.into_boxed_slice(),
                        v1.metadata.num_features as usize,
                        v1.metadata.num_groups as usize,
                    ))
                }
                ModelPayload::Gbdt(_) => Err(DeserializeError::TypeMismatch {
                    expected: ModelType::GbLinear,
                    actual: ModelType::Gbdt,
                }),
            },
        }
    }
}

// ============================================================================
// Tree Conversion Helpers
// ============================================================================

fn tree_to_payload(tree: &Tree<ScalarLeaf>) -> TreePayload {
    let n_nodes = tree.n_nodes();

    let mut split_features = Vec::with_capacity(n_nodes);
    let mut thresholds = Vec::with_capacity(n_nodes);
    let mut left_children = Vec::with_capacity(n_nodes);
    let mut right_children = Vec::with_capacity(n_nodes);
    let mut default_left = Vec::with_capacity(n_nodes);
    let mut is_leaf = Vec::with_capacity(n_nodes);
    let mut leaf_values = Vec::with_capacity(n_nodes);
    let mut split_types = Vec::with_capacity(n_nodes);

    use crate::repr::gbdt::TreeView;

    for node_idx in 0..n_nodes as u32 {
        split_features.push(tree.split_index(node_idx));
        thresholds.push(tree.split_threshold(node_idx));
        left_children.push(tree.left_child(node_idx));
        right_children.push(tree.right_child(node_idx));
        default_left.push(tree.default_left(node_idx));
        is_leaf.push(tree.is_leaf(node_idx));
        leaf_values.push(tree.leaf_value(node_idx).0);
        split_types.push(tree.split_type(node_idx) as u8);
    }

    TreePayload {
        num_nodes: n_nodes as u32,
        split_features,
        thresholds,
        left_children,
        right_children,
        default_left,
        is_leaf,
        leaf_values,
        split_types,
        gains: None,  // TODO: add when we have gains on Tree
        covers: None, // TODO: add when we have covers on Tree
    }
}

fn payload_to_tree(
    payload: TreePayload,
    categories: Option<&CategoriesStorage>,
    linear_coeffs: Option<&LeafCoefficients>,
) -> Tree<ScalarLeaf> {
    let leaf_values: Vec<ScalarLeaf> = payload.leaf_values.into_iter().map(ScalarLeaf).collect();

    let split_types: Vec<SplitType> = payload
        .split_types
        .into_iter()
        .map(SplitType::from)
        .collect();

    // Use the provided categories or empty
    let cat_storage = categories.cloned().unwrap_or_else(CategoriesStorage::empty);

    // Use the provided linear coefficients or empty
    let leaf_coeffs = linear_coeffs.cloned().unwrap_or_else(LeafCoefficients::empty);

    Tree::with_linear_leaves(
        payload.split_features,
        payload.thresholds,
        payload.left_children,
        payload.right_children,
        payload.default_left,
        payload.is_leaf,
        leaf_values,
        split_types,
        cat_storage,
        leaf_coeffs,
    )
}

// ============================================================================
// Categories Collection
// ============================================================================

fn collect_categories(forest: &Forest<ScalarLeaf>) -> Option<CategoriesPayload> {
    let mut tree_indices = Vec::new();
    let mut node_indices = Vec::new();
    let mut starts = Vec::new();
    let mut sizes = Vec::new();
    let mut bitsets = Vec::new();

    for (tree_idx, tree) in forest.trees().enumerate() {
        if tree.has_categorical() {
            let cat_storage = tree.categories();
            let segments = cat_storage.segments();
            let tree_bitsets = cat_storage.bitsets();

            for (node_idx, &(start, size)) in segments.iter().enumerate() {
                if size > 0 {
                    tree_indices.push(tree_idx as u32);
                    node_indices.push(node_idx as u32);
                    starts.push(bitsets.len() as u32);
                    sizes.push(size);
                    bitsets.extend_from_slice(
                        &tree_bitsets[start as usize..(start + size) as usize],
                    );
                }
            }
        }
    }

    if tree_indices.is_empty() {
        None
    } else {
        Some(CategoriesPayload {
            tree_indices,
            node_indices,
            starts,
            sizes,
            bitsets,
        })
    }
}

fn rebuild_categories_for_tree(
    tree_idx: usize,
    num_nodes: usize,
    categories: &CategoriesPayload,
) -> CategoriesStorage {
    // Find all segments for this tree
    let mut segments = vec![(0u32, 0u32); num_nodes];
    let mut local_bitsets = Vec::new();

    for i in 0..categories.tree_indices.len() {
        if categories.tree_indices[i] as usize == tree_idx {
            let node_idx = categories.node_indices[i] as usize;
            let start = categories.starts[i] as usize;
            let size = categories.sizes[i];

            segments[node_idx] = (local_bitsets.len() as u32, size);
            local_bitsets.extend_from_slice(
                &categories.bitsets[start..start + size as usize],
            );
        }
    }

    if local_bitsets.is_empty() {
        CategoriesStorage::empty()
    } else {
        CategoriesStorage::new(local_bitsets, segments)
    }
}

// ============================================================================
// Linear Leaves Collection
// ============================================================================

fn collect_linear_leaves(forest: &Forest<ScalarLeaf>) -> Option<LinearLeavesPayload> {
    let mut tree_indices = Vec::new();
    let mut node_indices = Vec::new();
    let mut intercepts = Vec::new();
    let mut starts = Vec::new();
    let mut sizes = Vec::new();
    let mut feature_indices = Vec::new();
    let mut coefficients = Vec::new();

    for (tree_idx, tree) in forest.trees().enumerate() {
        if tree.has_linear_leaves() {
            use crate::repr::gbdt::TreeView;
            for node_idx in 0..tree.n_nodes() as u32 {
                if let Some((feats, coefs)) = tree.leaf_terms(node_idx) {
                    tree_indices.push(tree_idx as u32);
                    node_indices.push(node_idx);
                    intercepts.push(tree.leaf_intercept(node_idx));
                    starts.push(feature_indices.len() as u32);
                    sizes.push(feats.len() as u32);
                    feature_indices.extend_from_slice(feats);
                    coefficients.extend(coefs.iter().copied());
                }
            }
        }
    }

    if tree_indices.is_empty() {
        None
    } else {
        Some(LinearLeavesPayload {
            tree_indices,
            node_indices,
            intercepts,
            starts,
            sizes,
            feature_indices,
            coefficients,
        })
    }
}

fn rebuild_linear_leaves_for_tree(
    tree_idx: usize,
    num_nodes: usize,
    linear_leaves: &LinearLeavesPayload,
) -> LeafCoefficients {
    let mut builder = LeafCoefficientsBuilder::new();

    for i in 0..linear_leaves.tree_indices.len() {
        if linear_leaves.tree_indices[i] as usize == tree_idx {
            let node_idx = linear_leaves.node_indices[i];
            let intercept = linear_leaves.intercepts[i];
            let start = linear_leaves.starts[i] as usize;
            let size = linear_leaves.sizes[i] as usize;

            let feats = &linear_leaves.feature_indices[start..start + size];
            let coefs = &linear_leaves.coefficients[start..start + size];

            builder.add(node_idx, feats, intercept, coefs);
        }
    }

    builder.build(num_nodes)
}

// ============================================================================
// Forest Reconstruction
// ============================================================================

fn payload_to_forest(
    gbdt: GbdtPayload,
    metadata: ModelMetadata,
) -> Result<Forest<ScalarLeaf>, DeserializeError> {
    let n_groups = metadata.num_groups;
    let base_scores = metadata.base_scores;

    let mut forest = Forest::new(n_groups).with_base_score(base_scores);

    for (tree_idx, (tree_payload, &group)) in gbdt
        .forest
        .trees
        .into_iter()
        .zip(gbdt.forest.tree_groups.iter())
        .enumerate()
    {
        let num_nodes = tree_payload.num_nodes as usize;

        // Rebuild categories for this tree if present
        let categories = gbdt.categories.as_ref().map(|c| {
            rebuild_categories_for_tree(tree_idx, num_nodes, c)
        });

        // Rebuild linear leaves for this tree if present
        let linear_leaves = gbdt.linear_leaves.as_ref().map(|l| {
            rebuild_linear_leaves_for_tree(tree_idx, num_nodes, l)
        });

        let tree = payload_to_tree(tree_payload, categories.as_ref(), linear_leaves.as_ref());
        forest.push_tree(tree, group);
    }

    Ok(forest)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forest_roundtrip_simple() {
        // Build a simple forest
        let tree = crate::scalar_tree! {
            0 => num(0, 0.5, L) -> 1, 2,
            1 => leaf(1.0),
            2 => leaf(2.0),
        };

        let mut forest = Forest::for_regression().with_base_score(vec![0.5]);
        forest.push_tree(tree, 0);

        // Convert to payload
        let payload = Payload::from_forest(&forest);

        // Convert back
        let restored = payload.into_forest().unwrap();

        // Verify
        assert_eq!(restored.n_trees(), 1);
        assert_eq!(restored.n_groups(), 1);
        assert_eq!(restored.base_score(), &[0.5]);

        // Predictions should match
        let pred1 = forest.predict_row(&[0.3]);
        let pred2 = restored.predict_row(&[0.3]);
        assert_eq!(pred1, pred2);

        let pred1 = forest.predict_row(&[0.7]);
        let pred2 = restored.predict_row(&[0.7]);
        assert_eq!(pred1, pred2);
    }

    #[test]
    fn forest_roundtrip_multiple_trees() {
        let tree1 = crate::scalar_tree! {
            0 => num(0, 0.5, L) -> 1, 2,
            1 => leaf(1.0),
            2 => leaf(2.0),
        };

        let tree2 = crate::scalar_tree! {
            0 => num(1, 0.3, R) -> 1, 2,
            1 => leaf(0.5),
            2 => leaf(1.5),
        };

        let mut forest = Forest::for_regression().with_base_score(vec![0.0]);
        forest.push_tree(tree1, 0);
        forest.push_tree(tree2, 0);

        let payload = Payload::from_forest(&forest);
        let restored = payload.into_forest().unwrap();

        assert_eq!(restored.n_trees(), 2);

        // Test predictions
        let test_row = [0.3, 0.5];
        let pred1 = forest.predict_row(&test_row);
        let pred2 = restored.predict_row(&test_row);
        assert_eq!(pred1, pred2);
    }

    #[test]
    fn forest_roundtrip_categorical() {
        let tree = crate::scalar_tree! {
            0 => cat(0, [1, 3], L) -> 1, 2,
            1 => leaf(-1.0),
            2 => leaf(1.0),
        };

        let mut forest = Forest::for_regression();
        forest.push_tree(tree, 0);

        let payload = Payload::from_forest(&forest);
        let restored = payload.into_forest().unwrap();

        // Categories 1,3 go right
        assert_eq!(forest.predict_row(&[0.0]), restored.predict_row(&[0.0]));
        assert_eq!(forest.predict_row(&[1.0]), restored.predict_row(&[1.0]));
        assert_eq!(forest.predict_row(&[2.0]), restored.predict_row(&[2.0]));
        assert_eq!(forest.predict_row(&[3.0]), restored.predict_row(&[3.0]));
    }

    #[test]
    fn linear_model_roundtrip() {
        let weights = vec![0.5, 0.3, 0.1].into_boxed_slice();
        let model = LinearModel::new(weights, 2, 1);

        let payload = Payload::from_linear_model(&model, 100);
        let restored = payload.into_linear_model().unwrap();

        assert_eq!(restored.num_features(), 2);
        assert_eq!(restored.num_groups(), 1);
        assert_eq!(restored.weight(0, 0), 0.5);
        assert_eq!(restored.weight(1, 0), 0.3);
        assert_eq!(restored.bias(0), 0.1);
    }

    #[test]
    fn type_mismatch_error() {
        let weights = vec![0.5, 0.3, 0.1].into_boxed_slice();
        let model = LinearModel::new(weights, 2, 1);
        let payload = Payload::from_linear_model(&model, 100);

        // Try to extract as forest
        let result = payload.into_forest();
        assert!(matches!(result, Err(DeserializeError::TypeMismatch { .. })));
    }

    // ========================================================================
    // High-Level API Tests (to_bytes/from_bytes, save/load)
    // ========================================================================

    #[test]
    fn forest_bytes_roundtrip() {
        let tree = crate::scalar_tree! {
            0 => num(0, 0.5, L) -> 1, 2,
            1 => leaf(1.0),
            2 => leaf(2.0),
        };

        let mut forest = Forest::for_regression().with_base_score(vec![0.5]);
        forest.push_tree(tree, 0);

        // Serialize to bytes
        let bytes = forest.to_bytes().unwrap();

        // Deserialize from bytes
        let restored = Forest::from_bytes(&bytes).unwrap();

        // Verify predictions match
        assert_eq!(forest.predict_row(&[0.3]), restored.predict_row(&[0.3]));
        assert_eq!(forest.predict_row(&[0.7]), restored.predict_row(&[0.7]));
    }

    #[test]
    fn linear_model_bytes_roundtrip() {
        let weights = vec![0.5, 0.3, 0.1].into_boxed_slice();
        let model = LinearModel::new(weights, 2, 1);

        // Serialize to bytes
        let bytes = model.to_bytes(100).unwrap();

        // Deserialize from bytes
        let restored = LinearModel::from_bytes(&bytes).unwrap();

        assert_eq!(restored.num_features(), 2);
        assert_eq!(restored.num_groups(), 1);
        assert_eq!(restored.weight(0, 0), 0.5);
    }

    #[test]
    fn forest_save_load() {
        use std::fs;

        let tree = crate::scalar_tree! {
            0 => num(0, 0.5, L) -> 1, 2,
            1 => leaf(1.0),
            2 => leaf(2.0),
        };

        let mut forest = Forest::for_regression().with_base_score(vec![0.5]);
        forest.push_tree(tree, 0);

        // Create a temp file path
        let path = std::env::temp_dir().join("boosters_test_forest.bstr");

        // Save
        forest.save(&path).unwrap();

        // Load
        let restored = Forest::load(&path).unwrap();

        // Cleanup
        fs::remove_file(&path).ok();

        // Verify
        assert_eq!(forest.predict_row(&[0.3]), restored.predict_row(&[0.3]));
    }

    #[test]
    fn linear_model_save_load() {
        use std::fs;

        let weights = vec![0.5, 0.3, 0.1].into_boxed_slice();
        let model = LinearModel::new(weights, 2, 1);

        // Create a temp file path
        let path = std::env::temp_dir().join("boosters_test_linear.bstr");

        // Save
        model.save(&path, 100).unwrap();

        // Load
        let restored = LinearModel::load(&path).unwrap();

        // Cleanup
        fs::remove_file(&path).ok();

        // Verify
        assert_eq!(restored.weight(0, 0), 0.5);
    }

    #[test]
    fn type_mismatch_high_level() {
        let weights = vec![0.5, 0.3, 0.1].into_boxed_slice();
        let model = LinearModel::new(weights, 2, 1);
        let bytes = model.to_bytes(100).unwrap();

        // Try to load as Forest
        let result = Forest::from_bytes(&bytes);
        assert!(matches!(result, Err(DeserializeError::TypeMismatch { .. })));
    }
}

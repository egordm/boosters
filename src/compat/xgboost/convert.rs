//! Conversion from XGBoost JSON types to native booste-rs types.

use crate::forest::SoAForest;
use crate::linear::LinearModel;
use crate::model::Booster;
use crate::trees::{ScalarLeaf, TreeBuilder};

use super::json::{GradientBooster, ModelTrees, Tree, XgbModel};

/// Error type for XGBoost model conversion.
#[derive(Debug, thiserror::Error)]
pub enum ConversionError {
    #[error("tree {0} has no nodes")]
    EmptyTree(usize),
    #[error("invalid node index in tree {tree}: node {node} references child {child} but tree has {num_nodes} nodes")]
    InvalidNodeIndex {
        tree: usize,
        node: usize,
        child: i32,
        num_nodes: usize,
    },
    #[error("gblinear weights length {actual} doesn't match (num_features + 1) * num_groups = {expected}")]
    InvalidLinearWeights { actual: usize, expected: usize },
}

/// Convert base_score from probability space to margin space based on objective.
///
/// XGBoost stores base_score in probability/original space in JSON, but the predictor
/// uses margin space. This replicates XGBoost's `ProbToMargin` logic.
fn prob_to_margin(base_score: f32, objective: &str) -> f32 {
    match objective {
        // Logistic objectives: use logit transform
        // logit(p) = log(p / (1 - p)) = -log(1/p - 1)
        "binary:logistic" | "reg:logistic" => {
            // Clamp to avoid infinity
            let p = base_score.clamp(1e-7, 1.0 - 1e-7);
            (p / (1.0 - p)).ln()
        }
        // Gamma/Tweedie: use log transform
        "reg:gamma" | "reg:tweedie" => base_score.max(1e-7).ln(),
        // For regression and other objectives, base_score is already in margin space
        _ => base_score,
    }
}

impl XgbModel {
    /// Convert to a native `SoAForest<ScalarLeaf>`.
    ///
    /// This only supports gbtree and dart boosters (tree-based models).
    /// For gblinear models, use [`to_booster()`](Self::to_booster) instead.
    ///
    /// Note: For DART models, this returns just the forest without weights.
    /// Use [`to_booster()`](Self::to_booster) to get proper DART weighting.
    pub fn to_forest(&self) -> Result<SoAForest<ScalarLeaf>, ConversionError> {
        if self.is_linear() {
            // For linear models, to_forest doesn't make sense
            // Users should use to_booster() instead
            return Err(ConversionError::InvalidLinearWeights {
                actual: 0,
                expected: 0,
            });
        }
        let (forest, _weights) = self.to_forest_with_weights()?;
        Ok(forest)
    }

    /// Convert to a native `Booster`.
    ///
    /// Returns:
    /// - `Booster::Tree` for gbtree models
    /// - `Booster::Dart` for DART models (with per-tree weights)
    /// - `Booster::Linear` for gblinear models
    pub fn to_booster(&self) -> Result<Booster, ConversionError> {
        match &self.learner.gradient_booster {
            GradientBooster::Gbtree { .. } | GradientBooster::Dart { .. } => {
                let (forest, weights) = self.to_forest_with_weights()?;
                match weights {
                    Some(w) => Ok(Booster::Dart {
                        forest,
                        weights: w.into_boxed_slice(),
                    }),
                    None => Ok(Booster::Tree(forest)),
                }
            }
            GradientBooster::Gblinear { model } => {
                let linear = self.convert_linear_model(&model.weights)?;
                Ok(Booster::Linear(linear))
            }
        }
    }

    /// Returns true if this model uses DART booster.
    pub fn is_dart(&self) -> bool {
        matches!(
            &self.learner.gradient_booster,
            GradientBooster::Dart { .. }
        )
    }

    /// Returns true if this model uses gblinear booster.
    pub fn is_linear(&self) -> bool {
        matches!(
            &self.learner.gradient_booster,
            GradientBooster::Gblinear { .. }
        )
    }

    /// Convert gblinear weights to LinearModel.
    ///
    /// XGBoost stores weights in column-major order: all weights for feature 0 across
    /// all groups, then all weights for feature 1, etc., followed by biases.
    /// Layout: [w(f0,g0), w(f0,g1), ..., w(fn,g0), w(fn,g1), ..., bias(g0), bias(g1), ...]
    fn convert_linear_model(&self, weights: &[f32]) -> Result<LinearModel, ConversionError> {
        let num_features = self.learner.learner_model_param.num_feature as usize;
        let num_class = self.learner.learner_model_param.num_class;
        let num_groups = if num_class <= 1 { 1 } else { num_class as usize };

        let expected_len = (num_features + 1) * num_groups;
        if weights.len() != expected_len {
            return Err(ConversionError::InvalidLinearWeights {
                actual: weights.len(),
                expected: expected_len,
            });
        }

        // XGBoost uses the same layout as our LinearModel: feature × group + bias
        Ok(LinearModel::new(
            weights.to_vec().into_boxed_slice(),
            num_features,
            num_groups,
        ))
    }

    /// Internal: Convert to forest with optional DART weights.
    fn to_forest_with_weights(
        &self,
    ) -> Result<(SoAForest<ScalarLeaf>, Option<Vec<f32>>), ConversionError> {
        let (model_trees, tree_weights) = match &self.learner.gradient_booster {
            GradientBooster::Gbtree { model } => (model, None),
            GradientBooster::Dart { gbtree, weight_drop } => {
                (&gbtree.model, Some(weight_drop.clone()))
            }
            GradientBooster::Gblinear { .. } => {
                // This shouldn't happen - callers should use convert_linear_model directly
                unreachable!("to_forest_with_weights called on gblinear model")
            }
        };

        // Determine number of groups (1 for regression, num_class for multiclass)
        let num_class = self.learner.learner_model_param.num_class;
        let num_groups = if num_class <= 1 { 1 } else { num_class as u32 };

        // Build the forest
        let mut forest = SoAForest::new(num_groups);

        // Get base score and convert to margin space based on objective
        let raw_base_score = self.learner.learner_model_param.base_score;
        let objective = self.learner.objective.name();
        let margin_base_score = prob_to_margin(raw_base_score, objective);
        forest = forest.with_base_score(vec![margin_base_score; num_groups as usize]);

        // Convert each tree
        for (tree_idx, xgb_tree) in model_trees.trees.iter().enumerate() {
            let tree_group = model_trees.tree_info.get(tree_idx).copied().unwrap_or(0) as u32;
            let native_tree = convert_tree(xgb_tree, tree_idx)?;
            forest.push_tree(native_tree, tree_group);
        }

        Ok((forest, tree_weights))
    }
}

/// Convert a single XGBoost tree to native SoATreeStorage.
fn convert_tree(
    xgb_tree: &Tree,
    tree_idx: usize,
) -> Result<crate::trees::SoATreeStorage<ScalarLeaf>, ConversionError> {
    let num_nodes = xgb_tree.tree_param.num_nodes as usize;
    if num_nodes == 0 {
        return Err(ConversionError::EmptyTree(tree_idx));
    }

    // Build a lookup map from node index to categorical bitset.
    // XGBoost stores categorical data in parallel arrays:
    // - categories_nodes: which node indices have categorical splits
    // - categories_segments: start index into categories array
    // - categories_sizes: number of u32 words for this node's bitset
    // - categories: flat array of bitset words
    let categorical_map = build_categorical_map(xgb_tree);

    let mut builder = TreeBuilder::new();

    // XGBoost stores nodes in BFS order, which matches our expected layout.
    // We need to iterate and add nodes in order.
    for node_idx in 0..num_nodes {
        let left_child = xgb_tree.left_children[node_idx];
        let right_child = xgb_tree.right_children[node_idx];

        // A node is a leaf if left_child == -1 (XGBoost convention)
        let is_leaf = left_child == -1;

        if is_leaf {
            // Leaf node: base_weights contains the leaf value
            let leaf_value = xgb_tree.base_weights[node_idx];
            builder.add_leaf(ScalarLeaf(leaf_value));
        } else {
            // Split node
            // Validate child indices
            if left_child < 0 || left_child as usize >= num_nodes {
                return Err(ConversionError::InvalidNodeIndex {
                    tree: tree_idx,
                    node: node_idx,
                    child: left_child,
                    num_nodes,
                });
            }
            if right_child < 0 || right_child as usize >= num_nodes {
                return Err(ConversionError::InvalidNodeIndex {
                    tree: tree_idx,
                    node: node_idx,
                    child: right_child,
                    num_nodes,
                });
            }

            let feature_index = xgb_tree.split_indices[node_idx] as u32;
            let default_left = xgb_tree.default_left[node_idx] != 0;

            // Check if this is a categorical split
            // XGBoost split_type: 0 = numeric, 1 = categorical
            let is_categorical = xgb_tree.split_type.get(node_idx).copied().unwrap_or(0) == 1;

            if is_categorical {
                // Get the category bitset for this node
                let bitset = categorical_map
                    .get(&node_idx)
                    .cloned()
                    .unwrap_or_default();

                builder.add_categorical_split(
                    feature_index,
                    bitset,
                    default_left,
                    left_child as u32,
                    right_child as u32,
                );
            } else {
                // Numeric split
                let threshold = xgb_tree.split_conditions[node_idx];

                builder.add_split(
                    feature_index,
                    threshold,
                    default_left,
                    left_child as u32,
                    right_child as u32,
                );
            }
        }
    }

    Ok(builder.build())
}

/// Build a map from node index to category bitset.
///
/// XGBoost JSON stores categories as integer values (not packed bitsets).
/// The `categories` array contains the actual category values, and we need to
/// convert them to a packed bitset where bit `c` is set if category `c` goes right.
///
/// XGBoost JSON format:
/// - categories_nodes: which node indices have categorical splits
/// - categories_segments: start index into categories array for each node
/// - categories_sizes: number of category VALUES for each node
/// - categories: flat array of category INTEGER VALUES (not bitset words)
fn build_categorical_map(xgb_tree: &Tree) -> std::collections::HashMap<usize, Vec<u32>> {
    let mut map = std::collections::HashMap::new();

    for i in 0..xgb_tree.categories_nodes.len() {
        let node_idx = xgb_tree.categories_nodes[i] as usize;
        let start = xgb_tree.categories_segments[i] as usize;
        let size = xgb_tree.categories_sizes[i] as usize;

        // Get the category VALUES (integers) for this node
        let category_values: Vec<u32> = xgb_tree.categories[start..start + size]
            .iter()
            .map(|&x| x as u32)
            .collect();

        // Convert category values to a packed bitset
        // Find max category to determine bitset size
        let max_cat = category_values.iter().copied().max().unwrap_or(0);
        let num_words = (max_cat / 32 + 1) as usize;
        let mut bitset = vec![0u32; num_words];

        // Set bits for each category that goes right
        for cat in category_values {
            let word_idx = (cat / 32) as usize;
            let bit_idx = cat % 32;
            bitset[word_idx] |= 1u32 << bit_idx;
        }

        map.insert(node_idx, bitset);
    }

    map
}

/// Extract model trees from a gradient booster.
impl ModelTrees {
    /// Number of trees in this model.
    pub fn num_trees(&self) -> usize {
        self.trees.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::path::PathBuf;

    fn load_gbtree(name: &str) -> XgbModel {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/test-cases/xgboost/gbtree/inference")
            .join(format!("{name}.model.json"));
        let file = File::open(&path).expect("Failed to open test model");
        serde_json::from_reader(file).expect("Failed to parse test model")
    }

    fn load_gblinear(name: &str) -> XgbModel {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/test-cases/xgboost/gblinear/inference")
            .join(format!("{name}.model.json"));
        let file = File::open(&path).expect("Failed to open test model");
        serde_json::from_reader(file).expect("Failed to parse test model")
    }

    fn load_dart(name: &str) -> XgbModel {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/test-cases/xgboost/dart/inference")
            .join(format!("{name}.model.json"));
        let file = File::open(&path).expect("Failed to open test model");
        serde_json::from_reader(file).expect("Failed to parse test model")
    }

    #[test]
    fn convert_gbtree_regression() {
        let model = load_gbtree("gbtree_regression");
        let forest = model.to_forest().expect("Conversion failed");

        assert_eq!(forest.num_groups(), 1);
        assert!(forest.num_trees() > 0);
    }

    #[test]
    fn convert_gbtree_binary_logistic() {
        let model = load_gbtree("gbtree_binary_logistic");
        let forest = model.to_forest().expect("Conversion failed");

        assert_eq!(forest.num_groups(), 1); // Binary uses single output
        assert!(forest.num_trees() > 0);
    }

    #[test]
    fn convert_gbtree_multiclass() {
        let model = load_gbtree("gbtree_multiclass");
        let forest = model.to_forest().expect("Conversion failed");

        assert_eq!(forest.num_groups(), 3); // 3-class
        assert!(forest.num_trees() > 0);
    }

    #[test]
    fn convert_dart_regression() {
        let model = load_dart("dart_regression");
        let forest = model.to_forest().expect("Conversion failed");

        assert_eq!(forest.num_groups(), 1);
        assert!(forest.num_trees() > 0);
    }

    #[test]
    fn to_booster_gbtree_returns_tree() {
        let model = load_gbtree("gbtree_regression");
        let booster = model.to_booster().expect("Conversion failed");

        assert!(!model.is_dart());
        match booster {
            Booster::Tree(_) => {}
            Booster::Dart { .. } => panic!("Expected Booster::Tree"),
            Booster::Linear(_) => panic!("Expected Booster::Tree"),
        }
    }

    #[test]
    fn to_booster_dart_returns_dart_with_weights() {
        let model = load_dart("dart_regression");
        let booster = model.to_booster().expect("Conversion failed");

        assert!(model.is_dart());
        match booster {
            Booster::Dart { forest, weights } => {
                assert!(forest.num_trees() > 0);
                assert_eq!(weights.len(), forest.num_trees());
            }
            Booster::Tree(_) => panic!("Expected Booster::Dart"),
            Booster::Linear(_) => panic!("Expected Booster::Dart"),
        }
    }

    #[test]
    fn converted_forest_can_predict() {
        let model = load_gbtree("gbtree_regression");
        let forest = model.to_forest().expect("Conversion failed");

        // Simple smoke test: predict on dummy features
        let features = vec![0.5; 5]; // 5 features
        let predictions = forest.predict_row(&features);

        assert_eq!(predictions.len(), 1);
        // The prediction should be a finite number
        assert!(predictions[0].is_finite());
    }

    #[test]
    fn convert_gblinear_regression() {
        let model = load_gblinear("gblinear_regression");
        let booster = model.to_booster().expect("Conversion failed");

        assert!(model.is_linear());
        match booster {
            Booster::Linear(linear) => {
                assert_eq!(linear.num_features(), 5);
                assert_eq!(linear.num_groups(), 1);
            }
            _ => panic!("Expected Booster::Linear"),
        }
    }

    #[test]
    fn convert_gblinear_binary() {
        let model = load_gblinear("gblinear_binary");
        let booster = model.to_booster().expect("Conversion failed");

        assert!(model.is_linear());
        match booster {
            Booster::Linear(linear) => {
                assert_eq!(linear.num_features(), 4);
                assert_eq!(linear.num_groups(), 1); // Binary is single output
            }
            _ => panic!("Expected Booster::Linear"),
        }
    }

    #[test]
    fn convert_gblinear_multiclass() {
        let model = load_gblinear("gblinear_multiclass");
        let booster = model.to_booster().expect("Conversion failed");

        assert!(model.is_linear());
        match booster {
            Booster::Linear(linear) => {
                assert_eq!(linear.num_features(), 4);
                assert_eq!(linear.num_groups(), 3); // 3-class multiclass
            }
            _ => panic!("Expected Booster::Linear"),
        }
    }

    #[test]
    fn gblinear_to_forest_fails() {
        let model = load_gblinear("gblinear_regression");

        // to_forest should fail for gblinear models
        assert!(model.to_forest().is_err());
    }
}

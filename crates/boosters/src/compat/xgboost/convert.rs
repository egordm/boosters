//! Conversion from XGBoost JSON types to native boosters types.

use crate::repr::gbdt::{Forest, MutableTree, ScalarLeaf, Tree, categories_to_bitset};
use crate::repr::gblinear::LinearModel;
use ndarray::Array2;

use super::json::{GradientBooster, ModelTrees, Tree as XgbTree, XgbModel};

/// A booster model converted from XGBoost.
///
/// This enum represents the different types of gradient boosting models
/// that can be loaded from XGBoost JSON format.
#[derive(Debug, Clone)]
pub enum Booster {
    /// Standard gradient boosted tree ensemble.
    Tree(Forest<ScalarLeaf>),
    /// DART (Dropout Additive Regression Trees) ensemble with per-tree weights.
    Dart {
        forest: Forest<ScalarLeaf>,
        weights: Box<[f32]>,
    },
    /// Linear (gblinear) booster model.
    Linear(LinearModel),
}

/// Error type for XGBoost model conversion.
#[derive(Debug, thiserror::Error)]
pub enum ConversionError {
    #[error("tree {0} has no nodes")]
    EmptyTree(usize),
    #[error(
        "invalid node index in tree {tree}: node {node} references child {child} but tree has {num_nodes} nodes"
    )]
    InvalidNodeIndex {
        tree: usize,
        node: usize,
        child: i32,
        num_nodes: usize,
    },
    #[error(
        "gblinear weights length {actual} doesn't match (num_features + 1) * num_groups = {expected}"
    )]
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
    /// Convert to a native `Forest<ScalarLeaf>`.
    ///
    /// This only supports gbtree and dart boosters (tree-based models).
    /// For gblinear models, use [`to_booster()`](Self::to_booster) instead.
    ///
    /// Note: For DART models, this returns just the forest without weights.
    /// Use [`to_booster()`](Self::to_booster) to get proper DART weighting.
    pub fn to_forest(&self) -> Result<Forest<ScalarLeaf>, ConversionError> {
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
        matches!(&self.learner.gradient_booster, GradientBooster::Dart { .. })
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
    /// XGBoost stores weights in row-major order: `[n_features + 1, n_groups]`
    /// where the last row contains biases. We bake the base_score into the
    /// bias so that prediction requires no additional parameters.
    fn convert_linear_model(&self, weights: &[f32]) -> Result<LinearModel, ConversionError> {
        let num_features = self.learner.learner_model_param.n_features as usize;
        let num_class = self.learner.learner_model_param.n_class;
        let num_groups = if num_class <= 1 {
            1
        } else {
            num_class as usize
        };

        let expected_len = (num_features + 1) * num_groups;
        if weights.len() != expected_len {
            return Err(ConversionError::InvalidLinearWeights {
                actual: weights.len(),
                expected: expected_len,
            });
        }

        // Get base_score and convert to margin space
        let raw_base_score = self.learner.learner_model_param.base_score;
        let objective = self.learner.objective.name();
        let margin_base_score = prob_to_margin(raw_base_score, objective);

        // Reshape flat weights into [n_features + 1, n_groups] array
        let mut arr = Array2::from_shape_vec((num_features + 1, num_groups), weights.to_vec())
            .expect("shape and weights length match");

        // Bake base_score into bias (last row)
        for group in 0..num_groups {
            arr[[num_features, group]] += margin_base_score;
        }

        Ok(LinearModel::new(arr))
    }

    /// Internal: Convert to forest with optional DART weights.
    fn to_forest_with_weights(
        &self,
    ) -> Result<(Forest<ScalarLeaf>, Option<Vec<f32>>), ConversionError> {
        let (model_trees, tree_weights) = match &self.learner.gradient_booster {
            GradientBooster::Gbtree { model } => (model, None),
            GradientBooster::Dart {
                gbtree,
                weight_drop,
            } => (&gbtree.model, Some(weight_drop.clone())),
            GradientBooster::Gblinear { .. } => {
                // This shouldn't happen - callers should use convert_linear_model directly
                unreachable!("to_forest_with_weights called on gblinear model")
            }
        };

        // Determine number of groups (1 for regression, num_class for multiclass)
        let num_class = self.learner.learner_model_param.n_class;
        let num_groups = if num_class <= 1 { 1 } else { num_class as u32 };

        // Build the forest
        let mut forest = Forest::new(num_groups);

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

/// Convert a single XGBoost tree to native `Tree`.
fn convert_tree(xgb_tree: &XgbTree, tree_idx: usize) -> Result<Tree<ScalarLeaf>, ConversionError> {
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

    let mut tree = MutableTree::<ScalarLeaf>::with_capacity(num_nodes);
    tree.init_root_with_n_nodes(num_nodes);

    // XGBoost stores nodes in BFS order, which matches our expected layout.
    // We need to iterate and add nodes in order.
    for node_idx in 0..num_nodes {
        let left_child = xgb_tree.left_children[node_idx];
        let right_child = xgb_tree.right_children[node_idx];

        // A node is a leaf if left_child == -1 (XGBoost convention)
        let node_is_leaf = left_child == -1;

        if node_is_leaf {
            // Leaf node: base_weights contains the leaf value
            let leaf_value = xgb_tree.base_weights[node_idx];
            tree.make_leaf(node_idx as u32, ScalarLeaf(leaf_value));
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
            let node_default_left = xgb_tree.default_left[node_idx] != 0;

            // Check if this is a categorical split
            // XGBoost split_type: 0 = numeric, 1 = categorical
            let is_categorical = xgb_tree.split_type.get(node_idx).copied().unwrap_or(0) == 1;

            if is_categorical {
                let bitset = categorical_map.get(&node_idx).cloned().unwrap_or_default();
                tree.set_categorical_split(
                    node_idx as u32,
                    feature_index,
                    bitset,
                    node_default_left,
                    left_child as u32,
                    right_child as u32,
                );
            } else {
                // Numeric split
                let threshold = xgb_tree.split_conditions[node_idx];
                tree.set_numeric_split(
                    node_idx as u32,
                    feature_index,
                    threshold,
                    node_default_left,
                    left_child as u32,
                    right_child as u32,
                );
            }
        }
    }

    Ok(tree.freeze())
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
fn build_categorical_map(xgb_tree: &XgbTree) -> std::collections::HashMap<usize, Vec<u32>> {
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

        let bitset = categories_to_bitset(&category_values);

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
    use crate::inference::gbdt::SimplePredictor;
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

        assert_eq!(forest.n_groups(), 1);
        assert!(forest.n_trees() > 0);
    }

    #[test]
    fn convert_gbtree_binary_logistic() {
        let model = load_gbtree("gbtree_binary_logistic");
        let forest = model.to_forest().expect("Conversion failed");

        assert_eq!(forest.n_groups(), 1); // Binary uses single output
        assert!(forest.n_trees() > 0);
    }

    #[test]
    fn convert_gbtree_multiclass() {
        let model = load_gbtree("gbtree_multiclass");
        let forest = model.to_forest().expect("Conversion failed");

        assert_eq!(forest.n_groups(), 3); // 3-class
        assert!(forest.n_trees() > 0);
    }

    #[test]
    fn convert_dart_regression() {
        let model = load_dart("dart_regression");
        let forest = model.to_forest().expect("Conversion failed");

        assert_eq!(forest.n_groups(), 1);
        assert!(forest.n_trees() > 0);
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
                assert!(forest.n_trees() > 0);
                assert_eq!(weights.len(), forest.n_trees());
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
        let features = vec![0.5f32; 5]; // 5 features
        let predictor = SimplePredictor::new(&forest);
        let mut predictions = vec![0.0f32; predictor.n_groups()];
        let sample = ndarray::ArrayView1::from(features.as_slice());
        predictor.predict_row_into(sample, None, &mut predictions);

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
                assert_eq!(linear.n_features(), 5);
                assert_eq!(linear.n_groups(), 1);
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
                assert_eq!(linear.n_features(), 4);
                assert_eq!(linear.n_groups(), 1); // Binary is single output
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
                assert_eq!(linear.n_features(), 4);
                assert_eq!(linear.n_groups(), 3); // 3-class multiclass
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

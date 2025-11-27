//! Conversion from XGBoost JSON types to native booste-rs types.

use crate::forest::SoAForest;
use crate::trees::{ScalarLeaf, TreeBuilder};

use super::json::{GradientBooster, ModelTrees, Tree, XgbModel};

/// Error type for XGBoost model conversion.
#[derive(Debug, thiserror::Error)]
pub enum ConversionError {
    #[error("unsupported booster type: gblinear models are not supported for tree prediction")]
    UnsupportedBooster,
    #[error("tree {0} has no nodes")]
    EmptyTree(usize),
    #[error("invalid node index in tree {tree}: node {node} references child {child} but tree has {num_nodes} nodes")]
    InvalidNodeIndex {
        tree: usize,
        node: usize,
        child: i32,
        num_nodes: usize,
    },
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
    /// gblinear models will return an error.
    pub fn to_forest(&self) -> Result<SoAForest<ScalarLeaf>, ConversionError> {
        let (model_trees, tree_weights) = match &self.learner.gradient_booster {
            GradientBooster::Gbtree { model } => (model, None),
            GradientBooster::Dart { gbtree, weight_drop } => (&gbtree.model, Some(weight_drop)),
            GradientBooster::Gblinear { .. } => return Err(ConversionError::UnsupportedBooster),
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
            let native_tree = convert_tree(xgb_tree, tree_idx, tree_weights)?;
            forest.push_tree(native_tree, tree_group);
        }

        Ok(forest)
    }
}

/// Convert a single XGBoost tree to native SoATreeStorage.
fn convert_tree(
    xgb_tree: &Tree,
    tree_idx: usize,
    _tree_weights: Option<&Vec<f32>>,
) -> Result<crate::trees::SoATreeStorage<ScalarLeaf>, ConversionError> {
    let num_nodes = xgb_tree.tree_param.num_nodes as usize;
    if num_nodes == 0 {
        return Err(ConversionError::EmptyTree(tree_idx));
    }

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
            let threshold = xgb_tree.split_conditions[node_idx];
            let default_left = xgb_tree.default_left[node_idx] != 0;

            builder.add_split(
                feature_index,
                threshold,
                default_left,
                left_child as u32,
                right_child as u32,
            );
        }
    }

    Ok(builder.build())
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

    fn load_test_model(name: &str) -> XgbModel {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/test-cases/xgboost")
            .join(format!("{name}.model.json"));
        let file = File::open(&path).expect("Failed to open test model");
        serde_json::from_reader(file).expect("Failed to parse test model")
    }

    #[test]
    fn convert_gbtree_regression() {
        let model = load_test_model("regression");
        let forest = model.to_forest().expect("Conversion failed");

        assert_eq!(forest.num_groups(), 1);
        assert!(forest.num_trees() > 0);
    }

    #[test]
    fn convert_gbtree_binary_logistic() {
        let model = load_test_model("binary_logistic");
        let forest = model.to_forest().expect("Conversion failed");

        assert_eq!(forest.num_groups(), 1); // Binary uses single output
        assert!(forest.num_trees() > 0);
    }

    #[test]
    fn convert_gbtree_multiclass() {
        let model = load_test_model("multiclass");
        let forest = model.to_forest().expect("Conversion failed");

        assert_eq!(forest.num_groups(), 3); // 3-class
        assert!(forest.num_trees() > 0);
    }

    #[test]
    fn convert_dart_regression() {
        let model = load_test_model("dart_regression");
        let forest = model.to_forest().expect("Conversion failed");

        assert_eq!(forest.num_groups(), 1);
        assert!(forest.num_trees() > 0);
    }

    #[test]
    fn converted_forest_can_predict() {
        let model = load_test_model("regression");
        let forest = model.to_forest().expect("Conversion failed");

        // Simple smoke test: predict on dummy features
        let features = vec![0.5; 5]; // 5 features
        let predictions = forest.predict_row(&features);

        assert_eq!(predictions.len(), 1);
        // The prediction should be a finite number
        assert!(predictions[0].is_finite());
    }
}

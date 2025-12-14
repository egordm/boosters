//! Conversion from LightGBM parsed types to native booste-rs types.

use crate::repr::gbdt::{Forest, MutableTree, ScalarLeaf, Tree};

use super::text::{DecisionType, LgbModel, LgbTree};

/// Error type for LightGBM model conversion.
#[derive(Debug, thiserror::Error)]
pub enum ConversionError {
    #[error("tree {0} has no nodes")]
    EmptyTree(usize),
    #[error("tree {tree}: invalid child index {child} at node {node}")]
    InvalidChildIndex {
        tree: usize,
        node: usize,
        child: i32,
    },
    #[error("linear trees are not supported")]
    LinearTreesNotSupported,
    #[error("conversion error: {0}")]
    Other(String),
}

impl LgbModel {
    /// Convert to a native `Forest<ScalarLeaf>`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let model = LgbModel::from_file("model.txt")?;
    /// let forest = model.to_forest()?;
    /// let predictions = forest.predict_row(&features);
    /// ```
    pub fn to_forest(&self) -> Result<Forest<ScalarLeaf>, ConversionError> {
        // Check for unsupported features
        for (idx, tree) in self.trees.iter().enumerate() {
            if tree.is_linear {
                return Err(ConversionError::LinearTreesNotSupported);
            }
            if tree.num_leaves == 0 {
                return Err(ConversionError::EmptyTree(idx));
            }
        }

        let num_groups = self.num_groups() as u32;
        let mut forest = Forest::new(num_groups);

        // LightGBM doesn't store base_score the same way XGBoost does.
        // The initial prediction is typically 0 for regression, or learned from data.
        // For now, we use 0 as the base score.
        let base_scores = vec![0.0f32; num_groups as usize];
        forest = forest.with_base_score(base_scores);

        // Convert each tree
        for (tree_idx, lgb_tree) in self.trees.iter().enumerate() {
            // Determine which group this tree belongs to
            // Trees are organized as: tree[iteration * num_groups + group_idx]
            let tree_group = (tree_idx % num_groups as usize) as u32;

            let native_tree = convert_tree(lgb_tree, tree_idx)?;
            forest.push_tree(native_tree, tree_group);
        }

        Ok(forest)
    }
}

/// Convert a single LightGBM tree to canonical [`Tree`].
fn convert_tree(
    lgb_tree: &LgbTree,
    _tree_idx: usize,
) -> Result<Tree<ScalarLeaf>, ConversionError> {
    // Special case: single-leaf tree
    if lgb_tree.num_leaves == 1 {
        let leaf_value = lgb_tree.leaf_value.first().copied().unwrap_or(0.0) as f32;
        let mut t = MutableTree::<ScalarLeaf>::with_capacity(1);
        let root = t.init_root_with_num_nodes(1);
        t.make_leaf(root, ScalarLeaf(leaf_value));
        return Ok(t.freeze());
    }

    // LightGBM tree structure:
    // - Internal nodes: indices 0 to num_leaves-2
    // - Leaves: referenced as negative values in left_child/right_child
    //   - Leaf index = ~child (bitwise NOT) when child < 0
    //
    // We need to convert to our BFS layout where nodes and leaves are
    // laid out in traversal order.

    let num_internal = lgb_tree.num_leaves - 1;
    let total_nodes = num_internal + lgb_tree.num_leaves;

    let mut tree = MutableTree::<ScalarLeaf>::with_capacity(total_nodes);
    tree.init_root_with_num_nodes(total_nodes);

    for node_idx in 0..num_internal {
        let left = lgb_tree.left_child[node_idx];
        let right = lgb_tree.right_child[node_idx];
        let dt = DecisionType::from_i8(lgb_tree.decision_type[node_idx]);

        // Convert child references:
        // - If child < 0: it's a leaf with index ~child
        // - If child >= 0: it's an internal node with that index

        let left_child = convert_child_ref(left, num_internal)?;
        let right_child = convert_child_ref(right, num_internal)?;

        let feature_index = lgb_tree.split_feature[node_idx] as u32;

        if dt.is_categorical {
            // Categorical split
            let cat_idx = lgb_tree.threshold[node_idx] as usize;
            let bitset = extract_categorical_bitset(lgb_tree, cat_idx);
            tree.set_categorical_split(
                node_idx as u32,
                feature_index,
                bitset,
                dt.default_left,
                left_child,
                right_child,
            );
        } else {
            // Numerical split
            // LightGBM uses `<=` for left; our traversal uses `<`.
            // Store `next_up(threshold)` so boundary decisions match.
            let threshold = next_up_f32(lgb_tree.threshold[node_idx] as f32);
            tree.set_numeric_split(
                node_idx as u32,
                feature_index,
                threshold,
                dt.default_left,
                left_child,
                right_child,
            );
        }
    }

    // Add leaf nodes
    // Note: leaf_value already has shrinkage applied during training
    // The shrinkage field in the model is just for informational purposes
    for leaf_idx in 0..lgb_tree.num_leaves {
        let node_idx = num_internal + leaf_idx;
        tree.make_leaf(node_idx as u32, ScalarLeaf(lgb_tree.leaf_value[leaf_idx] as f32));
    }

    Ok(tree.freeze())
}

/// Node or leaf in the conversion process.
#[allow(dead_code)]
enum NodeOrLeaf {
    Internal(usize),
    Leaf(usize),
}

/// Convert a LightGBM child reference to our format.
///
/// LightGBM uses negative values for leaves: leaf_idx = ~child
/// We convert to: internal nodes keep their index, leaves get index num_internal + leaf_idx
fn convert_child_ref(child: i32, num_internal: usize) -> Result<u32, ConversionError> {
    if child < 0 {
        // Leaf: ~child gives the leaf index
        let leaf_idx = !child;
        Ok((num_internal as i32 + leaf_idx) as u32)
    } else {
        // Internal node: keep index
        Ok(child as u32)
    }
}

#[inline]
fn next_up_f32(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }

    if x == f32::INFINITY {
        return x;
    }

    if x == -0.0 {
        // nextafter(-0.0, +inf) = smallest positive subnormal
        return f32::from_bits(1);
    }

    let bits = x.to_bits();
    if x >= 0.0 {
        f32::from_bits(bits + 1)
    } else {
        f32::from_bits(bits - 1)
    }
}

/// Extract categorical bitset for a given category index.
fn extract_categorical_bitset(lgb_tree: &LgbTree, cat_idx: usize) -> Vec<u32> {
    if lgb_tree.cat_boundaries.is_empty() {
        return Vec::new();
    }

    let start = lgb_tree.cat_boundaries.get(cat_idx).copied().unwrap_or(0) as usize;
    let end = lgb_tree
        .cat_boundaries
        .get(cat_idx + 1)
        .copied()
        .unwrap_or(start as i32) as usize;

    lgb_tree.cat_threshold[start..end].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn load_model(name: &str) -> LgbModel {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/test-cases/lightgbm/inference")
            .join(name)
            .join("model.txt");
        LgbModel::from_file(&path).expect("Failed to load model")
    }

    fn load_expected(name: &str) -> (Vec<Vec<f64>>, Vec<f64>) {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/test-cases/lightgbm/inference")
            .join(name)
            .join("input.json");
        let input: serde_json::Value =
            serde_json::from_reader(std::fs::File::open(&path).unwrap()).unwrap();
        let data: Vec<Vec<f64>> = serde_json::from_value(input["data"].clone()).unwrap();

        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/test-cases/lightgbm/inference")
            .join(name)
            .join("expected.json");
        let expected: serde_json::Value =
            serde_json::from_reader(std::fs::File::open(&path).unwrap()).unwrap();
        let raw: Vec<f64> = serde_json::from_value(expected["raw"].clone()).unwrap();

        (data, raw)
    }

    #[test]
    fn convert_small_tree() {
        let model = load_model("small_tree");
        let forest = model.to_forest().expect("Conversion failed");

        assert_eq!(forest.n_groups(), 1);
        assert_eq!(forest.n_trees(), 3);
    }

    #[test]
    fn predict_small_tree() {
        let model = load_model("small_tree");
        let forest = model.to_forest().expect("Conversion failed");
        let (data, expected) = load_expected("small_tree");

        for (i, (row, exp)) in data.iter().zip(expected.iter()).enumerate() {
            let features: Vec<f32> = row.iter().map(|x| *x as f32).collect();
            let pred = forest.predict_row(&features);
            assert_eq!(pred.len(), 1, "Row {}: expected 1 output", i);
            assert_abs_diff_eq!(
                pred[0],
                *exp as f32,
                epsilon = 0.01 // Allow 1% tolerance due to float precision
            );
        }
    }

    #[test]
    fn convert_regression() {
        let model = load_model("regression");
        let forest = model.to_forest().expect("Conversion failed");

        assert_eq!(forest.n_groups(), 1);
        assert!(forest.n_trees() > 0);
    }

    #[test]
    fn predict_regression() {
        let model = load_model("regression");
        let forest = model.to_forest().expect("Conversion failed");
        let (data, expected) = load_expected("regression");

        for (i, (row, exp)) in data.iter().zip(expected.iter()).enumerate() {
            let features: Vec<f32> = row.iter().map(|x| *x as f32).collect();
            let pred = forest.predict_row(&features);
            assert_eq!(pred.len(), 1, "Row {}: expected 1 output", i);
            assert_abs_diff_eq!(
                pred[0],
                *exp as f32,
                epsilon = 0.01
            );
        }
    }

    #[test]
    fn convert_binary_classification() {
        let model = load_model("binary_classification");
        let forest = model.to_forest().expect("Conversion failed");

        assert_eq!(forest.n_groups(), 1); // Binary uses single output
        assert!(forest.n_trees() > 0);
    }

    #[test]
    fn predict_binary_classification_raw() {
        let model = load_model("binary_classification");
        let forest = model.to_forest().expect("Conversion failed");
        let (data, expected) = load_expected("binary_classification");

        for (i, (row, exp)) in data.iter().zip(expected.iter()).enumerate() {
            let features: Vec<f32> = row.iter().map(|x| *x as f32).collect();
            let pred = forest.predict_row(&features);
            assert_eq!(pred.len(), 1, "Row {}: expected 1 output", i);
            assert_abs_diff_eq!(
                pred[0],
                *exp as f32,
                epsilon = 0.01
            );
        }
    }

    #[test]
    fn convert_multiclass() {
        let model = load_model("multiclass");
        let forest = model.to_forest().expect("Conversion failed");

        assert_eq!(forest.n_groups(), 3);
        assert!(forest.n_trees() > 0);
    }

    #[test]
    fn predict_multiclass_raw() {
        let model = load_model("multiclass");
        let forest = model.to_forest().expect("Conversion failed");

        // Load input data
        let input_path = format!(
            "{}/tests/test-cases/lightgbm/inference/multiclass/input.json",
            env!("CARGO_MANIFEST_DIR")
        );
        let input: serde_json::Value =
            serde_json::from_reader(std::fs::File::open(&input_path).unwrap()).unwrap();
        let data: Vec<Vec<f64>> = serde_json::from_value(input["data"].clone()).unwrap();

        // Load multiclass expected data (array of arrays)
        let expected_path = format!(
            "{}/tests/test-cases/lightgbm/inference/multiclass/expected.json",
            env!("CARGO_MANIFEST_DIR")
        );
        let expected: serde_json::Value =
            serde_json::from_reader(std::fs::File::open(&expected_path).unwrap()).unwrap();
        let raw: Vec<Vec<f64>> = serde_json::from_value(expected["raw"].clone()).unwrap();

        for (i, (row, exp_vec)) in data.iter().zip(raw.iter()).enumerate() {
            let features: Vec<f32> = row.iter().map(|x| *x as f32).collect();
            let pred = forest.predict_row(&features);
            // Multiclass returns one output per class
            assert_eq!(pred.len(), 3, "Row {}: expected 3 outputs for 3-class", i);

            for (class_idx, pred_val) in pred.iter().enumerate() {
                let exp_val = exp_vec[class_idx];
                assert_abs_diff_eq!(
                    *pred_val,
                    exp_val as f32,
                    epsilon = 0.01
                );
            }
        }
    }

    #[test]
    fn predict_regression_missing() {
        let model = load_model("regression_missing");
        let forest = model.to_forest().expect("Conversion failed");

        // Load input data (with null values for NaN)
        let input_path = format!(
            "{}/tests/test-cases/lightgbm/inference/regression_missing/input.json",
            env!("CARGO_MANIFEST_DIR")
        );
        let input: serde_json::Value =
            serde_json::from_reader(std::fs::File::open(&input_path).unwrap()).unwrap();
        let data_json = input["data"].as_array().unwrap();

        // Parse data, converting null to f32::NAN
        let data: Vec<Vec<f32>> = data_json
            .iter()
            .map(|row| {
                row.as_array()
                    .unwrap()
                    .iter()
                    .map(|v| {
                        if v.is_null() {
                            f32::NAN
                        } else {
                            v.as_f64().unwrap() as f32
                        }
                    })
                    .collect()
            })
            .collect();

        // Load expected values
        let expected_path = format!(
            "{}/tests/test-cases/lightgbm/inference/regression_missing/expected.json",
            env!("CARGO_MANIFEST_DIR")
        );
        let expected: serde_json::Value =
            serde_json::from_reader(std::fs::File::open(&expected_path).unwrap()).unwrap();
        let raw: Vec<f64> = serde_json::from_value(expected["raw"].clone()).unwrap();

        for (i, (row, exp)) in data.iter().zip(raw.iter()).enumerate() {
            let pred = forest.predict_row(row);
            assert_eq!(pred.len(), 1, "Row {}: expected 1 output", i);
            assert_abs_diff_eq!(
                pred[0],
                *exp as f32,
                epsilon = 0.02 // Slightly higher tolerance for missing value handling
            );
        }
    }
}

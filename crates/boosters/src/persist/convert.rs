//! Conversion between runtime types and schema types.
//!
//! This module provides conversions between the runtime model types
//! (e.g., `GBDTModel`, `Forest`, `Tree`) and their schema counterparts
//! (e.g., `GBDTModelSchema`, `ForestSchema`, `TreeSchema`).
//!
//! Conversions are implemented as `From` traits for lossless transformations.

use super::schema::{
    CategoriesSchema, FeatureTypeSchema, ForestSchema, GBDTModelSchema, GBLinearModelSchema,
    LeafValuesSchema, LinearCoefficientsSchema, LinearWeightsSchema, ModelMetaSchema, TreeSchema,
};
use crate::model::{FeatureType, GBDTModel, GBLinearModel, ModelMeta};
use crate::repr::gbdt::{Forest, ScalarLeaf, Tree};
use crate::repr::gblinear::LinearModel;

// =============================================================================
// FeatureType conversions
// =============================================================================

impl From<&FeatureType> for FeatureTypeSchema {
    fn from(ft: &FeatureType) -> Self {
        match ft {
            FeatureType::Numeric => FeatureTypeSchema::Numeric,
            FeatureType::Categorical { .. } => FeatureTypeSchema::Categorical,
        }
    }
}

impl From<FeatureTypeSchema> for FeatureType {
    fn from(ft: FeatureTypeSchema) -> Self {
        match ft {
            FeatureTypeSchema::Numeric => FeatureType::Numeric,
            FeatureTypeSchema::Categorical => FeatureType::Categorical { n_categories: None },
        }
    }
}

// =============================================================================
// ModelMeta conversions
// =============================================================================

impl From<&ModelMeta> for ModelMetaSchema {
    fn from(meta: &ModelMeta) -> Self {
        ModelMetaSchema {
            num_features: meta.n_features,
            num_groups: meta.n_groups,
            feature_names: meta.feature_names.clone(),
            feature_types: meta
                .feature_types
                .as_ref()
                .map(|types| types.iter().map(FeatureTypeSchema::from).collect()),
            objective_name: None, // Set by caller when saving
        }
    }
}

impl From<ModelMetaSchema> for ModelMeta {
    fn from(schema: ModelMetaSchema) -> Self {
        ModelMeta {
            n_features: schema.num_features,
            n_groups: schema.num_groups,
            feature_names: schema.feature_names,
            feature_types: schema
                .feature_types
                .map(|types| types.into_iter().map(FeatureType::from).collect()),
            base_scores: Vec::new(), // Filled from ForestSchema.base_score
            best_iteration: None,
        }
    }
}

// =============================================================================
// Tree conversions
// =============================================================================

impl From<&Tree<ScalarLeaf>> for TreeSchema {
    fn from(tree: &Tree<ScalarLeaf>) -> Self {
        use crate::repr::gbdt::tree_view::TreeView;

        let n_nodes = tree.n_nodes();

        // Extract arrays
        let mut split_indices = Vec::with_capacity(n_nodes);
        let mut thresholds = Vec::with_capacity(n_nodes);
        let mut children_left = Vec::with_capacity(n_nodes);
        let mut children_right = Vec::with_capacity(n_nodes);
        let mut default_left = Vec::with_capacity(n_nodes);
        let mut leaf_values_vec = Vec::with_capacity(n_nodes);

        for node_id in 0..n_nodes as u32 {
            split_indices.push(tree.split_index(node_id));
            thresholds.push(tree.split_threshold(node_id) as f64);
            children_left.push(tree.left_child(node_id));
            children_right.push(tree.right_child(node_id));
            default_left.push(tree.default_left(node_id));
            leaf_values_vec.push(tree.leaf_value(node_id).0 as f64);
        }

        // Convert categories
        let categories_storage = tree.categories();
        let categories = if categories_storage.is_empty() {
            CategoriesSchema::default()
        } else {
            let mut node_indices = Vec::new();
            let mut category_sets = Vec::new();

            for node_id in 0..n_nodes as u32 {
                let bitset = categories_storage.bitset_for_node(node_id);
                if !bitset.is_empty() {
                    // Convert bitset to list of categories
                    let cats: Vec<u32> = bitset_to_categories(bitset);
                    if !cats.is_empty() {
                        node_indices.push(node_id);
                        category_sets.push(cats);
                    }
                }
            }

            CategoriesSchema {
                node_indices,
                category_sets,
            }
        };

        // Convert linear coefficients if present
        let linear_coefficients = if tree.has_linear_leaves() {
            let mut node_indices = Vec::new();
            let mut coefficients = Vec::new();

            for node_id in 0..n_nodes as u32 {
                if let Some((feat_indices, coefs)) = tree.leaf_terms(node_id)
                    && !coefs.is_empty()
                {
                    node_indices.push(node_id);
                    // Pack intercept and coefficients together
                    let intercept = tree.leaf_intercept(node_id);
                    let mut coef_vec: Vec<f64> = vec![intercept as f64];
                    coef_vec.extend(feat_indices.iter().map(|&f| f as f64));
                    coef_vec.extend(coefs.iter().map(|&c| c as f64));
                    coefficients.push(coef_vec);
                }
            }

            LinearCoefficientsSchema {
                node_indices,
                coefficients,
            }
        } else {
            LinearCoefficientsSchema::default()
        };

        TreeSchema {
            num_nodes: n_nodes as u32,
            split_indices,
            thresholds,
            children_left,
            children_right,
            default_left,
            leaf_values: LeafValuesSchema::Scalar {
                values: leaf_values_vec,
            },
            categories,
            linear_coefficients,
            gains: tree.gains().map(|g| g.iter().map(|&v| v as f64).collect()),
            covers: tree.covers().map(|c| c.iter().map(|&v| v as f64).collect()),
        }
    }
}

/// Convert a bitset to a list of set category indices.
fn bitset_to_categories(bitset: &[u32]) -> Vec<u32> {
    let mut cats = Vec::new();
    for (word_idx, &word) in bitset.iter().enumerate() {
        if word == 0 {
            continue;
        }
        let base = (word_idx as u32) * 32;
        for bit in 0..32 {
            if (word >> bit) & 1 != 0 {
                cats.push(base + bit);
            }
        }
    }
    cats
}

/// Convert category list to bitset.
fn categories_to_bitset(categories: &[u32]) -> Vec<u32> {
    if categories.is_empty() {
        return vec![];
    }

    let max_cat = categories.iter().copied().max().unwrap_or(0);
    let num_words = ((max_cat >> 5) + 1) as usize;
    let mut bitset = vec![0u32; num_words];

    for &cat in categories {
        let word_idx = (cat >> 5) as usize;
        let bit_idx = cat & 31;
        bitset[word_idx] |= 1u32 << bit_idx;
    }

    bitset
}

impl TryFrom<TreeSchema> for Tree<ScalarLeaf> {
    type Error = super::error::ReadError;

    fn try_from(schema: TreeSchema) -> Result<Self, Self::Error> {
        use crate::repr::gbdt::categories::CategoriesStorage;
        use crate::repr::gbdt::coefficients::LeafCoefficients;
        use crate::repr::gbdt::types::SplitType;

        let n_nodes = schema.num_nodes as usize;

        // Basic length validation (avoid debug_assert panics in Tree::new).
        if schema.split_indices.len() != n_nodes
            || schema.thresholds.len() != n_nodes
            || schema.children_left.len() != n_nodes
            || schema.children_right.len() != n_nodes
            || schema.default_left.len() != n_nodes
        {
            return Err(super::error::ReadError::Validation(
                "Tree schema arrays must all have length num_nodes".into(),
            ));
        }

        if let Some(gains) = &schema.gains
            && gains.len() != n_nodes
        {
            return Err(super::error::ReadError::Validation(
                "Tree schema gains must have length num_nodes".into(),
            ));
        }

        if let Some(covers) = &schema.covers
            && covers.len() != n_nodes
        {
            return Err(super::error::ReadError::Validation(
                "Tree schema covers must have length num_nodes".into(),
            ));
        }

        if schema.categories.node_indices.len() != schema.categories.category_sets.len() {
            return Err(super::error::ReadError::Validation(
                "Tree schema categories.node_indices and categories.category_sets must have same length".into(),
            ));
        }
        for &node_id in &schema.categories.node_indices {
            if node_id as usize >= n_nodes {
                return Err(super::error::ReadError::Validation(
                    "Tree schema categories contains out-of-bounds node index".into(),
                ));
            }
        }

        if schema.linear_coefficients.node_indices.len()
            != schema.linear_coefficients.coefficients.len()
        {
            return Err(super::error::ReadError::Validation(
                "Tree schema linear_coefficients.node_indices and linear_coefficients.coefficients must have same length".into(),
            ));
        }
        for (idx, &node_id) in schema.linear_coefficients.node_indices.iter().enumerate() {
            if node_id as usize >= n_nodes {
                return Err(super::error::ReadError::Validation(
                    "Tree schema linear_coefficients contains out-of-bounds node index".into(),
                ));
            }
            let packed = &schema.linear_coefficients.coefficients[idx];
            if packed.len() > 1 {
                // packed = [intercept, feat_idx..., coef...]
                let remaining = packed.len() - 1;
                if !remaining.is_multiple_of(2) {
                    return Err(super::error::ReadError::Validation(
                        "Tree schema linear_coefficients packed array must have length 1 + 2*k"
                            .into(),
                    ));
                }
            }
        }

        // Convert leaf values
        let leaf_values: Vec<ScalarLeaf> = match schema.leaf_values {
            LeafValuesSchema::Scalar { values } => {
                if values.len() != n_nodes {
                    return Err(super::error::ReadError::Validation(
                        "Tree schema leaf_values must have length num_nodes".into(),
                    ));
                }
                values.into_iter().map(|v| ScalarLeaf(v as f32)).collect()
            }
            LeafValuesSchema::Vector { values } => {
                if values.len() != n_nodes {
                    return Err(super::error::ReadError::Validation(
                        "Tree schema leaf_values must have length num_nodes".into(),
                    ));
                }

                // Allow vector leaves only when K=1 (degenerate vector leaf). True multi-output
                // vector leaves are handled at ForestSchema level by expansion to per-group trees.
                let k = values.first().map(|v| v.len()).unwrap_or(0);
                if k != 1 || values.iter().any(|v| v.len() != k) {
                    return Err(super::error::ReadError::Validation(
                        "VectorLeaf requires K=1 for Tree<ScalarLeaf> (multi-output is forest-level)".into(),
                    ));
                }

                values
                    .into_iter()
                    .map(|v| ScalarLeaf(v.first().copied().unwrap_or(0.0) as f32))
                    .collect()
            }
        };

        // Determine split types from categories
        let mut split_types = vec![SplitType::Numeric; n_nodes];
        let categories = if schema.categories.node_indices.is_empty() {
            CategoriesStorage::empty()
        } else {
            // Build categories storage
            let mut all_bitsets = Vec::new();
            let mut segments = vec![(0u32, 0u32); n_nodes];

            for (idx, &node_id) in schema.categories.node_indices.iter().enumerate() {
                let cats = &schema.categories.category_sets[idx];
                let bitset = categories_to_bitset(cats);

                let start = all_bitsets.len() as u32;
                let size = bitset.len() as u32;
                all_bitsets.extend(bitset);
                segments[node_id as usize] = (start, size);
                split_types[node_id as usize] = SplitType::Categorical;
            }

            CategoriesStorage::new(all_bitsets, segments)
        };

        // Convert linear coefficients
        let leaf_coefficients = if schema.linear_coefficients.node_indices.is_empty() {
            LeafCoefficients::empty()
        } else {
            let mut all_features = Vec::new();
            let mut all_coefs = Vec::new();
            let mut segments = vec![(0u32, 0u16); n_nodes];
            let mut intercepts = vec![0.0f32; n_nodes];

            for (idx, &node_id) in schema.linear_coefficients.node_indices.iter().enumerate() {
                let packed = &schema.linear_coefficients.coefficients[idx];
                if !packed.is_empty() {
                    let intercept = packed[0] as f32;
                    intercepts[node_id as usize] = intercept;

                    if packed.len() > 1 {
                        // packed[1..half+1] are feature indices, packed[half+1..] are coefficients
                        let remaining = &packed[1..];
                        let half = remaining.len() / 2;
                        let feat_indices: Vec<u32> =
                            remaining[..half].iter().map(|&f| f as u32).collect();
                        let coefs: Vec<f32> = remaining[half..].iter().map(|&c| c as f32).collect();

                        let start = all_features.len() as u32;
                        let len = feat_indices.len() as u16;
                        all_features.extend(feat_indices);
                        all_coefs.extend(coefs);
                        segments[node_id as usize] = (start, len);
                    }
                }
            }

            LeafCoefficients::new(all_features, all_coefs, segments, intercepts)
        };

        // Compute is_leaf from children: a node is a leaf if left_child == 0
        // (0 is the sentinel for "no child" in our schema)
        let is_leaf: Vec<bool> = schema.children_left.iter().map(|&left| left == 0).collect();

        let tree = Tree::new(
            schema.split_indices,
            schema.thresholds.into_iter().map(|t| t as f32).collect(),
            schema.children_left,
            schema.children_right,
            schema.default_left,
            is_leaf,
            leaf_values,
            split_types,
            categories,
            leaf_coefficients,
        );

        // Add gains and covers if present
        let tree = if let Some(gains) = schema.gains {
            tree.with_gains(gains.into_iter().map(|g| g as f32).collect())
        } else {
            tree
        };

        let tree = if let Some(covers) = schema.covers {
            tree.with_covers(covers.into_iter().map(|c| c as f32).collect())
        } else {
            tree
        };

        tree.validate().map_err(|e| {
            super::error::ReadError::Validation(format!("Invalid tree structure: {e:?}"))
        })?;

        Ok(tree)
    }
}

// =============================================================================
// Forest conversions
// =============================================================================

impl From<&Forest<ScalarLeaf>> for ForestSchema {
    fn from(forest: &Forest<ScalarLeaf>) -> Self {
        ForestSchema {
            trees: forest.trees().map(TreeSchema::from).collect(),
            tree_groups: if forest.n_groups() > 1 {
                Some(forest.tree_groups().iter().map(|&g| g as usize).collect())
            } else {
                None
            },
            n_groups: forest.n_groups() as usize,
            base_score: forest.base_score().iter().map(|&s| s as f64).collect(),
        }
    }
}

impl TryFrom<ForestSchema> for Forest<ScalarLeaf> {
    type Error = super::error::ReadError;

    fn try_from(schema: ForestSchema) -> Result<Self, Self::Error> {
        let n_groups_usize = schema.n_groups;
        let n_groups = n_groups_usize as u32;

        if schema.base_score.len() != n_groups_usize {
            return Err(super::error::ReadError::Validation(
                "Forest schema base_score must have length n_groups".into(),
            ));
        }

        if let Some(groups) = &schema.tree_groups
            && groups.len() != schema.trees.len()
        {
            return Err(super::error::ReadError::Validation(
                "Forest schema tree_groups must have same length as trees".into(),
            ));
        }

        // Set base score
        let base_scores: Vec<f32> = schema.base_score.iter().map(|&s| s as f32).collect();
        let mut forest = Forest::new(n_groups).with_base_score(base_scores);

        // Add trees
        let tree_groups: Vec<u32> = schema
            .tree_groups
            .map(|groups| groups.into_iter().map(|g| g as u32).collect())
            .unwrap_or_else(|| vec![0; schema.trees.len()]);

        for (tree_idx, tree_schema) in schema.trees.into_iter().enumerate() {
            let group = tree_groups[tree_idx];
            if group >= n_groups {
                return Err(super::error::ReadError::Validation(
                    "Forest schema tree group out of range".into(),
                ));
            }

            match &tree_schema.leaf_values {
                LeafValuesSchema::Scalar { .. } => {
                    let tree = Tree::try_from(tree_schema)?;
                    forest.push_tree(tree, group);
                }
                LeafValuesSchema::Vector { values } => {
                    let n_nodes = tree_schema.num_nodes as usize;
                    if values.len() != n_nodes {
                        return Err(super::error::ReadError::Validation(
                            "Tree schema leaf_values must have length num_nodes".into(),
                        ));
                    }

                    let k = values.first().map(|v| v.len()).unwrap_or(0);
                    if k == 0 || values.iter().any(|v| v.len() != k) {
                        return Err(super::error::ReadError::Validation(
                            "VectorLeaf values must have consistent non-zero length".into(),
                        ));
                    }

                    if k == 1 {
                        // Degenerate vector leaf: treat as scalar.
                        let scalar_values: Vec<f64> = values
                            .iter()
                            .map(|v| v.first().copied().unwrap_or(0.0))
                            .collect();

                        let TreeSchema {
                            num_nodes,
                            split_indices,
                            thresholds,
                            children_left,
                            children_right,
                            default_left,
                            leaf_values: _,
                            categories,
                            linear_coefficients,
                            gains,
                            covers,
                        } = tree_schema;

                        let tree = Tree::try_from(TreeSchema {
                            num_nodes,
                            split_indices,
                            thresholds,
                            children_left,
                            children_right,
                            default_left,
                            leaf_values: LeafValuesSchema::Scalar {
                                values: scalar_values,
                            },
                            categories,
                            linear_coefficients,
                            gains,
                            covers,
                        })?;
                        forest.push_tree(tree, group);
                    } else if k == n_groups_usize {
                        // Multi-output vector leaf tree: expand into K scalar trees.
                        let TreeSchema {
                            num_nodes,
                            split_indices,
                            thresholds,
                            children_left,
                            children_right,
                            default_left,
                            leaf_values: _,
                            categories,
                            linear_coefficients,
                            gains,
                            covers,
                        } = tree_schema;

                        for out_group in 0..n_groups_usize {
                            let scalar_values: Vec<f64> =
                                values.iter().map(|v| v[out_group]).collect();

                            let tree = Tree::try_from(TreeSchema {
                                num_nodes,
                                split_indices: split_indices.clone(),
                                thresholds: thresholds.clone(),
                                children_left: children_left.clone(),
                                children_right: children_right.clone(),
                                default_left: default_left.clone(),
                                leaf_values: LeafValuesSchema::Scalar {
                                    values: scalar_values,
                                },
                                categories: categories.clone(),
                                linear_coefficients: linear_coefficients.clone(),
                                gains: gains.clone(),
                                covers: covers.clone(),
                            })?;
                            forest.push_tree(tree, out_group as u32);
                        }
                    } else {
                        return Err(super::error::ReadError::Validation(
                            "VectorLeaf length must be 1 or equal to forest n_groups".into(),
                        ));
                    }
                }
            }
        }

        forest
            .validate()
            .map_err(|e| super::error::ReadError::Validation(format!("Invalid forest: {e:?}")))?;

        Ok(forest)
    }
}

#[cfg(test)]
mod roundtrip_tests {
    use super::*;
    use crate::persist::error::ReadError;
    use crate::repr::gbdt::{Forest, ScalarLeaf, Tree};

    fn minimal_tree_schema_scalar() -> TreeSchema {
        TreeSchema {
            num_nodes: 1,
            split_indices: vec![0],
            thresholds: vec![0.0],
            children_left: vec![0],
            children_right: vec![0],
            default_left: vec![false],
            leaf_values: LeafValuesSchema::Scalar { values: vec![1.0] },
            categories: CategoriesSchema::default(),
            linear_coefficients: LinearCoefficientsSchema::default(),
            gains: None,
            covers: None,
        }
    }

    #[test]
    fn tree_schema_mismatched_lengths_rejected() {
        let bad = TreeSchema {
            num_nodes: 1,
            split_indices: vec![0],
            thresholds: vec![],
            children_left: vec![0],
            children_right: vec![0],
            default_left: vec![false],
            leaf_values: LeafValuesSchema::Scalar { values: vec![1.0] },
            categories: CategoriesSchema::default(),
            linear_coefficients: LinearCoefficientsSchema::default(),
            gains: None,
            covers: None,
        };

        let err = Tree::<ScalarLeaf>::try_from(bad).unwrap_err();
        assert!(matches!(err, ReadError::Validation(_)));
    }

    #[test]
    fn tree_schema_out_of_bounds_child_rejected() {
        let bad = TreeSchema {
            num_nodes: 3,
            split_indices: vec![0, 0, 0],
            thresholds: vec![0.5, 0.0, 0.0],
            children_left: vec![1, 0, 0],
            children_right: vec![99, 0, 0],
            default_left: vec![true, false, false],
            leaf_values: LeafValuesSchema::Scalar {
                values: vec![0.0, 1.0, 2.0],
            },
            categories: CategoriesSchema::default(),
            linear_coefficients: LinearCoefficientsSchema::default(),
            gains: None,
            covers: None,
        };

        let err = Tree::<ScalarLeaf>::try_from(bad).unwrap_err();
        assert!(matches!(err, ReadError::Validation(_)));
    }

    #[test]
    fn forest_schema_tree_groups_len_mismatch_rejected() {
        let schema = ForestSchema {
            trees: vec![minimal_tree_schema_scalar()],
            tree_groups: Some(vec![0, 0]),
            n_groups: 1,
            base_score: vec![0.0],
        };
        let err = Forest::<ScalarLeaf>::try_from(schema).unwrap_err();
        assert!(matches!(err, ReadError::Validation(_)));
    }

    #[test]
    fn forest_schema_base_score_len_mismatch_rejected() {
        let schema = ForestSchema {
            trees: vec![minimal_tree_schema_scalar()],
            tree_groups: None,
            n_groups: 2,
            base_score: vec![0.0],
        };
        let err = Forest::<ScalarLeaf>::try_from(schema).unwrap_err();
        assert!(matches!(err, ReadError::Validation(_)));
    }

    #[test]
    fn vector_leaf_k1_is_accepted_as_scalar() {
        let tree_schema = TreeSchema {
            leaf_values: LeafValuesSchema::Vector {
                values: vec![vec![1.0]],
            },
            ..minimal_tree_schema_scalar()
        };

        let tree = Tree::<ScalarLeaf>::try_from(tree_schema).unwrap();
        tree.validate().unwrap();
    }

    #[test]
    fn vector_leaf_expands_to_group_trees() {
        let tree_schema = TreeSchema {
            leaf_values: LeafValuesSchema::Vector {
                values: vec![vec![1.0, 2.0]],
            },
            ..minimal_tree_schema_scalar()
        };

        let schema = ForestSchema {
            trees: vec![tree_schema],
            tree_groups: None,
            n_groups: 2,
            base_score: vec![0.0, 0.0],
        };

        let forest = Forest::<ScalarLeaf>::try_from(schema).unwrap();
        assert_eq!(forest.n_groups(), 2);
        assert_eq!(forest.n_trees(), 2);
        assert_eq!(forest.tree_groups(), &[0, 1]);
    }
}

// =============================================================================
// GBDTModel conversions
// =============================================================================

impl From<&GBDTModel> for GBDTModelSchema {
    fn from(model: &GBDTModel) -> Self {
        let mut meta = ModelMetaSchema::from(model.meta());
        meta.objective_name = None;

        GBDTModelSchema {
            meta,
            forest: ForestSchema::from(model.forest()),
            output_transform: model.output_transform(),
        }
    }
}

impl TryFrom<GBDTModelSchema> for GBDTModel {
    type Error = super::error::ReadError;

    fn try_from(schema: GBDTModelSchema) -> Result<Self, Self::Error> {
        let forest = Forest::try_from(schema.forest)?;
        let mut meta = ModelMeta::from(schema.meta);

        // Fill base_scores from forest
        meta.base_scores = forest.base_score().to_vec();

        Ok(GBDTModel::from_parts(forest, meta, schema.output_transform))
    }
}

// =============================================================================
// GBLinear conversions
// =============================================================================

impl From<&LinearModel> for LinearWeightsSchema {
    fn from(model: &LinearModel) -> Self {
        LinearWeightsSchema {
            values: model.as_slice().iter().map(|&v| v as f64).collect(),
            num_features: model.n_features(),
            num_groups: model.n_groups(),
        }
    }
}

impl From<LinearWeightsSchema> for LinearModel {
    fn from(schema: LinearWeightsSchema) -> Self {
        use ndarray::Array2;

        let n_rows = schema.num_features + 1; // +1 for bias
        let n_cols = schema.num_groups;

        let weights: Vec<f32> = schema.values.iter().map(|&v| v as f32).collect();
        let arr = Array2::from_shape_vec((n_rows, n_cols), weights)
            .expect("weight dimensions should match");

        LinearModel::new(arr)
    }
}

impl From<&GBLinearModel> for GBLinearModelSchema {
    fn from(model: &GBLinearModel) -> Self {
        let mut meta = ModelMetaSchema::from(model.meta());
        meta.objective_name = None;

        GBLinearModelSchema {
            meta,
            weights: LinearWeightsSchema::from(model.linear()),
            base_score: model.meta().base_scores.iter().map(|&s| s as f64).collect(),
            output_transform: model.output_transform(),
        }
    }
}

impl TryFrom<GBLinearModelSchema> for GBLinearModel {
    type Error = super::error::ReadError;

    fn try_from(schema: GBLinearModelSchema) -> Result<Self, Self::Error> {
        let linear = LinearModel::from(schema.weights);
        let mut meta = ModelMeta::from(schema.meta);

        // Fill base_scores from schema
        meta.base_scores = schema.base_score.iter().map(|&s| s as f32).collect();

        Ok(GBLinearModel::from_parts(
            linear,
            meta,
            schema.output_transform,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::OutputTransform;
    use crate::repr::gbdt::tree_view::TreeView;
    use crate::scalar_tree;

    #[test]
    fn model_meta_roundtrip() {
        let meta = ModelMeta::for_regression(10).with_feature_names(vec!["a".into(), "b".into()]);

        let schema = ModelMetaSchema::from(&meta);
        let restored = ModelMeta::from(schema);

        assert_eq!(meta.n_features, restored.n_features);
        assert_eq!(meta.feature_names, restored.feature_names);
    }

    #[test]
    fn tree_roundtrip() {
        let tree = scalar_tree! {
            0 => num(0, 0.5, L) -> 1, 2,
            1 => leaf(1.0),
            2 => num(1, 0.3, R) -> 3, 4,
            3 => leaf(2.0),
            4 => leaf(3.0),
        };

        let schema = TreeSchema::from(&tree);
        assert_eq!(schema.num_nodes, 5);
        assert_eq!(schema.thresholds[0], 0.5);

        let restored = Tree::try_from(schema).unwrap();
        assert_eq!(restored.n_nodes(), 5);
    }

    #[test]
    fn forest_roundtrip() {
        let tree = scalar_tree! {
            0 => num(0, 0.5, L) -> 1, 2,
            1 => leaf(1.0),
            2 => leaf(2.0),
        };

        let mut forest = Forest::for_regression().with_base_score(vec![0.5]);
        forest.push_tree(tree, 0);

        let schema = ForestSchema::from(&forest);
        assert_eq!(schema.trees.len(), 1);
        assert_eq!(schema.base_score, vec![0.5]);

        let restored = Forest::try_from(schema).unwrap();
        assert_eq!(restored.n_trees(), 1);
        assert_eq!(restored.base_score(), &[0.5]);
    }

    #[test]
    fn gbdt_model_roundtrip() {
        let tree = scalar_tree! {
            0 => num(0, 0.5, L) -> 1, 2,
            1 => leaf(1.0),
            2 => leaf(2.0),
        };

        let mut forest = Forest::for_regression().with_base_score(vec![0.5]);
        forest.push_tree(tree, 0);

        let meta = ModelMeta::for_binary_classification(2);
        let model = GBDTModel::from_parts(forest, meta, OutputTransform::Sigmoid);

        let schema = GBDTModelSchema::from(&model);
        let restored = GBDTModel::try_from(schema).unwrap();
        assert_eq!(restored.forest().n_trees(), 1);
        assert_eq!(restored.output_transform(), OutputTransform::Sigmoid);
    }

    #[test]
    fn linear_model_roundtrip() {
        use ndarray::array;

        let weights = array![[0.5], [0.3], [0.1]];
        let model = LinearModel::new(weights);

        let schema = LinearWeightsSchema::from(&model);
        assert_eq!(schema.num_features, 2);
        assert_eq!(schema.num_groups, 1);

        let restored = LinearModel::from(schema);
        assert_eq!(restored.n_features(), 2);
        assert!((restored.weight(0, 0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn gblinear_model_roundtrip() {
        use ndarray::array;

        let weights = array![[0.5], [0.3], [0.1]];
        let linear = LinearModel::new(weights);
        let meta = ModelMeta::for_binary_classification(2);
        let model = GBLinearModel::from_parts(linear, meta, OutputTransform::Sigmoid);

        let schema = GBLinearModelSchema::from(&model);
        let restored = GBLinearModel::try_from(schema).unwrap();
        assert_eq!(restored.linear().n_features(), 2);
        assert_eq!(restored.output_transform(), OutputTransform::Sigmoid);
    }

    #[test]
    fn bitset_roundtrip() {
        let cats = vec![1, 3, 5, 32, 64];
        let bitset = categories_to_bitset(&cats);
        let restored = bitset_to_categories(&bitset);
        assert_eq!(restored, cats);
    }
}

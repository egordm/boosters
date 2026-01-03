//! Conversion between runtime types and schema types.
//!
//! This module provides conversions between the runtime model types
//! (e.g., `GBDTModel`, `Forest`, `Tree`) and their schema counterparts
//! (e.g., `GBDTModelSchema`, `ForestSchema`, `TreeSchema`).
//!
//! Conversions are implemented as `From` traits for lossless transformations.

use std::collections::BTreeMap;

use super::schema::{
    BinningConfigSchema, CategoriesSchema, FeatureSelectorSchema, FeatureTypeSchema, ForestSchema,
    GBDTConfigSchema, GBDTModelSchema, GBLinearConfigSchema, GBLinearModelSchema,
    GrowthStrategySchema, LeafValuesSchema, LinearCoefficientsSchema, LinearLeafConfigSchema,
    LinearWeightsSchema, MetricSchema, ModelMetaSchema, ObjectiveSchema, TaskKindSchema,
    TreeSchema, UpdateStrategySchema, VerbositySchema,
};
use crate::model::gbdt::GBDTConfig;
use crate::model::gblinear::GBLinearConfig;
use crate::model::{FeatureType, GBDTModel, GBLinearModel, ModelMeta, TaskKind};
use crate::repr::gbdt::{Forest, ScalarLeaf, Tree};
use crate::repr::gblinear::LinearModel;
use crate::training::gbdt::GrowthStrategy;

// =============================================================================
// Objective / Metric conversions (schema <-> runtime)
// =============================================================================

impl From<&crate::training::Objective> for ObjectiveSchema {
    fn from(obj: &crate::training::Objective) -> Self {
        match obj {
            crate::training::Objective::SquaredLoss(_) => Self::SquaredLoss,
            crate::training::Objective::AbsoluteLoss(_) => Self::AbsoluteLoss,
            crate::training::Objective::LogisticLoss(_) => Self::LogisticLoss,
            crate::training::Objective::HingeLoss(_) => Self::HingeLoss,
            crate::training::Objective::SoftmaxLoss(inner) => Self::SoftmaxLoss {
                n_classes: inner.n_classes,
            },
            crate::training::Objective::PinballLoss(inner) => Self::PinballLoss {
                alphas: inner.alphas.iter().map(|&a| a as f64).collect(),
            },
            crate::training::Objective::PseudoHuberLoss(inner) => Self::PseudoHuberLoss {
                delta: inner.delta as f64,
            },
            crate::training::Objective::PoissonLoss(_) => Self::PoissonLoss,
            crate::training::Objective::Custom(inner) => Self::Custom {
                name: inner.name().to_string(),
            },
        }
    }
}

impl TryFrom<ObjectiveSchema> for crate::training::Objective {
    type Error = super::error::ReadError;

    fn try_from(schema: ObjectiveSchema) -> Result<Self, Self::Error> {
        Ok(match schema {
            ObjectiveSchema::SquaredLoss => Self::squared(),
            ObjectiveSchema::AbsoluteLoss => Self::absolute(),
            ObjectiveSchema::LogisticLoss => Self::logistic(),
            ObjectiveSchema::HingeLoss => Self::hinge(),
            ObjectiveSchema::SoftmaxLoss { n_classes } => Self::softmax(n_classes),
            ObjectiveSchema::PinballLoss { alphas } => {
                Self::multi_quantile(alphas.into_iter().map(|a| a as f32).collect())
            }
            ObjectiveSchema::PseudoHuberLoss { delta } => Self::pseudo_huber(delta as f32),
            ObjectiveSchema::PoissonLoss => Self::poisson(),
            ObjectiveSchema::Custom { name } => {
                return Err(super::error::ReadError::Validation(format!(
                    "cannot deserialize custom objective: {name}"
                )));
            }
        })
    }
}

impl From<&crate::training::Metric> for MetricSchema {
    fn from(metric: &crate::training::Metric) -> Self {
        match metric {
            crate::training::Metric::None => Self::None,
            crate::training::Metric::Rmse(_) => Self::Rmse,
            crate::training::Metric::Mae(_) => Self::Mae,
            crate::training::Metric::Mape(_) => Self::Mape,
            crate::training::Metric::LogLoss(_) => Self::LogLoss,
            crate::training::Metric::Accuracy(inner) => Self::Accuracy {
                threshold: inner.threshold as f64,
            },
            crate::training::Metric::MarginAccuracy(_) => Self::MarginAccuracy,
            crate::training::Metric::Auc(_) => Self::Auc,
            crate::training::Metric::MulticlassLogLoss(_) => Self::MulticlassLogLoss,
            crate::training::Metric::MulticlassAccuracy(_) => Self::MulticlassAccuracy,
            crate::training::Metric::Quantile(inner) => Self::Quantile {
                alphas: inner.alphas.iter().map(|&a| a as f64).collect(),
            },
            crate::training::Metric::Huber(inner) => Self::Huber { delta: inner.delta },
            crate::training::Metric::PoissonDeviance(_) => Self::PoissonDeviance,
            crate::training::Metric::Custom(inner) => Self::Custom {
                name: inner.name().to_string(),
            },
        }
    }
}

impl TryFrom<MetricSchema> for crate::training::Metric {
    type Error = super::error::ReadError;

    fn try_from(schema: MetricSchema) -> Result<Self, Self::Error> {
        Ok(match schema {
            MetricSchema::None => Self::none(),
            MetricSchema::Rmse => Self::rmse(),
            MetricSchema::Mae => Self::mae(),
            MetricSchema::Mape => Self::mape(),
            MetricSchema::LogLoss => Self::logloss(),
            MetricSchema::Accuracy { threshold } => Self::accuracy_with_threshold(threshold as f32),
            MetricSchema::MarginAccuracy => Self::margin_accuracy(),
            MetricSchema::Auc => Self::auc(),
            MetricSchema::MulticlassLogLoss => Self::multiclass_logloss(),
            MetricSchema::MulticlassAccuracy => Self::multiclass_accuracy(),
            MetricSchema::Quantile { alphas } => {
                Self::multi_quantile(alphas.into_iter().map(|a| a as f32).collect())
            }
            MetricSchema::Huber { delta } => Self::huber(delta as f32),
            MetricSchema::PoissonDeviance => Self::poisson_deviance(),
            MetricSchema::Custom { name } => {
                return Err(super::error::ReadError::Validation(format!(
                    "cannot deserialize custom metric: {name}"
                )));
            }
        })
    }
}

// =============================================================================
// TaskKind conversions
// =============================================================================

impl From<TaskKind> for TaskKindSchema {
    fn from(task: TaskKind) -> Self {
        match task {
            TaskKind::Regression => TaskKindSchema::Regression,
            TaskKind::BinaryClassification => TaskKindSchema::BinaryClassification,
            TaskKind::MulticlassClassification { .. } => TaskKindSchema::MulticlassClassification,
            TaskKind::Ranking => TaskKindSchema::Ranking,
        }
    }
}

impl From<TaskKindSchema> for TaskKind {
    fn from(task: TaskKindSchema) -> Self {
        match task {
            TaskKindSchema::Regression => TaskKind::Regression,
            TaskKindSchema::BinaryClassification => TaskKind::BinaryClassification,
            // num_classes will be filled from ModelMetaSchema.num_classes
            TaskKindSchema::MulticlassClassification => {
                TaskKind::MulticlassClassification { n_classes: 0 }
            }
            TaskKindSchema::Ranking => TaskKind::Ranking,
        }
    }
}

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
        let num_classes = match meta.task {
            TaskKind::MulticlassClassification { n_classes } => Some(n_classes),
            _ => None,
        };

        ModelMetaSchema {
            task: meta.task.into(),
            num_features: meta.n_features,
            num_classes,
            feature_names: meta.feature_names.clone(),
            feature_types: meta
                .feature_types
                .as_ref()
                .map(|types| types.iter().map(FeatureTypeSchema::from).collect()),
        }
    }
}

impl From<ModelMetaSchema> for ModelMeta {
    fn from(schema: ModelMetaSchema) -> Self {
        let task = match schema.task {
            TaskKindSchema::MulticlassClassification => TaskKind::MulticlassClassification {
                n_classes: schema.num_classes.unwrap_or(0),
            },
            other => other.into(),
        };

        ModelMeta {
            task,
            n_features: schema.num_features,
            n_groups: task.n_groups(),
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

        // Convert leaf values
        let leaf_values: Vec<ScalarLeaf> = match schema.leaf_values {
            LeafValuesSchema::Scalar { values } => {
                values.into_iter().map(|v| ScalarLeaf(v as f32)).collect()
            }
            LeafValuesSchema::Vector { .. } => {
                return Err(super::error::ReadError::Validation(
                    "VectorLeaf not yet supported for Tree<ScalarLeaf>".into(),
                ));
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
        let n_groups = schema.n_groups as u32;
        let mut forest = Forest::new(n_groups);

        // Set base score
        let base_scores: Vec<f32> = schema.base_score.iter().map(|&s| s as f32).collect();
        forest = forest.with_base_score(base_scores);

        // Add trees
        let tree_groups: Vec<u32> = schema
            .tree_groups
            .map(|groups| groups.iter().map(|&g| g as u32).collect())
            .unwrap_or_else(|| vec![0; schema.trees.len()]);

        for (tree_schema, group) in schema.trees.into_iter().zip(tree_groups) {
            let tree = Tree::try_from(tree_schema)?;
            forest.push_tree(tree, group);
        }

        Ok(forest)
    }
}

// =============================================================================
// GBDTConfig conversions
// =============================================================================

impl From<GrowthStrategy> for GrowthStrategySchema {
    fn from(gs: GrowthStrategy) -> Self {
        match gs {
            GrowthStrategy::DepthWise { max_depth } => Self::DepthWise { max_depth },
            GrowthStrategy::LeafWise { max_leaves } => Self::LeafWise { max_leaves },
        }
    }
}

impl From<GrowthStrategySchema> for GrowthStrategy {
    fn from(gs: GrowthStrategySchema) -> Self {
        match gs {
            GrowthStrategySchema::DepthWise { max_depth } => Self::DepthWise { max_depth },
            GrowthStrategySchema::LeafWise { max_leaves } => Self::LeafWise { max_leaves },
        }
    }
}

impl From<&crate::data::BinningConfig> for BinningConfigSchema {
    fn from(cfg: &crate::data::BinningConfig) -> Self {
        Self {
            max_bins: cfg.max_bins,
            sparsity_threshold: cfg.sparsity_threshold as f64,
            enable_bundling: cfg.enable_bundling,
            max_categorical_cardinality: cfg.max_categorical_cardinality,
            sample_cnt: cfg.sample_cnt,
        }
    }
}

impl From<BinningConfigSchema> for crate::data::BinningConfig {
    fn from(cfg: BinningConfigSchema) -> Self {
        Self {
            max_bins: cfg.max_bins,
            sparsity_threshold: cfg.sparsity_threshold as f32,
            enable_bundling: cfg.enable_bundling,
            max_categorical_cardinality: cfg.max_categorical_cardinality,
            sample_cnt: cfg.sample_cnt,
        }
    }
}

impl From<&crate::training::gbdt::LinearLeafConfig> for LinearLeafConfigSchema {
    fn from(cfg: &crate::training::gbdt::LinearLeafConfig) -> Self {
        Self {
            lambda: cfg.lambda as f64,
            alpha: cfg.alpha as f64,
            max_iterations: cfg.max_iterations,
            tolerance: cfg.tolerance,
            min_samples: cfg.min_samples,
            coefficient_threshold: cfg.coefficient_threshold as f64,
            max_features: cfg.max_features,
        }
    }
}

impl From<LinearLeafConfigSchema> for crate::training::gbdt::LinearLeafConfig {
    fn from(cfg: LinearLeafConfigSchema) -> Self {
        crate::training::gbdt::LinearLeafConfig::builder()
            .lambda(cfg.lambda as f32)
            .alpha(cfg.alpha as f32)
            .max_iterations(cfg.max_iterations)
            .tolerance(cfg.tolerance)
            .min_samples(cfg.min_samples)
            .coefficient_threshold(cfg.coefficient_threshold as f32)
            .max_features(cfg.max_features)
            .build()
    }
}

impl From<crate::training::Verbosity> for VerbositySchema {
    fn from(v: crate::training::Verbosity) -> Self {
        match v {
            crate::training::Verbosity::Silent => Self::Silent,
            crate::training::Verbosity::Warning => Self::Warning,
            crate::training::Verbosity::Info => Self::Info,
            crate::training::Verbosity::Debug => Self::Debug,
        }
    }
}

impl From<VerbositySchema> for crate::training::Verbosity {
    fn from(v: VerbositySchema) -> Self {
        match v {
            VerbositySchema::Silent => Self::Silent,
            VerbositySchema::Warning => Self::Warning,
            VerbositySchema::Info => Self::Info,
            VerbositySchema::Debug => Self::Debug,
        }
    }
}

impl From<&GBDTConfig> for GBDTConfigSchema {
    fn from(config: &GBDTConfig) -> Self {
        GBDTConfigSchema {
            objective: ObjectiveSchema::from(&config.objective),
            metric: config.metric.as_ref().map(MetricSchema::from),
            n_trees: config.n_trees,
            learning_rate: config.learning_rate as f64,
            growth_strategy: GrowthStrategySchema::from(config.growth_strategy),
            max_onehot_cats: config.max_onehot_cats,
            lambda: config.lambda as f64,
            alpha: config.alpha as f64,
            min_child_weight: config.min_child_weight as f64,
            min_gain: config.min_gain as f64,
            min_samples_leaf: config.min_samples_leaf,
            subsample: config.subsample as f64,
            colsample_bytree: config.colsample_bytree as f64,
            colsample_bylevel: config.colsample_bylevel as f64,
            binning: BinningConfigSchema::from(&config.binning),
            linear_leaves: config
                .linear_leaves
                .as_ref()
                .map(LinearLeafConfigSchema::from),
            early_stopping_rounds: config.early_stopping_rounds,
            cache_size: config.cache_size,
            seed: config.seed,
            verbosity: VerbositySchema::from(config.verbosity),
            extra: BTreeMap::new(),
        }
    }
}

// =============================================================================
// GBDTModel conversions
// =============================================================================

impl From<&GBDTModel> for GBDTModelSchema {
    fn from(model: &GBDTModel) -> Self {
        GBDTModelSchema {
            meta: ModelMetaSchema::from(model.meta()),
            forest: ForestSchema::from(model.forest()),
            config: GBDTConfigSchema::from(model.config()),
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

        let cfg_schema = schema.config;
        let config = GBDTConfig {
            objective: crate::training::Objective::try_from(cfg_schema.objective)?,
            metric: match cfg_schema.metric {
                None => None,
                Some(m) => Some(crate::training::Metric::try_from(m)?),
            },
            n_trees: cfg_schema.n_trees,
            learning_rate: cfg_schema.learning_rate as f32,
            growth_strategy: GrowthStrategy::from(cfg_schema.growth_strategy),
            max_onehot_cats: cfg_schema.max_onehot_cats,
            lambda: cfg_schema.lambda as f32,
            alpha: cfg_schema.alpha as f32,
            min_child_weight: cfg_schema.min_child_weight as f32,
            min_gain: cfg_schema.min_gain as f32,
            min_samples_leaf: cfg_schema.min_samples_leaf,
            subsample: cfg_schema.subsample as f32,
            colsample_bytree: cfg_schema.colsample_bytree as f32,
            colsample_bylevel: cfg_schema.colsample_bylevel as f32,
            binning: crate::data::BinningConfig::from(cfg_schema.binning),
            linear_leaves: cfg_schema
                .linear_leaves
                .map(crate::training::gbdt::LinearLeafConfig::from),
            early_stopping_rounds: cfg_schema.early_stopping_rounds,
            cache_size: cfg_schema.cache_size,
            seed: cfg_schema.seed,
            verbosity: crate::training::Verbosity::from(cfg_schema.verbosity),
        };

        Ok(GBDTModel::from_parts(forest, meta, config))
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

impl From<&GBLinearConfig> for GBLinearConfigSchema {
    fn from(config: &GBLinearConfig) -> Self {
        GBLinearConfigSchema {
            objective: ObjectiveSchema::from(&config.objective),
            metric: config.metric.as_ref().map(MetricSchema::from),
            n_rounds: config.n_rounds,
            learning_rate: config.learning_rate as f64,
            alpha: config.alpha as f64,
            lambda: config.lambda as f64,
            update_strategy: UpdateStrategySchema::from(config.update_strategy),
            feature_selector: FeatureSelectorSchema::from(config.feature_selector),
            max_delta_step: config.max_delta_step as f64,
            early_stopping_rounds: config.early_stopping_rounds,
            seed: config.seed,
            verbosity: VerbositySchema::from(config.verbosity),
            extra: BTreeMap::new(),
        }
    }
}

impl From<crate::training::gblinear::UpdateStrategy> for UpdateStrategySchema {
    fn from(v: crate::training::gblinear::UpdateStrategy) -> Self {
        match v {
            crate::training::gblinear::UpdateStrategy::Shotgun => Self::Shotgun,
            crate::training::gblinear::UpdateStrategy::Sequential => Self::Sequential,
        }
    }
}

impl From<UpdateStrategySchema> for crate::training::gblinear::UpdateStrategy {
    fn from(v: UpdateStrategySchema) -> Self {
        match v {
            UpdateStrategySchema::Shotgun => Self::Shotgun,
            UpdateStrategySchema::Sequential => Self::Sequential,
        }
    }
}

impl From<crate::training::gblinear::FeatureSelectorKind> for FeatureSelectorSchema {
    fn from(v: crate::training::gblinear::FeatureSelectorKind) -> Self {
        use crate::training::gblinear::FeatureSelectorKind;
        match v {
            FeatureSelectorKind::Cyclic => Self::Cyclic,
            FeatureSelectorKind::Shuffle => Self::Shuffle,
            FeatureSelectorKind::Random => Self::Random,
            FeatureSelectorKind::Greedy { top_k } => Self::Greedy { top_k },
            FeatureSelectorKind::Thrifty { top_k } => Self::Thrifty { top_k },
        }
    }
}

impl From<FeatureSelectorSchema> for crate::training::gblinear::FeatureSelectorKind {
    fn from(v: FeatureSelectorSchema) -> Self {
        use crate::training::gblinear::FeatureSelectorKind;
        match v {
            FeatureSelectorSchema::Cyclic => FeatureSelectorKind::Cyclic,
            FeatureSelectorSchema::Shuffle => FeatureSelectorKind::Shuffle,
            FeatureSelectorSchema::Random => FeatureSelectorKind::Random,
            FeatureSelectorSchema::Greedy { top_k } => FeatureSelectorKind::Greedy { top_k },
            FeatureSelectorSchema::Thrifty { top_k } => FeatureSelectorKind::Thrifty { top_k },
        }
    }
}

impl From<&GBLinearModel> for GBLinearModelSchema {
    fn from(model: &GBLinearModel) -> Self {
        GBLinearModelSchema {
            meta: ModelMetaSchema::from(model.meta()),
            weights: LinearWeightsSchema::from(model.linear()),
            base_score: model.meta().base_scores.iter().map(|&s| s as f64).collect(),
            config: GBLinearConfigSchema::from(model.config()),
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

        let cfg_schema = schema.config;
        let config = GBLinearConfig {
            objective: crate::training::Objective::try_from(cfg_schema.objective)?,
            metric: match cfg_schema.metric {
                None => None,
                Some(m) => Some(crate::training::Metric::try_from(m)?),
            },
            n_rounds: cfg_schema.n_rounds,
            learning_rate: cfg_schema.learning_rate as f32,
            alpha: cfg_schema.alpha as f32,
            lambda: cfg_schema.lambda as f32,
            update_strategy: crate::training::gblinear::UpdateStrategy::from(
                cfg_schema.update_strategy,
            ),
            feature_selector: crate::training::gblinear::FeatureSelectorKind::from(
                cfg_schema.feature_selector,
            ),
            max_delta_step: cfg_schema.max_delta_step as f32,
            early_stopping_rounds: cfg_schema.early_stopping_rounds,
            seed: cfg_schema.seed,
            verbosity: crate::training::Verbosity::from(cfg_schema.verbosity),
        };

        Ok(GBLinearModel::from_parts(linear, meta, config))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repr::gbdt::tree_view::TreeView;
    use crate::scalar_tree;
    use crate::training::ObjectiveFn;

    #[test]
    fn task_kind_roundtrip() {
        let tasks = [
            TaskKind::Regression,
            TaskKind::BinaryClassification,
            TaskKind::MulticlassClassification { n_classes: 5 },
            TaskKind::Ranking,
        ];

        for task in tasks {
            let schema: TaskKindSchema = task.into();
            let restored: TaskKind = schema.into();

            match (task, restored) {
                (TaskKind::Regression, TaskKind::Regression) => {}
                (TaskKind::BinaryClassification, TaskKind::BinaryClassification) => {}
                (
                    TaskKind::MulticlassClassification { .. },
                    TaskKind::MulticlassClassification { .. },
                ) => {}
                (TaskKind::Ranking, TaskKind::Ranking) => {}
                _ => panic!("Task kind mismatch: {:?} vs {:?}", task, restored),
            }
        }
    }

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
        let config = GBDTConfig::builder()
            .objective(crate::training::Objective::logistic())
            .n_trees(10)
            .learning_rate(0.2)
            .build()
            .unwrap();

        let model = GBDTModel::from_parts(forest, meta, config);

        let schema = GBDTModelSchema::from(&model);
        let restored = GBDTModel::try_from(schema).unwrap();
        assert_eq!(restored.forest().n_trees(), 1);
        assert_eq!(restored.config().objective.name(), "logistic");
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
        let config = GBLinearConfig::builder()
            .objective(crate::training::Objective::logistic())
            .n_rounds(5)
            .learning_rate(0.3)
            .build()
            .unwrap();

        let model = GBLinearModel::from_parts(linear, meta, config);

        let schema = GBLinearModelSchema::from(&model);
        let restored = GBLinearModel::try_from(schema).unwrap();
        assert_eq!(restored.linear().n_features(), 2);
        assert_eq!(restored.config().objective.name(), "logistic");
    }

    #[test]
    fn bitset_roundtrip() {
        let cats = vec![1, 3, 5, 32, 64];
        let bitset = categories_to_bitset(&cats);
        let restored = bitset_to_categories(&bitset);
        assert_eq!(restored, cats);
    }
}

//! Gradient boosting trainer for tree ensembles.
//!
//! This module implements the full boosting loop for training GBTree models.
//! It coordinates tree building, gradient computation, and prediction updates.
//!
//! # Example
//!
//! The simplest way to train:
//!
//! ```ignore
//! use booste_rs::training::GBTreeTrainer;
//!
//! let trainer = GBTreeTrainer::default();
//! let forest = trainer.train(&data, &labels, None, &[]);
//! ```
//!
//! For more control, use the builder:
//!
//! ```ignore
//! use booste_rs::training::{GBTreeTrainer, LossFunction};
//!
//! let trainer = GBTreeTrainer::builder()
//!     .loss(LossFunction::SquaredError)
//!     .num_rounds(100)
//!     .max_depth(6)
//!     .learning_rate(0.1)
//!     .build()
//!     .unwrap();
//!
//! let forest = trainer.train(&data, &labels, None, &[]);
//! ```
//!
//! For multiclass/multi-output training, use `LossFunction::Softmax`:
//!
//! ```ignore
//! use booste_rs::training::{GBTreeTrainer, LossFunction};
//!
//! let trainer = GBTreeTrainer::builder()
//!     .loss(LossFunction::Softmax { num_classes: 3 })
//!     .build()
//!     .unwrap();
//! let forest = trainer.train(&data, &labels, None, &[]);
//! ```
//!
//! See RFC-0015 for design rationale.

use derive_builder::Builder;

use super::constraints::{InteractionConstraints, MonotonicConstraints};
use crate::data::ColumnAccess;
use crate::forest::SoAForest;
use crate::training::metric::EvalSet;
use crate::training::{
    EarlyStopping, EvalMetric, GradientBuffer, Loss, LossFunction, Metric, TrainingLogger,
    Verbosity,
};
use crate::trees::{
    ScalarLeaf, SoATreeStorage, TreeBuilder as SoATreeBuilder, categories_to_bitset,
};

use super::constraints::MonotonicConstraint;
use super::grower::{BuildingTree, GrowthStrategy, ParallelStrategy, TreeBuildParams, TreeGrower};
use super::partition::RowPartitioner;
use super::quantize::{BinCuts, BinIndex, ExactQuantileCuts, QuantizedMatrix, Quantizer};
use super::sampling::{ColumnSampler, RowSampler, RowSampling};
use super::split::GainParams;

// ============================================================================
// GBTreeTrainer
// ============================================================================

/// Gradient boosted tree trainer with all parameters inlined.
///
/// Use [`GBTreeTrainer::builder()`] for a fluent configuration API,
/// or [`GBTreeTrainer::default()`] for sensible defaults.
///
/// # Example
///
/// ```ignore
/// use booste_rs::training::{GBTreeTrainer, LossFunction};
///
/// // Simple usage with defaults
/// let trainer = GBTreeTrainer::default();
/// let forest = trainer.train(&data, &labels, None, &[]);
///
/// // Configured via builder
/// let trainer = GBTreeTrainer::builder()
///     .loss(LossFunction::Logistic)
///     .num_rounds(50)
///     .growth_strategy(GrowthStrategy::DepthWise { max_depth: 4 })
///     .learning_rate(0.1)
///     .build()
///     .unwrap();
/// ```
#[derive(Debug, Clone, Builder)]
#[builder(setter(into), default)]
pub struct GBTreeTrainer {
    // ========================================================================
    // Loss function
    // ========================================================================
    /// Loss function for gradient computation.
    #[builder(default)]
    pub loss: LossFunction,

    // ========================================================================
    // Training parameters
    // ========================================================================
    /// Number of boosting rounds (trees to build).
    #[builder(default = "100")]
    pub num_rounds: u32,

    /// Learning rate (shrinkage) applied to leaf weights.
    #[builder(default = "0.3")]
    pub learning_rate: f32,

    /// Maximum number of histogram bins per feature.
    #[builder(default = "256")]
    pub max_bins: usize,

    /// Random seed for reproducibility.
    #[builder(default = "0")]
    pub seed: u64,

    // ========================================================================
    // Tree structure
    // ========================================================================
    /// Growth strategy: depth-wise (XGBoost) or leaf-wise (LightGBM).
    ///
    /// - `GrowthStrategy::DepthWise { max_depth }`: Expand all nodes at each level
    /// - `GrowthStrategy::LeafWise { max_leaves }`: Always expand best-gain leaf
    #[builder(default)]
    pub growth_strategy: GrowthStrategy,

    /// Minimum samples required to split a node.
    #[builder(default = "2")]
    pub min_samples_split: u32,

    /// Minimum samples required in a leaf.
    #[builder(default = "1")]
    pub min_samples_leaf: u32,

    // ========================================================================
    // Regularization
    // ========================================================================
    /// L2 regularization on leaf weights (XGBoost's `lambda`).
    #[builder(default = "1.0")]
    pub reg_lambda: f32,

    /// L1 regularization on leaf weights (XGBoost's `alpha`).
    #[builder(default = "0.0")]
    pub reg_alpha: f32,

    /// Minimum loss reduction to make a split (XGBoost's `gamma`).
    #[builder(default = "0.0")]
    pub min_split_gain: f32,

    /// Minimum sum of hessians in a child.
    #[builder(default = "1.0")]
    pub min_child_weight: f32,

    // ========================================================================
    // Sampling
    // ========================================================================
    /// Row sampling strategy.
    ///
    /// Options:
    /// - `RowSampling::None` (default): Use all rows
    /// - `RowSampling::Random { rate: 0.8 }`: Random 80% subsample
    /// - `RowSampling::Goss { top_rate, other_rate }`: Gradient-based one-side sampling
    #[builder(default)]
    pub row_sampling: RowSampling,

    /// Column subsampling ratio per tree (0, 1].
    #[builder(default = "1.0")]
    pub colsample_bytree: f32,

    /// Column subsampling ratio per level (0, 1].
    #[builder(default = "1.0")]
    pub colsample_bylevel: f32,

    /// Column subsampling ratio per node (0, 1].
    #[builder(default = "1.0")]
    pub colsample_bynode: f32,

    // ========================================================================
    // Constraints
    // ========================================================================
    /// Monotonic constraints per feature (-1: decreasing, 0: none, 1: increasing).
    #[builder(default)]
    pub monotone_constraints: Vec<MonotonicConstraint>,

    /// Interaction constraints as groups of features.
    #[builder(default)]
    pub interaction_constraints: Vec<Vec<u32>>,

    // ========================================================================
    // Logging and callbacks
    // ========================================================================
    /// Verbosity level for logging.
    #[builder(default)]
    pub verbosity: Verbosity,

    /// Evaluation metric for logging (used when eval sets provided).
    #[builder(default)]
    pub eval_metric: EvalMetric,

    /// Early stopping patience (0 = disabled).
    #[builder(default = "0")]
    pub early_stopping_rounds: usize,

    /// Index of evaluation set to use for early stopping (0 = first eval set).
    #[builder(default = "0")]
    pub early_stopping_eval_set: usize,

    // ========================================================================
    // Internal / advanced
    // ========================================================================
    /// Parallelization strategy for histogram building.
    #[builder(default)]
    pub parallel_strategy: ParallelStrategy,
}

impl Default for GBTreeTrainer {
    fn default() -> Self {
        Self {
            loss: LossFunction::default(),
            num_rounds: 100,
            learning_rate: 0.3,
            max_bins: 256,
            seed: 0,
            growth_strategy: GrowthStrategy::default(),
            min_samples_split: 2,
            min_samples_leaf: 1,
            reg_lambda: 1.0,
            reg_alpha: 0.0,
            min_split_gain: 0.0,
            min_child_weight: 1.0,
            row_sampling: RowSampling::None,
            colsample_bytree: 1.0,
            colsample_bylevel: 1.0,
            colsample_bynode: 1.0,
            monotone_constraints: Vec::new(),
            interaction_constraints: Vec::new(),
            verbosity: Verbosity::default(),
            eval_metric: EvalMetric::default(),
            early_stopping_rounds: 0,
            early_stopping_eval_set: 0,
            parallel_strategy: ParallelStrategy::default(),
        }
    }
}

impl GBTreeTrainer {
    /// Create a new trainer with default parameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a builder for configuring the trainer.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let trainer = GBTreeTrainer::builder()
    ///     .num_rounds(100)
    ///     .max_depth(6)
    ///     .build()
    ///     .unwrap();
    /// ```
    pub fn builder() -> GBTreeTrainerBuilder {
        GBTreeTrainerBuilder::default()
    }

    /// Build internal TreeBuildParams from the flat config.
    fn tree_build_params(&self) -> TreeBuildParams {
        let (max_depth, max_leaves) = match self.growth_strategy {
            GrowthStrategy::DepthWise { max_depth } => (max_depth, 0),
            GrowthStrategy::LeafWise { max_leaves } => (0, max_leaves),
        };
        TreeBuildParams {
            gain: GainParams {
                lambda: self.reg_lambda,
                alpha: self.reg_alpha,
                min_split_gain: self.min_split_gain,
                min_child_weight: self.min_child_weight,
            },
            max_depth,
            max_leaves,
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
            parallel_strategy: self.parallel_strategy,
            min_rows_for_parallel: 1024,
            row_parallel_threshold: 4.0,
        }
    }

    /// Train a gradient boosted forest from raw (unquantized) data.
    ///
    /// This is the primary training API. It handles quantization automatically.
    /// The loss function determines whether single-output or multi-output training is used:
    /// - Single-output losses (`SquaredError`, `Logistic`, etc.): one tree per round
    /// - Multi-output losses (`Softmax`, `MultiQuantile`): K trees per round
    ///
    /// # Arguments
    ///
    /// * `data` - Raw feature matrix (any type implementing `ColumnAccess`)
    /// * `labels` - Target labels
    /// * `weights` - Optional per-sample weights. When provided, each sample's
    ///   contribution to the loss function is scaled by its weight. Pass `None`
    ///   for uniform weighting (equivalent to all weights = 1.0).
    /// * `eval_sets` - Evaluation sets for monitoring (pass `&[]` if not needed)
    ///
    /// # Sample Weights
    ///
    /// Weights allow you to control how much each training sample influences the
    /// model. Common use cases include:
    ///
    /// - **Class imbalance**: Give higher weights to minority class samples
    /// - **Importance sampling**: Emphasize certain samples over others
    /// - **Sample exclusion**: Set weight to 0.0 to exclude samples without
    ///   removing them from the dataset
    ///
    /// Weights must be non-negative. They are **not normalized** internally -
    /// the sum of weights affects regularization strength. To match XGBoost
    /// behavior, use unnormalized weights (this is XGBoost's default).
    ///
    /// # Example
    ///
    /// ```ignore
    /// use booste_rs::training::{GBTreeTrainer, LossFunction};
    ///
    /// // Regression (unweighted)
    /// let trainer = GBTreeTrainer::default();
    /// let forest = trainer.train(&data, &labels, None, &[]);
    ///
    /// // Classification with class imbalance handling
    /// // Give 10x weight to minority class (label=1)
    /// let weights: Vec<f32> = labels.iter()
    ///     .map(|&label| if label > 0.5 { 10.0 } else { 1.0 })
    ///     .collect();
    /// let trainer = GBTreeTrainer::builder()
    ///     .loss(LossFunction::Logistic)
    ///     .build()
    ///     .unwrap();
    /// let forest = trainer.train(&data, &labels, Some(&weights), &[]);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `weights.len() != labels.len()` when weights are provided.
    pub fn train<D>(
        &self,
        data: &D,
        labels: &[f32],
        weights: Option<&[f32]>,
        eval_sets: &[EvalSet<'_, D>],
    ) -> SoAForest<ScalarLeaf>
    where
        D: ColumnAccess<Element = f32> + Sync,
    {
        // Compute bin cuts from training data
        let cut_finder = ExactQuantileCuts::default();
        let quantizer = Quantizer::from_data(data, &cut_finder, self.max_bins);
        let cuts = quantizer.cuts().clone();

        // Quantize training data
        let quantized: QuantizedMatrix<u8> = quantizer.quantize(data);

        // Quantize evaluation sets
        let quantized_eval_sets: Vec<QuantizedMatrix<u8>> = eval_sets
            .iter()
            .map(|es| quantizer.quantize(es.data))
            .collect();

        // Create QuantizedEvalSet references
        let quantized_refs: Vec<QuantizedEvalSet<'_, u8>> = eval_sets
            .iter()
            .zip(quantized_eval_sets.iter())
            .map(|(es, q)| QuantizedEvalSet::new(es.name, q, es.labels))
            .collect();

        self.train_quantized(&quantized, labels, weights, &cuts, &quantized_refs)
    }

    // ========================================================================
    // Quantized training implementation (unified for single and multi-output)
    // ========================================================================

    /// Train on pre-quantized data.
    ///
    /// This is the core training method that works on pre-quantized data.
    /// Use [`train()`](Self::train) for the high-level API that handles quantization.
    ///
    /// Handles both single-output (regression, binary classification) and
    /// multi-output (multiclass, multi-quantile) training through the same code path.
    /// For single-output, `num_outputs = 1` and we build 1 tree per round.
    /// For multi-output, `num_outputs = K` and we build K trees per round.
    ///
    /// # Arguments
    ///
    /// * `quantized` - Pre-quantized feature matrix
    /// * `labels` - Target labels
    /// * `weights` - Optional sample weights (must match labels length)
    /// * `cuts` - Bin boundaries used for quantization
    /// * `eval_sets` - Evaluation sets for monitoring
    ///
    /// # Panics
    ///
    /// Panics if `weights.len() != labels.len()` when weights are provided.
    pub fn train_quantized<B>(
        &self,
        quantized: &QuantizedMatrix<B>,
        labels: &[f32],
        weights: Option<&[f32]>,
        cuts: &BinCuts,
        eval_sets: &[QuantizedEvalSet<'_, B>],
    ) -> SoAForest<ScalarLeaf>
    where
        B: BinIndex,
    {
        // Validate weights length
        if let Some(w) = weights {
            assert_eq!(
                w.len(),
                labels.len(),
                "weights length ({}) must match labels length ({})",
                w.len(),
                labels.len()
            );
        }

        let mut logger = TrainingLogger::new(self.verbosity);
        let num_rows = quantized.num_rows() as usize;
        let num_features = cuts.num_features() as u32;
        let num_outputs = self.loss.num_outputs();
        assert_eq!(labels.len(), num_rows, "labels length must match data rows");

        // Create pre-configured components once
        let col_sampler = ColumnSampler::new(
            num_features,
            self.colsample_bytree,
            self.colsample_bylevel,
            self.colsample_bynode,
        );
        let mono_constraints = MonotonicConstraints::new(&self.monotone_constraints, num_features as usize);
        let interaction_constraints =
            InteractionConstraints::new(&self.interaction_constraints, num_features);
        let tree_params = self.tree_build_params();

        // Create tree grower once (owns col_sampler, constraints - reused across all trees)
        let mut grower = TreeGrower::new(
            self.growth_strategy,
            cuts,
            tree_params,
            self.learning_rate,
            col_sampler,
            mono_constraints,
            interaction_constraints,
        );

        // Initialize base scores (per output) using loss-specific initialization
        // RFC-0024: init_base_score supports weights for proper initialization
        let base_scores: Vec<f32> = self.loss.init_base_score(labels, weights);

        // Initialize predictions: [num_rows * num_outputs] in column-major order
        // Layout: [output0_sample0, output0_sample1, ..., output1_sample0, ...]
        // Index: output * num_rows + sample
        let mut predictions: Vec<f32> = base_scores
            .iter()
            .flat_map(|&base| std::iter::repeat(base).take(num_rows))
            .collect();

        // Gradient buffer
        let mut grads = GradientBuffer::new(num_rows, num_outputs);

        // Row partitioner (reused per tree within a round)
        let mut partitioner = RowPartitioner::new(num_rows as u32);

        // Early stopping
        let mut early_stopping = if self.early_stopping_rounds > 0 && !eval_sets.is_empty() {
            Some(EarlyStopping::new(
                self.early_stopping_rounds,
                self.eval_metric.higher_is_better(),
            ))
        } else {
            None
        };

        // Trees per output: trees_per_output[output_idx][round]
        let mut trees_per_output: Vec<Vec<BuildingTree>> = (0..num_outputs)
            .map(|_| Vec::with_capacity(self.num_rounds as usize))
            .collect();

        // Log training start
        logger.info(&format!(
            "Starting training: {} rounds, {} samples, {} outputs",
            self.num_rounds, num_rows, num_outputs,
        ));

        // Main training loop
        for round in 0..self.num_rounds {
            // Compute gradients for all samples and outputs
            self.loss
                .compute_gradients(&predictions, labels, weights, &mut grads);

            let round_seed = self.seed.wrapping_add(round as u64);

            // Apply row sampling (same sample for all outputs in this round)
            // For GOSS with multi-output, uses L2 norm of gradient vectors
            if self.row_sampling.is_enabled() {
                let sample = self.row_sampling.sample(&grads, round_seed);
                partitioner.reset_with_rows(&sample.indices);

                // Apply GOSS weights if present (column-major optimized)
                if let Some(ref sample_weights) = sample.weights {
                    // Loop over outputs first (contiguous per-output), then samples
                    for output_idx in 0..num_outputs {
                        let (output_grads, output_hess) = grads.output_grads_hess_mut(output_idx);
                        for (idx_in_sample, &row_idx) in sample.indices.iter().enumerate() {
                            let weight = sample_weights[idx_in_sample];
                            if weight != 1.0 {
                                let row = row_idx as usize;
                                output_grads[row] *= weight;
                                output_hess[row] *= weight;
                            }
                        }
                    }
                }
            } else {
                partitioner.reset();
            }

            // Build one tree per output
            for output_idx in 0..num_outputs {
                let output_seed = round_seed.wrapping_add(output_idx as u64 * 1000);

                // Build tree using reusable grower (col_sampler.sample_tree called internally)
                let output_grads = grads.output_grads(output_idx);
                let output_hess = grads.output_hess(output_idx);
                let tree = grower.build_tree(
                    quantized,
                    output_grads,
                    output_hess,
                    &mut partitioner,
                    output_seed,
                );

                // Update predictions for this output (column-major)
                let output_start = output_idx * num_rows;
                for row in 0..num_rows {
                    let leaf_value = Self::predict_row(&tree, quantized, row as u32);
                    predictions[output_start + row] += leaf_value;
                }

                trees_per_output[output_idx].push(tree);
            }

            // Evaluation and early stopping
            let mut round_metrics: Vec<(String, f64)> = Vec::new();
            let mut early_stop_triggered = false;

            if self.verbosity >= Verbosity::Info {
                let train_metric =
                    self.eval_metric
                        .evaluate(&predictions, labels, weights, num_outputs);
                round_metrics.push(("train".to_string(), train_metric));
            }

            for (idx, eval_set) in eval_sets.iter().enumerate() {
                let eval_preds =
                    Self::predict_with_trees(&trees_per_output, eval_set.data, &base_scores);

                if self.verbosity >= Verbosity::Info {
                    let metric_value = self.eval_metric.evaluate(
                        &eval_preds,
                        eval_set.labels,
                        eval_set.weights,
                        num_outputs,
                    );
                    round_metrics.push((eval_set.name.to_string(), metric_value));
                }

                // Early stopping on configured eval set
                if idx == self.early_stopping_eval_set {
                    if let Some(ref mut es) = early_stopping {
                        // Compute metric for early stopping
                        let metric_value = self.eval_metric.evaluate(
                            &eval_preds,
                            eval_set.labels,
                            eval_set.weights,
                            num_outputs,
                        );
                        if es.should_stop(metric_value) {
                            logger.info(&format!(
                                "Early stopping at round {} (best: {})",
                                round,
                                es.best_round()
                            ));
                            early_stop_triggered = true;
                        }
                    }
                }
            }

            if self.verbosity >= Verbosity::Info && !round_metrics.is_empty() {
                logger.log_round(round as usize, &round_metrics);
            }

            if early_stop_triggered {
                break;
            }
        }

        let total_trees: usize = trees_per_output.iter().map(|t| t.len()).sum();
        logger.info(&format!("Training complete: {} trees built", total_trees));

        self.freeze_forest_unified(trees_per_output, base_scores)
    }

    /// Unified forest creation from trees per output.
    fn freeze_forest_unified(
        &self,
        trees_per_output: Vec<Vec<BuildingTree>>,
        base_scores: Vec<f32>,
    ) -> SoAForest<ScalarLeaf> {
        let num_outputs = trees_per_output.len();
        let mut forest = SoAForest::new(num_outputs as u32).with_base_score(base_scores);

        for (output_idx, output_trees) in trees_per_output.into_iter().enumerate() {
            for tree in output_trees {
                let soa_tree = self.convert_tree(&tree);
                forest.push_tree(soa_tree, output_idx as u32);
            }
        }

        forest
    }

    /// Predict with all trees for multi-output (used for eval sets).
    /// Returns predictions in column-major order: [output0_row0, output0_row1, ..., output1_row0, ...]
    fn predict_with_trees<B: BinIndex>(
        trees_per_output: &[Vec<BuildingTree>],
        data: &QuantizedMatrix<B>,
        base_scores: &[f32],
    ) -> Vec<f32> {
        let num_rows = data.num_rows() as usize;

        // Initialize with base scores (column-major)
        let mut predictions: Vec<f32> = base_scores
            .iter()
            .flat_map(|&base| std::iter::repeat(base).take(num_rows))
            .collect();

        // Add tree predictions for each output (column-major: contiguous per output)
        for (output_idx, trees) in trees_per_output.iter().enumerate() {
            let output_start = output_idx * num_rows;
            for tree in trees {
                for row in 0..num_rows {
                    let pred = Self::predict_row(tree, data, row as u32);
                    predictions[output_start + row] += pred;
                }
            }
        }

        predictions
    }

    // ========================================================================
    // Helper methods
    // ========================================================================

    fn predict_row<B: BinIndex>(
        tree: &BuildingTree,
        quantized: &QuantizedMatrix<B>,
        row: u32,
    ) -> f32 {
        let mut node_id = 0u32;

        loop {
            let node = tree.node(node_id);
            if node.is_leaf {
                return node.weight;
            }

            let split = node.split.as_ref().expect("Non-leaf must have split");
            let bin = quantized.get(row, split.feature).to_usize();

            let goes_left = if bin == 0 {
                split.default_left
            } else if split.is_categorical {
                split.categories_left.contains(&(bin as u32))
            } else {
                bin <= split.split_bin as usize
            };

            node_id = if goes_left { node.left } else { node.right };
        }
    }

    fn convert_tree(&self, building: &BuildingTree) -> SoATreeStorage<ScalarLeaf> {
        let mut builder = SoATreeBuilder::<ScalarLeaf>::new();

        for node_id in 0..building.num_nodes() as u32 {
            let node = building.node(node_id);

            if node.is_leaf {
                builder.add_leaf(ScalarLeaf(node.weight));
            } else {
                let split = node.split.as_ref().expect("Non-leaf must have split");
                if split.is_categorical {
                    let bitset = categories_to_bitset(&split.categories_left);
                    builder.add_categorical_split(
                        split.feature,
                        bitset,
                        split.default_left,
                        node.left,
                        node.right,
                    );
                } else {
                    builder.add_split(
                        split.feature,
                        split.threshold,
                        split.default_left,
                        node.left,
                        node.right,
                    );
                }
            }
        }

        builder.build()
    }
}

// ============================================================================
// QuantizedEvalSet
// ============================================================================

/// Evaluation set for GBTree training with quantized data.
pub struct QuantizedEvalSet<'a, B: BinIndex> {
    /// Dataset name (appears in logs).
    pub name: &'a str,
    /// Quantized feature matrix.
    pub data: &'a QuantizedMatrix<B>,
    /// Labels (length = n_samples).
    pub labels: &'a [f32],
    /// Optional sample weights for weighted metric computation.
    pub weights: Option<&'a [f32]>,
}

impl<'a, B: BinIndex> QuantizedEvalSet<'a, B> {
    /// Create a new quantized evaluation set without sample weights.
    pub fn new(name: &'a str, data: &'a QuantizedMatrix<B>, labels: &'a [f32]) -> Self {
        Self {
            name,
            data,
            labels,
            weights: None,
        }
    }

    /// Create a new quantized evaluation set with sample weights.
    pub fn with_weights(
        name: &'a str,
        data: &'a QuantizedMatrix<B>,
        labels: &'a [f32],
        weights: &'a [f32],
    ) -> Self {
        Self {
            name,
            data,
            labels,
            weights: Some(weights),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{ColMatrix, RowMatrix};
    use crate::training::gbtree::quantize::{CutFinder, ExactQuantileCuts, Quantizer};

    fn make_regression_data() -> (QuantizedMatrix<u8>, BinCuts, Vec<f32>) {
        let mut data = Vec::new();
        let mut labels = Vec::new();
        for i in 0..100 {
            let x0 = i as f32 / 10.0;
            let x1 = (i % 10) as f32;
            data.push(x0);
            data.push(x1);
            labels.push(x0 + 0.1);
        }

        // Create data in row-major format, then convert to col-major for ColumnAccess
        let row_matrix = RowMatrix::from_vec(data, 100, 2);
        let matrix: ColMatrix<f32> = (&row_matrix).into();

        let cuts_finder = ExactQuantileCuts::new(1);
        let cuts = cuts_finder.find_cuts(&matrix, 256);
        let quantizer = Quantizer::new(cuts.clone());
        let quantized = quantizer.quantize::<_, u8>(&matrix);

        (quantized, cuts, labels)
    }

    #[test]
    fn test_default_trainer() {
        let trainer = GBTreeTrainer::default();
        assert_eq!(trainer.num_rounds, 100);
        assert!(matches!(
            trainer.growth_strategy,
            GrowthStrategy::DepthWise { max_depth: 6 }
        ));
        assert_eq!(trainer.learning_rate, 0.3);
    }

    #[test]
    fn test_builder() {
        let trainer = GBTreeTrainer::builder()
            .num_rounds(50u32)
            .growth_strategy(GrowthStrategy::DepthWise { max_depth: 4 })
            .learning_rate(0.1)
            .build()
            .unwrap();

        assert_eq!(trainer.num_rounds, 50);
        assert!(matches!(
            trainer.growth_strategy,
            GrowthStrategy::DepthWise { max_depth: 4 }
        ));
        assert_eq!(trainer.learning_rate, 0.1);
    }

    #[test]
    fn test_train_simple() {
        let (quantized, cuts, labels) = make_regression_data();

        let trainer = GBTreeTrainer::builder()
            .num_rounds(10u32)
            .growth_strategy(GrowthStrategy::DepthWise { max_depth: 3 })
            .verbosity(Verbosity::Silent)
            .build()
            .unwrap();

        let forest = trainer.train_quantized(&quantized, &labels, None, &cuts, &[]);
        assert_eq!(forest.num_trees(), 10);
    }

    #[test]
    fn test_train_reduces_error() {
        let (quantized, cuts, labels) = make_regression_data();

        let trainer = GBTreeTrainer::builder()
            .num_rounds(20u32)
            .growth_strategy(GrowthStrategy::DepthWise { max_depth: 4 })
            .learning_rate(0.3)
            .verbosity(Verbosity::Silent)
            .build()
            .unwrap();

        let forest = trainer.train_quantized(&quantized, &labels, None, &cuts, &[]);

        // Compute final predictions
        use crate::data::RowMatrix;
        use crate::predict::{Predictor, StandardTraversal};

        let mut data = Vec::new();
        for i in 0..100 {
            let x0 = i as f32 / 10.0;
            let x1 = (i % 10) as f32;
            data.push(x0);
            data.push(x1);
        }
        let row_matrix = RowMatrix::from_vec(data, 100, 2);
        let predictor = Predictor::<StandardTraversal>::new(&forest);
        let predictions = predictor.predict(&row_matrix).into_vec();

        // Check RMSE is reasonable
        let rmse: f64 = predictions
            .iter()
            .zip(labels.iter())
            .map(|(p, l)| (p - l).powi(2) as f64)
            .sum::<f64>()
            / labels.len() as f64;
        let rmse = rmse.sqrt();

        assert!(rmse < 1.0, "RMSE should be < 1.0, got {}", rmse);
    }

    #[test]
    fn test_leaf_wise_growth() {
        let (quantized, cuts, labels) = make_regression_data();

        let trainer = GBTreeTrainer::builder()
            .num_rounds(10u32)
            .growth_strategy(GrowthStrategy::LeafWise { max_leaves: 8 })
            .verbosity(Verbosity::Silent)
            .build()
            .unwrap();

        let forest = trainer.train_quantized(&quantized, &labels, None, &cuts, &[]);
        assert_eq!(forest.num_trees(), 10);
    }

    #[test]
    fn test_loss_logistic() {
        let trainer = GBTreeTrainer::builder()
            .loss(LossFunction::Logistic)
            .num_rounds(5u32)
            .verbosity(Verbosity::Silent)
            .build()
            .unwrap();

        assert_eq!(trainer.loss, LossFunction::Logistic);
    }

    #[test]
    fn test_train_with_weights() {
        let (quantized, cuts, labels) = make_regression_data();

        // Create weights: emphasize first half of samples
        let mut weights = vec![2.0; labels.len() / 2];
        weights.extend(vec![0.5; labels.len() - labels.len() / 2]);

        let trainer = GBTreeTrainer::builder()
            .num_rounds(10u32)
            .growth_strategy(GrowthStrategy::DepthWise { max_depth: 3 })
            .verbosity(Verbosity::Silent)
            .build()
            .unwrap();

        let forest = trainer.train_quantized(&quantized, &labels, Some(&weights), &cuts, &[]);
        assert_eq!(forest.num_trees(), 10);
    }

    #[test]
    fn test_train_uniform_weights_matches_unweighted() {
        let (quantized, cuts, labels) = make_regression_data();

        let trainer = GBTreeTrainer::builder()
            .num_rounds(10u32)
            .growth_strategy(GrowthStrategy::DepthWise { max_depth: 3 })
            .seed(42u64)
            .verbosity(Verbosity::Silent)
            .build()
            .unwrap();

        // Train without weights
        let forest_unweighted = trainer.train_quantized(&quantized, &labels, None, &cuts, &[]);

        // Train with uniform weights
        let uniform_weights = vec![1.0; labels.len()];
        let forest_weighted =
            trainer.train_quantized(&quantized, &labels, Some(&uniform_weights), &cuts, &[]);

        // Both forests should have the same number of trees
        assert_eq!(forest_unweighted.num_trees(), forest_weighted.num_trees());
    }

    #[test]
    #[should_panic(expected = "weights length")]
    fn test_train_weights_length_mismatch() {
        let (quantized, cuts, labels) = make_regression_data();

        let trainer = GBTreeTrainer::builder()
            .num_rounds(5u32)
            .verbosity(Verbosity::Silent)
            .build()
            .unwrap();

        let bad_weights = vec![1.0; labels.len() / 2]; // Wrong length
        let _forest = trainer.train_quantized(&quantized, &labels, Some(&bad_weights), &cuts, &[]);
    }
}

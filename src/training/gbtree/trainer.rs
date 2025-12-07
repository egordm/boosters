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
//! let forest = trainer.train(&data, &labels, &[]);
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
//! let forest = trainer.train(&data, &labels, &[]);
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
//! let forest = trainer.train(&data, &labels, &[]);
//! ```
//!
//! See RFC-0015 for design rationale.

use derive_builder::Builder;

use crate::data::ColumnAccess;
use crate::forest::SoAForest;
use crate::training::metric::EvalSet;
use crate::training::{
    EarlyStopping, EvalMetric, GradientBuffer, Loss, LossFunction, TrainingLogger, Verbosity,
};
use crate::trees::{categories_to_bitset, ScalarLeaf, SoATreeStorage, TreeBuilder as SoATreeBuilder};

use super::constraints::MonotonicConstraint;
use super::grower::{BuildingTree, GrowthStrategy, TreeGrower, TreeParams};
use super::partition::RowPartitioner;
use super::quantize::{BinCuts, BinIndex, ExactQuantileCuts, QuantizedMatrix, Quantizer};
use super::sampling::{RowSampler, RowSampling};
use super::split::GainParams;

// ============================================================================
// GBTreeTrainer
// ============================================================================

/// Growth strategy for tree building.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GrowthMode {
    /// Depth-wise growth (XGBoost style): expand all nodes at each depth level.
    DepthWise,
    /// Leaf-wise growth (LightGBM style): always expand the best-gain leaf.
    LeafWise,
}

impl Default for GrowthMode {
    fn default() -> Self {
        Self::DepthWise
    }
}

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
/// let forest = trainer.train(&data, &labels);
///
/// // Configured via builder
/// let trainer = GBTreeTrainer::builder()
///     .loss(LossFunction::Logistic)
///     .num_rounds(50)
///     .max_depth(4)
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
    #[builder(default)]
    pub growth_mode: GrowthMode,

    /// Maximum tree depth (used by depth-wise growth, also as limit for leaf-wise).
    #[builder(default = "6")]
    pub max_depth: u32,

    /// Maximum number of leaves (used by leaf-wise growth).
    #[builder(default = "31")]
    pub max_leaves: u32,

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

    // ========================================================================
    // Internal / advanced
    // ========================================================================
    /// Use parallel histogram building.
    #[builder(default = "false")]
    pub parallel_histograms: bool,
}

impl Default for GBTreeTrainer {
    fn default() -> Self {
        Self {
            loss: LossFunction::default(),
            num_rounds: 100,
            learning_rate: 0.3,
            max_bins: 256,
            seed: 0,
            growth_mode: GrowthMode::default(),
            max_depth: 6,
            max_leaves: 31,
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
            parallel_histograms: false,
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

    /// Build internal TreeParams from the flat config.
    fn tree_params(&self) -> TreeParams {
        TreeParams {
            gain: GainParams {
                lambda: self.reg_lambda,
                alpha: self.reg_alpha,
                min_split_gain: self.min_split_gain,
                min_child_weight: self.min_child_weight,
            },
            max_depth: self.max_depth,
            max_leaves: self.max_leaves,
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
            learning_rate: self.learning_rate,
            parallel_histograms: self.parallel_histograms,
            colsample_bytree: self.colsample_bytree,
            colsample_bylevel: self.colsample_bylevel,
            colsample_bynode: self.colsample_bynode,
            monotone_constraints: self.monotone_constraints.clone(),
            interaction_constraints: self.interaction_constraints.clone(),
        }
    }

    /// Build GrowthStrategy from config.
    fn growth_strategy(&self) -> GrowthStrategy {
        match self.growth_mode {
            GrowthMode::DepthWise => GrowthStrategy::DepthWise {
                max_depth: self.max_depth,
            },
            GrowthMode::LeafWise => GrowthStrategy::LeafWise {
                max_leaves: self.max_leaves,
            },
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
    /// * `eval_sets` - Evaluation sets for monitoring (pass `&[]` if not needed)
    ///
    /// # Example
    ///
    /// ```ignore
    /// use booste_rs::training::{GBTreeTrainer, LossFunction};
    ///
    /// // Regression
    /// let trainer = GBTreeTrainer::default();
    /// let forest = trainer.train(&data, &labels, &[]);
    ///
    /// // Multiclass (3 classes)
    /// let trainer = GBTreeTrainer::builder()
    ///     .loss(LossFunction::Softmax { num_classes: 3 })
    ///     .build()
    ///     .unwrap();
    /// let forest = trainer.train(&data, &labels, &[]);
    /// ```
    pub fn train<D>(
        &self,
        data: &D,
        labels: &[f32],
        eval_sets: &[EvalSet<'_, D>],
    ) -> SoAForest<ScalarLeaf>
    where
        D: ColumnAccess<Element = f32> + Sync,
    {
        let mut logger = TrainingLogger::new(self.verbosity);

        // Compute bin cuts from training data
        logger.info("Computing bin cuts...");
        let cut_finder = ExactQuantileCuts::default();
        let quantizer = Quantizer::from_data(data, &cut_finder, self.max_bins);
        let cuts = quantizer.cuts().clone();

        // Quantize training data
        logger.info("Quantizing training data...");
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

        self.train_internal(&quantized, labels, &cuts, &quantized_refs, &mut logger)
    }

    /// Train on pre-quantized data (advanced API).
    pub fn train_quantized<B>(
        &self,
        quantized: &QuantizedMatrix<B>,
        labels: &[f32],
        cuts: &BinCuts,
        eval_sets: &[QuantizedEvalSet<'_, B>],
    ) -> SoAForest<ScalarLeaf>
    where
        B: BinIndex,
    {
        let mut logger = TrainingLogger::new(self.verbosity);
        self.train_internal(quantized, labels, cuts, eval_sets, &mut logger)
    }

    // ========================================================================
    // Internal training implementation (unified for single and multi-output)
    // ========================================================================

    /// Unified internal training implementation.
    ///
    /// Handles both single-output (regression, binary classification) and
    /// multi-output (multiclass, multi-quantile) training through the same code path.
    /// For single-output, `num_outputs = 1` and we build 1 tree per round.
    /// For multi-output, `num_outputs = K` and we build K trees per round.
    fn train_internal<B>(
        &self,
        quantized: &QuantizedMatrix<B>,
        labels: &[f32],
        cuts: &BinCuts,
        eval_sets: &[QuantizedEvalSet<'_, B>],
        logger: &mut TrainingLogger,
    ) -> SoAForest<ScalarLeaf>
    where
        B: BinIndex,
    {
        let num_rows = quantized.num_rows() as usize;
        let num_outputs = self.loss.num_outputs();
        assert_eq!(labels.len(), num_rows, "labels length must match data rows");

        let tree_params = self.tree_params();
        let growth_strategy = self.growth_strategy();

        // Initialize base scores (per output) using loss-specific initialization
        let base_scores: Vec<f32> = self.loss.init_base_score(labels, None);

        // Initialize predictions: [num_rows * num_outputs] in row-major order
        let mut predictions: Vec<f32> = (0..num_rows)
            .flat_map(|_| base_scores.iter().copied())
            .collect();

        // Gradient buffer
        let mut grads = GradientBuffer::new(num_rows, num_outputs);

        // Row partitioner (reused per tree within a round)
        let mut partitioner = RowPartitioner::new(num_rows as u32);

        // Early stopping
        let mut early_stopping = if self.early_stopping_rounds > 0 && !eval_sets.is_empty() {
            Some(EarlyStopping::new(
                self.eval_metric.clone(),
                self.early_stopping_rounds,
            ))
        } else {
            None
        };

        // Trees per output: trees_per_output[output_idx][round]
        let mut trees_per_output: Vec<Vec<BuildingTree>> = (0..num_outputs)
            .map(|_| Vec::with_capacity(self.num_rounds as usize))
            .collect();

        // Log training start
        let sampling_info = if self.row_sampling.is_enabled() {
            format!(" ({})", self.row_sampling)
        } else {
            String::new()
        };

        let outputs_info = if num_outputs > 1 {
            format!(", {} outputs", num_outputs)
        } else {
            String::new()
        };

        logger.info(&format!(
            "Starting training: {} rounds, {} samples{}{}",
            self.num_rounds, num_rows, outputs_info, sampling_info,
        ));
        
        let base_scores_str: Vec<String> = base_scores.iter()
            .map(|s| format!("{:.6}", s))
            .collect();
        logger.info(&format!("Base scores: [{}]", base_scores_str.join(", ")));

        // Main training loop
        for round in 0..self.num_rounds {
            // Compute gradients for all samples and outputs
            self.loss.compute_gradients(&predictions, labels, &mut grads);

            let round_seed = self.seed.wrapping_add(round as u64);

            // Apply row sampling (same sample for all outputs in this round)
            // For GOSS with multi-output, uses L2 norm of gradient vectors
            let row_sample = if self.row_sampling.is_enabled() {
                let sample = self.row_sampling.sample_multioutput(&grads, round_seed);
                partitioner.reset_with_rows(&sample.indices);

                // Apply GOSS weights if present
                if let Some(ref weights) = sample.weights {
                    for (idx_in_sample, &row_idx) in sample.indices.iter().enumerate() {
                        let weight = weights[idx_in_sample];
                        if weight != 1.0 {
                            for output_idx in 0..num_outputs {
                                let (grad, hess) = grads.get(row_idx as usize, output_idx);
                                grads.set(row_idx as usize, output_idx, grad * weight, hess * weight);
                            }
                        }
                    }
                }
                Some(sample)
            } else {
                partitioner.reset();
                None
            };
            let _ = row_sample; // Silence unused warning

            // Build one tree per output
            for output_idx in 0..num_outputs {
                // Build tree
                let output_seed = round_seed.wrapping_add(output_idx as u64 * 1000);
                let tree = self.build_tree(
                    quantized,
                    &grads,
                    output_idx,
                    cuts,
                    &tree_params,
                    growth_strategy,
                    &mut partitioner,
                    output_seed,
                );

                // Update predictions for this output
                Self::update_predictions_for_output(
                    &tree,
                    quantized,
                    &mut predictions,
                    num_outputs,
                    output_idx,
                );

                trees_per_output[output_idx].push(tree);
            }

            // Evaluation and early stopping
            let mut round_metrics: Vec<(String, f64)> = Vec::new();
            let mut early_stop_triggered = false;

            if self.verbosity >= Verbosity::Info {
                let train_metric = self.compute_train_metric_multioutput(
                    &predictions, labels, num_outputs,
                );
                round_metrics.push(("train".to_string(), train_metric));
            }

            for (idx, eval_set) in eval_sets.iter().enumerate() {
                let eval_preds = Self::predict_with_trees_multioutput(
                    &trees_per_output,
                    eval_set.data,
                    &base_scores,
                );

                if self.verbosity >= Verbosity::Info {
                    let metric_value = self.compute_train_metric_multioutput(
                        &eval_preds, eval_set.labels, num_outputs,
                    );
                    round_metrics.push((eval_set.name.to_string(), metric_value));
                }

                // Early stopping on first eval set
                if idx == 0 {
                    if let Some(ref mut es) = early_stopping {
                        // For multi-output, use averaged predictions for early stopping
                        let eval_for_es = if num_outputs == 1 {
                            eval_preds.clone()
                        } else {
                            // Average predictions across outputs for early stopping metric
                            Self::average_predictions(&eval_preds, num_outputs)
                        };
                        if es.should_stop(&eval_for_es, eval_set.labels) {
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

    /// Build a single tree for the given output.
    fn build_tree<B: BinIndex>(
        &self,
        quantized: &QuantizedMatrix<B>,
        grads: &GradientBuffer,
        output_idx: usize,
        cuts: &BinCuts,
        tree_params: &TreeParams,
        growth_strategy: GrowthStrategy,
        partitioner: &mut RowPartitioner,
        seed: u64,
    ) -> BuildingTree {
        let num_rows = quantized.num_rows() as usize;
        let num_outputs = grads.n_outputs();

        // For multi-output, extract gradients for this output into a temporary buffer
        let grads_to_use = if num_outputs > 1 {
            // Copy gradients for this output
            let mut temp = GradientBuffer::new(num_rows, 1);
            for row in 0..num_rows {
                let (grad, hess) = grads.get(row, output_idx);
                temp.set(row, 0, grad, hess);
            }
            temp
        } else {
            // Single-output: copy to new buffer (TreeGrower expects owned)
            let mut temp = GradientBuffer::new(num_rows, 1);
            for row in 0..num_rows {
                let (grad, hess) = grads.get(row, 0);
                temp.set(row, 0, grad, hess);
            }
            temp
        };

        match growth_strategy {
            GrowthStrategy::DepthWise { max_depth } => {
                let policy = super::grower::DepthWisePolicy { max_depth };
                let mut grower = TreeGrower::new(policy, cuts, tree_params.clone());
                grower.build_tree_with_seed(quantized, &grads_to_use, partitioner, seed)
            }
            GrowthStrategy::LeafWise { max_leaves } => {
                let policy = super::grower::LeafWisePolicy { max_leaves };
                let mut grower = TreeGrower::new(policy, cuts, tree_params.clone());
                grower.build_tree_with_seed(quantized, &grads_to_use, partitioner, seed)
            }
        }
    }

    /// Update predictions for a specific output after building a tree.
    fn update_predictions_for_output<B: BinIndex>(
        tree: &BuildingTree,
        quantized: &QuantizedMatrix<B>,
        predictions: &mut [f32],
        num_outputs: usize,
        output_idx: usize,
    ) {
        for row in 0..quantized.num_rows() as usize {
            let leaf_value = Self::predict_row(tree, quantized, row as u32);
            predictions[row * num_outputs + output_idx] += leaf_value;
        }
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
    /// Returns predictions in row-major order: [row0_out0, row0_out1, ..., row1_out0, ...]
    fn predict_with_trees_multioutput<B: BinIndex>(
        trees_per_output: &[Vec<BuildingTree>],
        data: &QuantizedMatrix<B>,
        base_scores: &[f32],
    ) -> Vec<f32> {
        let num_rows = data.num_rows() as usize;
        let num_outputs = trees_per_output.len();

        // Initialize with base scores
        let mut predictions: Vec<f32> = (0..num_rows)
            .flat_map(|_| base_scores.iter().copied())
            .collect();

        // Add tree predictions for each output
        for (output_idx, trees) in trees_per_output.iter().enumerate() {
            for tree in trees {
                for row in 0..num_rows {
                    let pred = Self::predict_row(tree, data, row as u32);
                    predictions[row * num_outputs + output_idx] += pred;
                }
            }
        }

        predictions
    }

    /// Average predictions across outputs (for early stopping with multi-output).
    fn average_predictions(predictions: &[f32], num_outputs: usize) -> Vec<f32> {
        let num_rows = predictions.len() / num_outputs;
        (0..num_rows)
            .map(|row| {
                let sum: f32 = (0..num_outputs)
                    .map(|k| predictions[row * num_outputs + k])
                    .sum();
                sum / num_outputs as f32
            })
            .collect()
    }

    // ========================================================================
    // Helper methods
    // ========================================================================

    fn predict_row<B: BinIndex>(tree: &BuildingTree, quantized: &QuantizedMatrix<B>, row: u32) -> f32 {
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

    fn compute_train_metric(&self, predictions: &[f32], labels: &[f32]) -> f64 {
        let sum_sq_err: f64 = predictions
            .iter()
            .zip(labels.iter())
            .map(|(p, l)| (p - l).powi(2) as f64)
            .sum();
        (sum_sq_err / predictions.len() as f64).sqrt()
    }

    /// Compute training metric for multi-output.
    /// For single-output, uses standard RMSE.
    /// For multi-output, computes RMSE averaged across outputs.
    fn compute_train_metric_multioutput(
        &self,
        predictions: &[f32],
        labels: &[f32],
        num_outputs: usize,
    ) -> f64 {
        if num_outputs == 1 {
            return self.compute_train_metric(predictions, labels);
        }

        // For multi-output: average RMSE across outputs
        // predictions: [row0_out0, row0_out1, ..., row1_out0, ...]
        // labels: [row0_label, row1_label, ...]
        let num_rows = labels.len();

        // Compute per-output MSE and average
        let mut total_mse = 0.0f64;
        for output_idx in 0..num_outputs {
            let mse: f64 = (0..num_rows)
                .map(|row| {
                    let pred = predictions[row * num_outputs + output_idx] as f64;
                    // For classification outputs, we compare raw scores
                    // For quantile outputs, we compare predictions to label
                    // In both cases, lower error is better
                    let label = labels[row] as f64;
                    (pred - label).powi(2)
                })
                .sum::<f64>()
                / num_rows as f64;
            total_mse += mse;
        }

        // Return average RMSE
        (total_mse / num_outputs as f64).sqrt()
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
}

impl<'a, B: BinIndex> QuantizedEvalSet<'a, B> {
    /// Create a new quantized evaluation set.
    pub fn new(name: &'a str, data: &'a QuantizedMatrix<B>, labels: &'a [f32]) -> Self {
        Self { name, data, labels }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{RowMatrix, ColMatrix};
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
        assert_eq!(trainer.max_depth, 6);
        assert_eq!(trainer.learning_rate, 0.3);
    }

    #[test]
    fn test_builder() {
        let trainer = GBTreeTrainer::builder()
            .num_rounds(50u32)
            .max_depth(4u32)
            .learning_rate(0.1)
            .build()
            .unwrap();

        assert_eq!(trainer.num_rounds, 50);
        assert_eq!(trainer.max_depth, 4);
        assert_eq!(trainer.learning_rate, 0.1);
    }

    #[test]
    fn test_base_score_auto_regression() {
        // Auto for regression should use mean
        let labels = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let trainer = GBTreeTrainer::default();
        let base_scores = trainer.compute_base_scores(&labels);
        assert_eq!(base_scores.len(), 1);
        assert!((base_scores[0] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_base_score_auto_softmax() {
        // Auto for softmax should use log-priors
        let labels = vec![0.0, 0.0, 0.0, 1.0, 2.0]; // 60% class 0, 20% class 1, 20% class 2
        let trainer = GBTreeTrainer::builder()
            .loss(LossFunction::Softmax { num_classes: 3 })
            .build()
            .unwrap();
        let base_scores = trainer.compute_base_scores(&labels);
        assert_eq!(base_scores.len(), 3);
        // Class 0 should have highest base score (most frequent)
        assert!(base_scores[0] > base_scores[1]);
        assert!(base_scores[0] > base_scores[2]);
        // log(0.6) ≈ -0.51, log(0.2) ≈ -1.61
        assert!((base_scores[0] - (-0.51)).abs() < 0.1);
        assert!((base_scores[1] - (-1.61)).abs() < 0.1);
    }

    #[test]
    fn test_base_score_auto_logistic() {
        // All positive labels
        let labels = vec![1.0, 1.0, 1.0, 1.0];
        let trainer = GBTreeTrainer::builder()
            .loss(LossFunction::Logistic)
            .build()
            .unwrap();
        let base_scores = trainer.compute_base_scores(&labels);
        assert_eq!(base_scores.len(), 1);
        // 100% positive → very large positive log-odds
        assert!(base_scores[0] > 5.0);
        
        // 50/50 labels
        let labels = vec![0.0, 0.0, 1.0, 1.0];
        let base_scores = trainer.compute_base_scores(&labels);
        // 50% → log-odds ≈ 0
        assert!(base_scores[0].abs() < 0.1);
    }

    #[test]
    fn test_train_simple() {
        let (quantized, cuts, labels) = make_regression_data();
        
        let trainer = GBTreeTrainer::builder()
            .num_rounds(10u32)
            .max_depth(3u32)
            .verbosity(Verbosity::Silent)
            .build()
            .unwrap();

        let forest = trainer.train_quantized(&quantized, &labels, &cuts, &[]);
        assert_eq!(forest.num_trees(), 10);
    }

    #[test]
    fn test_train_reduces_error() {
        let (quantized, cuts, labels) = make_regression_data();
        
        let trainer = GBTreeTrainer::builder()
            .num_rounds(20u32)
            .max_depth(4u32)
            .learning_rate(0.3)
            .verbosity(Verbosity::Silent)
            .build()
            .unwrap();

        let forest = trainer.train_quantized(&quantized, &labels, &cuts, &[]);
        
        // Compute final predictions
        use crate::predict::{Predictor, StandardTraversal};
        use crate::data::RowMatrix;
        
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
            .growth_mode(GrowthMode::LeafWise)
            .max_leaves(8u32)
            .verbosity(Verbosity::Silent)
            .build()
            .unwrap();

        let forest = trainer.train_quantized(&quantized, &labels, &cuts, &[]);
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
}

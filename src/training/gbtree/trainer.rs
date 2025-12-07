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
use super::sampling::{GossParams, GossSampler, RowSampler};
use super::split::GainParams;

// ============================================================================
// GBTreeTrainer
// ============================================================================

/// Strategy for initializing base score.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum BaseScore {
    /// Use mean of labels (good for regression).
    #[default]
    Mean,
    /// Use a fixed value.
    Fixed(f32),
    /// Use zero (raw model starts from 0).
    Zero,
}

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

    /// Base score initialization strategy.
    #[builder(default)]
    pub base_score: BaseScore,

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
    /// Row subsampling ratio per tree (0, 1].
    #[builder(default = "1.0")]
    pub subsample: f32,

    /// Column subsampling ratio per tree (0, 1].
    #[builder(default = "1.0")]
    pub colsample_bytree: f32,

    /// Column subsampling ratio per level (0, 1].
    #[builder(default = "1.0")]
    pub colsample_bylevel: f32,

    /// Column subsampling ratio per node (0, 1].
    #[builder(default = "1.0")]
    pub colsample_bynode: f32,

    /// GOSS sampling parameters (None = disabled).
    /// When enabled, uses gradient-based one-side sampling instead of random.
    #[builder(default)]
    pub goss: Option<GossParams>,

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
            base_score: BaseScore::default(),
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
            subsample: 1.0,
            colsample_bytree: 1.0,
            colsample_bylevel: 1.0,
            colsample_bynode: 1.0,
            goss: None,
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
            subsample: self.subsample,
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

        // Dispatch based on number of outputs
        if self.loss.num_outputs() == 1 {
            self.train_quantized_internal(&quantized, labels, &cuts, &quantized_refs, &mut logger)
        } else {
            self.train_multioutput_quantized(&quantized, labels, &cuts, &mut logger)
        }
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
        if self.loss.num_outputs() == 1 {
            self.train_quantized_internal(quantized, labels, cuts, eval_sets, &mut logger)
        } else {
            self.train_multioutput_quantized(quantized, labels, cuts, &mut logger)
        }
    }

    /// Internal training implementation for single-output losses.
    fn train_quantized_internal<B>(
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
        assert_eq!(labels.len(), num_rows, "labels length must match data rows");

        let tree_params = self.tree_params();
        let growth_strategy = self.growth_strategy();

        // Initialize base score
        let base_score_value = self.compute_base_score(labels);
        logger.info(&format!("Base score: {:.6}", base_score_value));

        // Initialize predictions
        let mut predictions = vec![base_score_value; num_rows];

        // Gradient buffer (single output for regression/binary)
        let mut grads = GradientBuffer::new(num_rows, 1);

        // Row partitioner (reused per tree)
        let mut partitioner = RowPartitioner::new(num_rows as u32);

        // Sampling setup
        let goss_sampler = self.goss.map(GossSampler::new);
        let row_sampler = if goss_sampler.is_none() && self.subsample < 1.0 {
            Some(RowSampler::new(num_rows as u32, self.subsample))
        } else {
            None
        };

        // Early stopping
        let mut early_stopping = if self.early_stopping_rounds > 0 && !eval_sets.is_empty() {
            Some(EarlyStopping::new(
                Box::new(self.eval_metric.clone()),
                self.early_stopping_rounds,
            ))
        } else {
            None
        };

        // Trees built during training
        let mut trees: Vec<BuildingTree> = Vec::with_capacity(self.num_rounds as usize);

        // Log sampling strategy
        let sampling_info = if let Some(ref goss) = self.goss {
            format!(" (GOSS: top={:.2}, other={:.2})", goss.top_rate, goss.other_rate)
        } else if self.subsample < 1.0 {
            format!(" (subsample={:.2})", self.subsample)
        } else {
            String::new()
        };

        logger.info(&format!(
            "Starting training: {} rounds, {} samples{}",
            self.num_rounds, num_rows, sampling_info,
        ));

        for round in 0..self.num_rounds {
            // Compute gradients
            self.loss.compute_gradients(&predictions, labels, &mut grads);

            // Sample rows for this round
            let round_seed = self.seed.wrapping_add(round as u64);

            // Apply row sampling
            if let Some(ref goss) = goss_sampler {
                let gradient_slice: Vec<f32> = (0..num_rows)
                    .map(|i| grads.get(i, 0).0)
                    .collect();
                let sample = goss.sample(&gradient_slice, round_seed);
                partitioner.reset_with_rows(&sample.indices);

                // Apply GOSS weights
                for (idx_in_sample, &row_idx) in sample.indices.iter().enumerate() {
                    let weight = sample.weights[idx_in_sample];
                    if weight != 1.0 {
                        let (grad, hess) = grads.get(row_idx as usize, 0);
                        grads.set(row_idx as usize, 0, grad * weight, hess * weight);
                    }
                }
            } else if let Some(ref random_sampler) = row_sampler {
                let sampled_rows = random_sampler.sample(round_seed);
                partitioner.reset_with_rows(&sampled_rows);
            } else {
                partitioner.reset();
            }

            // Build tree using appropriate policy
            let tree = match growth_strategy {
                GrowthStrategy::DepthWise { max_depth } => {
                    let policy = super::grower::DepthWisePolicy { max_depth };
                    let mut grower = TreeGrower::new(policy, cuts, tree_params.clone());
                    grower.build_tree_with_seed(quantized, &grads, &mut partitioner, round_seed)
                }
                GrowthStrategy::LeafWise { max_leaves } => {
                    let policy = super::grower::LeafWisePolicy { max_leaves };
                    let mut grower = TreeGrower::new(policy, cuts, tree_params.clone());
                    grower.build_tree_with_seed(quantized, &grads, &mut partitioner, round_seed)
                }
            };

            // Update predictions
            Self::update_predictions(&tree, quantized, &mut predictions);

            // Compute metrics and check early stopping
            let mut round_metrics: Vec<(String, f64)> = Vec::new();
            let mut early_stop_triggered = false;

            if self.verbosity >= Verbosity::Info {
                let train_metric = self.compute_train_metric(&predictions, labels);
                round_metrics.push(("train".to_string(), train_metric));
            }

            for (idx, eval_set) in eval_sets.iter().enumerate() {
                let eval_preds = Self::predict_with_trees(&trees, &tree, eval_set.data, base_score_value);

                if self.verbosity >= Verbosity::Info {
                    let metric_value = self.compute_train_metric(&eval_preds, eval_set.labels);
                    round_metrics.push((eval_set.name.to_string(), metric_value));
                }

                if idx == 0 {
                    if let Some(ref mut es) = early_stopping {
                        if es.should_stop(&eval_preds, eval_set.labels) {
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

            trees.push(tree);

            if early_stop_triggered {
                break;
            }
        }

        logger.info(&format!("Training complete: {} trees built", trees.len()));

        self.freeze_forest(trees, base_score_value)
    }

    /// Internal training implementation for multi-output losses.
    fn train_multioutput_quantized<B>(
        &self,
        quantized: &QuantizedMatrix<B>,
        labels: &[f32],
        cuts: &BinCuts,
        logger: &mut TrainingLogger,
    ) -> SoAForest<ScalarLeaf>
    where
        B: BinIndex,
    {
        let num_rows = quantized.num_rows() as usize;
        let num_outputs = self.loss.num_outputs();
        let tree_params = self.tree_params();
        let growth_strategy = self.growth_strategy();

        assert_eq!(labels.len(), num_rows, "labels length must match data rows");

        // Initialize predictions
        let base_scores: Vec<f32> = vec![0.0; num_outputs];
        let mut predictions: Vec<f32> = vec![0.0; num_rows * num_outputs];

        // Gradient buffer (K outputs per sample)
        let mut grads = GradientBuffer::new(num_rows, num_outputs);

        // Partitioners (one per output)
        let mut partitioners: Vec<RowPartitioner> = (0..num_outputs)
            .map(|_| RowPartitioner::new(num_rows as u32))
            .collect();

        // Trees per output
        let mut trees_per_output: Vec<Vec<BuildingTree>> = (0..num_outputs)
            .map(|_| Vec::with_capacity(self.num_rounds as usize))
            .collect();

        logger.info(&format!(
            "Starting multioutput training: {} rounds, {} samples, {} outputs",
            self.num_rounds, num_rows, num_outputs
        ));

        for round in 0..self.num_rounds {
            // Compute gradients for all samples and outputs
            self.loss.compute_gradients(&predictions, labels, &mut grads);

            let round_seed = self.seed.wrapping_add(round as u64);

            // Train one tree per output
            for output_idx in 0..num_outputs {
                // Extract gradients for this output
                let mut output_grads = GradientBuffer::new(num_rows, 1);
                for row in 0..num_rows {
                    let (grad, hess) = grads.get(row, output_idx);
                    output_grads.set(row, 0, grad, hess);
                }

                partitioners[output_idx].reset();

                // Build tree
                let output_seed = round_seed.wrapping_add(output_idx as u64 * 1000);
                let tree = match growth_strategy {
                    GrowthStrategy::DepthWise { max_depth } => {
                        let policy = super::grower::DepthWisePolicy { max_depth };
                        let mut grower = TreeGrower::new(policy, cuts, tree_params.clone());
                        grower.build_tree_with_seed(
                            quantized,
                            &output_grads,
                            &mut partitioners[output_idx],
                            output_seed,
                        )
                    }
                    GrowthStrategy::LeafWise { max_leaves } => {
                        let policy = super::grower::LeafWisePolicy { max_leaves };
                        let mut grower = TreeGrower::new(policy, cuts, tree_params.clone());
                        grower.build_tree_with_seed(
                            quantized,
                            &output_grads,
                            &mut partitioners[output_idx],
                            output_seed,
                        )
                    }
                };

                // Update predictions for this output
                for row in 0..num_rows {
                    let pred = Self::predict_row(&tree, quantized, row as u32);
                    predictions[row * num_outputs + output_idx] += pred;
                }

                trees_per_output[output_idx].push(tree);
            }

            if (round + 1) % 10 == 0 || round == 0 {
                logger.info(&format!(
                    "Round {} complete, {} trees per output",
                    round + 1,
                    trees_per_output[0].len()
                ));
            }
        }

        self.freeze_multioutput_forest(trees_per_output, base_scores)
    }

    // ========================================================================
    // Helper methods
    // ========================================================================

    fn compute_base_score(&self, labels: &[f32]) -> f32 {
        match self.base_score {
            BaseScore::Mean => {
                let sum: f32 = labels.iter().sum();
                sum / labels.len() as f32
            }
            BaseScore::Fixed(v) => v,
            BaseScore::Zero => 0.0,
        }
    }

    fn update_predictions<B: BinIndex>(
        tree: &BuildingTree,
        quantized: &QuantizedMatrix<B>,
        predictions: &mut [f32],
    ) {
        for row in 0..quantized.num_rows() as usize {
            let leaf_value = Self::predict_row(tree, quantized, row as u32);
            predictions[row] += leaf_value;
        }
    }

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

    fn predict_with_trees<B: BinIndex>(
        trees: &[BuildingTree],
        new_tree: &BuildingTree,
        data: &QuantizedMatrix<B>,
        base_score: f32,
    ) -> Vec<f32> {
        let num_rows = data.num_rows() as usize;
        let mut predictions = vec![base_score; num_rows];

        for tree in trees {
            for row in 0..num_rows {
                predictions[row] += Self::predict_row(tree, data, row as u32);
            }
        }

        for row in 0..num_rows {
            predictions[row] += Self::predict_row(new_tree, data, row as u32);
        }

        predictions
    }

    fn compute_train_metric(&self, predictions: &[f32], labels: &[f32]) -> f64 {
        let sum_sq_err: f64 = predictions
            .iter()
            .zip(labels.iter())
            .map(|(p, l)| (p - l).powi(2) as f64)
            .sum();
        (sum_sq_err / predictions.len() as f64).sqrt()
    }

    fn freeze_forest(&self, trees: Vec<BuildingTree>, base_score: f32) -> SoAForest<ScalarLeaf> {
        let mut forest = SoAForest::for_regression().with_base_score(vec![base_score]);

        for building_tree in trees {
            let soa_tree = self.convert_tree(&building_tree);
            forest.push_tree(soa_tree, 0);
        }

        forest
    }

    fn freeze_multioutput_forest(
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
    fn test_base_score_mean() {
        let labels = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let trainer = GBTreeTrainer::default();
        let base = trainer.compute_base_score(&labels);
        assert!((base - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_base_score_fixed() {
        let labels = vec![1.0, 2.0, 3.0];
        let trainer = GBTreeTrainer::builder()
            .base_score(BaseScore::Fixed(0.5))
            .build()
            .unwrap();
        let base = trainer.compute_base_score(&labels);
        assert_eq!(base, 0.5);
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

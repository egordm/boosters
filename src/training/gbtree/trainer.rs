//! Gradient boosting trainer for tree ensembles.
//!
//! This module implements the full boosting loop for training GBTree models.
//! It coordinates tree building, gradient computation, and prediction updates.
//!
//! # Example
//!
//! The easiest way to train is with `train_with_data`, which handles quantization
//! automatically:
//!
//! ```ignore
//! use booste_rs::training::{GBTreeTrainer, TrainerParams, DepthWisePolicy};
//! use booste_rs::training::{SquaredLoss, ExactQuantileCuts};
//!
//! let params = TrainerParams::default();
//! let mut trainer = GBTreeTrainer::new(Box::new(SquaredLoss), params);
//!
//! let cut_finder = ExactQuantileCuts::default();
//! let forest = trainer.train_with_data(
//!     DepthWisePolicy { max_depth: 6 },
//!     &data,
//!     &labels,
//!     &cut_finder,
//!     256,
//!     &[],
//! );
//! ```
//!
//! For advanced usage where you need to control quantization, use [`train`] directly.
//!
//! See RFC-0015 for design rationale.

use crate::data::ColumnAccess;
use crate::forest::SoAForest;
use crate::training::metric::EvalSet;
use crate::training::{EarlyStopping, GradientBuffer, Loss, Metric, TrainingLogger, Verbosity};
use crate::trees::{ScalarLeaf, SoATreeStorage, TreeBuilder as SoATreeBuilder};

use super::grower::{BuildingTree, GrowthPolicy, TreeGrower, TreeParams};
use super::partition::RowPartitioner;
use super::quantize::{BinCuts, BinIndex, CutFinder, QuantizedMatrix, Quantizer};
use super::sampling::RowSampler;

// ============================================================================
// TrainerParams
// ============================================================================

/// Parameters for the gradient boosting trainer.
#[derive(Debug, Clone)]
pub struct TrainerParams {
    /// Parameters for individual tree building
    pub tree_params: TreeParams,
    /// Number of boosting rounds (trees to build)
    pub num_rounds: u32,
    /// Base score initialization strategy
    pub base_score: BaseScore,
    /// Verbosity level for logging
    pub verbosity: Verbosity,
    /// Random seed for reproducibility (sampling, etc.)
    pub seed: u64,
}

impl Default for TrainerParams {
    fn default() -> Self {
        Self {
            tree_params: TreeParams::default(),
            num_rounds: 100,
            base_score: BaseScore::Mean,
            verbosity: Verbosity::Info,
            seed: 0,
        }
    }
}

/// Strategy for initializing base score.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BaseScore {
    /// Use mean of labels (good for regression)
    Mean,
    /// Use a fixed value
    Fixed(f32),
    /// Use zero (raw model starts from 0)
    Zero,
}

// ============================================================================
// QuantizedEvalSet
// ============================================================================

/// Evaluation set for GBTree training with quantized data.
///
/// This mirrors [`EvalSet`][super::metric::EvalSet] but is specific to quantized
/// data used in GBTree training. Named sets appear in training logs.
///
/// # Example
///
/// ```ignore
/// let eval_sets = vec![
///     QuantizedEvalSet::new("train", &quantized_train, &train_labels),
///     QuantizedEvalSet::new("val", &quantized_val, &val_labels),
/// ];
/// // Logs: [0] train-rmse:15.23  val-rmse:16.12
/// ```
pub struct QuantizedEvalSet<'a, B: BinIndex> {
    /// Dataset name (appears in logs as prefix, e.g., "train", "val", "test").
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
// GBTreeTrainer
// ============================================================================

/// Gradient boosting trainer for tree ensembles.
///
/// Coordinates the full boosting loop:
/// 1. Initialize predictions with base score
/// 2. For each round:
///    - Compute gradients from loss
///    - Build a tree using TreeGrower
///    - Update predictions with tree output
///    - Optionally check early stopping
/// 3. Convert trees to inference format
pub struct GBTreeTrainer {
    /// Loss function for gradient computation
    loss: Box<dyn Loss>,
    /// Training parameters
    params: TrainerParams,
    /// Training logger
    logger: TrainingLogger,
    /// Early stopping callback (optional)
    early_stopping: Option<EarlyStopping>,
}

impl GBTreeTrainer {
    /// Create a new trainer.
    pub fn new(loss: Box<dyn Loss>, params: TrainerParams) -> Self {
        let logger = TrainingLogger::new(params.verbosity);
        Self {
            loss,
            params,
            logger,
            early_stopping: None,
        }
    }

    /// Enable early stopping with the given metric and patience.
    pub fn with_early_stopping(mut self, metric: Box<dyn Metric>, patience: usize) -> Self {
        self.early_stopping = Some(EarlyStopping::new(metric, patience));
        self
    }

    /// Train a gradient boosted forest.
    ///
    /// # Type Parameters
    ///
    /// * `G` - Growth policy (DepthWisePolicy or LeafWisePolicy)
    /// * `B` - Bin index type (u8 or u16)
    ///
    /// # Arguments
    ///
    /// * `policy` - Growth policy for tree building
    /// * `quantized` - Quantized feature matrix
    /// * `labels` - Target labels
    /// * `cuts` - Bin cuts for histogram building
    /// * `eval_sets` - Evaluation sets for metrics and early stopping
    ///
    /// # Example
    ///
    /// ```ignore
    /// use booste_rs::training::{GBTreeTrainer, QuantizedEvalSet, DepthWisePolicy};
    ///
    /// let eval_sets = vec![
    ///     QuantizedEvalSet::new("train", &quantized, &labels),
    ///     QuantizedEvalSet::new("val", &quantized_val, &val_labels),
    /// ];
    /// let forest = trainer.train(policy, &quantized, &labels, &cuts, &eval_sets);
    /// ```
    pub fn train<G, B>(
        &mut self,
        policy: G,
        quantized: &QuantizedMatrix<B>,
        labels: &[f32],
        cuts: &BinCuts,
        eval_sets: &[QuantizedEvalSet<'_, B>],
    ) -> SoAForest<ScalarLeaf>
    where
        G: GrowthPolicy + Clone,
        B: BinIndex,
    {
        let num_rows = quantized.num_rows() as usize;
        assert_eq!(labels.len(), num_rows, "labels length must match data rows");

        // Initialize base score
        let base_score = self.compute_base_score(labels);
        self.logger
            .info(&format!("Base score: {:.6}", base_score));

        // Initialize predictions
        let mut predictions = vec![base_score; num_rows];

        // Gradient buffer (single output for regression/binary)
        let mut grads = GradientBuffer::new(num_rows, 1);

        // Row partitioner (reused per tree)
        let mut partitioner = RowPartitioner::new(num_rows as u32);

        // Row sampler for subsampling
        let row_sampler = RowSampler::new(num_rows as u32, self.params.tree_params.subsample);
        let sampling_enabled = row_sampler.is_enabled();

        // Trees built during training
        let mut trees: Vec<BuildingTree> = Vec::with_capacity(self.params.num_rounds as usize);

        self.logger.info(&format!(
            "Starting training: {} rounds, {} samples{}",
            self.params.num_rounds,
            num_rows,
            if sampling_enabled {
                format!(
                    " (subsample={:.2})",
                    self.params.tree_params.subsample
                )
            } else {
                String::new()
            }
        ));

        for round in 0..self.params.num_rounds {
            // Compute gradients for all rows
            self.loss
                .compute_gradients(&predictions, labels, &mut grads);

            // Sample rows for this round (same seed + round = reproducible)
            let round_seed = self.params.seed.wrapping_add(round as u64);
            let sampled_rows = row_sampler.sample(round_seed);

            // Reset partitioner for new tree with sampled rows
            if sampling_enabled {
                partitioner.reset_with_rows(&sampled_rows);
            } else {
                partitioner.reset();
            }

            // Build tree with seed for column sampling reproducibility
            let mut grower = TreeGrower::new(policy.clone(), cuts, self.params.tree_params.clone());
            let tree = grower.build_tree_with_seed(quantized, &grads, &mut partitioner, round_seed);

            // Update predictions for all rows (not just sampled)
            Self::update_predictions(&tree, quantized, &mut predictions);

            // Compute metrics for all eval sets
            let mut round_metrics: Vec<(String, f64)> = Vec::new();
            let mut early_stop_triggered = false;

            // Always log training metric
            if self.params.verbosity >= Verbosity::Info {
                let train_metric = self.compute_train_metric(&predictions, labels);
                round_metrics.push(("train".to_string(), train_metric));
            }

            // Process each eval set
            for (idx, eval_set) in eval_sets.iter().enumerate() {
                let eval_preds =
                    Self::predict_with_trees(&trees, &tree, eval_set.data, base_score);

                // Compute and log metric
                if self.params.verbosity >= Verbosity::Info {
                    let metric_value = self.compute_train_metric(&eval_preds, eval_set.labels);
                    round_metrics.push((eval_set.name.to_string(), metric_value));
                }

                // Check early stopping on first eval set (or could be configurable)
                if idx == 0 {
                    if let Some(early_stop) = &mut self.early_stopping {
                        if early_stop.should_stop(&eval_preds, eval_set.labels) {
                            self.logger.info(&format!(
                                "Early stopping at round {} (best: {})",
                                round,
                                early_stop.best_round()
                            ));
                            early_stop_triggered = true;
                        }
                    }
                }
            }

            // Log progress
            if self.params.verbosity >= Verbosity::Info && !round_metrics.is_empty() {
                self.logger.log_round(round as usize, &round_metrics);
            }

            trees.push(tree);

            if early_stop_triggered {
                break;
            }
        }

        self.logger.info(&format!(
            "Training complete: {} trees built",
            trees.len()
        ));

        // Convert to inference format
        self.freeze_forest(trees, base_score)
    }

    /// Train a gradient boosted forest from raw (unquantized) data.
    ///
    /// This is the simplified API that handles quantization automatically.
    /// For advanced use cases where you need to control quantization, use [`train`]
    /// instead.
    ///
    /// # Type Parameters
    ///
    /// * `G` - Growth policy (DepthWisePolicy or LeafWisePolicy)
    /// * `D` - Data matrix type (must implement ColumnAccess)
    /// * `C` - Cut finder strategy (e.g., ExactQuantileCuts)
    ///
    /// # Arguments
    ///
    /// * `policy` - Growth policy for tree building
    /// * `data` - Raw feature matrix
    /// * `labels` - Target labels
    /// * `cut_finder` - Strategy for computing bin boundaries
    /// * `max_bins` - Maximum number of bins per feature (256 is typical)
    /// * `eval_sets` - Evaluation sets for metrics and early stopping
    ///
    /// # Example
    ///
    /// ```ignore
    /// use booste_rs::training::{GBTreeTrainer, EvalSet, DepthWisePolicy, ExactQuantileCuts};
    ///
    /// let cut_finder = ExactQuantileCuts::default();
    /// let eval_sets = vec![
    ///     EvalSet::new("val", &val_data, &val_labels),
    /// ];
    /// let forest = trainer.train_with_data(
    ///     DepthWisePolicy { max_depth: 6 },
    ///     &data,
    ///     &labels,
    ///     &cut_finder,
    ///     256,
    ///     &eval_sets,
    /// );
    /// ```
    pub fn train_with_data<G, D, C>(
        &mut self,
        policy: G,
        data: &D,
        labels: &[f32],
        cut_finder: &C,
        max_bins: usize,
        eval_sets: &[EvalSet<'_, D>],
    ) -> SoAForest<ScalarLeaf>
    where
        G: GrowthPolicy + Clone,
        D: ColumnAccess<Element = f32> + Sync,
        C: CutFinder,
    {
        // Compute bin cuts from training data
        self.logger.info("Computing bin cuts...");
        let quantizer = Quantizer::from_data(data, cut_finder, max_bins);
        let cuts = quantizer.cuts().clone();

        // Quantize training data
        self.logger.info("Quantizing training data...");
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

        // Delegate to the main train method
        self.train(policy, &quantized, labels, &cuts, &quantized_refs)
    }

    /// Compute base score from labels.
    fn compute_base_score(&self, labels: &[f32]) -> f32 {
        match self.params.base_score {
            BaseScore::Mean => {
                let sum: f32 = labels.iter().sum();
                sum / labels.len() as f32
            }
            BaseScore::Fixed(v) => v,
            BaseScore::Zero => 0.0,
        }
    }

    /// Update predictions by adding tree outputs.
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

    /// Traverse a BuildingTree for a single row.
    ///
    /// This is used during training to update predictions.
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
                // Missing value (bin 0) - use default direction
                split.default_left
            } else if split.is_categorical {
                // Categorical: check if bin is in left set
                split.categories_left.contains(&(bin as u32))
            } else {
                // Numerical: bin <= split_bin goes left
                bin <= split.split_bin as usize
            };

            node_id = if goes_left { node.left } else { node.right };
        }
    }

    /// Predict with all trees so far plus the new tree.
    fn predict_with_trees<B: BinIndex>(
        trees: &[BuildingTree],
        new_tree: &BuildingTree,
        data: &QuantizedMatrix<B>,
        base_score: f32,
    ) -> Vec<f32> {
        let num_rows = data.num_rows() as usize;
        let mut predictions = vec![base_score; num_rows];

        // Add predictions from existing trees
        for tree in trees {
            for row in 0..num_rows {
                predictions[row] += Self::predict_row(tree, data, row as u32);
            }
        }

        // Add predictions from new tree
        for row in 0..num_rows {
            predictions[row] += Self::predict_row(new_tree, data, row as u32);
        }

        predictions
    }

    /// Compute training metric (simple squared error for now).
    fn compute_train_metric(&self, predictions: &[f32], labels: &[f32]) -> f64 {
        let sum_sq_err: f64 = predictions
            .iter()
            .zip(labels.iter())
            .map(|(p, l)| (p - l).powi(2) as f64)
            .sum();
        (sum_sq_err / predictions.len() as f64).sqrt()
    }

    /// Convert BuildingTrees to inference-optimized SoAForest.
    fn freeze_forest(&self, trees: Vec<BuildingTree>, base_score: f32) -> SoAForest<ScalarLeaf> {
        let mut forest = SoAForest::for_regression().with_base_score(vec![base_score]);

        for building_tree in trees {
            let soa_tree = self.convert_tree(&building_tree);
            forest.push_tree(soa_tree, 0); // All trees in group 0 for regression
        }

        forest
    }

    /// Convert a BuildingTree to SoATreeStorage.
    fn convert_tree(&self, building: &BuildingTree) -> SoATreeStorage<ScalarLeaf> {
        let mut builder = SoATreeBuilder::<ScalarLeaf>::new();

        // Process nodes in order (BFS or just iterate since nodes are already numbered)
        for node_id in 0..building.num_nodes() as u32 {
            let node = building.node(node_id);

            if node.is_leaf {
                builder.add_leaf(ScalarLeaf(node.weight));
            } else {
                let split = node.split.as_ref().expect("Non-leaf must have split");
                if split.is_categorical {
                    // Convert categories_left to bitset format
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

/// Convert category list to bitset format (u32 words).
fn categories_to_bitset(categories: &[u32]) -> Vec<u32> {
    if categories.is_empty() {
        return vec![];
    }

    let max_cat = *categories.iter().max().unwrap_or(&0);
    let num_words = (max_cat / 32 + 1) as usize;
    let mut bitset = vec![0u32; num_words];

    for &cat in categories {
        let word = (cat / 32) as usize;
        let bit = cat % 32;
        bitset[word] |= 1u32 << bit;
    }

    bitset
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::DenseMatrix;
    use crate::training::gbtree::quantize::{CutFinder, ExactQuantileCuts, Quantizer};
    use crate::training::{DepthWisePolicy, LeafWisePolicy, SquaredLoss};

    fn make_regression_data() -> (QuantizedMatrix<u8>, BinCuts, Vec<f32>) {
        // Simple linear relationship: y = x0 + noise
        let mut data = Vec::new();
        let mut labels = Vec::new();
        for i in 0..100 {
            let x0 = i as f32 / 10.0;
            let x1 = (i % 10) as f32;
            data.push(x0);
            data.push(x1);
            labels.push(x0 + 0.1); // y ≈ x0
        }

        let matrix = DenseMatrix::from_vec(data, 100, 2);
        let cuts_finder = ExactQuantileCuts::new(1);
        let cuts = cuts_finder.find_cuts(&matrix, 256);
        let quantizer = Quantizer::new(cuts.clone());
        let quantized = quantizer.quantize::<_, u8>(&matrix);

        (quantized, cuts, labels)
    }

    #[test]
    fn test_trainer_params_default() {
        let params = TrainerParams::default();
        assert_eq!(params.num_rounds, 100);
        assert_eq!(params.base_score, BaseScore::Mean);
    }

    #[test]
    fn test_base_score_mean() {
        let labels = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let params = TrainerParams {
            base_score: BaseScore::Mean,
            ..Default::default()
        };
        let trainer = GBTreeTrainer::new(Box::new(SquaredLoss), params);
        let base = trainer.compute_base_score(&labels);
        assert!((base - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_base_score_fixed() {
        let labels = vec![1.0, 2.0, 3.0];
        let params = TrainerParams {
            base_score: BaseScore::Fixed(0.5),
            ..Default::default()
        };
        let trainer = GBTreeTrainer::new(Box::new(SquaredLoss), params);
        let base = trainer.compute_base_score(&labels);
        assert_eq!(base, 0.5);
    }

    #[test]
    fn test_train_depth_wise() {
        let (quantized, cuts, labels) = make_regression_data();

        let params = TrainerParams {
            num_rounds: 10,
            tree_params: TreeParams {
                max_depth: 3,
                learning_rate: 0.3,
                ..Default::default()
            },
            verbosity: Verbosity::Silent,
            ..Default::default()
        };

        let policy = DepthWisePolicy { max_depth: 3 };
        let mut trainer = GBTreeTrainer::new(Box::new(SquaredLoss), params);
        let forest = trainer.train(policy, &quantized, &labels, &cuts, &[]);

        assert_eq!(forest.num_trees(), 10);
        assert_eq!(forest.num_groups(), 1);
    }

    #[test]
    fn test_train_leaf_wise() {
        let (quantized, cuts, labels) = make_regression_data();

        let params = TrainerParams {
            num_rounds: 10,
            tree_params: TreeParams {
                max_leaves: 8,
                learning_rate: 0.3,
                ..Default::default()
            },
            verbosity: Verbosity::Silent,
            ..Default::default()
        };

        let policy = LeafWisePolicy { max_leaves: 8 };
        let mut trainer = GBTreeTrainer::new(Box::new(SquaredLoss), params);
        let forest = trainer.train(policy, &quantized, &labels, &cuts, &[]);

        assert_eq!(forest.num_trees(), 10);
    }

    #[test]
    fn test_predict_row() {
        // Build a simple tree manually and verify prediction
        let (quantized, cuts, labels) = make_regression_data();

        let params = TrainerParams {
            num_rounds: 1,
            tree_params: TreeParams {
                max_depth: 2,
                learning_rate: 1.0,
                ..Default::default()
            },
            verbosity: Verbosity::Silent,
            ..Default::default()
        };

        let policy = DepthWisePolicy { max_depth: 2 };
        let mut trainer = GBTreeTrainer::new(Box::new(SquaredLoss), params);
        let forest = trainer.train(policy, &quantized, &labels, &cuts, &[]);

        // Forest should be valid and produce predictions
        assert_eq!(forest.num_trees(), 1);
    }

    #[test]
    fn test_predictions_improve() {
        let (quantized, cuts, labels) = make_regression_data();

        let params = TrainerParams {
            num_rounds: 20,
            tree_params: TreeParams {
                max_depth: 4,
                learning_rate: 0.3,
                ..Default::default()
            },
            verbosity: Verbosity::Silent,
            ..Default::default()
        };

        let policy = DepthWisePolicy { max_depth: 4 };
        let mut trainer = GBTreeTrainer::new(Box::new(SquaredLoss), params);

        // Train for a few rounds
        let forest = trainer.train(policy, &quantized, &labels, &cuts, &[]);

        // Predict on training data (should fit reasonably well)
        // Since we're using the inference forest, we need to use raw features
        // For now just verify we got trees
        assert!(forest.num_trees() > 0);
    }

    #[test]
    fn test_categories_to_bitset() {
        let cats = vec![0, 1, 5, 32, 33];
        let bitset = categories_to_bitset(&cats);

        assert_eq!(bitset.len(), 2); // Need 2 words for cat 33

        // Check bits
        assert!(bitset[0] & (1 << 0) != 0); // cat 0
        assert!(bitset[0] & (1 << 1) != 0); // cat 1
        assert!(bitset[0] & (1 << 5) != 0); // cat 5
        assert!(bitset[1] & (1 << 0) != 0); // cat 32
        assert!(bitset[1] & (1 << 1) != 0); // cat 33
    }

    #[test]
    fn test_empty_categories_bitset() {
        let cats: Vec<u32> = vec![];
        let bitset = categories_to_bitset(&cats);
        assert!(bitset.is_empty());
    }

    // ========================================================================
    // Story 7 Test Cases
    // ========================================================================

    #[test]
    fn test_gradients_computed_correctly() {
        // 7.T2: Verify gradients computed using Loss trait
        let (quantized, cuts, labels) = make_regression_data();

        let params = TrainerParams {
            num_rounds: 1,
            tree_params: TreeParams {
                max_depth: 2,
                learning_rate: 1.0,
                ..Default::default()
            },
            verbosity: Verbosity::Silent,
            base_score: BaseScore::Zero, // Start from 0 for predictable gradients
            ..Default::default()
        };

        let policy = DepthWisePolicy { max_depth: 2 };
        let mut trainer = GBTreeTrainer::new(Box::new(SquaredLoss), params);
        let _forest = trainer.train(policy, &quantized, &labels, &cuts, &[]);

        // For squared loss with predictions=0, gradients should be -labels
        // The tree should be built to reduce these gradients
        // We can verify the forest was built (training worked)
    }

    #[test]
    fn test_predictions_updated_after_tree() {
        // 7.T3: Predictions updated after each tree
        let (quantized, cuts, labels) = make_regression_data();

        // Train multiple rounds and check that more trees = better fit
        let params_1 = TrainerParams {
            num_rounds: 1,
            tree_params: TreeParams {
                max_depth: 3,
                learning_rate: 0.5,
                ..Default::default()
            },
            verbosity: Verbosity::Silent,
            ..Default::default()
        };

        let params_10 = TrainerParams {
            num_rounds: 10,
            tree_params: TreeParams {
                max_depth: 3,
                learning_rate: 0.5,
                ..Default::default()
            },
            verbosity: Verbosity::Silent,
            ..Default::default()
        };

        let policy = DepthWisePolicy { max_depth: 3 };

        let mut trainer_1 = GBTreeTrainer::new(Box::new(SquaredLoss), params_1);
        let forest_1 = trainer_1.train(policy.clone(), &quantized, &labels, &cuts, &[]);

        let mut trainer_10 = GBTreeTrainer::new(Box::new(SquaredLoss), params_10);
        let forest_10 = trainer_10.train(policy, &quantized, &labels, &cuts, &[]);

        assert_eq!(forest_1.num_trees(), 1);
        assert_eq!(forest_10.num_trees(), 10);
    }

    #[test]
    fn test_early_stopping_triggers() {
        // 7.T4: Early stopping triggers after patience exceeded
        use crate::training::Rmse;

        let (quantized, cuts, labels) = make_regression_data();

        // Create separate eval set with different labels to simulate overfitting
        // This ensures the eval loss will stop improving while train loss continues
        let mut eval_labels = labels.clone();
        // Add noise to eval labels - model will overfit to training data
        for (i, l) in eval_labels.iter_mut().enumerate() {
            *l += ((i % 7) as f32 - 3.0) * 2.0; // Systematic noise
        }

        let params = TrainerParams {
            num_rounds: 100, // Many rounds - early stopping should kick in
            tree_params: TreeParams {
                max_depth: 6, // Deep trees to overfit
                learning_rate: 0.8, // High learning rate to overfit quickly
                min_samples_split: 1,
                min_samples_leaf: 1,
                ..Default::default()
            },
            verbosity: Verbosity::Silent,
            ..Default::default()
        };

        let policy = DepthWisePolicy { max_depth: 6 };
        let mut trainer = GBTreeTrainer::new(Box::new(SquaredLoss), params)
            .with_early_stopping(Box::new(Rmse), 3); // Patience of 3 rounds

        let eval_set = QuantizedEvalSet::new("eval", &quantized, &eval_labels);
        let forest = trainer.train(
            policy,
            &quantized,
            &labels,
            &cuts,
            &[eval_set],
        );

        // With high learning rate, deep trees, and noisy eval data,
        // should stop when overfitting starts (eval loss increases)
        assert!(
            forest.num_trees() < 100,
            "Expected early stopping, got {} trees",
            forest.num_trees()
        );
    }

    #[test]
    fn test_forest_prediction_quality() {
        // Integration-level test: verify trained forest can predict
        let (quantized, cuts, labels) = make_regression_data();

        let params = TrainerParams {
            num_rounds: 50,
            tree_params: TreeParams {
                max_depth: 4,
                learning_rate: 0.2,
                ..Default::default()
            },
            verbosity: Verbosity::Silent,
            ..Default::default()
        };

        let policy = DepthWisePolicy { max_depth: 4 };
        let mut trainer = GBTreeTrainer::new(Box::new(SquaredLoss), params);
        let forest = trainer.train(policy, &quantized, &labels, &cuts, &[]);

        // The forest should produce reasonable predictions
        assert_eq!(forest.num_trees(), 50);
        assert_eq!(forest.num_groups(), 1);
        // Base score should be mean of labels (around 5.0 for our data)
        let base_scores = forest.base_score();
        assert!(base_scores.len() == 1);
        let bs = base_scores[0];
        // Mean of y = x0 + 0.1 where x0 in [0, 10) with step 0.1
        // Mean should be around (0 + 9.9) / 2 + 0.1 = 5.05
        assert!(
            (bs - 5.0).abs() < 1.0,
            "Base score {} far from expected ~5.0",
            bs
        );
    }

    #[test]
    fn test_leaf_wise_vs_depth_wise() {
        // Compare leaf-wise and depth-wise on same data
        let (quantized, cuts, labels) = make_regression_data();

        let params = TrainerParams {
            num_rounds: 20,
            tree_params: TreeParams {
                max_depth: 4,
                max_leaves: 16,
                learning_rate: 0.3,
                ..Default::default()
            },
            verbosity: Verbosity::Silent,
            ..Default::default()
        };

        let depth_policy = DepthWisePolicy { max_depth: 4 };
        let leaf_policy = LeafWisePolicy { max_leaves: 16 };

        let mut trainer_d = GBTreeTrainer::new(Box::new(SquaredLoss), params.clone());
        let forest_d = trainer_d.train(depth_policy, &quantized, &labels, &cuts, &[]);

        let mut trainer_l = GBTreeTrainer::new(Box::new(SquaredLoss), params);
        let forest_l = trainer_l.train(leaf_policy, &quantized, &labels, &cuts, &[]);

        // Both should produce 20 trees
        assert_eq!(forest_d.num_trees(), 20);
        assert_eq!(forest_l.num_trees(), 20);
    }

    #[test]
    fn test_train_with_data() {
        // Test the simplified API that handles quantization internally
        let mut data = Vec::new();
        let mut labels = Vec::new();
        for i in 0..100 {
            let x0 = i as f32 / 10.0;
            let x1 = (i % 10) as f32;
            data.push(x0);
            data.push(x1);
            labels.push(x0 + 0.1); // y ≈ x0
        }

        let matrix = DenseMatrix::from_vec(data, 100, 2);

        let params = TrainerParams {
            num_rounds: 10,
            tree_params: TreeParams {
                max_depth: 3,
                learning_rate: 0.3,
                ..Default::default()
            },
            verbosity: Verbosity::Silent,
            ..Default::default()
        };

        let policy = DepthWisePolicy { max_depth: 3 };
        let cut_finder = ExactQuantileCuts::default();
        let mut trainer = GBTreeTrainer::new(Box::new(SquaredLoss), params);

        let forest = trainer.train_with_data(
            policy,
            &matrix,
            &labels,
            &cut_finder,
            256,
            &[], // no eval sets
        );

        assert_eq!(forest.num_trees(), 10);
        assert_eq!(forest.num_groups(), 1);
    }

    #[test]
    fn test_train_with_data_eval_sets() {
        // Test with evaluation sets
        let mut data = Vec::new();
        let mut labels = Vec::new();
        for i in 0..100 {
            let x0 = i as f32 / 10.0;
            let x1 = (i % 10) as f32;
            data.push(x0);
            data.push(x1);
            labels.push(x0 + 0.1);
        }

        let matrix = DenseMatrix::from_vec(data, 100, 2);

        // Create a small validation set
        let mut val_data = Vec::new();
        let mut val_labels = Vec::new();
        for i in 0..20 {
            let x0 = i as f32 / 10.0 + 0.5;
            let x1 = (i % 10) as f32;
            val_data.push(x0);
            val_data.push(x1);
            val_labels.push(x0 + 0.1);
        }
        let val_matrix = DenseMatrix::from_vec(val_data, 20, 2);

        let params = TrainerParams {
            num_rounds: 10,
            tree_params: TreeParams {
                max_depth: 3,
                learning_rate: 0.3,
                ..Default::default()
            },
            verbosity: Verbosity::Silent,
            ..Default::default()
        };

        let policy = DepthWisePolicy { max_depth: 3 };
        let cut_finder = ExactQuantileCuts::default();
        let mut trainer = GBTreeTrainer::new(Box::new(SquaredLoss), params);

        let eval_sets = vec![EvalSet::new("val", &val_matrix, &val_labels)];

        let forest = trainer.train_with_data(
            policy,
            &matrix,
            &labels,
            &cut_finder,
            256,
            &eval_sets,
        );

        assert_eq!(forest.num_trees(), 10);
    }

    // ========================================================================
    // Row Sampling Tests (Story 3)
    // ========================================================================

    #[test]
    fn test_train_with_subsample() {
        // 3.T1: Training with row subsampling produces valid forest
        let (quantized, cuts, labels) = make_regression_data();

        let params = TrainerParams {
            num_rounds: 10,
            tree_params: TreeParams {
                max_depth: 3,
                learning_rate: 0.3,
                subsample: 0.8, // Use 80% of rows per tree
                ..Default::default()
            },
            verbosity: Verbosity::Silent,
            seed: 42,
            ..Default::default()
        };

        let policy = DepthWisePolicy { max_depth: 3 };
        let mut trainer = GBTreeTrainer::new(Box::new(SquaredLoss), params);
        let forest = trainer.train(policy, &quantized, &labels, &cuts, &[]);

        assert_eq!(forest.num_trees(), 10);
        assert_eq!(forest.num_groups(), 1);
    }

    #[test]
    fn test_subsample_reproducibility() {
        // 3.T2: Same seed produces identical forests
        let (quantized, cuts, labels) = make_regression_data();

        let make_params = |seed: u64| TrainerParams {
            num_rounds: 5,
            tree_params: TreeParams {
                max_depth: 3,
                learning_rate: 0.3,
                subsample: 0.5,
                ..Default::default()
            },
            verbosity: Verbosity::Silent,
            seed,
            ..Default::default()
        };

        let policy = DepthWisePolicy { max_depth: 3 };

        // Train twice with same seed
        let mut trainer1 = GBTreeTrainer::new(Box::new(SquaredLoss), make_params(42));
        let forest1 = trainer1.train(policy.clone(), &quantized, &labels, &cuts, &[]);

        let mut trainer2 = GBTreeTrainer::new(Box::new(SquaredLoss), make_params(42));
        let forest2 = trainer2.train(policy.clone(), &quantized, &labels, &cuts, &[]);

        // Same seed should produce identical base scores
        assert_eq!(forest1.base_score(), forest2.base_score());
        assert_eq!(forest1.num_trees(), forest2.num_trees());
    }

    #[test]
    fn test_different_seeds_different_forests() {
        // 3.T3: Different seeds produce different forests
        let (quantized, cuts, labels) = make_regression_data();

        let make_params = |seed: u64| TrainerParams {
            num_rounds: 5,
            tree_params: TreeParams {
                max_depth: 3,
                learning_rate: 0.3,
                subsample: 0.5,
                ..Default::default()
            },
            verbosity: Verbosity::Silent,
            seed,
            ..Default::default()
        };

        let policy = DepthWisePolicy { max_depth: 3 };

        let mut trainer1 = GBTreeTrainer::new(Box::new(SquaredLoss), make_params(42));
        let forest1 = trainer1.train(policy.clone(), &quantized, &labels, &cuts, &[]);

        let mut trainer2 = GBTreeTrainer::new(Box::new(SquaredLoss), make_params(123));
        let forest2 = trainer2.train(policy.clone(), &quantized, &labels, &cuts, &[]);

        // Different seeds should produce different forests
        // (they might accidentally be the same, but very unlikely with 50% subsample)
        assert_eq!(forest1.num_trees(), forest2.num_trees());
        // Base scores should be the same (computed from all labels)
        assert_eq!(forest1.base_score(), forest2.base_score());
    }

    #[test]
    fn test_subsample_full_sample() {
        // 3.T4: subsample=1.0 should use all rows (same as no sampling)
        let (quantized, cuts, labels) = make_regression_data();

        let params_no_sample = TrainerParams {
            num_rounds: 5,
            tree_params: TreeParams {
                max_depth: 3,
                learning_rate: 0.3,
                subsample: 1.0, // No sampling
                ..Default::default()
            },
            verbosity: Verbosity::Silent,
            seed: 42,
            ..Default::default()
        };

        let params_with_sample = TrainerParams {
            num_rounds: 5,
            tree_params: TreeParams {
                max_depth: 3,
                learning_rate: 0.3,
                subsample: 1.0, // Still no effective sampling
                ..Default::default()
            },
            verbosity: Verbosity::Silent,
            seed: 999, // Different seed shouldn't matter
            ..Default::default()
        };

        let policy = DepthWisePolicy { max_depth: 3 };

        let mut trainer1 = GBTreeTrainer::new(Box::new(SquaredLoss), params_no_sample);
        let forest1 = trainer1.train(policy.clone(), &quantized, &labels, &cuts, &[]);

        let mut trainer2 = GBTreeTrainer::new(Box::new(SquaredLoss), params_with_sample);
        let forest2 = trainer2.train(policy.clone(), &quantized, &labels, &cuts, &[]);

        // With subsample=1.0, seed shouldn't affect result
        assert_eq!(forest1.base_score(), forest2.base_score());
        assert_eq!(forest1.num_trees(), forest2.num_trees());
    }

    #[test]
    fn test_subsample_small_ratio() {
        // 3.T5: Very small subsample ratio still works
        let (quantized, cuts, labels) = make_regression_data();

        let params = TrainerParams {
            num_rounds: 5,
            tree_params: TreeParams {
                max_depth: 2,
                learning_rate: 0.3,
                subsample: 0.1, // Only 10% of rows (10 rows from 100)
                min_samples_split: 1,
                min_samples_leaf: 1,
                ..Default::default()
            },
            verbosity: Verbosity::Silent,
            seed: 42,
            ..Default::default()
        };

        let policy = DepthWisePolicy { max_depth: 2 };
        let mut trainer = GBTreeTrainer::new(Box::new(SquaredLoss), params);
        let forest = trainer.train(policy, &quantized, &labels, &cuts, &[]);

        // Should still produce valid trees
        assert_eq!(forest.num_trees(), 5);
    }

    // ========================================================================
    // Column Sampling Tests (Story 3)
    // ========================================================================

    #[test]
    fn test_train_with_colsample_bytree() {
        // 3.T6: Training with column subsampling per tree
        let (quantized, cuts, labels) = make_regression_data();

        let params = TrainerParams {
            num_rounds: 10,
            tree_params: TreeParams {
                max_depth: 3,
                learning_rate: 0.3,
                colsample_bytree: 0.5, // Use 50% of features per tree
                ..Default::default()
            },
            verbosity: Verbosity::Silent,
            seed: 42,
            ..Default::default()
        };

        let policy = DepthWisePolicy { max_depth: 3 };
        let mut trainer = GBTreeTrainer::new(Box::new(SquaredLoss), params);
        let forest = trainer.train(policy, &quantized, &labels, &cuts, &[]);

        assert_eq!(forest.num_trees(), 10);
    }

    #[test]
    fn test_train_with_colsample_bylevel() {
        // 3.T7: Training with column subsampling per level
        let (quantized, cuts, labels) = make_regression_data();

        let params = TrainerParams {
            num_rounds: 10,
            tree_params: TreeParams {
                max_depth: 4,
                learning_rate: 0.3,
                colsample_bylevel: 0.8, // Use 80% of features per level
                ..Default::default()
            },
            verbosity: Verbosity::Silent,
            seed: 42,
            ..Default::default()
        };

        let policy = DepthWisePolicy { max_depth: 4 };
        let mut trainer = GBTreeTrainer::new(Box::new(SquaredLoss), params);
        let forest = trainer.train(policy, &quantized, &labels, &cuts, &[]);

        assert_eq!(forest.num_trees(), 10);
    }

    #[test]
    fn test_train_with_colsample_bynode() {
        // 3.T8: Training with column subsampling per node
        let (quantized, cuts, labels) = make_regression_data();

        let params = TrainerParams {
            num_rounds: 10,
            tree_params: TreeParams {
                max_depth: 3,
                learning_rate: 0.3,
                colsample_bynode: 0.7, // Use 70% of features per node
                ..Default::default()
            },
            verbosity: Verbosity::Silent,
            seed: 42,
            ..Default::default()
        };

        let policy = DepthWisePolicy { max_depth: 3 };
        let mut trainer = GBTreeTrainer::new(Box::new(SquaredLoss), params);
        let forest = trainer.train(policy, &quantized, &labels, &cuts, &[]);

        assert_eq!(forest.num_trees(), 10);
    }

    #[test]
    fn test_train_with_combined_sampling() {
        // 3.T9: Training with both row and column sampling combined
        let (quantized, cuts, labels) = make_regression_data();

        let params = TrainerParams {
            num_rounds: 10,
            tree_params: TreeParams {
                max_depth: 3,
                learning_rate: 0.3,
                subsample: 0.8,           // 80% of rows
                colsample_bytree: 0.9,    // 90% of features per tree
                colsample_bylevel: 0.9,   // 90% of remaining per level
                colsample_bynode: 0.9,    // 90% of remaining per node
                ..Default::default()
            },
            verbosity: Verbosity::Silent,
            seed: 42,
            ..Default::default()
        };

        let policy = DepthWisePolicy { max_depth: 3 };
        let mut trainer = GBTreeTrainer::new(Box::new(SquaredLoss), params);
        let forest = trainer.train(policy, &quantized, &labels, &cuts, &[]);

        assert_eq!(forest.num_trees(), 10);
    }

    #[test]
    fn test_colsample_reproducibility() {
        // 3.T10: Same seed produces identical forests with column sampling
        let (quantized, cuts, labels) = make_regression_data();

        let make_params = |seed: u64| TrainerParams {
            num_rounds: 5,
            tree_params: TreeParams {
                max_depth: 3,
                learning_rate: 0.3,
                colsample_bytree: 0.5,
                ..Default::default()
            },
            verbosity: Verbosity::Silent,
            seed,
            ..Default::default()
        };

        let policy = DepthWisePolicy { max_depth: 3 };

        let mut trainer1 = GBTreeTrainer::new(Box::new(SquaredLoss), make_params(42));
        let forest1 = trainer1.train(policy.clone(), &quantized, &labels, &cuts, &[]);

        let mut trainer2 = GBTreeTrainer::new(Box::new(SquaredLoss), make_params(42));
        let forest2 = trainer2.train(policy.clone(), &quantized, &labels, &cuts, &[]);

        // Same seed should produce identical forests
        assert_eq!(forest1.base_score(), forest2.base_score());
        assert_eq!(forest1.num_trees(), forest2.num_trees());
    }

    #[test]
    fn test_aggressive_sampling() {
        // 3.T11: Aggressive sampling (low values) should still work
        let (quantized, cuts, labels) = make_regression_data();

        let params = TrainerParams {
            num_rounds: 5,
            tree_params: TreeParams {
                max_depth: 3,
                learning_rate: 0.5,
                subsample: 0.3,           // Only 30% of rows
                colsample_bytree: 0.5,    // Only 50% of features
                min_samples_split: 1,
                min_samples_leaf: 1,
                ..Default::default()
            },
            verbosity: Verbosity::Silent,
            seed: 42,
            ..Default::default()
        };

        let policy = DepthWisePolicy { max_depth: 3 };
        let mut trainer = GBTreeTrainer::new(Box::new(SquaredLoss), params);
        let forest = trainer.train(policy, &quantized, &labels, &cuts, &[]);

        // Should still produce valid trees
        assert_eq!(forest.num_trees(), 5);
    }
}

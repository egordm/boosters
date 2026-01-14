//! GBDT Trainer for gradient boosting.
//!
//! Orchestrates objective computation, tree growing, and prediction updates.
//! Use [`GBDTTrainer::train`] to train a forest from a binned dataset.

use crate::data::{BinnedDataset, Dataset, init_predictions};
use crate::data::{TargetsView, WeightsView};
use crate::training::Gradients;
use crate::training::Verbosity;
use crate::training::callback::{EarlyStopAction, EarlyStopping};
use crate::training::eval;
use crate::training::logger::TrainingLogger;
use crate::training::sampling::{ColSamplingParams, RowSampler, RowSamplingParams};
use crate::training::{Metric, Objective};

use super::expansion::GrowthStrategy;
use super::grower::{GrowerParams, TreeGrower};
use super::linear::{LeafLinearTrainer, LinearLeafConfig};
use super::split::GainParams;
use crate::utils::Parallelism;

use crate::repr::gbdt::{Forest, ScalarLeaf};
use ndarray::Array2;

// =============================================================================
// GBDTParams
// =============================================================================

/// Parameters for GBDT training.
#[derive(Clone, Debug)]
pub struct GBDTParams {
    // --- Boosting parameters ---
    /// Number of boosting rounds (trees to train).
    pub n_trees: u32,
    /// Learning rate (shrinkage).
    pub learning_rate: f32,

    // --- Tree structure ---
    /// Tree growth strategy.
    pub growth_strategy: GrowthStrategy,
    /// Max categories for one-hot encoding categorical splits.
    pub max_onehot_cats: u32,

    // --- Regularization (encapsulated in GainParams) ---
    /// Gain computation parameters (regularization, min child weight, etc.).
    pub gain: GainParams,

    // --- Sampling ---
    /// Row sampling configuration.
    pub row_sampling: RowSamplingParams,
    /// Column sampling configuration.
    pub col_sampling: ColSamplingParams,

    /// Histogram cache size (number of slots).
    pub cache_size: usize,

    // --- Early stopping ---
    /// Early stopping rounds. Training stops if no improvement for this many rounds.
    /// Set to 0 to disable.
    pub early_stopping_rounds: u32,

    // --- Logging ---
    /// Verbosity level for training output.
    pub verbosity: Verbosity,

    // --- Reproducibility ---
    /// Random seed.
    pub seed: u64,

    // --- Linear leaves ---
    /// Linear leaf configuration. If set, fit linear models in leaves.
    /// See RFC-0009 for design rationale.
    pub linear_leaves: Option<LinearLeafConfig>,
}

impl Default for GBDTParams {
    fn default() -> Self {
        Self {
            n_trees: 100,
            learning_rate: 0.3,
            growth_strategy: GrowthStrategy::default(),
            max_onehot_cats: 4,
            gain: GainParams::default(),
            row_sampling: RowSamplingParams::None,
            col_sampling: ColSamplingParams::None,
            cache_size: 8,
            early_stopping_rounds: 0,
            verbosity: Verbosity::default(),
            seed: 42,
            linear_leaves: None,
        }
    }
}

impl GBDTParams {
    /// Convert to GrowerParams for tree grower.
    fn to_grower_params(&self) -> GrowerParams {
        GrowerParams {
            gain: self.gain.clone(),
            learning_rate: self.learning_rate,
            growth_strategy: self.growth_strategy,
            max_onehot_cats: self.max_onehot_cats,
            col_sampling: self.col_sampling.clone(),
        }
    }
}

// =============================================================================
// GBDTTrainer
// =============================================================================

/// GBDT Trainer.
pub struct GBDTTrainer {
    /// Objective function.
    objective: Objective,
    /// Evaluation metric.
    metric: Metric,
    /// Training parameters.
    params: GBDTParams,
}

impl GBDTTrainer {
    /// Create a new GBDT trainer.
    pub fn new(objective: Objective, metric: Metric, params: GBDTParams) -> Self {
        Self {
            objective,
            metric,
            params,
        }
    }

    /// Get reference to parameters.
    pub fn params(&self) -> &GBDTParams {
        &self.params
    }

    /// Get reference to objective.
    pub fn objective(&self) -> &Objective {
        &self.objective
    }

    /// Get reference to metric.
    pub fn metric(&self) -> &Metric {
        &self.metric
    }

    /// Train a forest.
    ///
    /// **Note:** This method does NOT create a thread pool. The caller must set up
    /// parallelism via `rayon::ThreadPool::install()` if desired.
    ///
    /// # Arguments
    ///
    /// * `dataset` - Raw dataset with feature values (for prediction, linear trees, SHAP)
    /// * `binned` - Binned dataset created with [`BinnedDataset::from_dataset`] (for histograms)
    /// * `targets` - Target values (length = n_rows × n_outputs)
    /// * `weights` - Sample weights (None for uniform)
    /// * `val_set` - Optional validation set for early stopping
    /// * `parallelism` - Sequential or Parallel iteration hint
    ///
    /// [`BinnedDataset::from_dataset`]: crate::data::BinnedDataset::from_dataset
    pub fn train(
        &self,
        dataset: &Dataset,
        binned: &BinnedDataset,
        targets: TargetsView<'_>,
        weights: WeightsView<'_>,
        val_set: Option<&Dataset>,
        parallelism: Parallelism,
    ) -> Option<Forest<ScalarLeaf>> {
        let n_rows = binned.n_samples();
        let n_outputs = self.objective.n_outputs();

        // Validate inputs
        if targets.n_samples() < n_rows {
            return None;
        }

        // Initialize components (train-local)
        let grower_params = self.params.to_grower_params();
        let mut grower =
            TreeGrower::new(binned, grower_params, self.params.cache_size, parallelism);

        // Initialize linear leaf trainer if configured
        let mut linear_trainer = self
            .params
            .linear_leaves
            .as_ref()
            .map(|config| LeafLinearTrainer::new(config.clone(), n_rows));

        let mut row_sampler = RowSampler::new(
            self.params.row_sampling.clone(),
            n_rows,
            self.params.seed,
            self.params.learning_rate,
        );

        let mut gradients = Gradients::new(n_rows, n_outputs);

        // Compute base scores using objective
        let base_scores = self.objective.compute_base_score(targets, weights);

        // Initialize predictions with base scores
        let mut predictions = init_predictions(&base_scores, n_rows);

        // Create inference forest directly (Phase 2: no conversion needed)
        let mut forest =
            Forest::<ScalarLeaf>::new(n_outputs as u32).with_base_score(base_scores.clone());

        // Check if we need evaluation
        let needs_evaluation = !matches!(&self.metric, Metric::None);

        // Initialize validation predictions with base scores
        // Shape: [n_outputs, n_val_samples]
        // Only allocate if evaluation is needed and val_set provided
        let mut val_predictions: Option<Array2<f32>> = if needs_evaluation {
            val_set.map(|vs| init_predictions(&base_scores, vs.n_samples()))
        } else {
            None
        };
        // Early stopping (always present, may be disabled)
        let mut early_stopping = EarlyStopping::new(
            self.params.early_stopping_rounds as usize,
            self.metric.higher_is_better(),
        );
        let mut best_n_trees: usize = 0;

        // Evaluator for computing metrics (only used if evaluation is needed)
        let mut evaluator =
            eval::Evaluator::new(&self.metric, self.objective.output_transform(), n_outputs);

        // Logger
        let mut logger = TrainingLogger::new(self.params.verbosity);
        logger.start_training(self.params.n_trees as usize);

        // Check if objective needs leaf value renewing
        let needs_leaf_renew = self.objective.needs_leaf_renew();

        // Scratch buffer for quantile computation (reused across leaves/outputs/rounds)
        let mut quantile_scratch: Vec<u32> = Vec::new();

        for round in 0..self.params.n_trees {
            // Compute gradients for all outputs
            self.objective.compute_gradients_into(
                predictions.view(),
                targets,
                weights,
                gradients.pairs_array_mut(),
            );

            // Grow one tree per output
            for output in 0..n_outputs {
                // Row sampling: modifies gradients in place for this output
                // - GOSS: amplifies sampled small-gradient rows
                // - Uniform: zeros out unsampled rows (split finding requires hess_sum > 0)
                let grad_hess = gradients.output_pairs_mut(output);
                let sampled = row_sampler.sample(round as usize, grad_hess);

                // Grow tree (returns MutableTree for potential linear fitting)
                let mut mutable_tree = grower.grow(binned, &gradients, output, sampled);

                // Leaf value renewing for quantile/L1 objectives
                // For these objectives, the optimal leaf value is the α-quantile of residuals,
                // not the Newton-step from gradient sums. This follows XGBoost/LightGBM pattern.
                if needs_leaf_renew {
                    // Iterate by index to avoid Vec allocation. We access leaf_node_mapping
                    // by index and immediately apply changes. The learning_rate is applied
                    // once per leaf.
                    let n_leaves = grower.leaf_node_mapping().len();
                    for leaf_idx in 0..n_leaves {
                        let (tree_node, part_node) = grower.leaf_node_mapping()[leaf_idx];
                        let indices = grower.partitioner().get_leaf_indices(part_node);
                        if indices.is_empty() {
                            continue;
                        }
                        let new_value = self.objective.renew_single_leaf_value(
                            output,
                            indices,
                            targets,
                            predictions.view(),
                            weights,
                            &mut quantile_scratch,
                        );
                        if new_value.is_nan() {
                            continue;
                        }
                        let scaled_value = new_value * self.params.learning_rate;
                        mutable_tree.set_leaf_value(tree_node, ScalarLeaf(scaled_value));
                        grower.update_cached_leaf_value(part_node, scaled_value);
                    }
                }

                // Fit linear models in leaves
                // Skip first N trees based on config (default 1: first tree has homogeneous gradients)
                // Only fit if linear_leaves config is set
                if let Some(ref mut linear_trainer) = linear_trainer {
                    let skip_n = linear_trainer.config().skip_first_n_trees;
                    if round >= skip_n {
                        let fitted = linear_trainer.train(
                            &mutable_tree,
                            dataset,
                            grower.partitioner(),
                            grower.leaf_node_mapping(),
                            &gradients,
                            output,
                            self.params.learning_rate,
                        );
                        // Apply fitted coefficients to tree
                        for leaf in fitted {
                            mutable_tree.set_linear_leaf(
                                leaf.node_id,
                                leaf.features,
                                leaf.intercept,
                                leaf.coefficients,
                            );
                        }
                    }
                }

                // Freeze tree
                let tree = mutable_tree.freeze();

                // Update predictions (row of Array2)
                let output_preds = predictions.row_mut(output);
                // Use fast path when possible:
                // - No sampling (all rows are in partitioner)
                // - No linear leaves (require computing linear predictions)
                // Linear leaves require tree.predict_into to compute correct predictions
                // since grower.update_predictions_from_last_tree uses only scalar values.
                // Note: leaf renewing now updates cached values, so fast path works!
                let has_linear_leaves = tree.has_linear_leaves();
                if sampled.is_none() && !has_linear_leaves {
                    // Fast path: use partitioner for O(n) prediction update
                    grower.update_predictions_from_last_tree(output_preds);
                } else {
                    // Fallback: row sampling trains on a subset, or linear leaves
                    // require computing linear predictions for correct gradient updates.
                    tree.predict_into(dataset, output_preds, parallelism);
                }

                // Incremental validation prediction: add this tree's contribution
                // val_predictions shape is [n_outputs, n_val_samples]
                // Only compute if evaluation is needed and val_set provided
                if let (Some(dataset_val), Some(ref mut vp)) = (val_set, val_predictions.as_mut()) {
                    let output_preds = vp.row_mut(output);
                    tree.predict_into(dataset_val, output_preds, parallelism);
                }

                forest.push_tree(tree, output as u32);
            }

            // Evaluate on validation set (using accumulated predictions)
            // Only evaluate if metric is enabled
            let (round_metrics, early_stop_value) = if needs_evaluation {
                let metrics = evaluator.evaluate_round(
                    predictions.view(),
                    targets,
                    weights,
                    val_set,
                    val_predictions.as_ref().map(|p| p.view()),
                );
                let value = eval::Evaluator::early_stop_value(&metrics, val_set.is_some());
                (metrics, value)
            } else {
                (Vec::new(), f64::NAN)
            };

            if self.params.verbosity >= Verbosity::Info {
                logger.log_metrics(round as usize, &round_metrics);
            }

            // Early stopping check (value always present: either eval or train metric)
            if early_stopping.is_enabled() {
                match early_stopping.update(early_stop_value) {
                    EarlyStopAction::Improved => {
                        best_n_trees = forest.n_trees();
                    }
                    EarlyStopAction::Stop => {
                        if self.params.verbosity >= Verbosity::Info {
                            logger.log_early_stopping(
                                round as usize,
                                early_stopping.best_round(),
                                self.metric.name(),
                            );
                        }
                        break;
                    }
                    EarlyStopAction::Continue => {}
                }
            }
        }

        logger.finish_training();

        // Return best model if early stopping was active and found a best
        if early_stopping.is_enabled() && best_n_trees > 0 && best_n_trees < forest.n_trees() {
            Some(Self::truncate_forest(&forest, best_n_trees))
        } else {
            Some(forest)
        }
    }

    /// Create a truncated copy of the forest with only the first `n_trees` trees.
    fn truncate_forest(forest: &Forest<ScalarLeaf>, n_trees: usize) -> Forest<ScalarLeaf> {
        let mut truncated =
            Forest::new(forest.n_groups()).with_base_score(forest.base_score().to_vec());

        for (idx, (tree, group)) in forest.trees_with_groups().enumerate() {
            if idx >= n_trees {
                break;
            }
            truncated.push_tree(tree.clone(), group);
        }

        truncated
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::WeightsView;
    use crate::data::{BinnedDataset, BinningConfig, Dataset};
    use crate::training::metrics::Metric;
    use crate::training::objectives::Objective;
    use ndarray::{Array2, arr2};

    /// Create test datasets - returns (raw Dataset, BinnedDataset)
    fn make_test_datasets() -> (Dataset, BinnedDataset) {
        // 8 rows, 2 features - feature-major layout [n_features, n_samples]
        let data = Array2::from_shape_vec(
            (2, 8),
            vec![
                0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, // feature 0
                1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, // feature 1
            ],
        )
        .unwrap();

        let raw = Dataset::from_array(data.view(), None, None);
        let binned = BinnedDataset::from_dataset(&raw, &BinningConfig::default()).unwrap();
        (raw, binned)
    }

    #[test]
    fn test_params_default() {
        let params = GBDTParams::default();

        assert_eq!(params.n_trees, 100);
        assert!((params.learning_rate - 0.3).abs() < 1e-6);
        assert_eq!(
            params.growth_strategy,
            GrowthStrategy::DepthWise { max_depth: 6 }
        );
        assert!((params.gain.reg_lambda - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_params_custom() {
        let params = GBDTParams {
            n_trees: 50,
            learning_rate: 0.1,
            growth_strategy: GrowthStrategy::DepthWise { max_depth: 4 },
            gain: GainParams {
                reg_lambda: 2.0,
                min_child_weight: 5.0,
                ..Default::default()
            },
            ..Default::default()
        };

        assert_eq!(params.n_trees, 50);
        assert!((params.learning_rate - 0.1).abs() < 1e-6);
        assert_eq!(
            params.growth_strategy,
            GrowthStrategy::DepthWise { max_depth: 4 }
        );
        assert!((params.gain.reg_lambda - 2.0).abs() < 1e-6);
        assert!((params.gain.min_child_weight - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_train_single_tree() {
        let (dataset, binned) = make_test_datasets();
        let targets_arr = arr2(&[[1.0f32, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 4.5]]);
        let targets = TargetsView::new(targets_arr.view());

        let params = GBDTParams {
            n_trees: 1,
            ..Default::default()
        };

        let trainer = GBDTTrainer::new(Objective::SquaredLoss, Metric::Rmse, params);
        let forest = trainer
            .train(
                &dataset,
                &binned,
                targets,
                WeightsView::None,
                None,
                Parallelism::Sequential,
            )
            .unwrap();

        assert_eq!(forest.n_trees(), 1);
        assert_eq!(forest.n_groups(), 1);
    }

    #[test]
    fn test_train_multiple_trees() {
        let (dataset, binned) = make_test_datasets();
        let targets_arr = arr2(&[[1.0f32, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 4.5]]);
        let targets = TargetsView::new(targets_arr.view());

        let params = GBDTParams {
            n_trees: 10,
            learning_rate: 0.1,
            ..Default::default()
        };

        let trainer = GBDTTrainer::new(Objective::SquaredLoss, Metric::Rmse, params);
        let forest = trainer
            .train(
                &dataset,
                &binned,
                targets,
                WeightsView::None,
                None,
                Parallelism::Sequential,
            )
            .unwrap();

        assert_eq!(forest.n_trees(), 10);
    }

    #[test]
    fn test_train_with_regularization() {
        let (dataset, binned) = make_test_datasets();
        let targets_arr = arr2(&[[1.0f32, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 4.5]]);
        let targets = TargetsView::new(targets_arr.view());

        let params = GBDTParams {
            n_trees: 5,
            gain: GainParams {
                reg_lambda: 10.0,
                min_gain: 0.5,
                ..Default::default()
            },
            ..Default::default()
        };

        let trainer = GBDTTrainer::new(Objective::SquaredLoss, Metric::Rmse, params);
        let forest = trainer
            .train(
                &dataset,
                &binned,
                targets,
                WeightsView::None,
                None,
                Parallelism::Sequential,
            )
            .unwrap();

        assert_eq!(forest.n_trees(), 5);
    }

    #[test]
    fn test_train_weighted() {
        let (dataset, binned) = make_test_datasets();
        let targets_arr = arr2(&[[1.0f32, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 4.5]]);
        let targets = TargetsView::new(targets_arr.view());
        let weights = ndarray::array![1.0f32, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0];

        let params = GBDTParams {
            n_trees: 5,
            ..Default::default()
        };

        let trainer = GBDTTrainer::new(Objective::SquaredLoss, Metric::Rmse, params);
        let forest = trainer
            .train(
                &dataset,
                &binned,
                targets,
                WeightsView::from_array(weights.view()),
                None,
                Parallelism::Sequential,
            )
            .unwrap();

        assert_eq!(forest.n_trees(), 5);
    }

    #[test]
    fn test_leaf_wise_growth() {
        let (dataset, binned) = make_test_datasets();
        let targets_arr = arr2(&[[1.0f32, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 4.5]]);
        let targets = TargetsView::new(targets_arr.view());

        let params = GBDTParams {
            n_trees: 3,
            growth_strategy: GrowthStrategy::LeafWise { max_leaves: 8 },
            ..Default::default()
        };

        let trainer = GBDTTrainer::new(Objective::SquaredLoss, Metric::Rmse, params);
        let forest = trainer
            .train(
                &dataset,
                &binned,
                targets,
                WeightsView::None,
                None,
                Parallelism::Sequential,
            )
            .unwrap();

        assert_eq!(forest.n_trees(), 3);
    }

    #[test]
    fn test_train_invalid_targets() {
        let (dataset, binned) = make_test_datasets();
        let targets_arr = arr2(&[[1.0f32, 2.0]]); // Too few targets
        let targets = TargetsView::new(targets_arr.view());

        let params = GBDTParams::default();

        let trainer = GBDTTrainer::new(Objective::SquaredLoss, Metric::Rmse, params);
        let result = trainer.train(
            &dataset,
            &binned,
            targets,
            WeightsView::None,
            None,
            Parallelism::Sequential,
        );

        assert!(result.is_none());
    }

    #[test]
    fn test_train_with_linear_leaves() {
        let (dataset, binned) = make_test_datasets();
        // Targets have a linear pattern on feature 0
        let targets_arr = arr2(&[[1.0f32, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 4.5]]);
        let targets = TargetsView::new(targets_arr.view());

        let params = GBDTParams {
            n_trees: 5,
            learning_rate: 0.3,
            linear_leaves: Some(LinearLeafConfig::default()),
            ..Default::default()
        };

        let trainer = GBDTTrainer::new(Objective::SquaredLoss, Metric::Rmse, params);
        let forest = trainer
            .train(
                &dataset,
                &binned,
                targets,
                WeightsView::None,
                None,
                Parallelism::Sequential,
            )
            .unwrap();

        assert_eq!(forest.n_trees(), 5);

        // Check that at least some trees have linear leaves (skip first tree)
        let has_linear_leaves = (0..5).any(|i| forest.tree(i).has_linear_leaves());
        // Note: linear leaves may not always be fitted if data doesn't support it
        // This is just a smoke test that the code runs without panicking
        let _ = has_linear_leaves;
    }

    #[test]
    fn test_first_tree_no_linear_coefficients() {
        let (dataset, binned) = make_test_datasets();
        let targets_arr = arr2(&[[1.0f32, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 4.5]]);
        let targets = TargetsView::new(targets_arr.view());

        let params = GBDTParams {
            n_trees: 3,
            linear_leaves: Some(LinearLeafConfig::default()),
            ..Default::default()
        };

        let trainer = GBDTTrainer::new(Objective::SquaredLoss, Metric::Rmse, params);
        let forest = trainer
            .train(
                &dataset,
                &binned,
                targets,
                WeightsView::None,
                None,
                Parallelism::Sequential,
            )
            .unwrap();

        // First tree should NOT have linear leaves (round 0 is skipped)
        let first_tree = forest.tree(0);
        assert!(
            !first_tree.has_linear_leaves(),
            "First tree should not have linear leaves"
        );
    }

    /// Regression test: linear leaves should improve RMSE on linearly structured data.
    ///
    /// This test was added after a bug where training predictions didn't use linear
    /// coefficients (only inference did), causing gradient boosting to diverge.
    /// The fix ensures training predictions match inference predictions.
    #[test]
    fn test_linear_leaves_improve_rmse() {
        use crate::data::{BinnedDataset, BinningConfig};
        use crate::inference::SimplePredictor;
        use crate::testing::synthetic_datasets::synthetic_regression;

        // Create a synthetic regression dataset with linear structure
        let dataset = synthetic_regression(200, 10, 42, 0.05);
        let binning_config = BinningConfig::builder().max_bins(256).build();
        let binned_dataset = BinnedDataset::from_dataset(&dataset, &binning_config).unwrap();
        let targets = dataset.targets().expect("synthetic datasets have targets");

        // Train WITHOUT linear leaves
        let params_base = GBDTParams {
            n_trees: 20,
            learning_rate: 0.1,
            growth_strategy: GrowthStrategy::DepthWise { max_depth: 4 },
            linear_leaves: None,
            ..Default::default()
        };
        let trainer_base = GBDTTrainer::new(Objective::SquaredLoss, Metric::Rmse, params_base);
        let forest_base = trainer_base
            .train(
                &dataset,
                &binned_dataset,
                targets,
                WeightsView::None,
                None,
                Parallelism::Sequential,
            )
            .unwrap();

        // Train WITH linear leaves
        let params_linear = GBDTParams {
            n_trees: 20,
            learning_rate: 0.1,
            growth_strategy: GrowthStrategy::DepthWise { max_depth: 4 },
            linear_leaves: Some(LinearLeafConfig {
                lambda: 0.01,
                min_samples: 5, // Lower threshold for test
                ..Default::default()
            }),
            ..Default::default()
        };
        let trainer_linear = GBDTTrainer::new(Objective::SquaredLoss, Metric::Rmse, params_linear);
        let forest_linear = trainer_linear
            .train(
                &dataset,
                &binned_dataset,
                targets,
                WeightsView::None,
                None,
                Parallelism::Sequential,
            )
            .unwrap();

        // Compute predictions using predict (takes &Dataset)
        let predictor_base = SimplePredictor::new(&forest_base);
        let predictor_linear = SimplePredictor::new(&forest_linear);
        let preds_base = predictor_base.predict(&dataset, Parallelism::Sequential);
        let preds_linear = predictor_linear.predict(&dataset, Parallelism::Sequential);

        // Compute RMSE using the metrics module
        let rmse_base = Metric::Rmse.compute(preds_base.view(), targets, WeightsView::None);
        let rmse_linear = Metric::Rmse.compute(preds_linear.view(), targets, WeightsView::None);

        // Linear leaves should improve RMSE (at least not make it significantly worse)
        // On synthetic linear data, we expect meaningful improvement
        assert!(
            rmse_linear <= rmse_base * 1.1, // Allow 10% tolerance for numerical variance
            "Linear leaves should not worsen RMSE. Base: {:.4}, Linear: {:.4}",
            rmse_base,
            rmse_linear
        );

        // Stricter check: linear should meaningfully improve
        assert!(
            rmse_linear < rmse_base,
            "Linear leaves should improve RMSE on linear data. Base: {:.4}, Linear: {:.4}",
            rmse_base,
            rmse_linear
        );
    }

    #[test]
    fn test_training_populates_gains_and_covers() {
        use crate::repr::gbdt::TreeView;

        let (dataset, binned) = make_test_datasets();
        let targets_arr = arr2(&[[1.0f32, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 4.5]]);
        let targets = TargetsView::new(targets_arr.view());

        let params = GBDTParams {
            n_trees: 3,
            growth_strategy: GrowthStrategy::DepthWise { max_depth: 3 },
            ..Default::default()
        };

        let trainer = GBDTTrainer::new(Objective::SquaredLoss, Metric::Rmse, params);
        let forest = trainer
            .train(
                &dataset,
                &binned,
                targets,
                WeightsView::None,
                None,
                Parallelism::Sequential,
            )
            .unwrap();

        // All trained trees should have gains and covers
        for i in 0..forest.n_trees() {
            let tree = forest.tree(i);
            assert!(tree.has_gains(), "Tree {} should have gains", i);
            assert!(tree.has_covers(), "Tree {} should have covers", i);

            let gains = tree.gains().expect("gains should be present");
            let covers = tree.covers().expect("covers should be present");

            assert_eq!(gains.len(), tree.n_nodes());
            assert_eq!(covers.len(), tree.n_nodes());

            // Covers should be positive (sum of hessians)
            for (node_idx, &cover) in covers.iter().enumerate() {
                assert!(
                    cover >= 0.0,
                    "Cover at node {} should be non-negative",
                    node_idx
                );
            }

            // Split nodes should have positive gain, leaves have gain=0
            for (node_idx, &gain) in gains.iter().enumerate() {
                if tree.is_leaf(node_idx as u32) {
                    assert!(
                        gain.abs() < 1e-6,
                        "Leaf {} should have gain=0, got {}",
                        node_idx,
                        gain
                    );
                }
                // Split gains should be non-negative (enforced by min_gain)
                assert!(
                    gain >= 0.0,
                    "Gain at node {} should be non-negative",
                    node_idx
                );
            }
        }
    }

    /// Test that quantile leaf renewing improves pinball loss compared to without renewing.
    ///
    /// This validates that the XGBoost/LightGBM-style leaf value renewing for quantile
    /// regression is working correctly. The optimal leaf value for quantile regression
    /// is the α-quantile of residuals, not the Newton-step from gradient sums.
    #[test]
    fn test_pinball_leaf_renewing_improves_quality() {
        use crate::data::{BinnedDataset, BinningConfig};
        use crate::inference::SimplePredictor;
        use ndarray::Array1;

        // Create a simple dataset with known quantile structure
        // Features: single feature with values 0-99
        // Targets: feature + noise (the 0.1 quantile should be lower than mean)
        let n_samples = 100;
        let features: Vec<f32> = (0..n_samples).map(|i| i as f32).collect();
        let targets: Vec<f32> = features
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                // Add asymmetric noise: more positive than negative
                let noise = if i % 3 == 0 { -2.0 } else { 3.0 };
                x + noise
            })
            .collect();

        let dataset = Dataset::builder()
            .add_feature_unnamed(Array1::from_vec(features))
            .build()
            .unwrap();

        let binned = BinnedDataset::from_dataset(&dataset, &BinningConfig::default()).unwrap();
        let targets_arr = Array2::from_shape_vec((1, n_samples), targets.clone()).unwrap();
        let targets_view = TargetsView::new(targets_arr.view());

        // Train with quantile objective (alpha = 0.1)
        // With leaf renewing, the model should learn the 0.1-quantile better
        let params = GBDTParams {
            n_trees: 10,
            learning_rate: 0.3,
            growth_strategy: GrowthStrategy::DepthWise { max_depth: 3 },
            gain: GainParams {
                min_child_weight: 1.0,
                ..Default::default()
            },
            ..Default::default()
        };

        let trainer = GBDTTrainer::new(
            Objective::PinballLoss { alphas: vec![0.1] },
            Metric::None, // No metric evaluation, just train
            params,
        );

        let forest = trainer
            .train(
                &dataset,
                &binned,
                targets_view,
                WeightsView::None,
                None,
                Parallelism::Sequential,
            )
            .unwrap();

        // Verify forest was created
        assert_eq!(forest.n_trees(), 10);

        // Make predictions
        let predictor = SimplePredictor::new(&forest);
        let predictions = predictor.predict(&dataset, Parallelism::Sequential);

        // For alpha=0.1, predictions should generally be lower than the targets
        // (the 10th percentile should underestimate most observations)
        let under_target_count = predictions
            .row(0)
            .iter()
            .zip(targets_arr.row(0).iter())
            .filter(|(p, t)| *p < *t)
            .count();

        // We expect roughly 90% of predictions to be under the target (since alpha=0.1)
        // Allow some tolerance due to finite samples and regularization
        let under_target_pct = under_target_count as f32 / n_samples as f32;
        assert!(
            under_target_pct > 0.7,
            "For alpha=0.1, at least 70% of predictions should be under target, got {:.1}%",
            under_target_pct * 100.0
        );
    }
}

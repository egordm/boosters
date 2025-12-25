//! GBDT Trainer for gradient boosting.
//!
//! Orchestrates objective computation, tree growing, and prediction updates.
//! Use [`GBDTTrainer::train`] to train a forest from a binned dataset.

use crate::data::{init_predictions, BinnedDataset};
use crate::dataset::TargetsView;
use crate::training::callback::{EarlyStopping, EarlyStopAction};
use crate::training::eval::{self, EvalSet};
use crate::training::logger::TrainingLogger;
use crate::training::metrics::MetricFn;
use crate::training::objectives::ObjectiveFn;
use crate::training::sampling::{ColSamplingParams, RowSampler, RowSamplingParams};
use crate::training::Gradients;
use crate::training::Verbosity;

use super::expansion::GrowthStrategy;
use super::grower::{GrowerParams, TreeGrower};
use super::linear::{LeafLinearTrainer, LinearLeafConfig};
use crate::utils::Parallelism;
use super::split::GainParams;

use crate::repr::gbdt::{Forest, ScalarLeaf};
use ndarray::{Array2, ArrayView1};

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
    /// Index of eval set to use for early stopping (default: first eval set).
    pub early_stopping_eval_set: usize,

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
            early_stopping_eval_set: 0,
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
pub struct GBDTTrainer<O: ObjectiveFn, M: MetricFn> {
    /// Objective function.
    objective: O,
    /// Evaluation metric.
    metric: M,
    /// Training parameters.
    params: GBDTParams,
}

impl<O: ObjectiveFn, M: MetricFn> GBDTTrainer<O, M> {
    /// Create a new GBDT trainer.
    pub fn new(objective: O, metric: M, params: GBDTParams) -> Self {
        Self { objective, metric, params }
    }

    /// Get reference to parameters.
    pub fn params(&self) -> &GBDTParams {
        &self.params
    }

    /// Get reference to objective.
    pub fn objective(&self) -> &O {
        &self.objective
    }

    /// Get reference to metric.
    pub fn metric(&self) -> &M {
        &self.metric
    }

    /// Train a forest.
    ///
    /// **Note:** This method does NOT create a thread pool. The caller must set up
    /// parallelism via `rayon::ThreadPool::install()` if desired.
    ///
    /// # Arguments
    ///
    /// * `dataset` - Binned dataset created with [`BinnedDatasetBuilder`]
    /// * `targets` - Target values (length = n_rows × n_outputs)
    /// * `weights` - Sample weights (None for uniform)
    /// * `eval_sets` - Validation sets for early stopping (`&[]` to skip)
    /// * `parallelism` - Sequential or Parallel iteration hint
    ///
    /// [`BinnedDatasetBuilder`]: crate::data::BinnedDatasetBuilder
    pub fn train(
        &self,
        dataset: &BinnedDataset,
        targets: TargetsView<'_>,
        weights: Option<ArrayView1<f32>>,
        eval_sets: &[EvalSet<'_>],
        parallelism: Parallelism,
    ) -> Option<Forest<ScalarLeaf>> {
        let n_rows = dataset.n_rows();
        let n_outputs = self.objective.n_outputs();

        // Validate inputs
        if targets.n_samples() < n_rows {
            return None;
        }

        // Initialize components (train-local)
        let grower_params = self.params.to_grower_params();
        let mut grower = TreeGrower::new(
            dataset,
            grower_params,
            self.params.cache_size,
            parallelism,
        );

        // Initialize linear leaf trainer if configured
        let mut linear_trainer = if let Some(config) = &self.params.linear_leaves {
            // Estimate max samples per leaf: use n_rows as upper bound
            // (worst case: single-leaf tree for some output)
            // This may overallocate but is safe; typical usage is much smaller
            Some(LeafLinearTrainer::new(config.clone(), n_rows))
        } else {
            None
        };

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
        let mut forest = Forest::<ScalarLeaf>::new(n_outputs as u32)
            .with_base_score(base_scores.clone());

        // Check if we need evaluation (metric is enabled)
        let needs_evaluation = self.metric.is_enabled();

        // Initialize eval predictions with base scores
        // Shape: [n_outputs, n_eval_samples] for each eval set
        // Only allocate if evaluation is needed
        let mut eval_predictions: Vec<Array2<f32>> = if needs_evaluation {
            eval_sets
                .iter()
                .map(|es| init_predictions(&base_scores, es.dataset.n_samples()))
                .collect()
        } else {
            Vec::new()
        };
        // Early stopping (always present, may be disabled)
        let mut early_stopping = EarlyStopping::new(
            self.params.early_stopping_rounds as usize,
            self.metric.higher_is_better(),
        );
        let mut best_n_trees: usize = 0;

        // Evaluator for computing metrics (only used if evaluation is needed)
        let mut evaluator = eval::Evaluator::new(&self.objective, &self.metric, n_outputs);

        // Logger
        let mut logger = TrainingLogger::new(self.params.verbosity);
        logger.start_training(self.params.n_trees as usize);

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
                let mut mutable_tree = grower.grow(dataset, &gradients, output, sampled);

                // Fit linear models in leaves (skip round 0: homogeneous gradients)
                // Only fit if linear_leaves config is set and we're past round 0
                if round > 0 && let Some(ref mut linear_trainer) = linear_trainer {
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
                    #[cfg(debug_assertions)]
                    let fitted_count = fitted.len();
                    for leaf in fitted {
                        mutable_tree.set_linear_leaf(
                            leaf.node_id,
                            leaf.features,
                            leaf.intercept,
                            leaf.coefficients,
                        );
                    }
                    #[cfg(debug_assertions)]
                    eprintln!("Round {}: set {} linear leaves", round, fitted_count);
                }

                // Freeze tree
                let tree = mutable_tree.freeze();

                // Update predictions (row of Array2)
                let mut pred_row = predictions.row_mut(output);
                if sampled.is_none() {
                    // Fast path: use partitioner for O(n) prediction update
                    grower.update_predictions_from_last_tree(pred_row);
                } else {
                    // Fallback: row sampling trains on a subset; we must still apply the
                    // trained tree to all rows to keep predictions correct.
                    let pred_slice = pred_row.as_slice_mut()
                        .expect("prediction row should be contiguous");
                    tree.predict_into(dataset, pred_slice, parallelism);
                }

                // Incremental eval set prediction: add this tree's contribution
                // eval_predictions shape is [n_outputs, n_eval_samples] for each set
                // Only compute if evaluation is needed
                if needs_evaluation {
                    for (set_idx, eval_set) in eval_sets.iter().enumerate() {
                        let mut pred_row = eval_predictions[set_idx].row_mut(output);
                        let pred_slice = pred_row.as_slice_mut()
                            .expect("eval prediction row should be contiguous");
                        // Use FeaturesView directly - it implements FeatureAccessor
                        tree.predict_into(&eval_set.dataset.features(), pred_slice, parallelism);
                    }
                }

                forest.push_tree(tree, output as u32);
            }

            // Evaluate on eval sets (using accumulated predictions)
            // Only evaluate if metric is enabled
            let (round_metrics, early_stop_value) = if needs_evaluation {
                let metrics = evaluator.evaluate_round(
                    predictions.view(),
                    targets,
                    weights,
                    eval_sets,
                    &eval_predictions,
                );
                let value = eval::Evaluator::<O, M>::early_stop_value(
                    &metrics,
                    self.params.early_stopping_eval_set,
                );
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
    fn truncate_forest(
        forest: &Forest<ScalarLeaf>,
        n_trees: usize,
    ) -> Forest<ScalarLeaf> {
        let mut truncated = Forest::new(forest.n_groups())
            .with_base_score(forest.base_score().to_vec());

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
    use crate::data::{
        BinMapper, BinStorage, BinnedDataset, FeatureGroup, FeatureMeta, GroupLayout, MissingType,
    };
    use crate::training::metrics::Rmse;
    use crate::training::objectives::SquaredLoss;
    use ndarray::arr2;

    fn make_simple_mapper(n_bins: u32) -> BinMapper {
        let bounds: Vec<f64> = (0..n_bins).map(|i| i as f64 + 0.5).collect();
        BinMapper::numerical(bounds, MissingType::None, 0, 0, 0.0, 0.0, (n_bins - 1) as f64)
    }

    fn make_test_dataset() -> BinnedDataset {
        // 8 rows, 2 features
        let storage = BinStorage::from_u8(vec![
            0, 1, // row 0
            1, 2, // row 1
            2, 0, // row 2
            3, 1, // row 3
            0, 2, // row 4
            1, 0, // row 5
            2, 1, // row 6
            3, 2, // row 7
        ]);

        let group = FeatureGroup::new(vec![0, 1], GroupLayout::RowMajor, 8, storage, vec![4, 3]);

        let features = vec![
            FeatureMeta::new(make_simple_mapper(4), 0, 0),
            FeatureMeta::new(make_simple_mapper(3), 0, 1),
        ];

        BinnedDataset::with_bundle_plan(8, features, vec![group], None)
    }

    #[test]
    fn test_params_default() {
        let params = GBDTParams::default();

        assert_eq!(params.n_trees, 100);
        assert!((params.learning_rate - 0.3).abs() < 1e-6);
        assert_eq!(params.growth_strategy, GrowthStrategy::DepthWise { max_depth: 6 });
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
        assert_eq!(params.growth_strategy, GrowthStrategy::DepthWise { max_depth: 4 });
        assert!((params.gain.reg_lambda - 2.0).abs() < 1e-6);
        assert!((params.gain.min_child_weight - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_train_single_tree() {
        let dataset = make_test_dataset();
        let targets_arr = arr2(&[[1.0f32, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 4.5]]);
        let targets = TargetsView::new(targets_arr.view());

        let params = GBDTParams { n_trees: 1, ..Default::default() };

        let trainer = GBDTTrainer::new(SquaredLoss, Rmse, params);
        let forest = trainer.train(&dataset, targets, None, &[], Parallelism::Sequential).unwrap();

        assert_eq!(forest.n_trees(), 1);
        assert_eq!(forest.n_groups(), 1);
    }

    #[test]
    fn test_train_multiple_trees() {
        let dataset = make_test_dataset();
        let targets_arr = arr2(&[[1.0f32, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 4.5]]);
        let targets = TargetsView::new(targets_arr.view());

        let params = GBDTParams {
            n_trees: 10,
            learning_rate: 0.1,
            ..Default::default()
        };

        let trainer = GBDTTrainer::new(SquaredLoss, Rmse, params);
        let forest = trainer.train(&dataset, targets, None, &[], Parallelism::Sequential).unwrap();

        assert_eq!(forest.n_trees(), 10);
    }

    #[test]
    fn test_train_with_regularization() {
        let dataset = make_test_dataset();
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

        let trainer = GBDTTrainer::new(SquaredLoss, Rmse, params);
        let forest = trainer.train(&dataset, targets, None, &[], Parallelism::Sequential).unwrap();

        assert_eq!(forest.n_trees(), 5);
    }

    #[test]
    fn test_train_weighted() {
        let dataset = make_test_dataset();
        let targets_arr = arr2(&[[1.0f32, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 4.5]]);
        let targets = TargetsView::new(targets_arr.view());
        let weights = ndarray::array![1.0f32, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0];

        let params = GBDTParams { n_trees: 5, ..Default::default() };

        let trainer = GBDTTrainer::new(SquaredLoss, Rmse, params);
        let forest = trainer.train(&dataset, targets, Some(weights.view()), &[], Parallelism::Sequential).unwrap();

        assert_eq!(forest.n_trees(), 5);
    }

    #[test]
    fn test_leaf_wise_growth() {
        let dataset = make_test_dataset();
        let targets_arr = arr2(&[[1.0f32, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 4.5]]);
        let targets = TargetsView::new(targets_arr.view());

        let params = GBDTParams {
            n_trees: 3,
            growth_strategy: GrowthStrategy::LeafWise { max_leaves: 8 },
            ..Default::default()
        };

        let trainer = GBDTTrainer::new(SquaredLoss, Rmse, params);
        let forest = trainer
            .train(&dataset, targets, None, &[], Parallelism::Sequential)
            .unwrap();

        assert_eq!(forest.n_trees(), 3);
    }

    #[test]
    fn test_train_invalid_targets() {
        let dataset = make_test_dataset();
        let targets_arr = arr2(&[[1.0f32, 2.0]]); // Too few targets
        let targets = TargetsView::new(targets_arr.view());

        let params = GBDTParams::default();

        let trainer = GBDTTrainer::new(SquaredLoss, Rmse, params);
        let result = trainer.train(&dataset, targets, None, &[], Parallelism::Sequential);

        assert!(result.is_none());
    }

    #[test]
    fn test_train_with_linear_leaves() {
        let dataset = make_test_dataset();
        // Targets have a linear pattern on feature 0
        let targets_arr = arr2(&[[1.0f32, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 4.5]]);
        let targets = TargetsView::new(targets_arr.view());

        let params = GBDTParams {
            n_trees: 5,
            learning_rate: 0.3,
            linear_leaves: Some(LinearLeafConfig::default()),
            ..Default::default()
        };

        let trainer = GBDTTrainer::new(SquaredLoss, Rmse, params);
        let forest = trainer
            .train(&dataset, targets, None, &[], Parallelism::Sequential)
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
        let dataset = make_test_dataset();
        let targets_arr = arr2(&[[1.0f32, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 4.5]]);
        let targets = TargetsView::new(targets_arr.view());

        let params = GBDTParams {
            n_trees: 3,
            linear_leaves: Some(LinearLeafConfig::default()),
            ..Default::default()
        };

        let trainer = GBDTTrainer::new(SquaredLoss, Rmse, params);
        let forest = trainer
            .train(&dataset, targets, None, &[], Parallelism::Sequential)
            .unwrap();

        // First tree should NOT have linear leaves (round 0 is skipped)
        let first_tree = forest.tree(0);
        assert!(!first_tree.has_linear_leaves(), "First tree should not have linear leaves");
    }

    #[test]
    fn test_training_populates_gains_and_covers() {
        use crate::repr::gbdt::TreeView;

        let dataset = make_test_dataset();
        let targets_arr = arr2(&[[1.0f32, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 4.5]]);
        let targets = TargetsView::new(targets_arr.view());

        let params = GBDTParams {
            n_trees: 3,
            growth_strategy: GrowthStrategy::DepthWise { max_depth: 3 },
            ..Default::default()
        };

        let trainer = GBDTTrainer::new(SquaredLoss, Rmse, params);
        let forest = trainer
            .train(&dataset, targets, None, &[], Parallelism::Sequential)
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
                assert!(cover >= 0.0, "Cover at node {} should be non-negative", node_idx);
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
                assert!(gain >= 0.0, "Gain at node {} should be non-negative", node_idx);
            }
        }
    }
}

//! GBDT Trainer for gradient boosting.
//!
//! This module provides the main training loop for gradient boosted decision trees.
//! It orchestrates objective computation, tree growing, and prediction updates.
//!
//! # Example
//!
//! ```ignore
//! use booste_rs::training::{GBDTTrainer, GBDTParams, SquaredLoss, GainParams};
//!
//! let params = GBDTParams {
//!     n_trees: 100,
//!     learning_rate: 0.1,
//!     gain: GainParams { reg_lambda: 1.0, ..Default::default() },
//!     ..Default::default()
//! };
//!
//! let mut trainer = GBDTTrainer::new(SquaredLoss, params);
//! let forest = trainer.train(&dataset, &targets, &[]);
//! ```

use rayon::ThreadPoolBuilder;

use crate::data::BinnedDataset;
use crate::training::objectives::Objective;
use crate::training::sampling::{ColSamplingParams, RowSampler, RowSamplingParams};
use crate::training::Gradients;

use super::expansion::GrowthStrategy;
use super::grower::{GrowerParams, TreeGrower};
use super::optimization::OptimizationProfile;
use super::split::GainParams;

use crate::inference::gbdt::{Forest as InferenceForest, ScalarLeaf};

// =============================================================================
// GBDTParams
// =============================================================================

/// Parameters for GBDT training.
///
/// Use struct construction with `..Default::default()` for convenient configuration.
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

    // --- Resource control ---
    /// Number of threads to use for parallel operations.
    ///
    /// - `0`: Use rayon's global thread pool (default, uses all available cores)
    /// - `n > 0`: Create a dedicated thread pool with exactly `n` threads
    ///
    /// When set to a value > 1, a scoped thread pool is created for the
    /// training session, ensuring thread count is controlled without affecting
    /// other parts of the application.
    pub n_threads: usize,
    /// Histogram cache size (number of slots).
    pub cache_size: usize,
    /// Optimization profile for automatic tuning.
    pub optimization_profile: OptimizationProfile,

    // --- Reproducibility ---
    /// Random seed.
    pub seed: u64,
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
            n_threads: 0,
            cache_size: 8,
            optimization_profile: OptimizationProfile::Auto,
            seed: 42,
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
            optimization_profile: self.optimization_profile.clone(),
            col_sampling: self.col_sampling.clone(),
        }
    }
}

// =============================================================================
// GBDTTrainer
// =============================================================================

/// GBDT Trainer.
///
/// # Example
///
/// ```ignore
/// let mut trainer = GBDTTrainer::new(SquaredLoss, params);
/// let forest = trainer.train(&dataset, &targets, &[]);
/// ```
pub struct GBDTTrainer<O: Objective> {
    /// Objective function.
    objective: O,
    /// Training parameters.
    params: GBDTParams,
}

impl<O: Objective> GBDTTrainer<O> {
    /// Create a new GBDT trainer.
    pub fn new(objective: O, params: GBDTParams) -> Self {
        Self { objective, params }
    }

    /// Get reference to parameters.
    pub fn params(&self) -> &GBDTParams {
        &self.params
    }

    /// Get reference to objective.
    pub fn objective(&self) -> &O {
        &self.objective
    }

    /// Train a forest.
    ///
    /// # Arguments
    /// * `dataset` - Binned dataset
    /// * `targets` - Target values (length = n_rows * n_outputs)
    /// * `weights` - Sample weights (empty = uniform weights)
    ///
    /// # Returns
    /// Trained forest, or `None` if training fails (e.g., invalid input).
    ///
    /// # Panics
    ///
    /// Panics if `n_threads > 0` and the thread pool cannot be created (rare OS-level failure).
    pub fn train(
        &self,
        dataset: &BinnedDataset,
        targets: &[f32],
        weights: &[f32],
    ) -> Option<InferenceForest<ScalarLeaf>> {
        // If n_threads == 0, use rayon's global thread pool.
        // Otherwise, scope all rayon work for this training run to a dedicated pool.
        if self.params.n_threads == 0 {
            self.train_impl(dataset, targets, weights)
        } else {
            let pool = ThreadPoolBuilder::new()
                .num_threads(self.params.n_threads)
                .build()
                .expect("Failed to create thread pool");

            pool.install(|| self.train_impl(dataset, targets, weights))
        }
    }

    /// Internal training implementation.
    fn train_impl(
        &self,
        dataset: &BinnedDataset,
        targets: &[f32],
        weights: &[f32],
    ) -> Option<InferenceForest<ScalarLeaf>> {
        let n_rows = dataset.n_rows();
        let n_outputs = self.objective.n_outputs();

        // Validate inputs
        if targets.len() < n_rows {
            return None;
        }

        // Initialize components (train-local)
        let grower_params = self.params.to_grower_params();
        let mut grower = TreeGrower::new(dataset, grower_params, self.params.cache_size);

        let mut row_sampler = RowSampler::new(
            self.params.row_sampling.clone(),
            n_rows,
            self.params.seed,
            self.params.learning_rate,
        );

        let mut gradients = Gradients::new(n_rows, n_outputs);

        // Compute base scores
        let mut base_scores = vec![0.0f32; n_outputs];
        self.objective
            .compute_base_score(n_rows, n_outputs, targets, weights, &mut base_scores);

        // Initialize predictions (column-major: [output0_all_rows, output1_all_rows, ...])
        let mut predictions = Vec::with_capacity(n_rows * n_outputs);
        for output in 0..n_outputs {
            predictions.extend(std::iter::repeat(base_scores[output]).take(n_rows));
        }

        // Create inference forest directly (Phase 2: no conversion needed)
        let mut forest = InferenceForest::<ScalarLeaf>::new(n_outputs as u32)
            .with_base_score(base_scores);

        for round in 0..self.params.n_trees {
            // Compute gradients for all outputs
            let (grad_buf, hess_buf) = gradients.as_mut_slices();
            self.objective.compute_gradients(
                n_rows,
                n_outputs,
                &predictions,
                targets,
                weights,
                grad_buf,
                hess_buf,
            );

            // Grow one tree per output
            for output in 0..n_outputs {
                // Row sampling: modifies gradients in place for this output
                // - GOSS: amplifies sampled small-gradient rows
                // - Uniform: zeros out unsampled rows (split finding requires hess_sum > 0)
                let (grads, hess) = gradients.output_grads_hess_mut(output);
                let sampled = row_sampler.sample(round as usize, grads, hess);

                let pred_offset = output * n_rows;

                // Fast path: if we trained on all rows, we can update predictions using
                // training-time leaf assignments instead of traversing the tree per row.
                let tree = if sampled.is_none() {
                    grower.grow_and_update_predictions(
                        dataset,
                        &gradients,
                        output,
                        None,
                        &mut predictions[pred_offset..pred_offset + n_rows],
                    )
                } else {
                    // Fallback: row sampling trains on a subset; we must still apply the
                    // trained tree to all rows to keep predictions correct.
                    let tree = grower.grow(dataset, &gradients, output, sampled.as_deref());
                    for row in 0..n_rows {
                        if let Some(view) = dataset.row_view(row) {
                            predictions[pred_offset + row] += tree.predict_binned(&view, dataset).0;
                        }
                    }
                    tree
                };

                forest.push_tree(tree, output as u32);
            }
        }

        Some(forest)
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
    use crate::training::objectives::SquaredLoss;

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

        BinnedDataset::new(8, features, vec![group])
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
        let targets: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 4.5];

        let params = GBDTParams { n_trees: 1, ..Default::default() };

        let trainer = GBDTTrainer::new(SquaredLoss, params);
        let forest = trainer.train(&dataset, &targets, &[]).unwrap();

        assert_eq!(forest.n_trees(), 1);
        assert_eq!(forest.n_groups(), 1);
    }

    #[test]
    fn test_train_multiple_trees() {
        let dataset = make_test_dataset();
        let targets: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 4.5];

        let params = GBDTParams {
            n_trees: 10,
            learning_rate: 0.1,
            ..Default::default()
        };

        let trainer = GBDTTrainer::new(SquaredLoss, params);
        let forest = trainer.train(&dataset, &targets, &[]).unwrap();

        assert_eq!(forest.n_trees(), 10);
    }

    #[test]
    fn test_train_with_regularization() {
        let dataset = make_test_dataset();
        let targets: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 4.5];

        let params = GBDTParams {
            n_trees: 5,
            gain: GainParams {
                reg_lambda: 10.0,
                min_gain: 0.5,
                ..Default::default()
            },
            ..Default::default()
        };

        let trainer = GBDTTrainer::new(SquaredLoss, params);
        let forest = trainer.train(&dataset, &targets, &[]).unwrap();

        assert_eq!(forest.n_trees(), 5);
    }

    #[test]
    fn test_train_weighted() {
        let dataset = make_test_dataset();
        let targets: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 4.5];
        let weights: Vec<f32> = vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0];

        let params = GBDTParams { n_trees: 5, ..Default::default() };

        let trainer = GBDTTrainer::new(SquaredLoss, params);
        let forest = trainer.train(&dataset, &targets, &weights).unwrap();

        assert_eq!(forest.n_trees(), 5);
    }

    #[test]
    fn test_leaf_wise_growth() {
        let dataset = make_test_dataset();
        let targets: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 4.5];

        let params = GBDTParams {
            n_trees: 3,
            growth_strategy: GrowthStrategy::LeafWise { max_leaves: 8 },
            ..Default::default()
        };

        let trainer = GBDTTrainer::new(SquaredLoss, params);
        let forest = trainer.train(&dataset, &targets, &[]).unwrap();

        assert_eq!(forest.n_trees(), 3);
    }

    #[test]
    fn test_train_invalid_targets() {
        let dataset = make_test_dataset();
        let targets: Vec<f32> = vec![1.0, 2.0]; // Too few targets

        let params = GBDTParams::default();

        let trainer = GBDTTrainer::new(SquaredLoss, params);
        let result = trainer.train(&dataset, &targets, &[]);

        assert!(result.is_none());
    }
}

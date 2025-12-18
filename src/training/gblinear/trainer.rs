//! Gradient boosting trainer for linear models.
//!
//! This module implements coordinate descent training for GBLinear models.
//! It supports both single-output (regression, binary classification) and
//! multi-output (multiclass) training through the `Objective` trait.
//!
//! # Example
//!
//! ```ignore
//! use booste_rs::training::{GBLinearTrainer, GBLinearParams, SquaredLoss};
//!
//! let params = GBLinearParams {
//!     n_rounds: 100,
//!     learning_rate: 0.5,
//!     lambda: 1.0,
//!     ..Default::default()
//! };
//!
//! let trainer = GBLinearTrainer::new(SquaredLoss, params);
//! let model = trainer.train(&data, &labels, None, &[]);
//! ```
//!
//! For multiclass training:
//!
//! ```ignore
//! use booste_rs::training::{GBLinearTrainer, GBLinearParams, SoftmaxLoss};
//!
//! let params = GBLinearParams {
//!     n_rounds: 100,
//!     ..Default::default()
//! };
//!
//! let trainer = GBLinearTrainer::new(SoftmaxLoss::new(3), params);
//! let model = trainer.train(&data, &labels, None, &[]);
//! ```

use rayon::ThreadPoolBuilder;

use crate::data::{ColMatrix, Dataset};
use crate::inference::gblinear::LinearModel;
use crate::training::eval;
use crate::training::{
    EarlyStopping, EarlyStopAction, EvalSet, Gradients, Metric, Objective, ObjectiveExt, TrainingLogger, Verbosity,
};

use super::selector::FeatureSelectorKind;
use super::updater::{Updater, UpdateConfig, UpdaterKind};

// ============================================================================
// GBLinearParams
// ============================================================================

/// Parameters for GBLinear training.
///
/// Use struct construction with `..Default::default()` for convenient configuration.
///
/// # Example
///
/// ```ignore
/// use booste_rs::training::GBLinearParams;
///
/// let params = GBLinearParams {
///     n_rounds: 200,
///     learning_rate: 0.3,
///     lambda: 2.0,
///     ..Default::default()
/// };
/// ```
#[derive(Clone, Debug)]
pub struct GBLinearParams {
    // --- Training parameters ---
    /// Number of boosting rounds.
    pub n_rounds: u32,

    /// Learning rate (eta). Controls step size for weight updates.
    pub learning_rate: f32,

    /// L1 regularization (alpha). Encourages sparse weights.
    pub alpha: f32,

    /// L2 regularization (lambda). Prevents large weights.
    pub lambda: f32,

    // --- Update strategy ---
    /// Use parallel (shotgun) coordinate descent updates.
    /// When true, features are updated in parallel within each round.
    pub parallel: bool,

    /// Feature selection strategy for coordinate descent.
    pub feature_selector: FeatureSelectorKind,

    /// Random seed for feature shuffling.
    pub seed: u64,

    // --- Evaluation and early stopping ---
    /// Early stopping rounds. Training stops if no improvement for this many rounds.
    /// Set to 0 to disable.
    pub early_stopping_rounds: u32,

    /// Index of eval set to use for early stopping (default: first eval set).
    pub early_stopping_eval_set: usize,

    // --- Logging ---
    /// Verbosity level for training output.
    pub verbosity: Verbosity,

    // --- Resource control ---
    /// Number of threads to use for rayon-based parallel operations.
    ///
    /// - `0`: Use rayon's global thread pool (default)
    /// - `n > 0`: Create a dedicated thread pool with exactly `n` threads
    pub n_threads: usize,
}

impl Default for GBLinearParams {
    fn default() -> Self {
        Self {
            n_rounds: 100,
            learning_rate: 0.5,
            alpha: 0.0,
            lambda: 1.0,
            parallel: true,
            feature_selector: FeatureSelectorKind::default(),
            seed: 42,
            early_stopping_rounds: 0,
            early_stopping_eval_set: 0,
            verbosity: Verbosity::default(),
            n_threads: 0,
        }
    }
}

// ============================================================================
// GBLinearTrainer
// ============================================================================

/// Gradient boosted linear model trainer.
///
/// Generic over the objective function `O`, which determines the loss
/// and gradient computation strategy.
///
/// # Example
///
/// ```ignore
/// use booste_rs::training::{GBLinearTrainer, GBLinearParams, SquaredLoss, LogisticLoss, Rmse, LogLoss};
///
/// // Regression
/// let params = GBLinearParams::default();
/// let trainer = GBLinearTrainer::new(SquaredLoss, Rmse, params);
/// let model = trainer.train(&data, &labels, None, &[]);
///
/// // Binary classification
/// let trainer = GBLinearTrainer::new(LogisticLoss, LogLoss, GBLinearParams::default());
/// let model = trainer.train(&data, &labels, None, &[]);
/// ```
#[derive(Clone, Debug)]
pub struct GBLinearTrainer<O: Objective, M: Metric> {
    objective: O,
    metric: M,
    params: GBLinearParams,
}

impl<O: Objective, M: Metric> GBLinearTrainer<O, M> {
    /// Create a new trainer with the given objective and parameters.
    pub fn new(objective: O, metric: M, params: GBLinearParams) -> Self {
        Self {
            objective,
            metric,
            params,
        }
    }

    /// Train a linear model on column-accessible data.
    ///
    /// # Arguments
    ///
    /// * `train_data` - Training features (column-major `ColMatrix`)
    /// * `train_labels` - Training labels
    /// * `weights` - Optional sample weights (None = uniform)
    /// * `eval_sets` - Evaluation sets for monitoring (pass `&[]` if not needed)
    ///
    /// # Example
    ///
    /// ```ignore
    /// use booste_rs::data::ColMatrix;
    /// use booste_rs::training::{GBLinearTrainer, GBLinearParams, SquaredLoss};
    ///
    /// // Regression (unweighted)
    /// let params = GBLinearParams::default();
    /// let trainer = GBLinearTrainer::new(SquaredLoss, params);
    /// let model = trainer.train(&data, &labels, None, &[]);
    ///
    /// // With sample weights
    /// let weights: Vec<f32> = labels.iter()
    ///     .map(|&label| if label > 0.5 { 10.0 } else { 1.0 })
    ///     .collect();
    /// let model = trainer.train(&data, &labels, Some(&weights), &[]);
    /// ```
    ///
    /// # Panics
    ///
    /// - Panics if `weights.len() != labels.len()` when weights are provided.
    /// - Panics if `n_threads > 0` and the thread pool cannot be created (rare OS-level failure).
    pub fn train(&self, train: &Dataset, eval_sets: &[EvalSet<'_>]) -> Option<LinearModel> {
        if self.params.n_threads == 0 {
            return self.train_impl(train, eval_sets);
        }

        let pool = ThreadPoolBuilder::new()
            .num_threads(self.params.n_threads)
            .build()
            .expect("Failed to create thread pool");

        pool.install(|| self.train_impl(train, eval_sets))
    }

    fn train_impl(&self, train: &Dataset, eval_sets: &[EvalSet<'_>]) -> Option<LinearModel> {
        let train_data = train.for_gblinear().ok()?;
        let train_labels = train.targets();
        let weights = train.weights();

        let num_features = train_data.num_columns();
        let num_samples = train_data.num_rows();
        let num_outputs = self.objective.num_outputs();

        assert!(
            num_outputs >= 1,
            "Objective must have at least 1 output, got {}",
            num_outputs
        );
        debug_assert_eq!(train_labels.len(), num_samples);
        debug_assert!(weights.map_or(true, |w| w.len() == num_samples));

        // Compute base scores from objective (optimal constant prediction)
        let w = weights.unwrap_or(&[]);
        let base_scores = self.objective.base_scores(num_samples, train_labels, w);

        // Initialize model with base scores as biases
        let mut model = LinearModel::zeros(num_features, num_outputs);
        for (group, &base_score) in base_scores.iter().enumerate() {
            model.set_bias(group, base_score);
        }

        // Create updater and selector
        let updater_kind = if self.params.parallel {
            UpdaterKind::Parallel
        } else {
            UpdaterKind::Sequential
        };
        let mut selector = self.params.feature_selector.create_state(self.params.seed);

        let update_config = UpdateConfig {
            alpha: self.params.alpha,
            lambda: self.params.lambda,
            learning_rate: self.params.learning_rate,
        };
        let updater = Updater::new(updater_kind, update_config);

        // Gradient and prediction buffers
        let mut gradients = Gradients::new(num_samples, num_outputs);
        // Initialize predictions with base scores (column-major)
        let mut predictions = vec![0.0f32; num_samples * num_outputs];
        for (group, &base_score) in base_scores.iter().enumerate() {
            let start = group * num_samples;
            predictions[start..start + num_samples].fill(base_score);
        }
        // Initialize eval predictions with base scores
        let eval_data: Vec<ColMatrix<f32>> = eval_sets
            .iter()
            .map(|es| es.dataset.for_gblinear().ok())
            .collect::<Option<Vec<_>>>()?;

        let mut eval_predictions: Vec<Vec<f32>> = eval_data
            .iter()
            .map(|m| {
                let eval_rows = m.num_rows();
                let mut preds = vec![0.0f32; eval_rows * num_outputs];
                for (group, &base_score) in base_scores.iter().enumerate() {
                    let start = group * eval_rows;
                    preds[start..start + eval_rows].fill(base_score);
                }
                preds
            })
            .collect();

        // Early stopping (always present, may be disabled)
        let mut early_stopping = EarlyStopping::new(
            self.params.early_stopping_rounds as usize,
            self.metric.higher_is_better(),
        );
        let mut best_model: Option<LinearModel> = None;

        // Logger
        let mut logger = TrainingLogger::new(self.params.verbosity);
        logger.start_training(self.params.n_rounds as usize);

        // Evaluator for computing metrics
        let mut evaluator = eval::Evaluator::new(&self.objective, &self.metric, num_outputs);

        // Training loop
        for round in 0..self.params.n_rounds {
            // NOTE: We don't call predict_col_major() here anymore!
            // Predictions are maintained incrementally by applying weight deltas.
            // On first round (round == 0), predictions are already initialized with base scores.

            // Compute gradients from current predictions
            self.objective.compute_gradients_buffer(&predictions, train_labels, w, &mut gradients);

            // Update each output
            for output in 0..num_outputs {
                let bias_delta = updater.update_bias(&mut model, &gradients, output);
                
                // Apply bias delta to predictions incrementally
                if bias_delta.abs() > 1e-10 {
                    updater.apply_bias_delta_to_predictions(bias_delta, output, num_samples, &mut predictions);
                    
                    // Also update eval predictions
                    for (set_idx, matrix) in eval_data.iter().enumerate() {
                        let eval_rows = matrix.num_rows();
                        updater.apply_bias_delta_to_predictions(
                            bias_delta,
                            output,
                            eval_rows,
                            &mut eval_predictions[set_idx],
                        );
                    }
                    
                    // Recompute gradients after bias update
                    self.objective.compute_gradients_buffer(&predictions, train_labels, w, &mut gradients);
                }

                selector.setup_round(
                    &model,
                    &train_data,
                    &gradients,
                    output,
                    self.params.alpha,
                    self.params.lambda,
                );

                let weight_deltas = updater.update_round(
                    &mut model,
                    &train_data,
                    &gradients,
                    &mut selector,
                    output,
                );
                
                // Apply weight deltas to predictions incrementally
                if !weight_deltas.is_empty() {
                    updater.apply_weight_deltas_to_predictions(
                        &train_data,
                        &weight_deltas,
                        output,
                        num_samples,
                        &mut predictions,
                    );
                    
                    // Also update eval predictions
                    for (set_idx, matrix) in eval_data.iter().enumerate() {
                        let eval_rows = matrix.num_rows();
                        updater.apply_weight_deltas_to_predictions(
                            matrix,
                            &weight_deltas,
                            output,
                            eval_rows,
                            &mut eval_predictions[set_idx],
                        );
                    }
                }
            }

            // Evaluation using Evaluator
            let round_metrics = evaluator.evaluate_round(
                &predictions,
                train_labels,
                weights.unwrap_or(&[]),
                num_samples,
                eval_sets,
                &eval_predictions,
            );
            let early_stop_value = eval::Evaluator::<O, M>::early_stop_value(
                &round_metrics,
                self.params.early_stopping_eval_set,
            );

            if self.params.verbosity >= Verbosity::Info {
                logger.log_metrics(round as usize, &round_metrics);
            }

            // Early stopping check (value always present: either eval or train metric)
            if early_stopping.is_enabled() {
                match early_stopping.update(early_stop_value) {
                    EarlyStopAction::Improved => {
                        best_model = Some(model.clone());
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
        if early_stopping.is_enabled() && best_model.is_some() {
            best_model
        } else {
            Some(model)
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
    use crate::training::{LogLoss, MulticlassLogLoss, Rmse, SquaredLoss, LogisticLoss, SoftmaxLoss};

    #[test]
    fn test_params_default() {
        let params = GBLinearParams::default();
        assert_eq!(params.n_rounds, 100);
        assert_eq!(params.learning_rate, 0.5);
        assert_eq!(params.lambda, 1.0);
    }

    #[test]
    fn test_params_custom() {
        let params = GBLinearParams {
            n_rounds: 50,
            learning_rate: 0.3,
            lambda: 2.0,
            alpha: 0.1,
            ..Default::default()
        };

        assert_eq!(params.n_rounds, 50);
        assert_eq!(params.learning_rate, 0.3);
        assert_eq!(params.lambda, 2.0);
        assert_eq!(params.alpha, 0.1);
    }

    #[test]
    fn train_simple_regression() {
        // y = 2*x + 1
        let row_data = RowMatrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 4, 1);
        let train_data: ColMatrix = (&row_data).into();
        let train_labels = vec![3.0, 5.0, 7.0, 9.0];
        let train = Dataset::from_numeric(&train_data, train_labels).unwrap();

        let params = GBLinearParams {
            n_rounds: 100,
            learning_rate: 0.5,
            alpha: 0.0,
            lambda: 0.0,
            parallel: false,
            verbosity: Verbosity::Silent,
            ..Default::default()
        };

        let trainer = GBLinearTrainer::new(SquaredLoss, Rmse, params);
        let model = trainer.train(&train, &[]).unwrap();

        // Check predictions
        let pred1 = model.predict_row(&[1.0], &[0.0])[0];
        let pred2 = model.predict_row(&[2.0], &[0.0])[0];

        assert!((pred1 - 3.0).abs() < 0.5);
        assert!((pred2 - 5.0).abs() < 0.5);
    }

    #[test]
    fn train_with_regularization() {
        let row_data = RowMatrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 4, 1);
        let train_data: ColMatrix = (&row_data).into();
        let train_labels = vec![3.0, 5.0, 7.0, 9.0];
        let train = Dataset::from_numeric(&train_data, train_labels).unwrap();

        // Train without regularization
        let params_no_reg = GBLinearParams {
            n_rounds: 50,
            lambda: 0.0,
            parallel: false,
            verbosity: Verbosity::Silent,
            ..Default::default()
        };
        let trainer_no_reg = GBLinearTrainer::new(SquaredLoss, Rmse, params_no_reg);
        let model_no_reg = trainer_no_reg.train(&train, &[]).unwrap();

        // Train with L2 regularization
        let params_l2 = GBLinearParams {
            n_rounds: 50,
            lambda: 10.0,
            parallel: false,
            verbosity: Verbosity::Silent,
            ..Default::default()
        };
        let trainer_l2 = GBLinearTrainer::new(SquaredLoss, Rmse, params_l2);
        let model_l2 = trainer_l2.train(&train, &[]).unwrap();

        // L2 should produce smaller weights
        let w_no_reg = model_no_reg.weight(0, 0).abs();
        let w_l2 = model_l2.weight(0, 0).abs();
        assert!(w_l2 < w_no_reg);
    }

    #[test]
    fn train_multifeature() {
        // y = x0 + 2*x1
        let row_data = RowMatrix::from_vec(
            vec![
                1.0, 1.0, // y=3
                2.0, 1.0, // y=4
                1.0, 2.0, // y=5
                2.0, 2.0, // y=6
            ],
            4,
            2,
        );
        let train_data: ColMatrix = (&row_data).into();
        let train_labels = vec![3.0, 4.0, 5.0, 6.0];
        let train = Dataset::from_numeric(&train_data, train_labels).unwrap();

        let params = GBLinearParams {
            n_rounds: 200,
            learning_rate: 0.3,
            lambda: 0.0,
            parallel: false,
            verbosity: Verbosity::Silent,
            ..Default::default()
        };

        let trainer = GBLinearTrainer::new(SquaredLoss, Rmse, params);
        let model = trainer.train(&train, &[]).unwrap();

        let w0 = model.weight(0, 0);
        let w1 = model.weight(1, 0);

        // w0 should be ~1, w1 should be ~2
        assert!((w0 - 1.0).abs() < 0.3);
        assert!((w1 - 2.0).abs() < 0.3);
    }

    #[test]
    fn train_multiclass() {
        // Simple 3-class classification
        let row_data = RowMatrix::from_vec(
            vec![
                2.0, 1.0, // Class 0
                0.0, 1.0, // Class 1
                3.0, 1.0, // Class 0
                1.0, 3.0, // Class 2
                0.5, 0.5, // Class 1
                2.0, 2.0, // Class 2
            ],
            6,
            2,
        );
        let train_data: ColMatrix = (&row_data).into();
        let train_labels = vec![0.0, 1.0, 0.0, 2.0, 1.0, 2.0];
        let train = Dataset::from_numeric(&train_data, train_labels).unwrap();

        let params = GBLinearParams {
            n_rounds: 200,
            learning_rate: 0.3,
            lambda: 0.1,
            parallel: false,
            verbosity: Verbosity::Silent,
            ..Default::default()
        };

        let trainer = GBLinearTrainer::new(SoftmaxLoss::new(3), MulticlassLogLoss, params);
        let model = trainer.train(&train, &[]).unwrap();

        // Model should have 3 output groups
        assert_eq!(model.num_groups(), 3);

        // Verify model produces different outputs for different classes
        let preds0 = model.predict_row(&[2.0, 1.0], &[0.0, 0.0, 0.0]);
        let preds1 = model.predict_row(&[0.0, 1.0], &[0.0, 0.0, 0.0]);

        let diff: f32 = preds0
            .iter()
            .zip(preds1.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.1);
    }

    #[test]
    fn train_binary_classification() {
        let row_data = RowMatrix::from_vec(
            vec![
                0.0, 1.0, // Class 0
                1.0, 0.0, // Class 1
                0.5, 1.0, // Class 0
                1.0, 0.5, // Class 1
            ],
            4,
            2,
        );
        let train_data: ColMatrix = (&row_data).into();
        let train_labels = vec![0.0, 1.0, 0.0, 1.0];
        let train = Dataset::from_numeric(&train_data, train_labels).unwrap();

        let params = GBLinearParams {
            n_rounds: 50,
            verbosity: Verbosity::Silent,
            ..Default::default()
        };

        let trainer = GBLinearTrainer::new(LogisticLoss, LogLoss, params);
        let model = trainer.train(&train, &[]).unwrap();

        assert_eq!(model.num_groups(), 1);
    }
}

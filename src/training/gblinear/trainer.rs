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
use crate::inference::common::{PredictionKind, PredictionOutput};
use crate::inference::gblinear::LinearModel;
use crate::training::{
    EarlyStopping, EvalSet, Gradients, Metric, Objective, ObjectiveExt, TrainingLogger, Verbosity,
};

use super::selector::FeatureSelectorKind;
use super::updater::{update_bias, UpdateConfig, UpdaterKind};

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
        let base_scores = self.objective.init_base_score(train_labels, weights);

        // Initialize model with base scores as biases
        let mut model = LinearModel::zeros(num_features, num_outputs);
        for (group, &base_score) in base_scores.iter().enumerate() {
            model.set_bias(group, base_score);
        }

        // Create updater and selector
        let updater = if self.params.parallel {
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

        // Gradient and prediction buffers
        let mut gradients = Gradients::new(num_samples, num_outputs);
        // Initialize predictions with base scores (column-major)
        let mut predictions = vec![0.0f32; num_samples * num_outputs];
        for (group, &base_score) in base_scores.iter().enumerate() {
            for i in 0..num_samples {
                predictions[group * num_samples + i] = base_score;
            }
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
                    for i in 0..eval_rows {
                        preds[group * eval_rows + i] = base_score;
                    }
                }
                preds
            })
            .collect();

        // Early stopping state
        let mut early_stopping = if self.params.early_stopping_rounds > 0 && !eval_sets.is_empty() {
            Some(EarlyStopping::new(
                self.params.early_stopping_rounds as usize,
                self.metric.higher_is_better(),
            ))
        } else {
            None
        };

        // Logger
        let mut logger = TrainingLogger::new(self.params.verbosity);
        logger.start_training(self.params.n_rounds as usize);

        // Training loop
        for round in 0..self.params.n_rounds {
            // Compute predictions
            Self::compute_predictions(&model, &train_data, &mut predictions);

            // Compute gradients
            self.objective.compute_gradients_buffer(&predictions, train_labels, weights, &mut gradients);

            // Update each output
            for output in 0..num_outputs {
                update_bias(&mut model, &gradients, output, self.params.learning_rate);

                selector.setup_round(
                    &model,
                    &train_data,
                    &gradients,
                    output,
                    self.params.alpha,
                    self.params.lambda,
                );

                updater.update_round(
                    &mut model,
                    &train_data,
                    &gradients,
                    &mut selector,
                    output,
                    &update_config,
                );
            }

            // Evaluation
            let (round_metrics, early_stop_value) = self.evaluate_round(
                &model,
                train_labels,
                weights,
                &predictions,
                eval_sets,
                &eval_data,
                &mut eval_predictions,
                num_outputs,
                self.params.early_stopping_eval_set,
            );

            if self.params.verbosity >= Verbosity::Info {
                logger.log_round(round as usize, &round_metrics);
            }

            // Early stopping check
            if let Some(ref mut es) = early_stopping {
                if let Some(value) = early_stop_value {
                    if es.should_stop(value) {
                        if self.params.verbosity >= Verbosity::Info {
                            logger.log_early_stopping(round as usize, es.best_round(), self.metric.name());
                        }
                        break;
                    }
                }
            }
        }

        logger.finish_training();
        Some(model)
    }

    // ========================================================================
    // Helper methods
    // ========================================================================

    /// Compute predictions for all samples.
    ///
    /// Handles both single-output and multi-output models.
    /// Output is column-major: index = group * num_rows + row_idx
    fn compute_predictions(model: &LinearModel, data: &ColMatrix<f32>, output: &mut [f32]) {
        let num_rows = data.num_rows();
        let num_groups = model.num_groups();
        let num_features = model.num_features();

        // Initialize with bias (column-major: group-first)
        for group in 0..num_groups {
            let group_start = group * num_rows;
            for row_idx in 0..num_rows {
                output[group_start + row_idx] = model.bias(group);
            }
        }

        // Add weighted features (iterate by group for column-major writes)
        for feat_idx in 0..num_features {
            for (row_idx, value) in data.column(feat_idx) {
                for group in 0..num_groups {
                    output[group * num_rows + row_idx] += value * model.weight(feat_idx, group);
                }
            }
        }
    }

    fn evaluate_round(
        &self,
        model: &LinearModel,
        _train_labels: &[f32],
        _train_weights: Option<&[f32]>,
        _train_predictions: &[f32],
        eval_sets: &[EvalSet<'_>],
        eval_data: &[ColMatrix<f32>],
        eval_predictions: &mut [Vec<f32>],
        num_outputs: usize,
        early_stopping_eval_idx: usize,
    ) -> (Vec<(String, f64)>, Option<f64>) {
        let mut round_metrics = Vec::new();
        let mut early_stop_value = None;

        // Determine if we need to transform predictions for this metric
        let metric_kind = self.metric.expected_prediction_kind();
        let needs_transform = metric_kind != PredictionKind::Margin;

        // Training set metric is not computed here because the training ColMatrix
        // is not passed to this method. Eval-set metrics suffice for monitoring.

        for (set_idx, eval_set) in eval_sets.iter().enumerate() {
            // Compute raw predictions in column-major buffer
            Self::compute_predictions(model, &eval_data[set_idx], &mut eval_predictions[set_idx]);

            let n_rows = eval_data[set_idx].num_rows();

            // Convert column-major → row-major PredictionOutput and optionally transform
            let mut output = Self::col_major_to_row_major(&eval_predictions[set_idx], n_rows, num_outputs);
            if needs_transform {
                self.objective.transform_prediction_inplace(&mut output);
            }

            let targets = eval_set.dataset.targets();
            let w = eval_set.dataset.weights().unwrap_or(&[]);
            let value = self
                .metric
                .compute(n_rows, num_outputs, output.as_slice(), targets, w);

            round_metrics.push((format!("{}-{}", eval_set.name, self.metric.name()), value));

            if set_idx == early_stopping_eval_idx {
                early_stop_value = Some(value);
            }
        }

        (round_metrics, early_stop_value)
    }

    /// Convert column-major predictions to row-major `PredictionOutput`.
    fn col_major_to_row_major(col_major: &[f32], n_rows: usize, n_outputs: usize) -> PredictionOutput {
        let mut row_major = vec![0.0f32; n_rows * n_outputs];
        for out in 0..n_outputs {
            for row in 0..n_rows {
                row_major[row * n_outputs + out] = col_major[out * n_rows + row];
            }
        }
        PredictionOutput::new(row_major, n_rows, n_outputs)
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

//! High-level linear model trainer.
//!
//! Uses column-based access ([`ColumnAccess`]) for efficient coordinate descent.
//! Accepts data in any format that implements `ColumnAccess`:
//!
//! - [`ColMatrix`]: Best for dense data (columns are contiguous)
//! - [`CSCMatrix`]: Best for sparse data (only stores non-zeros)
//!
//! # Gradient Storage
//!
//! Training uses Structure-of-Arrays (SoA) gradient storage via [`GradientBuffer`]:
//! - Shape `[n_samples, n_outputs]` for unified single/multi-output handling
//! - Separate `grads[]` and `hess[]` arrays for cache efficiency
//!
//! # Evaluation Sets
//!

// Allow nested ifs in early stopping checks for readability.
// Also allow many arguments for internal utility functions.
// Allow range loops with index when we need the index for multiple arrays.
#![allow(clippy::collapsible_if, clippy::too_many_arguments, clippy::needless_range_loop)]
//! Use [`EvalSet`] to track metrics on multiple datasets during training.
//! The evaluation metric is configured via [`LinearTrainerConfig::eval_metric`]:
//!
//! ```ignore
//! let eval_sets = vec![
//!     EvalSet::new("train", &train_data, &train_labels),
//!     EvalSet::new("val", &val_data, &val_labels),
//! ];
//! let config = LinearTrainerConfig {
//!     eval_metric: EvalMetric::Rmse,
//!     ..Default::default()
//! };
//! let trainer = LinearTrainer::new(config);
//! let model = trainer.train_with_evals(&train_data, &train_labels, &eval_sets, &loss);
//! ```
//!
//! For row-major input, convert to column-major first:
//! ```ignore
//! let col_matrix: ColMatrix = row_matrix.to_layout();
//! trainer.train(&col_matrix, labels, loss);
//! ```

use crate::data::ColumnAccess;
use crate::linear::LinearModel;
use crate::training::{
    EvalMetric, EvalSet, GradientBuffer, Loss, MulticlassLoss, TrainingLogger, Verbosity,
};

use super::selector::FeatureSelectorKind;
use super::updater::{update_bias, UpdateConfig, UpdaterKind};

/// Configuration for linear model training.
#[derive(Debug, Clone)]
pub struct LinearTrainerConfig {
    /// Number of boosting rounds.
    pub num_rounds: usize,
    /// Learning rate (eta).
    pub learning_rate: f32,
    /// L1 regularization (alpha).
    pub alpha: f32,
    /// L2 regularization (lambda).
    pub lambda: f32,
    /// Use parallel (shotgun) updates.
    pub parallel: bool,
    /// Random seed for feature shuffling.
    pub seed: u64,
    /// Feature selector strategy.
    pub feature_selector: FeatureSelectorKind,
    /// Evaluation metric for logging and early stopping.
    /// Default: `EvalMetric::Rmse` for regression.
    /// For multiclass, consider using `EvalMetric::MulticlassLogLoss`.
    pub eval_metric: EvalMetric,
    /// Early stopping rounds (0 = disabled).
    pub early_stopping_rounds: usize,
    /// Index of eval set to use for early stopping (default: last eval set).
    /// Only used when `early_stopping_rounds > 0`.
    pub early_stopping_eval_set: Option<usize>,
    /// Verbosity level.
    pub verbosity: Verbosity,
}

impl Default for LinearTrainerConfig {
    fn default() -> Self {
        Self {
            num_rounds: 100,
            learning_rate: 0.5,
            alpha: 0.0,
            lambda: 1.0,
            parallel: true,
            seed: 42,
            feature_selector: FeatureSelectorKind::Shuffle,
            eval_metric: EvalMetric::Rmse,
            early_stopping_rounds: 0,
            early_stopping_eval_set: None,
            verbosity: Verbosity::Info,
        }
    }
}

/// Linear model trainer using coordinate descent.
pub struct LinearTrainer {
    config: LinearTrainerConfig,
}

impl LinearTrainer {
    /// Create a new trainer with the given configuration.
    pub fn new(config: LinearTrainerConfig) -> Self {
        Self { config }
    }

    /// Create a trainer with default configuration.
    pub fn default_config() -> Self {
        Self::new(LinearTrainerConfig::default())
    }

    /// Train a linear model on column-accessible data.
    ///
    /// This is the primary training method. It accepts any data type implementing
    /// [`ColumnAccess`], including:
    /// - [`ColMatrix`](crate::data::ColMatrix): Best for dense data
    /// - [`CSCMatrix`](crate::data::CSCMatrix): Best for sparse data
    ///
    /// # Arguments
    ///
    /// * `train_data` - Training features (must implement `ColumnAccess`)
    /// * `train_labels` - Training labels
    /// * `loss` - Loss function for gradient computation
    ///
    /// # Returns
    ///
    /// Trained `LinearModel`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use booste_rs::data::{ColMatrix, RowMatrix};
    /// use booste_rs::linear::training::LinearTrainer;
    /// use booste_rs::training::SquaredLoss;
    ///
    /// // Convert row-major to column-major for training
    /// let row_data = RowMatrix::from_vec(data, num_rows, num_features);
    /// let col_data: ColMatrix = row_data.to_layout();
    ///
    /// let trainer = LinearTrainer::default_config();
    /// let model = trainer.train(&col_data, &labels, &SquaredLoss);
    /// ```
    pub fn train<C, L>(
        &self,
        train_data: &C,
        train_labels: &[f32],
        loss: &L,
    ) -> LinearModel
    where
        C: ColumnAccess<Element = f32> + Sync,
        L: Loss,
    {
        self.train_with_evals::<C, C, L>(train_data, train_labels, &[], loss)
    }

    /// Train a linear model with multiple evaluation sets.
    ///
    /// This is the most flexible training method, supporting:
    /// - Multiple named evaluation sets (train, val, test)
    /// - Early stopping based on any eval set
    ///
    /// The evaluation metric is configured via [`LinearTrainerConfig::eval_metric`].
    /// Defaults to RMSE if not specified.
    ///
    /// # Arguments
    ///
    /// * `train_data` - Training features (must implement `ColumnAccess`)
    /// * `train_labels` - Training labels
    /// * `eval_sets` - Named datasets for evaluation (can include training set)
    /// * `loss` - Loss function for gradient computation
    ///
    /// # Returns
    ///
    /// Trained `LinearModel`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use booste_rs::data::ColMatrix;
    /// use booste_rs::linear::training::{EvalMetric, LinearTrainer, LinearTrainerConfig};
    /// use booste_rs::training::{EvalSet, SquaredLoss};
    ///
    /// let eval_sets = vec![
    ///     EvalSet::new("train", &train_data, &train_labels),
    ///     EvalSet::new("val", &val_data, &val_labels),
    /// ];
    ///
    /// let config = LinearTrainerConfig {
    ///     early_stopping_rounds: 10,
    ///     eval_metric: EvalMetric::Rmse,
    ///     ..Default::default()
    /// };
    /// let trainer = LinearTrainer::new(config);
    /// let model = trainer.train_with_evals(
    ///     &train_data,
    ///     &train_labels,
    ///     &eval_sets,
    ///     &SquaredLoss,
    /// );
    /// // Logs: [0] train-rmse:15.23 val-rmse:16.12
    /// ```
    pub fn train_with_evals<C, E, L>(
        &self,
        train_data: &C,
        train_labels: &[f32],
        eval_sets: &[EvalSet<'_, E>],
        loss: &L,
    ) -> LinearModel
    where
        C: ColumnAccess<Element = f32> + Sync,
        E: ColumnAccess<Element = f32>,
        L: Loss,
    {
        let num_features = train_data.num_columns();
        let num_samples = train_data.num_rows();
        let num_outputs = 1; // Loss trait is single-output

        assert_eq!(
            train_labels.len(),
            num_samples,
            "Labels length must match number of samples"
        );

        // Initialize model
        let mut model = LinearModel::zeros(num_features, num_outputs);

        // Create updater
        let updater = if self.config.parallel {
            UpdaterKind::Parallel
        } else {
            UpdaterKind::Sequential
        };

        // Create selector from config
        let mut selector = self.config.feature_selector.create_state(self.config.seed);

        // Create update config
        let update_config = UpdateConfig {
            alpha: self.config.alpha,
            lambda: self.config.lambda,
            learning_rate: self.config.learning_rate,
        };

        // SoA gradient storage
        let mut gradients = GradientBuffer::new(num_samples, num_outputs);

        // Predictions buffer for training
        let mut predictions = vec![0.0f32; num_samples];

        // Prediction buffers for each eval set
        let mut eval_predictions: Vec<Vec<f32>> = eval_sets
            .iter()
            .map(|es| vec![0.0f32; es.data.num_rows()])
            .collect();

        // Early stopping state
        let early_stopping_eval_idx = self
            .config
            .early_stopping_eval_set
            .unwrap_or(eval_sets.len().saturating_sub(1));
        let mut best_metric_value: Option<f64> = None;
        let mut best_round = 0;
        let mut rounds_without_improvement = 0;

        // Get metric from config
        let eval_metric = &self.config.eval_metric;
        let higher_is_better = eval_metric.higher_is_better();

        // Logger
        let mut logger = TrainingLogger::new(self.config.verbosity);
        logger.start_training(self.config.num_rounds);

        // Training loop
        for round in 0..self.config.num_rounds {
            // Compute predictions on training data
            Self::compute_predictions_col(&model, train_data, &mut predictions);

            // Compute gradients using SoA buffer
            loss.compute_gradients(&predictions, train_labels, &mut gradients);

            // Update bias
            update_bias(&mut model, &gradients, 0, self.config.learning_rate);

            // Setup selector (handles Greedy/Thrifty gradient-based ranking)
            selector.setup_round(
                &model,
                train_data,
                &gradients,
                0,
                self.config.alpha,
                self.config.lambda,
            );

            // Update feature weights
            updater.update_round(
                &mut model,
                train_data,
                &gradients,
                &mut selector,
                0, // output
                &update_config,
            );

            // Compute predictions and metrics for all eval sets
            let mut round_metrics = Vec::new();
            let mut early_stop_value: Option<f64> = None;

            for (set_idx, eval_set) in eval_sets.iter().enumerate() {
                // Compute predictions for this eval set
                Self::compute_predictions_col(&model, eval_set.data, &mut eval_predictions[set_idx]);

                // Compute the configured metric
                let value = eval_metric.evaluate(
                    &eval_predictions[set_idx],
                    eval_set.labels,
                    num_outputs,
                );

                let metric_display_name = format!("{}-{}", eval_set.name, eval_metric.name());
                round_metrics.push((metric_display_name, value));

                // Track early stopping metric value
                if set_idx == early_stopping_eval_idx {
                    early_stop_value = Some(value);
                }
            }

            // Logging
            if self.config.verbosity >= Verbosity::Info {
                logger.log_round(round, &round_metrics);
            }

            // Early stopping check
            if self.config.early_stopping_rounds > 0 {
                if let Some(current_value) = early_stop_value {
                    let is_improvement = match best_metric_value {
                        None => true,
                        Some(best) => {
                            if higher_is_better {
                                current_value > best
                            } else {
                                current_value < best
                            }
                        }
                    };

                    if is_improvement {
                        best_metric_value = Some(current_value);
                        best_round = round;
                        rounds_without_improvement = 0;
                    } else {
                        rounds_without_improvement += 1;
                    }

                    if rounds_without_improvement > self.config.early_stopping_rounds {
                        if self.config.verbosity >= Verbosity::Info {
                            logger.log_early_stopping(round, best_round, eval_metric.name());
                        }
                        break;
                    }
                }
            }
        }

        logger.finish_training();
        model
    }

    /// Train a multiclass linear model.
    ///
    /// This method is specifically for multiclass classification with K > 2 classes.
    /// It uses the [`MulticlassLoss`] trait which computes proper per-class gradients
    /// using the full prediction vector (e.g., softmax needs all class logits).
    ///
    /// # Arguments
    ///
    /// * `train_data` - Training features
    /// * `train_labels` - Class labels (0 to K-1)
    /// * `loss` - Multiclass loss function (e.g., `SoftmaxLoss`)
    ///
    /// # Example
    ///
    /// ```ignore
    /// use booste_rs::linear::training::LinearTrainer;
    /// use booste_rs::training::SoftmaxLoss;
    ///
    /// let loss = SoftmaxLoss::new(3); // 3 classes
    /// let model = trainer.train_multiclass(&data, &labels, &loss);
    /// ```
    pub fn train_multiclass<C, L>(
        &self,
        train_data: &C,
        train_labels: &[f32],
        loss: &L,
    ) -> LinearModel
    where
        C: ColumnAccess<Element = f32> + Sync,
        L: MulticlassLoss,
    {
        self.train_multiclass_with_evals::<C, C, L>(train_data, train_labels, &[], loss)
    }

    /// Train a multiclass linear model with multiple eval sets.
    ///
    /// This is the most flexible multiclass training method, supporting:
    /// - Multiple named evaluation sets (train, val, test)
    /// - Early stopping based on any eval set
    ///
    /// The evaluation metric is configured via [`LinearTrainerConfig::eval_metric`].
    /// For multiclass, use `EvalMetric::MulticlassLogLoss`.
    ///
    /// # Arguments
    ///
    /// * `train_data` - Training features
    /// * `train_labels` - Class labels (0 to K-1)
    /// * `eval_sets` - Named datasets for evaluation
    /// * `loss` - Multiclass loss function
    ///
    /// # Example
    ///
    /// ```ignore
    /// use booste_rs::linear::training::{EvalMetric, LinearTrainer, LinearTrainerConfig};
    /// use booste_rs::training::{EvalSet, SoftmaxLoss};
    ///
    /// let eval_sets = vec![
    ///     EvalSet::new("train", &train_data, &train_labels),
    ///     EvalSet::new("val", &val_data, &val_labels),
    /// ];
    /// let loss = SoftmaxLoss::new(3);
    ///
    /// let config = LinearTrainerConfig {
    ///     eval_metric: EvalMetric::MulticlassLogLoss,
    ///     ..Default::default()
    /// };
    /// let trainer = LinearTrainer::new(config);
    /// let model = trainer.train_multiclass_with_evals(
    ///     &train_data,
    ///     &train_labels,
    ///     &eval_sets,
    ///     &loss,
    /// );
    /// ```
    pub fn train_multiclass_with_evals<C, E, L>(
        &self,
        train_data: &C,
        train_labels: &[f32],
        eval_sets: &[EvalSet<'_, E>],
        loss: &L,
    ) -> LinearModel
    where
        C: ColumnAccess<Element = f32> + Sync,
        E: ColumnAccess<Element = f32>,
        L: MulticlassLoss,
    {
        let num_features = train_data.num_columns();
        let num_samples = train_data.num_rows();
        let num_outputs = loss.num_classes();

        assert!(
            num_outputs >= 2,
            "Multiclass requires at least 2 classes, got {}",
            num_outputs
        );
        assert_eq!(
            train_labels.len(),
            num_samples,
            "Labels length must match number of samples"
        );

        // Initialize model
        let mut model = LinearModel::zeros(num_features, num_outputs);

        // Create updater
        let updater = if self.config.parallel {
            UpdaterKind::Parallel
        } else {
            UpdaterKind::Sequential
        };

        let mut selector = self.config.feature_selector.create_state(self.config.seed);

        // Create update config
        let update_config = UpdateConfig {
            alpha: self.config.alpha,
            lambda: self.config.lambda,
            learning_rate: self.config.learning_rate,
        };

        // SoA gradient storage: K outputs per sample
        let mut gradients = GradientBuffer::new(num_samples, num_outputs);

        // Predictions buffer for training: K predictions per sample
        let mut predictions = vec![0.0f32; num_samples * num_outputs];

        // Prediction buffers for each eval set
        let mut eval_predictions: Vec<Vec<f32>> = eval_sets
            .iter()
            .map(|es| vec![0.0f32; es.data.num_rows() * num_outputs])
            .collect();

        // Early stopping state
        let early_stopping_eval_idx = self
            .config
            .early_stopping_eval_set
            .unwrap_or(eval_sets.len().saturating_sub(1));
        let mut best_metric_value: Option<f64> = None;
        let mut best_round = 0;
        let mut rounds_without_improvement = 0;

        // Get metric from config
        let eval_metric = &self.config.eval_metric;
        let higher_is_better = eval_metric.higher_is_better();

        // Logger
        let mut logger = TrainingLogger::new(self.config.verbosity);
        logger.start_training(self.config.num_rounds);

        // Training loop
        for round in 0..self.config.num_rounds {
            // Compute predictions on training data
            Self::compute_predictions_col_multiclass(&model, train_data, &mut predictions);

            // Compute multiclass gradients using SoA buffer
            loss.compute_gradients(&predictions, train_labels, &mut gradients);

            // Update each output (class)
            for output in 0..num_outputs {
                // Update bias
                update_bias(&mut model, &gradients, output, self.config.learning_rate);

                // Setup selector for this round (handles Greedy/Thrifty gradient ranking)
                selector.setup_round(
                    &model,
                    train_data,
                    &gradients,
                    output,
                    self.config.alpha,
                    self.config.lambda,
                );

                // Update feature weights
                updater.update_round(
                    &mut model,
                    train_data,
                    &gradients,
                    &mut selector,
                    output,
                    &update_config,
                );
            }

            // Compute predictions and metrics for all eval sets
            let mut round_metrics = Vec::new();
            let mut early_stop_value: Option<f64> = None;

            for (set_idx, eval_set) in eval_sets.iter().enumerate() {
                // Compute predictions for this eval set
                Self::compute_predictions_col_multiclass(
                    &model,
                    eval_set.data,
                    &mut eval_predictions[set_idx],
                );

                // Compute the configured metric
                let value = eval_metric.evaluate(
                    &eval_predictions[set_idx],
                    eval_set.labels,
                    num_outputs,
                );

                let metric_display_name = format!("{}-{}", eval_set.name, eval_metric.name());
                round_metrics.push((metric_display_name, value));

                // Track early stopping metric value
                if set_idx == early_stopping_eval_idx {
                    early_stop_value = Some(value);
                }
            }

            // Logging
            if self.config.verbosity >= Verbosity::Info {
                logger.log_round(round, &round_metrics);
            }

            // Early stopping check
            if self.config.early_stopping_rounds > 0 {
                if let Some(current_value) = early_stop_value {
                    let is_improvement = match best_metric_value {
                        None => true,
                        Some(best) => {
                            if higher_is_better {
                                current_value > best
                            } else {
                                current_value < best
                            }
                        }
                    };

                    if is_improvement {
                        best_metric_value = Some(current_value);
                        best_round = round;
                        rounds_without_improvement = 0;
                    } else {
                        rounds_without_improvement += 1;
                    }

                    if rounds_without_improvement > self.config.early_stopping_rounds {
                        if self.config.verbosity >= Verbosity::Info {
                            logger.log_early_stopping(round, best_round, eval_metric.name());
                        }
                        break;
                    }
                }
            }
        }

        logger.finish_training();
        model
    }

    /// Compute predictions for all samples (single-output).
    fn compute_predictions_col<C: ColumnAccess<Element = f32>>(
        model: &LinearModel,
        data: &C,
        output: &mut [f32],
    ) {
        let num_rows = data.num_rows();
        let num_features = model.num_features();

        // Initialize with bias
        for row_idx in 0..num_rows {
            output[row_idx] = model.bias(0);
        }

        // Add weighted features column by column
        for feat_idx in 0..num_features {
            let weight = model.weight(feat_idx, 0);
            for (row_idx, value) in data.column(feat_idx) {
                output[row_idx] += value * weight;
            }
        }
    }

    /// Compute predictions for all samples (multiclass).
    fn compute_predictions_col_multiclass<C: ColumnAccess<Element = f32>>(
        model: &LinearModel,
        data: &C,
        output: &mut [f32],
    ) {
        let num_rows = data.num_rows();
        let num_groups = model.num_groups();
        let num_features = model.num_features();

        // Initialize with bias
        for row_idx in 0..num_rows {
            for group in 0..num_groups {
                output[row_idx * num_groups + group] = model.bias(group);
            }
        }

        // Add weighted features column by column
        for feat_idx in 0..num_features {
            for (row_idx, value) in data.column(feat_idx) {
                for group in 0..num_groups {
                    output[row_idx * num_groups + group] += value * model.weight(feat_idx, group);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{ColMatrix, RowMatrix};
    use crate::training::SquaredLoss;

    #[test]
    fn train_simple_regression() {
        // y = 2*x + 1
        let row_data = RowMatrix::from_vec(
            vec![
                1.0, // x=1 → y=3
                2.0, // x=2 → y=5
                3.0, // x=3 → y=7
                4.0, // x=4 → y=9
            ],
            4,
            1,
        );
        let train_data: ColMatrix = row_data.to_layout();
        let train_labels = vec![3.0, 5.0, 7.0, 9.0];

        let config = LinearTrainerConfig {
            num_rounds: 100,
            learning_rate: 0.5,
            alpha: 0.0,
            lambda: 0.0,
            parallel: false,
            verbosity: Verbosity::Silent,
            ..Default::default()
        };

        let trainer = LinearTrainer::new(config);
        let model = trainer.train(&train_data, &train_labels, &SquaredLoss);

        // Check predictions
        let pred1 = model.predict_row(&[1.0], &[0.0])[0];
        let pred2 = model.predict_row(&[2.0], &[0.0])[0];

        // Should be close to true values
        assert!((pred1 - 3.0).abs() < 0.5);
        assert!((pred2 - 5.0).abs() < 0.5);
    }

    #[test]
    fn train_with_regularization() {
        let row_data = RowMatrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 4, 1);
        let train_data: ColMatrix = row_data.to_layout();
        let train_labels = vec![3.0, 5.0, 7.0, 9.0];

        // Train without regularization
        let config_no_reg = LinearTrainerConfig {
            num_rounds: 50,
            learning_rate: 0.5,
            alpha: 0.0,
            lambda: 0.0,
            parallel: false,
            verbosity: Verbosity::Silent,
            ..Default::default()
        };
        let model_no_reg = LinearTrainer::new(config_no_reg).train(
            &train_data,
            &train_labels,
            &SquaredLoss,
        );

        // Train with L2 regularization
        let config_l2 = LinearTrainerConfig {
            num_rounds: 50,
            learning_rate: 0.5,
            alpha: 0.0,
            lambda: 10.0,
            parallel: false,
            verbosity: Verbosity::Silent,
            ..Default::default()
        };
        let model_l2 =
            LinearTrainer::new(config_l2).train(&train_data, &train_labels, &SquaredLoss);

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
                1.0, 1.0, // x0=1, x1=1 → y=3
                2.0, 1.0, // x0=2, x1=1 → y=4
                1.0, 2.0, // x0=1, x1=2 → y=5
                2.0, 2.0, // x0=2, x1=2 → y=6
            ],
            4,
            2,
        );
        let train_data: ColMatrix = row_data.to_layout();
        let train_labels = vec![3.0, 4.0, 5.0, 6.0];

        let config = LinearTrainerConfig {
            num_rounds: 200,
            learning_rate: 0.3,
            alpha: 0.0,
            lambda: 0.0,
            parallel: false,
            verbosity: Verbosity::Silent,
            ..Default::default()
        };

        let trainer = LinearTrainer::new(config);
        let model = trainer.train(&train_data, &train_labels, &SquaredLoss);

        // Check weights are roughly correct
        let w0 = model.weight(0, 0);
        let w1 = model.weight(1, 0);

        // w0 should be ~1, w1 should be ~2
        assert!((w0 - 1.0).abs() < 0.3);
        assert!((w1 - 2.0).abs() < 0.3);
    }

    #[test]
    fn parallel_vs_sequential_similar() {
        let row_data = RowMatrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 4, 1);
        let train_data: ColMatrix = row_data.to_layout();
        let train_labels = vec![3.0, 5.0, 7.0, 9.0];

        let config_seq = LinearTrainerConfig {
            num_rounds: 50,
            parallel: false,
            verbosity: Verbosity::Silent,
            ..Default::default()
        };
        let model_seq =
            LinearTrainer::new(config_seq).train(&train_data, &train_labels, &SquaredLoss);

        let config_par = LinearTrainerConfig {
            num_rounds: 50,
            parallel: true,
            verbosity: Verbosity::Silent,
            ..Default::default()
        };
        let model_par =
            LinearTrainer::new(config_par).train(&train_data, &train_labels, &SquaredLoss);

        // Results should be similar (not identical due to race conditions in shotgun)
        let pred_seq = model_seq.predict_row(&[2.5], &[0.0])[0];
        let pred_par = model_par.predict_row(&[2.5], &[0.0])[0];

        assert!((pred_seq - pred_par).abs() < 1.0);
    }

    #[test]
    fn train_multiclass_simple() {
        use crate::training::SoftmaxLoss;

        // Simple 3-class classification
        let row_data = RowMatrix::from_vec(
            vec![
                2.0, 1.0, // Class 0: x0=2 > x1=1
                0.0, 1.0, // Class 1: x0=0 < x1=1
                3.0, 1.0, // Class 0: x0=3 > x1=1
                1.0, 3.0, // Class 2: sum=4 >= 3
                0.5, 0.5, // Class 1
                2.0, 2.0, // Class 2: sum=4 >= 3
            ],
            6,
            2,
        );
        let train_data: ColMatrix = row_data.to_layout();
        let train_labels = vec![0.0, 1.0, 0.0, 2.0, 1.0, 2.0];

        let config = LinearTrainerConfig {
            num_rounds: 200,
            learning_rate: 0.3,
            alpha: 0.0,
            lambda: 0.1,
            parallel: false,
            verbosity: Verbosity::Silent,
            ..Default::default()
        };

        let trainer = LinearTrainer::new(config);
        let loss = SoftmaxLoss::new(3);
        let model = trainer.train_multiclass(&train_data, &train_labels, &loss);

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
        assert!(
            diff > 0.1,
            "Multiclass model should produce different outputs: {:?} vs {:?}",
            preds0,
            preds1
        );

        // Check that training accuracy is reasonable (> 50% for 3 classes)
        let mut correct = 0;
        for (i, &label) in train_labels.iter().enumerate() {
            let row = &row_data.as_slice()[i * 2..(i + 1) * 2];
            let preds = model.predict_row(row, &[0.0, 0.0, 0.0]);
            let predicted = preds
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();
            if predicted == label as usize {
                correct += 1;
            }
        }
        let accuracy = correct as f64 / train_labels.len() as f64;
        assert!(
            accuracy > 0.5,
            "Training accuracy should be > 50%, got {}",
            accuracy
        );
    }

    #[test]
    fn train_with_evals_multiple_eval_sets() {
        use crate::training::EvalSet;

        // y = 2*x + 1
        let row_data = RowMatrix::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            6,
            1,
        );
        let train_data: ColMatrix = row_data.to_layout();
        let train_labels = vec![3.0, 5.0, 7.0, 9.0, 11.0, 13.0];

        // Split into train (first 4) and val (last 2) for eval sets
        let row_train = RowMatrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 4, 1);
        let col_train: ColMatrix = row_train.to_layout();
        let labels_train = vec![3.0, 5.0, 7.0, 9.0];

        let row_val = RowMatrix::from_vec(vec![5.0, 6.0], 2, 1);
        let col_val: ColMatrix = row_val.to_layout();
        let labels_val = vec![11.0, 13.0];

        let eval_sets = vec![
            EvalSet::new("train", &col_train, &labels_train),
            EvalSet::new("val", &col_val, &labels_val),
        ];

        let config = LinearTrainerConfig {
            num_rounds: 100,
            learning_rate: 0.5,
            alpha: 0.0,
            lambda: 0.0,
            parallel: false,
            verbosity: Verbosity::Silent,
            eval_metric: EvalMetric::Rmse,
            ..Default::default()
        };

        let trainer = LinearTrainer::new(config);
        let model = trainer.train_with_evals(
            &train_data,
            &train_labels,
            &eval_sets,
            &SquaredLoss,
        );

        // Check predictions
        let pred1 = model.predict_row(&[1.0], &[0.0])[0];
        let pred2 = model.predict_row(&[2.0], &[0.0])[0];

        // Should be close to true values
        assert!((pred1 - 3.0).abs() < 0.5, "pred1={}", pred1);
        assert!((pred2 - 5.0).abs() < 0.5, "pred2={}", pred2);
    }

    #[test]
    fn train_with_evals_early_stopping() {
        use crate::training::EvalSet;

        // y = 2*x + 1
        let row_data = RowMatrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 4, 1);
        let train_data: ColMatrix = row_data.to_layout();
        let train_labels = vec![3.0, 5.0, 7.0, 9.0];

        // Validation set
        let row_val = RowMatrix::from_vec(vec![5.0, 6.0], 2, 1);
        let col_val: ColMatrix = row_val.to_layout();
        let labels_val = vec![11.0, 13.0];

        let eval_sets = vec![
            EvalSet::new("train", &train_data, &train_labels),
            EvalSet::new("val", &col_val, &labels_val),
        ];

        let config = LinearTrainerConfig {
            num_rounds: 1000, // High number, but early stopping should kick in
            learning_rate: 0.5,
            alpha: 0.0,
            lambda: 0.0,
            parallel: false,
            early_stopping_rounds: 10,
            early_stopping_eval_set: Some(1), // Use validation set
            eval_metric: EvalMetric::Rmse,
            verbosity: Verbosity::Silent,
            ..Default::default()
        };

        let trainer = LinearTrainer::new(config);
        let model = trainer.train_with_evals(
            &train_data,
            &train_labels,
            &eval_sets,
            &SquaredLoss,
        );

        // Model should still work reasonably (training completed)
        let pred = model.predict_row(&[2.0], &[0.0])[0];
        assert!((pred - 5.0).abs() < 1.0, "pred={}", pred);
    }
}

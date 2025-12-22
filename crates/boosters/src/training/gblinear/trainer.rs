//! Gradient boosting trainer for linear models.
//!
//! Implements coordinate descent training for GBLinear models.
//! Supports single-output and multi-output objectives.

use ndarray::{Array2, ArrayView1, ArrayView2};

use crate::data::{Dataset, FeaturesView};
use crate::repr::gblinear::LinearModel;
use crate::training::eval;
use crate::training::{
    EarlyStopping, EarlyStopAction, EvalSet, Gradients, MetricFn, ObjectiveFn, TrainingLogger, Verbosity,
};

use super::selector::FeatureSelectorKind;
use super::updater::{Updater, UpdateConfig, UpdaterKind};

// ============================================================================
// GBLinearParams
// ============================================================================

/// Parameters for GBLinear training.
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
        }
    }
}

// ============================================================================
// GBLinearTrainer
// ============================================================================

/// Gradient boosted linear model trainer.
#[derive(Clone, Debug)]
pub struct GBLinearTrainer<O: ObjectiveFn, M: MetricFn> {
    objective: O,
    metric: M,
    params: GBLinearParams,
}

impl<O: ObjectiveFn, M: MetricFn> GBLinearTrainer<O, M> {
    /// Create a new trainer with the given objective and parameters.
    pub fn new(objective: O, metric: M, params: GBLinearParams) -> Self {
        Self {
            objective,
            metric,
            params,
        }
    }

    /// Train a linear model.
    ///
    /// **Note:** This method does NOT create a thread pool. The caller must set up
    /// parallelism via `rayon::ThreadPool::install()` if desired.
    ///
    /// # Arguments
    ///
    /// * `train` - Training dataset (features, targets, optional weights)
    /// * `eval_sets` - Evaluation sets for monitoring (`&[]` if not needed)
    pub fn train(
        &self, 
        train: &Dataset, eval_sets: &[EvalSet<'_>]
    ) -> Option<LinearModel> {
        let train_data = train.for_gblinear().ok()?;
        let train_labels = train.targets();
        let weights = train.weights();

        let num_features = train_data.nrows();
        let num_samples = train_data.ncols();
        let num_outputs = self.objective.n_outputs();

        assert!(
            num_outputs >= 1,
            "Objective must have at least 1 output, got {}",
            num_outputs
        );
        debug_assert_eq!(train_labels.len(), num_samples);
        debug_assert!(weights.is_none_or(|w| w.len() == num_samples));

        // Compute base scores from objective (optimal constant prediction)
        let weights_opt = weights.map(ArrayView1::from);
        let targets_1d = ArrayView1::from(train_labels);
        let mut base_scores = vec![0.0f32; num_outputs];
        self.objective.compute_base_score(
            targets_1d.view(),
            weights_opt,
            &mut base_scores,
        );

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
        
        // Check if we need evaluation (metric is enabled)
        let needs_evaluation = self.metric.is_enabled();
        
        // Initialize eval predictions with base scores (only if evaluation is needed)
        let eval_data: Vec<Array2<f32>> = if needs_evaluation {
            eval_sets
                .iter()
                .map(|es| es.dataset.for_gblinear().ok())
                .collect::<Option<Vec<_>>>()?
        } else {
            Vec::new()
        };

        // TODO: Keep as Vec<Vec<f32>> for compatibility with updater methods
        // We'll convert to Array2 views when calling evaluator
        let mut eval_predictions: Vec<Vec<f32>> = if needs_evaluation {
            eval_data
                .iter()
                .map(|m| {
                    let eval_rows = m.ncols();
                    let mut preds = vec![0.0f32; eval_rows * num_outputs];
                    for (group, &base_score) in base_scores.iter().enumerate() {
                        let start = group * eval_rows;
                        preds[start..start + eval_rows].fill(base_score);
                    }
                    preds
                })
                .collect()
        } else {
            Vec::new()
        };

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
            let predictions_view = ArrayView2::from_shape((num_outputs, num_samples), &predictions)
                .expect("predictions shape mismatch");
            self.objective.compute_gradients(
                predictions_view,
                targets_1d.view(),
                weights_opt,
                gradients.pairs_array_mut(),
            );

            // Update each output
            for output in 0..num_outputs {
                let bias_delta = updater.update_bias(&mut model, &gradients, output);
                
                // Apply bias delta to predictions incrementally
                if bias_delta.abs() > 1e-10 {
                    updater.apply_bias_delta_to_predictions(bias_delta, output, num_samples, &mut predictions);
                    
                    // Also update eval predictions (only if evaluation is needed)
                    if needs_evaluation {
                        for (set_idx, matrix) in eval_data.iter().enumerate() {
                            let eval_rows = matrix.ncols();
                            updater.apply_bias_delta_to_predictions(
                                bias_delta,
                                output,
                                eval_rows,
                                &mut eval_predictions[set_idx],
                            );
                        }
                    }
                    
                    // Recompute gradients after bias update
                    let predictions_view = ArrayView2::from_shape((num_outputs, num_samples), &predictions)
                        .expect("predictions shape mismatch");
                    self.objective.compute_gradients(
                        predictions_view,
                        targets_1d.view(),
                        weights_opt,
                        gradients.pairs_array_mut(),
                    );
                }

                let train_features = FeaturesView::from_array(train_data.view());
                selector.setup_round(
                    &model,
                    &train_features,
                    &gradients,
                    output,
                    self.params.alpha,
                    self.params.lambda,
                );

                let weight_deltas = updater.update_round(
                    &mut model,
                    &train_features,
                    &gradients,
                    &mut selector,
                    output,
                );
                
                // Apply weight deltas to predictions incrementally
                if !weight_deltas.is_empty() {
                    updater.apply_weight_deltas_to_predictions(
                        &train_features,
                        &weight_deltas,
                        output,
                        num_samples,
                        &mut predictions,
                    );
                    
                    // Also update eval predictions (only if evaluation is needed)
                    if needs_evaluation {
                        for (set_idx, matrix) in eval_data.iter().enumerate() {
                            let eval_rows = matrix.ncols();
                            let eval_features = FeaturesView::from_array(matrix.view());
                            updater.apply_weight_deltas_to_predictions(
                                &eval_features,
                                &weight_deltas,
                                output,
                                eval_rows,
                                &mut eval_predictions[set_idx],
                            );
                        }
                    }
                }
            }

            // Evaluation using Evaluator (only if metric is enabled)
            let (round_metrics, early_stop_value) = if needs_evaluation {
                // Convert predictions Vec to Array2 view
                let predictions_view = ArrayView2::from_shape((num_outputs, num_samples), &predictions)
                    .expect("predictions shape mismatch");
                
                // Convert eval_predictions Vec<Vec<f32>> to Vec<Array2<f32>>
                let eval_preds_arrays: Vec<Array2<f32>> = eval_predictions.iter()
                    .zip(eval_data.iter())
                    .map(|(preds, matrix)| {
                        let eval_rows = matrix.ncols();
                        Array2::from_shape_vec((num_outputs, eval_rows), preds.clone())
                            .expect("eval predictions shape mismatch")
                    })
                    .collect();
                
                let metrics = evaluator.evaluate_round(
                    predictions_view,
                    targets_1d.view(),
                    weights_opt,
                    eval_sets,
                    &eval_preds_arrays,
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
    use crate::training::{LogLoss, MulticlassLogLoss, Rmse, SquaredLoss, LogisticLoss, SoftmaxLoss};

    /// Helper to transpose row-major data to feature-major for FeaturesView.
    /// Input: [s0_f0, s0_f1, s1_f0, s1_f1, ...] (row-major)
    /// Output: [f0_s0, f0_s1, ..., f1_s0, f1_s1, ...] (feature-major)
    fn transpose_to_feature_major(data: &[f32], n_samples: usize, n_features: usize) -> Vec<f32> {
        let mut result = vec![0.0; data.len()];
        for sample in 0..n_samples {
            for feature in 0..n_features {
                result[feature * n_samples + sample] = data[sample * n_features + feature];
            }
        }
        result
    }

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
        // 4 samples, 1 feature - already feature-major since single feature
        let feature_data = vec![1.0, 2.0, 3.0, 4.0];
        let train_features = FeaturesView::from_slice(&feature_data, 4, 1).unwrap();
        let train_labels = vec![3.0, 5.0, 7.0, 9.0];
        let train = Dataset::from_numeric(&train_features, train_labels).unwrap();

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
        // 4 samples, 1 feature
        let feature_data = vec![1.0, 2.0, 3.0, 4.0];
        let train_features = FeaturesView::from_slice(&feature_data, 4, 1).unwrap();
        let train_labels = vec![3.0, 5.0, 7.0, 9.0];
        let train = Dataset::from_numeric(&train_features, train_labels).unwrap();

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
        // Row-major: [s0_f0, s0_f1, s1_f0, s1_f1, ...]
        let row_data = vec![
            1.0, 1.0, // y=3
            2.0, 1.0, // y=4
            1.0, 2.0, // y=5
            2.0, 2.0, // y=6
        ];
        let feature_data = transpose_to_feature_major(&row_data, 4, 2);
        let train_features = FeaturesView::from_slice(&feature_data, 4, 2).unwrap();
        let train_labels = vec![3.0, 4.0, 5.0, 6.0];
        let train = Dataset::from_numeric(&train_features, train_labels).unwrap();

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
        // Row-major data
        let row_data = vec![
            2.0, 1.0, // Class 0
            0.0, 1.0, // Class 1
            3.0, 1.0, // Class 0
            1.0, 3.0, // Class 2
            0.5, 0.5, // Class 1
            2.0, 2.0, // Class 2
        ];
        let feature_data = transpose_to_feature_major(&row_data, 6, 2);
        let train_features = FeaturesView::from_slice(&feature_data, 6, 2).unwrap();
        let train_labels = vec![0.0, 1.0, 0.0, 2.0, 1.0, 2.0];
        let train = Dataset::from_numeric(&train_features, train_labels).unwrap();

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
        assert_eq!(model.n_groups(), 3);

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
        // Row-major data
        let row_data = vec![
            0.0, 1.0, // Class 0
            1.0, 0.0, // Class 1
            0.5, 1.0, // Class 0
            1.0, 0.5, // Class 1
        ];
        let feature_data = transpose_to_feature_major(&row_data, 4, 2);
        let train_features = FeaturesView::from_slice(&feature_data, 4, 2).unwrap();
        let train_labels = vec![0.0, 1.0, 0.0, 1.0];
        let train = Dataset::from_numeric(&train_features, train_labels).unwrap();

        let params = GBLinearParams {
            n_rounds: 50,
            verbosity: Verbosity::Silent,
            ..Default::default()
        };

        let trainer = GBLinearTrainer::new(LogisticLoss, LogLoss, params);
        let model = trainer.train(&train, &[]).unwrap();

        assert_eq!(model.n_groups(), 1);
    }
}

//! Gradient boosting trainer for linear models.
//!
//! Implements coordinate descent training for GBLinear models.
//! Supports single-output and multi-output objectives.

use ndarray::Array2;

use crate::data::init_predictions;
use crate::data::{Dataset, TargetsView, WeightsView};
use crate::repr::gblinear::LinearModel;
use crate::training::eval;
use crate::training::{
    EarlyStopAction, EarlyStopping, Gradients, MetricFn, ObjectiveFn, TrainingLogger,
    Verbosity,
};

use super::selector::{FeatureSelector, FeatureSelectorKind};
use super::updater::{compute_weight_update, UpdateConfig, Updater, UpdateStrategy};

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
    /// Coordinate descent update strategy.
    pub update_strategy: UpdateStrategy,

    /// Feature selection strategy for coordinate descent.
    pub feature_selector: FeatureSelectorKind,

    /// Random seed for feature shuffling.
    pub seed: u64,

    /// Maximum per-coordinate Newton step (stability), in absolute value.
    ///
    /// Set to `0.0` to disable.
    pub max_delta_step: f32,

    // --- Evaluation and early stopping ---
    /// Early stopping rounds. Training stops if no improvement for this many rounds.
    /// Set to 0 to disable.
    pub early_stopping_rounds: u32,

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
            update_strategy: UpdateStrategy::default(),
            feature_selector: FeatureSelectorKind::default(),
            seed: 42,
            max_delta_step: 0.0,
            early_stopping_rounds: 0,
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
    /// * `train` - Training dataset (must have numeric features only)
    /// * `targets` - Training targets
    /// * `weights` - Optional sample weights
    /// * `val_set` - Optional validation set for early stopping
    ///
    /// # Returns
    ///
    /// Returns `None` if:
    /// - Dataset contains categorical features (not supported by GBLinear)
    /// - Validation set has categorical features
    pub fn train(
        &self,
        train: &Dataset,
        targets: TargetsView<'_>,
        weights: WeightsView<'_>,
        val_set: Option<&Dataset>,
    ) -> Option<LinearModel> {
        // Validate: GBLinear doesn't support categorical features
        if train.has_categorical() {
            return None;
        }
        if let Some(vs) = val_set {
            if vs.has_categorical() {
                return None;
            }
        }

        let n_features = train.n_features();
        let n_samples = train.n_samples();
        let n_outputs = self.objective.n_outputs();

        assert!(
            n_outputs >= 1,
            "Objective must have at least 1 output, got {}",
            n_outputs
        );
        debug_assert_eq!(targets.n_samples(), n_samples);
        debug_assert!(
            weights.is_none() || weights.as_array().is_some_and(|w| w.len() == n_samples)
        );

        // Compute base scores using objective
        let base_scores = self.objective.compute_base_score(targets, weights);

        // Initialize model with base scores as biases
        let mut model = LinearModel::zeros(n_features, n_outputs);
        for (group, &base_score) in base_scores.iter().enumerate() {
            model.set_bias(group, base_score);
        }

        // Create updater and selector
        let mut selector = self.params.feature_selector.create_state(self.params.seed);

        let update_config = UpdateConfig {
            alpha: self.params.alpha,
            lambda: self.params.lambda,
            learning_rate: self.params.learning_rate,
            max_delta_step: self.params.max_delta_step,
        };
        let updater = Updater::new(self.params.update_strategy, update_config.clone());

        // Gradient and prediction buffers
        let mut gradients = Gradients::new(n_samples, n_outputs);
        // Initialize predictions with base scores using helper
        // Shape: [n_outputs, n_samples] - column-major for efficient group access
        let mut predictions = init_predictions(&base_scores, n_samples);

        // Check if we need evaluation (metric is enabled)
        let needs_evaluation = self.metric.is_enabled();

        // Initialize validation predictions with base scores
        // Shape: [n_outputs, n_val_samples]
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
        let mut best_model: Option<LinearModel> = None;

        // Logger
        let mut logger = TrainingLogger::new(self.params.verbosity);
        logger.start_training(self.params.n_rounds as usize);

        // Evaluator for computing metrics
        let mut evaluator = eval::Evaluator::new(&self.objective, &self.metric, n_outputs);

        // Training loop
        for round in 0..self.params.n_rounds {
            // NOTE: We don't call predict_col_major() here anymore!
            // Predictions are maintained incrementally by applying weight deltas.
            // On first round (round == 0), predictions are already initialized with base scores.

            // Compute gradients from current predictions
            self.objective.compute_gradients_into(
                predictions.view(),
                targets,
                weights,
                gradients.pairs_array_mut(),
            );

            // Update each output
            for output in 0..n_outputs {
                let bias_delta = updater.update_bias(&mut model, &gradients, output);

                // Apply bias delta to predictions incrementally
                if bias_delta.abs() > 1e-10 {
                    updater.apply_bias_delta_to_predictions(
                        bias_delta,
                        output,
                        predictions.view_mut(),
                    );

                    // Also update validation predictions (only if evaluation is needed)
                    if let Some(ref mut vp) = val_predictions {
                        updater.apply_bias_delta_to_predictions(
                            bias_delta,
                            output,
                            vp.view_mut(),
                        );
                    }

                    // Recompute gradients after bias update
                    self.objective.compute_gradients_into(
                        predictions.view(),
                        targets,
                        weights,
                        gradients.pairs_array_mut(),
                    );
                }

                selector.setup_round(
                    &model,
                    train,
                    &gradients,
                    output,
                    self.params.alpha,
                    self.params.lambda,
                );

                // True coordinate descent for the Sequential strategy.
                //
                // Note: This recomputes gradients after each coordinate update. It's slower,
                // but can be substantially more stable/accurate for some objectives.
                if self.params.update_strategy == UpdateStrategy::Sequential {
                    while let Some(feature) = selector.next() {
                        let delta = compute_weight_update(
                            &model,
                            train,
                            &gradients,
                            feature,
                            output,
                            &update_config,
                        );

                        if delta.abs() <= 1e-10 {
                            continue;
                        }

                        model.add_weight(feature, output, delta);

                        // Incrementally update train predictions for this feature.
                        {
                            let mut output_row = predictions.row_mut(output);
                            train.for_each_feature_value(feature, |row, value| {
                                output_row[row] += value * delta;
                            });
                        }

                        // Keep validation predictions in sync (only if evaluation is needed).
                        if let (Some(dataset_val), Some(ref mut predictions_val)) =
                            (val_set, val_predictions.as_mut())
                        {
                            let mut output_row = predictions_val.row_mut(output);
                            dataset_val.for_each_feature_value(feature, |row, value| {
                                output_row[row] += value * delta;
                            });
                        }

                        // Recompute gradients after each coordinate update.
                        self.objective.compute_gradients_into(
                            predictions.view(),
                            targets,
                            weights,
                            gradients.pairs_array_mut(),
                        );
                    }
                } else {
                    let weight_deltas = updater.update_round(
                        &mut model,
                        train,
                        &gradients,
                        &mut selector,
                        output,
                    );

                    // Apply weight deltas to predictions incrementally
                    if !weight_deltas.is_empty() {
                        updater.apply_weight_deltas_to_predictions(
                            train,
                            &weight_deltas,
                            output,
                            predictions.view_mut(),
                        );

                        // Keep validation predictions in sync for correct metrics/early stopping.
                        if let (Some(dataset_val), Some(ref mut predictions_val)) =
                            (val_set, val_predictions.as_mut())
                        {
                            updater.apply_weight_deltas_to_predictions(
                                dataset_val,
                                &weight_deltas,
                                output,
                                predictions_val.view_mut(),
                            );
                        }
                    }
                }
            }

            // Evaluation using Evaluator (only if metric is enabled)
            let (round_metrics, early_stop_value) = if needs_evaluation {
                let metrics = evaluator.evaluate_round(
                    predictions.view(),
                    targets,
                    weights,
                    val_set,
                    val_predictions.as_ref().map(|p| p.view()),
                );
                let value = eval::Evaluator::<O, M>::early_stop_value(
                    &metrics,
                    val_set.is_some(),
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
    use crate::data::{transpose_to_c_order, Dataset, TargetsView, WeightsView};
    use crate::training::{
        LogLoss, LogisticLoss, MulticlassLogLoss, Rmse, SoftmaxLoss, SquaredLoss,
    };
    use ndarray::{Array2, array};

    /// Helper to create a Dataset and TargetsView from row-major feature data.
    /// Accepts features in [n_samples, n_features] layout (standard user format)
    /// and targets in [n_samples, n_outputs], then converts to feature-major internally.
    /// Returns (Dataset, targets_array) where targets_array is in [n_outputs, n_samples].
    fn make_dataset(features: Array2<f32>, targets: Array2<f32>) -> (Dataset, Array2<f32>) {
        // Transpose features to [n_features, n_samples] (feature-major)
        let features_fm = transpose_to_c_order(features.view());
        // Transpose targets to [n_outputs, n_samples]
        let targets_fm = transpose_to_c_order(targets.view());

        let dataset = Dataset::from_array(features_fm.view(), Some(targets_fm.clone()), None);

        (dataset, targets_fm)
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
        // 4 samples, 1 feature - [n_samples, n_features]
        let features = array![[1.0], [2.0], [3.0], [4.0]]; // [4, 1]
        let targets = array![[3.0], [5.0], [7.0], [9.0]]; // [4, 1]
        let (train, targets_fm) = make_dataset(features, targets);
        let targets_view = TargetsView::new(targets_fm.view());

        let params = GBLinearParams {
            n_rounds: 100,
            learning_rate: 0.5,
            alpha: 0.0,
            lambda: 0.0,
            update_strategy: UpdateStrategy::Sequential,
            verbosity: Verbosity::Silent,
            ..Default::default()
        };

        let trainer = GBLinearTrainer::new(SquaredLoss, Rmse, params);
        let model = trainer
            .train(&train, targets_view, WeightsView::None, None)
            .unwrap();

        // Check predictions
        let mut output = [0.0f32; 1];
        model.predict_row_into(&[1.0], &mut output);
        let pred1 = output[0];
        model.predict_row_into(&[2.0], &mut output);
        let pred2 = output[0];

        assert!((pred1 - 3.0).abs() < 0.5);
        assert!((pred2 - 5.0).abs() < 0.5);
    }

    #[test]
    fn train_with_regularization() {
        // 4 samples, 1 feature
        let features = array![[1.0], [2.0], [3.0], [4.0]]; // [4, 1]
        let targets = array![[3.0], [5.0], [7.0], [9.0]]; // [4, 1]
        let (train, targets_fm) = make_dataset(features, targets);
        let targets_view = TargetsView::new(targets_fm.view());

        // Train without regularization
        let params_no_reg = GBLinearParams {
            n_rounds: 50,
            lambda: 0.0,
            update_strategy: UpdateStrategy::Sequential,
            verbosity: Verbosity::Silent,
            ..Default::default()
        };
        let trainer_no_reg = GBLinearTrainer::new(SquaredLoss, Rmse, params_no_reg);
        let model_no_reg = trainer_no_reg
            .train(&train, targets_view.clone(), WeightsView::None, None)
            .unwrap();

        // Train with L2 regularization
        let params_l2 = GBLinearParams {
            n_rounds: 50,
            lambda: 10.0,
            update_strategy: UpdateStrategy::Sequential,
            verbosity: Verbosity::Silent,
            ..Default::default()
        };
        let trainer_l2 = GBLinearTrainer::new(SquaredLoss, Rmse, params_l2);
        let model_l2 = trainer_l2
            .train(&train, targets_view, WeightsView::None, None)
            .unwrap();

        // L2 should produce smaller weights
        let w_no_reg = model_no_reg.weight(0, 0).abs();
        let w_l2 = model_l2.weight(0, 0).abs();
        assert!(w_l2 < w_no_reg);
    }

    #[test]
    fn train_multifeature() {
        // y = x0 + 2*x1
        // 4 samples, 2 features - [n_samples, n_features]
        let features = array![
            [1.0, 1.0], // y=3
            [2.0, 1.0], // y=4
            [1.0, 2.0], // y=5
            [2.0, 2.0], // y=6
        ];
        let targets = array![[3.0], [4.0], [5.0], [6.0]];
        let (train, targets_fm) = make_dataset(features, targets);
        let targets_view = TargetsView::new(targets_fm.view());

        let params = GBLinearParams {
            n_rounds: 200,
            learning_rate: 0.3,
            lambda: 0.0,
            update_strategy: UpdateStrategy::Sequential,
            verbosity: Verbosity::Silent,
            ..Default::default()
        };

        let trainer = GBLinearTrainer::new(SquaredLoss, Rmse, params);
        let model = trainer
            .train(&train, targets_view, WeightsView::None, None)
            .unwrap();

        let w0 = model.weight(0, 0);
        let w1 = model.weight(1, 0);

        // w0 should be ~1, w1 should be ~2
        assert!((w0 - 1.0).abs() < 0.3);
        assert!((w1 - 2.0).abs() < 0.3);
    }

    #[test]
    fn train_multiclass() {
        // Simple 3-class classification
        // [n_samples, n_features] layout
        let features = array![
            [2.0, 1.0], // Class 0
            [0.0, 1.0], // Class 1
            [3.0, 1.0], // Class 0
            [1.0, 3.0], // Class 2
            [0.5, 0.5], // Class 1
            [2.0, 2.0], // Class 2
        ];
        let targets = array![[0.0], [1.0], [0.0], [2.0], [1.0], [2.0]];
        let (train, targets_fm) = make_dataset(features, targets);
        let targets_view = TargetsView::new(targets_fm.view());

        let params = GBLinearParams {
            n_rounds: 200,
            learning_rate: 0.3,
            lambda: 0.1,
            update_strategy: UpdateStrategy::Sequential,
            verbosity: Verbosity::Silent,
            ..Default::default()
        };

        let trainer = GBLinearTrainer::new(SoftmaxLoss::new(3), MulticlassLogLoss, params);
        let model = trainer
            .train(&train, targets_view, WeightsView::None, None)
            .unwrap();

        // Model should have 3 output groups
        assert_eq!(model.n_groups(), 3);

        // Verify model produces different outputs for different classes
        let mut preds0 = [0.0f32; 3];
        let mut preds1 = [0.0f32; 3];
        model.predict_row_into(&[2.0, 1.0], &mut preds0);
        model.predict_row_into(&[0.0, 1.0], &mut preds1);

        let diff: f32 = preds0
            .iter()
            .zip(preds1.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.1);
    }

    #[test]
    fn train_binary_classification() {
        // [n_samples, n_features] layout
        let features = array![
            [0.0, 1.0], // Class 0
            [1.0, 0.0], // Class 1
            [0.5, 1.0], // Class 0
            [1.0, 0.5], // Class 1
        ];
        let targets = array![[0.0], [1.0], [0.0], [1.0]];
        let (train, targets_fm) = make_dataset(features, targets);
        let targets_view = TargetsView::new(targets_fm.view());

        let params = GBLinearParams {
            n_rounds: 50,
            verbosity: Verbosity::Silent,
            ..Default::default()
        };

        let trainer = GBLinearTrainer::new(LogisticLoss, LogLoss, params);
        let model = trainer
            .train(&train, targets_view, WeightsView::None, None)
            .unwrap();

        assert_eq!(model.n_groups(), 1);
    }
}

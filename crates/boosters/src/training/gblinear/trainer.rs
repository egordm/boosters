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
    EarlyStopAction, EarlyStopping, Gradients, Metric, Objective, TrainingLogger, Verbosity,
};

use super::GBLinearTrainError;
use super::selector::FeatureSelectorKind;
use super::updater::{
    UpdateStrategy, Updater, apply_bias_delta_to_predictions, apply_weight_deltas_to_predictions,
};

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
            lambda: 0.0,
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
#[derive(Debug)]
pub struct GBLinearTrainer {
    objective: Objective,
    metric: Metric,
    params: GBLinearParams,
    updater: Updater,
}

impl GBLinearTrainer {
    /// Create a new trainer with the given objective and parameters.
    pub fn new(objective: Objective, metric: Metric, params: GBLinearParams) -> Self {
        let updater = Updater::new(
            params.update_strategy,
            params.alpha,
            params.lambda,
            params.learning_rate,
            params.max_delta_step,
        );

        Self {
            objective,
            metric,
            params,
            updater,
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
    /// Returns an error if:
    /// - Dataset contains categorical features (not supported by GBLinear)
    /// - Targets contain NaNs
    /// - Weights contain NaNs
    /// - Validation set has categorical features, NaN targets/weights, or is missing targets
    pub fn train(
        &self,
        train: &Dataset,
        targets: TargetsView<'_>,
        weights: WeightsView<'_>,
        val_set: Option<&Dataset>,
    ) -> Result<LinearModel, GBLinearTrainError> {
        // Validate: GBLinear doesn't support categorical features
        if train.has_categorical() {
            return Err(GBLinearTrainError::CategoricalFeaturesNotSupported);
        }
        if let Some(vs) = val_set
            && vs.has_categorical()
        {
            return Err(GBLinearTrainError::CategoricalFeaturesNotSupported);
        }

        // Feature NaNs are allowed and treated as missing (0 contribution), matching XGBoost gblinear.
        validate_no_nans_in_targets(targets, "train")?;
        validate_no_nans_in_weights(weights, "train")?;

        if let Some(vs) = val_set {
            let vs_targets = vs
                .targets()
                .ok_or(GBLinearTrainError::ValidationSetMissingTargets)?;
            validate_no_nans_in_targets(vs_targets, "valid")?;
            validate_no_nans_in_weights(vs.weights(), "valid")?;
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

        // Runtime context (data/weights dependent)
        let sum_instance_weight: f32 = weights.iter(n_samples).sum();

        // Gradient and prediction buffers
        let mut gradients = Gradients::new(n_samples, n_outputs);
        // Initialize predictions with base scores using helper
        // Shape: [n_outputs, n_samples] - column-major for efficient group access
        let mut predictions = init_predictions(&base_scores, n_samples);

        // Check if we need evaluation
        let needs_evaluation = !matches!(&self.metric, Metric::None);

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
        let mut evaluator =
            eval::Evaluator::new(&self.metric, self.objective.output_transform(), n_outputs);

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
                // Compute denormalized regularization once per output (see updater docs)
                let reg = self
                    .updater
                    .regularization()
                    .denormalize(sum_instance_weight);

                let bias_delta = {
                    let weights_and_bias = model.weights_and_bias_mut(output);
                    let grad_pairs = gradients.output_pairs_mut(output);
                    let predictions_row = predictions.row_mut(output);
                    self.updater
                        .update_bias_inplace(weights_and_bias, grad_pairs, predictions_row)
                };

                if bias_delta.abs() > 1e-10
                    && let Some(ref mut vp) = val_predictions
                {
                    apply_bias_delta_to_predictions(bias_delta, vp.row_mut(output));
                }

                selector.setup_round(
                    model.weights_and_bias(output),
                    train,
                    gradients.output_pairs(output),
                    reg,
                );

                let weight_deltas = {
                    let weights_and_bias = model.weights_and_bias_mut(output);
                    let grad_pairs = gradients.output_pairs_mut(output);
                    let predictions_row = predictions.row_mut(output);
                    self.updater.update_round_inplace(
                        train,
                        weights_and_bias,
                        grad_pairs,
                        &mut selector,
                        reg,
                        predictions_row,
                    )
                };

                // Keep validation predictions in sync for correct metrics/early stopping.
                if !weight_deltas.is_empty()
                    && let (Some(dataset_val), Some(ref mut predictions_val)) =
                        (val_set, val_predictions.as_mut())
                {
                    apply_weight_deltas_to_predictions(
                        dataset_val,
                        &weight_deltas,
                        predictions_val.row_mut(output),
                    );
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
        if early_stopping.is_enabled() {
            if let Some(best_model) = best_model {
                Ok(best_model)
            } else {
                Ok(model)
            }
        } else {
            Ok(model)
        }
    }
}

fn validate_no_nans_in_targets(
    targets: TargetsView<'_>,
    dataset_name: &'static str,
) -> Result<(), GBLinearTrainError> {
    if targets.view().iter().any(|v| v.is_nan()) {
        return Err(GBLinearTrainError::NaNInTargets {
            dataset: dataset_name,
        });
    }
    Ok(())
}

fn validate_no_nans_in_weights(
    weights: WeightsView<'_>,
    dataset_name: &'static str,
) -> Result<(), GBLinearTrainError> {
    if let Some(w) = weights.as_array()
        && w.iter().any(|v| v.is_nan())
    {
        return Err(GBLinearTrainError::NaNInWeights {
            dataset: dataset_name,
        });
    }
    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{Dataset, TargetsView, WeightsView, transpose_to_c_order};
    use crate::training::{Metric, Objective};
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
        assert_eq!(params.lambda, 0.0);
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

        let trainer = GBLinearTrainer::new(Objective::SquaredLoss, Metric::Rmse, params);
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
        let trainer_no_reg =
            GBLinearTrainer::new(Objective::SquaredLoss, Metric::Rmse, params_no_reg);
        let model_no_reg = trainer_no_reg
            .train(&train, targets_view, WeightsView::None, None)
            .unwrap();

        // Train with L2 regularization
        let params_l2 = GBLinearParams {
            n_rounds: 50,
            lambda: 10.0,
            update_strategy: UpdateStrategy::Sequential,
            verbosity: Verbosity::Silent,
            ..Default::default()
        };
        let trainer_l2 = GBLinearTrainer::new(Objective::SquaredLoss, Metric::Rmse, params_l2);
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

        let trainer = GBLinearTrainer::new(Objective::SquaredLoss, Metric::Rmse, params);
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

        let trainer = GBLinearTrainer::new(
            Objective::SoftmaxLoss { n_classes: 3 },
            Metric::MulticlassLogLoss,
            params,
        );
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

        let trainer = GBLinearTrainer::new(Objective::LogisticLoss, Metric::LogLoss, params);
        let model = trainer
            .train(&train, targets_view, WeightsView::None, None)
            .unwrap();

        assert_eq!(model.n_groups(), 1);
    }

    #[test]
    fn train_allows_nan_features_as_missing() {
        // One feature regression; treat NaN as 0 contribution.
        // y = 2*x + 1, with one missing x.
        let features = array![[1.0], [2.0], [f32::NAN], [4.0]]; // [n_samples=4, n_features=1]
        let targets = array![[3.0], [5.0], [1.0], [9.0]]; // For NaN row: x treated as 0 => y=1
        let (train, targets_fm) = make_dataset(features, targets);
        let targets_view = TargetsView::new(targets_fm.view());

        let params = GBLinearParams {
            n_rounds: 50,
            learning_rate: 0.5,
            alpha: 0.0,
            lambda: 0.0,
            update_strategy: UpdateStrategy::Sequential,
            verbosity: Verbosity::Silent,
            ..Default::default()
        };

        let trainer = GBLinearTrainer::new(Objective::SquaredLoss, Metric::Rmse, params);
        let model = trainer
            .train(&train, targets_view, WeightsView::None, None)
            .unwrap();

        let mut output = [0.0f32; 1];
        model.predict_row_into(&[f32::NAN], &mut output);
        assert!(output[0].is_finite());
    }
}

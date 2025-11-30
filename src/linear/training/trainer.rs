//! High-level linear model trainer.

use crate::data::{CSCMatrix, DataMatrix, RowMatrix, RowView};
use crate::linear::LinearModel;
use crate::training::{GradientPair, Loss, TrainingLogger, Verbosity};

use super::selector::ShuffleSelector;
use super::updater::{update_bias, CoordinateUpdater, ShotgunUpdater, UpdateConfig, Updater};

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
    /// Early stopping rounds (0 = disabled).
    pub early_stopping_rounds: usize,
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
            early_stopping_rounds: 0,
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

    /// Train a linear model.
    ///
    /// # Arguments
    ///
    /// * `train_data` - Training features
    /// * `train_labels` - Training labels
    /// * `loss` - Loss function for gradient computation
    /// * `num_groups` - Number of output groups (1 for regression, K for K-class)
    ///
    /// # Returns
    ///
    /// Trained `LinearModel`.
    pub fn train<M, L>(
        &self,
        train_data: &M,
        train_labels: &[f32],
        loss: &L,
        num_groups: usize,
    ) -> LinearModel
    where
        M: DataMatrix<Element = f32> + Sync,
        L: Loss,
    {
        self.train_with_validation(train_data, train_labels, None, None, loss, num_groups)
    }

    /// Train a linear model with validation data.
    ///
    /// # Arguments
    ///
    /// * `train_data` - Training features
    /// * `train_labels` - Training labels
    /// * `val_data` - Optional validation features
    /// * `val_labels` - Optional validation labels
    /// * `loss` - Loss function
    /// * `num_groups` - Number of output groups
    pub fn train_with_validation<M, L>(
        &self,
        train_data: &M,
        train_labels: &[f32],
        val_data: Option<&M>,
        val_labels: Option<&[f32]>,
        loss: &L,
        num_groups: usize,
    ) -> LinearModel
    where
        M: DataMatrix<Element = f32> + Sync,
        L: Loss,
    {
        let num_features = train_data.num_features();
        let num_samples = train_data.num_rows();

        assert_eq!(
            train_labels.len(),
            num_samples,
            "Labels length must match number of samples"
        );

        // Initialize model
        let mut model = LinearModel::zeros(num_features, num_groups);

        // Convert to CSC for efficient column access
        let csc = self.to_csc(train_data);

        // Create updater and selector
        let updater: Box<dyn Updater> = if self.config.parallel {
            Box::new(ShotgunUpdater::new())
        } else {
            Box::new(CoordinateUpdater::new())
        };

        let mut selector = ShuffleSelector::new(self.config.seed);

        // Create update config
        let update_config = UpdateConfig {
            alpha: self.config.alpha,
            lambda: self.config.lambda,
            learning_rate: self.config.learning_rate,
        };

        // Gradient storage
        let mut gradients = vec![GradientPair::ZERO; num_samples];

        // Predictions buffer
        let mut predictions = vec![0.0f32; num_samples * num_groups];

        // Logger
        let mut logger = TrainingLogger::new(self.config.verbosity);
        logger.start_training(self.config.num_rounds);

        // Early stopping (using train loss as metric if no validation)
        let use_early_stopping =
            self.config.early_stopping_rounds > 0 && val_data.is_some() && val_labels.is_some();

        // Training loop
        for round in 0..self.config.num_rounds {
            // Compute predictions
            self.compute_predictions(&model, train_data, &mut predictions);

            // Compute gradients
            self.compute_gradients(&predictions, train_labels, loss, num_groups, &mut gradients);

            // Update each output group
            for group in 0..num_groups {
                // Update bias (no regularization)
                update_bias(&mut model, &gradients, group, self.config.learning_rate);

                // Update feature weights
                updater.update_round(
                    &mut model,
                    &csc,
                    &gradients,
                    &mut selector,
                    group,
                    &update_config,
                );
            }

            // Logging
            if self.config.verbosity >= Verbosity::Info {
                let train_loss = self.compute_loss(&predictions, train_labels, num_groups);
                let mut metrics = vec![("train_loss".to_string(), train_loss)];

                if let (Some(vd), Some(vl)) = (val_data, val_labels) {
                    let mut val_preds = vec![0.0f32; vd.num_rows() * num_groups];
                    self.compute_predictions(&model, vd, &mut val_preds);
                    let val_loss = self.compute_loss(&val_preds, vl, num_groups);
                    metrics.push(("val_loss".to_string(), val_loss));
                }

                logger.log_round(round, &metrics);
            }

            // Early stopping check
            if use_early_stopping {
                // TODO: Implement proper early stopping with validation metric
                // For now, we just train for num_rounds
            }
        }

        logger.finish_training();
        model
    }

    /// Convert data matrix to CSC format.
    fn to_csc<M: DataMatrix<Element = f32>>(&self, data: &M) -> CSCMatrix<f32> {
        // First convert to dense, then to CSC
        let num_rows = data.num_rows();
        let num_features = data.num_features();

        let mut dense_data = vec![0.0f32; num_rows * num_features];
        for row_idx in 0..num_rows {
            let row = data.row(row_idx);
            for feat_idx in 0..num_features {
                dense_data[row_idx * num_features + feat_idx] = row.get(feat_idx).unwrap_or(0.0);
            }
        }

        let dense = RowMatrix::from_vec(dense_data, num_rows, num_features);
        CSCMatrix::from_dense_full(&dense)
    }

    /// Compute predictions for all samples.
    fn compute_predictions<M: DataMatrix<Element = f32>>(
        &self,
        model: &LinearModel,
        data: &M,
        output: &mut [f32],
    ) {
        let num_rows = data.num_rows();
        let num_groups = model.num_groups();
        let num_features = model.num_features();

        for row_idx in 0..num_rows {
            let row = data.row(row_idx);
            for group in 0..num_groups {
                let mut sum = model.bias(group);
                for feat_idx in 0..num_features {
                    let value = row.get(feat_idx).unwrap_or(0.0);
                    sum += value * model.weight(feat_idx, group);
                }
                output[row_idx * num_groups + group] = sum;
            }
        }
    }

    /// Compute gradients from predictions and labels.
    fn compute_gradients<L: Loss>(
        &self,
        predictions: &[f32],
        labels: &[f32],
        loss: &L,
        num_groups: usize,
        gradients: &mut [GradientPair],
    ) {
        // For single-output (regression, binary), just compute gradients directly
        if num_groups == 1 {
            for (i, (pred, label)) in predictions.iter().zip(labels.iter()).enumerate() {
                gradients[i] = loss.compute_gradient(*pred, *label);
            }
        } else {
            // For multiclass, we need to handle differently
            // For now, use first group's prediction vs label
            // TODO: Proper multiclass gradient handling
            for (i, label) in labels.iter().enumerate() {
                let pred = predictions[i * num_groups]; // Use first group
                gradients[i] = loss.compute_gradient(pred, *label);
            }
        }
    }

    /// Compute average loss.
    fn compute_loss(&self, predictions: &[f32], labels: &[f32], num_groups: usize) -> f64 {
        if num_groups == 1 {
            // Simple MSE for regression
            let mse: f64 = predictions
                .iter()
                .zip(labels.iter())
                .map(|(p, l)| {
                    let diff = (*p as f64) - (*l as f64);
                    diff * diff
                })
                .sum::<f64>()
                / predictions.len() as f64;
            mse.sqrt() // RMSE
        } else {
            // For multiclass, compute accuracy or cross-entropy
            // Simplified: just return RMSE of first group
            let n = labels.len();
            let mse: f64 = (0..n)
                .map(|i| {
                    let p = predictions[i * num_groups] as f64;
                    let l = labels[i] as f64;
                    (p - l) * (p - l)
                })
                .sum::<f64>()
                / n as f64;
            mse.sqrt()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::SquaredLoss;

    #[test]
    fn train_simple_regression() {
        // y = 2*x + 1
        let train_data = RowMatrix::from_vec(
            vec![
                1.0, // x=1 → y=3
                2.0, // x=2 → y=5
                3.0, // x=3 → y=7
                4.0, // x=4 → y=9
            ],
            4,
            1,
        );
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
        let model = trainer.train(&train_data, &train_labels, &SquaredLoss, 1);

        // Check predictions
        let pred1 = model.predict_row(&[1.0], &[0.0])[0];
        let pred2 = model.predict_row(&[2.0], &[0.0])[0];

        // Should be close to true values
        assert!((pred1 - 3.0).abs() < 0.5);
        assert!((pred2 - 5.0).abs() < 0.5);
    }

    #[test]
    fn train_with_regularization() {
        let train_data = RowMatrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 4, 1);
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
            1,
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
            LinearTrainer::new(config_l2).train(&train_data, &train_labels, &SquaredLoss, 1);

        // L2 should produce smaller weights
        let w_no_reg = model_no_reg.weight(0, 0).abs();
        let w_l2 = model_l2.weight(0, 0).abs();
        assert!(w_l2 < w_no_reg);
    }

    #[test]
    fn train_multifeature() {
        // y = x0 + 2*x1
        let train_data = RowMatrix::from_vec(
            vec![
                1.0, 1.0, // x0=1, x1=1 → y=3
                2.0, 1.0, // x0=2, x1=1 → y=4
                1.0, 2.0, // x0=1, x1=2 → y=5
                2.0, 2.0, // x0=2, x1=2 → y=6
            ],
            4,
            2,
        );
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
        let model = trainer.train(&train_data, &train_labels, &SquaredLoss, 1);

        // Check weights are roughly correct
        let w0 = model.weight(0, 0);
        let w1 = model.weight(1, 0);

        // w0 should be ~1, w1 should be ~2
        assert!((w0 - 1.0).abs() < 0.3);
        assert!((w1 - 2.0).abs() < 0.3);
    }

    #[test]
    fn parallel_vs_sequential_similar() {
        let train_data = RowMatrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 4, 1);
        let train_labels = vec![3.0, 5.0, 7.0, 9.0];

        let config_seq = LinearTrainerConfig {
            num_rounds: 50,
            parallel: false,
            verbosity: Verbosity::Silent,
            ..Default::default()
        };
        let model_seq =
            LinearTrainer::new(config_seq).train(&train_data, &train_labels, &SquaredLoss, 1);

        let config_par = LinearTrainerConfig {
            num_rounds: 50,
            parallel: true,
            verbosity: Verbosity::Silent,
            ..Default::default()
        };
        let model_par =
            LinearTrainer::new(config_par).train(&train_data, &train_labels, &SquaredLoss, 1);

        // Results should be similar (not identical due to race conditions in shotgun)
        let pred_seq = model_seq.predict_row(&[2.5], &[0.0])[0];
        let pred_par = model_par.predict_row(&[2.5], &[0.0])[0];

        assert!((pred_seq - pred_par).abs() < 1.0);
    }
}

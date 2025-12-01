//! High-level linear model trainer.
//!
//! Uses column-based access ([`ColumnAccess`]) for efficient coordinate descent.
//! Accepts data in any format that implements `ColumnAccess`:
//!
//! - [`ColMatrix`]: Best for dense data (columns are contiguous)
//! - [`CSCMatrix`]: Best for sparse data (only stores non-zeros)
//!
//! For row-major input, convert to column-major first:
//! ```ignore
//! let col_matrix: ColMatrix = row_matrix.to_layout();
//! trainer.train(&col_matrix, labels, loss, 1);
//! ```

use crate::data::ColumnAccess;
use crate::linear::LinearModel;
use crate::training::{GradientPair, Loss, TrainingLogger, Verbosity};

use super::selector::ShuffleSelector;
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
    /// * `num_groups` - Number of output groups (1 for regression, K for K-class)
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
    /// let model = trainer.train(&col_data, &labels, &SquaredLoss, 1);
    /// ```
    pub fn train<C, L>(
        &self,
        train_data: &C,
        train_labels: &[f32],
        loss: &L,
        num_groups: usize,
    ) -> LinearModel
    where
        C: ColumnAccess<Element = f32> + Sync,
        L: Loss,
    {
        self.train_with_validation::<C, C, L>(train_data, train_labels, None, None, loss, num_groups)
    }

    /// Train a linear model with validation data.
    ///
    /// # Arguments
    ///
    /// * `train_data` - Training features (must implement `ColumnAccess`)
    /// * `train_labels` - Training labels
    /// * `val_data` - Optional validation features (same type as train_data)
    /// * `val_labels` - Optional validation labels
    /// * `loss` - Loss function
    /// * `num_groups` - Number of output groups
    pub fn train_with_validation<C, V, L>(
        &self,
        train_data: &C,
        train_labels: &[f32],
        val_data: Option<&V>,
        val_labels: Option<&[f32]>,
        loss: &L,
        num_groups: usize,
    ) -> LinearModel
    where
        C: ColumnAccess<Element = f32> + Sync,
        V: ColumnAccess<Element = f32>,
        L: Loss,
    {
        let num_features = train_data.num_columns();
        let num_samples = train_data.num_rows();

        assert_eq!(
            train_labels.len(),
            num_samples,
            "Labels length must match number of samples"
        );

        // Initialize model
        let mut model = LinearModel::zeros(num_features, num_groups);

        // Create updater
        let updater = if self.config.parallel {
            UpdaterKind::Parallel
        } else {
            UpdaterKind::Sequential
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
            // Compute predictions using column access
            Self::compute_predictions_col(&model, train_data, &mut predictions);

            // Compute gradients
            Self::compute_gradients(&predictions, train_labels, loss, num_groups, &mut gradients);

            // Update each output group
            for group in 0..num_groups {
                // Update bias (no regularization)
                update_bias(&mut model, &gradients, group, self.config.learning_rate);

                // Update feature weights
                updater.update_round(
                    &mut model,
                    train_data,
                    &gradients,
                    &mut selector,
                    group,
                    &update_config,
                );
            }

            // Logging
            if self.config.verbosity >= Verbosity::Info {
                let train_loss = Self::compute_loss(&predictions, train_labels, num_groups);
                let mut metrics = vec![("train_loss".to_string(), train_loss)];

                if let (Some(vd), Some(vl)) = (val_data, val_labels) {
                    let mut val_preds = vec![0.0f32; vd.num_rows() * num_groups];
                    Self::compute_predictions_col(&model, vd, &mut val_preds);
                    let val_loss = Self::compute_loss(&val_preds, vl, num_groups);
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

    /// Compute predictions for all samples using column-based access.
    ///
    /// This is less efficient than row-based prediction but works with any
    /// `ColumnAccess` type without requiring `DataMatrix`.
    fn compute_predictions_col<C: ColumnAccess<Element = f32>>(
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

    /// Compute gradients from predictions and labels.
    fn compute_gradients<L: Loss>(
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
    fn compute_loss(predictions: &[f32], labels: &[f32], num_groups: usize) -> f64 {
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

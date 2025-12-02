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
//! For row-major input, convert to column-major first:
//! ```ignore
//! let col_matrix: ColMatrix = row_matrix.to_layout();
//! trainer.train(&col_matrix, labels, loss);
//! ```

use crate::data::ColumnAccess;
use crate::linear::LinearModel;
use crate::training::{GradientBuffer, Loss, MulticlassLoss, TrainingLogger, Verbosity};

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
        self.train_with_validation::<C, C, L>(train_data, train_labels, None, None, loss)
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
    pub fn train_with_validation<C, V, L>(
        &self,
        train_data: &C,
        train_labels: &[f32],
        val_data: Option<&V>,
        val_labels: Option<&[f32]>,
        loss: &L,
    ) -> LinearModel
    where
        C: ColumnAccess<Element = f32> + Sync,
        V: ColumnAccess<Element = f32>,
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

        let mut selector = ShuffleSelector::new(self.config.seed);

        // Create update config
        let update_config = UpdateConfig {
            alpha: self.config.alpha,
            lambda: self.config.lambda,
            learning_rate: self.config.learning_rate,
        };

        // SoA gradient storage
        let mut gradients = GradientBuffer::new(num_samples, num_outputs);

        // Predictions buffer
        let mut predictions = vec![0.0f32; num_samples];

        // Logger
        let mut logger = TrainingLogger::new(self.config.verbosity);
        logger.start_training(self.config.num_rounds);

        // Training loop
        for round in 0..self.config.num_rounds {
            // Compute predictions
            Self::compute_predictions_col(&model, train_data, &mut predictions);

            // Compute gradients using SoA buffer
            loss.compute_gradients(&predictions, train_labels, &mut gradients);

            // Update bias
            update_bias(&mut model, &gradients, 0, self.config.learning_rate);

            // Update feature weights
            updater.update_round(
                &mut model,
                train_data,
                &gradients,
                &mut selector,
                0, // output
                &update_config,
            );

            // Logging
            if self.config.verbosity >= Verbosity::Info {
                let train_loss = Self::compute_loss(&predictions, train_labels);
                let mut metrics = vec![("train_loss".to_string(), train_loss)];

                if let (Some(vd), Some(vl)) = (val_data, val_labels) {
                    let mut val_preds = vec![0.0f32; vd.num_rows()];
                    Self::compute_predictions_col(&model, vd, &mut val_preds);
                    let val_loss = Self::compute_loss(&val_preds, vl);
                    metrics.push(("val_loss".to_string(), val_loss));
                }

                logger.log_round(round, &metrics);
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
        self.train_multiclass_with_validation::<C, C, L>(train_data, train_labels, None, None, loss)
    }

    /// Train a multiclass linear model with validation data.
    ///
    /// # Arguments
    ///
    /// * `train_data` - Training features
    /// * `train_labels` - Class labels (0 to K-1)
    /// * `val_data` - Optional validation features
    /// * `val_labels` - Optional validation labels
    /// * `loss` - Multiclass loss function
    pub fn train_multiclass_with_validation<C, V, L>(
        &self,
        train_data: &C,
        train_labels: &[f32],
        val_data: Option<&V>,
        val_labels: Option<&[f32]>,
        loss: &L,
    ) -> LinearModel
    where
        C: ColumnAccess<Element = f32> + Sync,
        V: ColumnAccess<Element = f32>,
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

        let mut selector = ShuffleSelector::new(self.config.seed);

        // Create update config
        let update_config = UpdateConfig {
            alpha: self.config.alpha,
            lambda: self.config.lambda,
            learning_rate: self.config.learning_rate,
        };

        // SoA gradient storage: K outputs per sample
        let mut gradients = GradientBuffer::new(num_samples, num_outputs);

        // Predictions buffer: K predictions per sample
        let mut predictions = vec![0.0f32; num_samples * num_outputs];

        // Logger
        let mut logger = TrainingLogger::new(self.config.verbosity);
        logger.start_training(self.config.num_rounds);

        // Training loop
        for round in 0..self.config.num_rounds {
            // Compute predictions
            Self::compute_predictions_col_multiclass(&model, train_data, &mut predictions);

            // Compute multiclass gradients using SoA buffer
            loss.compute_gradients(&predictions, train_labels, &mut gradients);

            // Update each output (class)
            for output in 0..num_outputs {
                // Update bias
                update_bias(&mut model, &gradients, output, self.config.learning_rate);

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

            // Logging
            if self.config.verbosity >= Verbosity::Info {
                let train_acc =
                    Self::compute_multiclass_accuracy(&predictions, train_labels, num_outputs);
                let mut metrics = vec![("train_acc".to_string(), train_acc)];

                if let (Some(vd), Some(vl)) = (val_data, val_labels) {
                    let mut val_preds = vec![0.0f32; vd.num_rows() * num_outputs];
                    Self::compute_predictions_col_multiclass(&model, vd, &mut val_preds);
                    let val_acc = Self::compute_multiclass_accuracy(&val_preds, vl, num_outputs);
                    metrics.push(("val_acc".to_string(), val_acc));
                }

                logger.log_round(round, &metrics);
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

    /// Compute RMSE loss (single-output).
    fn compute_loss(predictions: &[f32], labels: &[f32]) -> f64 {
        let mse: f64 = predictions
            .iter()
            .zip(labels.iter())
            .map(|(p, l)| {
                let diff = (*p as f64) - (*l as f64);
                diff * diff
            })
            .sum::<f64>()
            / predictions.len() as f64;
        mse.sqrt()
    }

    /// Compute multiclass accuracy.
    fn compute_multiclass_accuracy(predictions: &[f32], labels: &[f32], num_groups: usize) -> f64 {
        let num_samples = labels.len();
        let mut correct = 0usize;

        for i in 0..num_samples {
            let preds_start = i * num_groups;
            let preds_end = preds_start + num_groups;
            let sample_preds = &predictions[preds_start..preds_end];

            // Find predicted class (argmax)
            let predicted_class = sample_preds
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            let true_class = labels[i] as usize;
            if predicted_class == true_class {
                correct += 1;
            }
        }

        correct as f64 / num_samples as f64
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
}

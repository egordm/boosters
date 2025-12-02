//! Evaluation metrics for model quality.
//!
//! Metrics are separate from loss functions — a model might be trained with
//! one loss but evaluated with different metrics.
//!
//! # Multi-Output Support
//!
//! Metrics support multi-output models (multiclass, multi-quantile) via the
//! `n_outputs` parameter. The predictions buffer has shape `[n_samples, n_outputs]`
//! in row-major order.
//!
//! # Evaluation Sets
//!
//! Use [`EvalSet`] to define named datasets for evaluation during training:
//!
//! ```ignore
//! let eval_sets = vec![
//!     EvalSet { name: "train", data: &train_data, labels: &train_labels },
//!     EvalSet { name: "val", data: &val_data, labels: &val_labels },
//! ];
//! ```

use crate::data::DataMatrix;

// =============================================================================
// Evaluation Set
// =============================================================================

/// Named dataset for evaluation during training.
///
/// Evaluation sets are used to track model performance on multiple datasets
/// (train, validation, test) during training. Each set has a name that appears
/// as a prefix in training logs.
///
/// # Example
///
/// ```ignore
/// let eval_sets = vec![
///     EvalSet { name: "train", data: &train_data, labels: &train_labels },
///     EvalSet { name: "val", data: &val_data, labels: &val_labels },
/// ];
/// // Logs: [0] train-rmse:15.23  val-rmse:16.12
/// ```
pub struct EvalSet<'a, D> {
    /// Dataset name (appears in logs as prefix, e.g., "train", "val", "test").
    pub name: &'a str,
    /// Feature matrix.
    pub data: &'a D,
    /// Labels (length = n_samples for single-output, or n_samples for multi-output
    /// where labels are class indices or target values).
    pub labels: &'a [f32],
}

impl<'a, D: DataMatrix> EvalSet<'a, D> {
    /// Create a new evaluation set.
    pub fn new(name: &'a str, data: &'a D, labels: &'a [f32]) -> Self {
        Self { name, data, labels }
    }
}

// =============================================================================
// Metric Trait
// =============================================================================

/// A metric for evaluating model quality.
///
/// Unlike [`super::Loss`] which computes gradients for optimization,
/// metrics compute scalar values for model evaluation and monitoring.
///
/// # Multi-Output Support
///
/// For multi-output models (multiclass, multi-quantile), pass `n_outputs > 1`.
/// The predictions buffer has shape `[n_samples, n_outputs]` in row-major order.
///
/// # Implementation Notes
///
/// - `evaluate`: Called with predictions, labels, and n_outputs
/// - Higher is better for some metrics (accuracy, AUC), lower for others (RMSE, logloss)
/// - Use `higher_is_better()` to determine the direction
pub trait Metric: Send + Sync {
    /// Evaluate the metric.
    ///
    /// # Arguments
    ///
    /// * `predictions` - Model predictions, shape `[n_samples, n_outputs]` flattened
    /// * `labels` - Ground truth labels, shape `[n_samples]`
    /// * `n_outputs` - Number of outputs per sample (1 for regression/binary, K for multiclass)
    ///
    /// # Returns
    ///
    /// Scalar metric value.
    fn evaluate(&self, predictions: &[f32], labels: &[f32], n_outputs: usize) -> f64;

    /// Whether higher values indicate better performance.
    ///
    /// - `true`: Higher is better (accuracy, AUC)
    /// - `false`: Lower is better (RMSE, MAE, logloss)
    fn higher_is_better(&self) -> bool;

    /// Name of the metric (for logging).
    fn name(&self) -> &str;
}

// Backward compatibility: implement evaluate via compute for single-output metrics
// This lets existing code work without changes

/// Helper trait for simple single-output metrics.
///
/// Implement this for metrics that only work with single-output models.
/// The `Metric` trait will be auto-implemented.
pub trait SimpleMetric: Send + Sync {
    /// Compute the metric value (single-output only).
    fn compute(&self, preds: &[f32], labels: &[f32]) -> f64;

    /// Whether higher values indicate better performance.
    fn higher_is_better(&self) -> bool;

    /// Name of the metric.
    fn name(&self) -> &str;
}

impl<T: SimpleMetric> Metric for T {
    fn evaluate(&self, predictions: &[f32], labels: &[f32], n_outputs: usize) -> f64 {
        if n_outputs == 1 {
            self.compute(predictions, labels)
        } else {
            // For multi-output, average metric across outputs (default behavior)
            // Specific metrics can override this
            self.compute(predictions, labels)
        }
    }

    fn higher_is_better(&self) -> bool {
        SimpleMetric::higher_is_better(self)
    }

    fn name(&self) -> &str {
        SimpleMetric::name(self)
    }
}

// =============================================================================
// RMSE (Root Mean Squared Error)
// =============================================================================

/// Root Mean Squared Error: sqrt(mean((pred - label)²))
///
/// Lower is better. Used for regression tasks.
#[derive(Debug, Clone, Copy, Default)]
pub struct Rmse;

impl SimpleMetric for Rmse {
    fn compute(&self, preds: &[f32], labels: &[f32]) -> f64 {
        debug_assert_eq!(preds.len(), labels.len());

        if preds.is_empty() {
            return 0.0;
        }

        let mse: f64 = preds
            .iter()
            .zip(labels.iter())
            .map(|(p, l)| {
                let diff = (*p as f64) - (*l as f64);
                diff * diff
            })
            .sum::<f64>()
            / preds.len() as f64;

        mse.sqrt()
    }

    fn higher_is_better(&self) -> bool {
        false
    }

    fn name(&self) -> &str {
        "rmse"
    }
}

// =============================================================================
// MAE (Mean Absolute Error)
// =============================================================================

/// Mean Absolute Error: mean(|pred - label|)
///
/// Lower is better. More robust to outliers than RMSE.
#[derive(Debug, Clone, Copy, Default)]
pub struct Mae;

impl SimpleMetric for Mae {
    fn compute(&self, preds: &[f32], labels: &[f32]) -> f64 {
        debug_assert_eq!(preds.len(), labels.len());

        if preds.is_empty() {
            return 0.0;
        }

        preds
            .iter()
            .zip(labels.iter())
            .map(|(p, l)| ((*p as f64) - (*l as f64)).abs())
            .sum::<f64>()
            / preds.len() as f64
    }

    fn higher_is_better(&self) -> bool {
        false
    }

    fn name(&self) -> &str {
        "mae"
    }
}

// =============================================================================
// LogLoss (Binary Cross-Entropy)
// =============================================================================

/// Binary cross-entropy: -mean(y*log(p) + (1-y)*log(1-p))
///
/// Lower is better. Used for binary classification.
/// Expects predictions to be probabilities in (0, 1).
#[derive(Debug, Clone, Copy, Default)]
pub struct LogLoss;

impl SimpleMetric for LogLoss {
    fn compute(&self, preds: &[f32], labels: &[f32]) -> f64 {
        debug_assert_eq!(preds.len(), labels.len());

        if preds.is_empty() {
            return 0.0;
        }

        let eps = 1e-15f64; // Clip to avoid log(0)

        preds
            .iter()
            .zip(labels.iter())
            .map(|(p, l)| {
                let p = (*p as f64).clamp(eps, 1.0 - eps);
                let l = *l as f64;
                -(l * p.ln() + (1.0 - l) * (1.0 - p).ln())
            })
            .sum::<f64>()
            / preds.len() as f64
    }

    fn higher_is_better(&self) -> bool {
        false
    }

    fn name(&self) -> &str {
        "logloss"
    }
}

// =============================================================================
// Accuracy
// =============================================================================

/// Classification accuracy: proportion of correct predictions.
///
/// Higher is better. For binary classification, uses 0.5 threshold.
/// For multiclass, expects predictions to be class indices.
#[derive(Debug, Clone, Copy)]
pub struct Accuracy {
    /// Threshold for binary classification (default: 0.5).
    pub threshold: f32,
}

impl Default for Accuracy {
    fn default() -> Self {
        Self { threshold: 0.5 }
    }
}

impl Accuracy {
    /// Create accuracy metric with custom threshold.
    pub fn with_threshold(threshold: f32) -> Self {
        Self { threshold }
    }
}

impl SimpleMetric for Accuracy {
    fn compute(&self, preds: &[f32], labels: &[f32]) -> f64 {
        debug_assert_eq!(preds.len(), labels.len());

        if preds.is_empty() {
            return 0.0;
        }

        let correct = preds
            .iter()
            .zip(labels.iter())
            .filter(|(p, l)| {
                let pred_class = if **p >= self.threshold { 1.0 } else { 0.0 };
                (pred_class - **l).abs() < 0.5
            })
            .count();

        correct as f64 / preds.len() as f64
    }

    fn higher_is_better(&self) -> bool {
        true
    }

    fn name(&self) -> &str {
        "accuracy"
    }
}

// =============================================================================
// Multiclass Accuracy
// =============================================================================

/// Multiclass accuracy: expects predictions as class indices.
///
/// Higher is better.
#[derive(Debug, Clone, Copy, Default)]
pub struct MulticlassAccuracy;

impl SimpleMetric for MulticlassAccuracy {
    fn compute(&self, preds: &[f32], labels: &[f32]) -> f64 {
        debug_assert_eq!(preds.len(), labels.len());

        if preds.is_empty() {
            return 0.0;
        }

        let correct = preds
            .iter()
            .zip(labels.iter())
            .filter(|(p, l)| (p.round() - l.round()).abs() < 0.5)
            .count();

        correct as f64 / preds.len() as f64
    }

    fn higher_is_better(&self) -> bool {
        true
    }

    fn name(&self) -> &str {
        "multiclass_accuracy"
    }
}

// =============================================================================
// AUC (Area Under ROC Curve)
// =============================================================================

/// Area Under the ROC Curve for binary classification.
///
/// Higher is better. Measures ranking quality.
/// Expects predictions to be scores (higher = more likely positive).
#[derive(Debug, Clone, Copy, Default)]
pub struct Auc;

impl SimpleMetric for Auc {
    fn compute(&self, preds: &[f32], labels: &[f32]) -> f64 {
        debug_assert_eq!(preds.len(), labels.len());

        if preds.is_empty() {
            return 0.5;
        }

        // Count positives and negatives
        let n_pos = labels.iter().filter(|&&l| l > 0.5).count();
        let n_neg = labels.len() - n_pos;

        if n_pos == 0 || n_neg == 0 {
            return 0.5; // Undefined, return random
        }

        // AUC via Mann-Whitney U statistic:
        // For each positive-negative pair, count when positive has higher score
        let mut concordant = 0.0f64;

        for (p_pos, l_pos) in preds.iter().zip(labels.iter()) {
            if *l_pos < 0.5 {
                continue; // Skip negatives in outer loop
            }

            for (p_neg, l_neg) in preds.iter().zip(labels.iter()) {
                if *l_neg > 0.5 {
                    continue; // Skip positives in inner loop
                }

                if p_pos > p_neg {
                    concordant += 1.0;
                } else if (p_pos - p_neg).abs() < 1e-10 {
                    concordant += 0.5; // Tie
                }
            }
        }

        concordant / (n_pos as f64 * n_neg as f64)
    }

    fn higher_is_better(&self) -> bool {
        true
    }

    fn name(&self) -> &str {
        "auc"
    }
}

// =============================================================================
// MAPE (Mean Absolute Percentage Error)
// =============================================================================

/// Mean Absolute Percentage Error: mean(|pred - label| / |label|) * 100
///
/// Lower is better. Used for regression tasks when relative error matters.
/// Undefined when label is 0; uses epsilon to avoid division by zero.
#[derive(Debug, Clone, Copy, Default)]
pub struct Mape;

impl SimpleMetric for Mape {
    fn compute(&self, preds: &[f32], labels: &[f32]) -> f64 {
        debug_assert_eq!(preds.len(), labels.len());

        if preds.is_empty() {
            return 0.0;
        }

        let eps = 1e-15f64; // Avoid division by zero

        let mape: f64 = preds
            .iter()
            .zip(labels.iter())
            .map(|(p, l)| {
                let p = *p as f64;
                let l = *l as f64;
                (p - l).abs() / l.abs().max(eps)
            })
            .sum::<f64>()
            / preds.len() as f64;

        mape * 100.0 // Return as percentage
    }

    fn higher_is_better(&self) -> bool {
        false
    }

    fn name(&self) -> &str {
        "mape"
    }
}

// =============================================================================
// Multiclass LogLoss
// =============================================================================

/// Multiclass cross-entropy: -mean(sum_k y_k * log(p_k))
///
/// Lower is better. Used for multiclass classification.
/// Expects predictions to be probabilities with shape `[n_samples, n_classes]`
/// in row-major order, and labels to be class indices in `0..n_classes`.
#[derive(Debug, Clone, Copy, Default)]
pub struct MulticlassLogLoss;

impl Metric for MulticlassLogLoss {
    fn evaluate(&self, predictions: &[f32], labels: &[f32], n_outputs: usize) -> f64 {
        if labels.is_empty() || n_outputs == 0 {
            return 0.0;
        }

        let n_samples = labels.len();
        debug_assert_eq!(predictions.len(), n_samples * n_outputs);

        let eps = 1e-15f64; // Clip to avoid log(0)

        let total_loss: f64 = labels
            .iter()
            .enumerate()
            .map(|(i, &label)| {
                let class_idx = label.round() as usize;
                debug_assert!(class_idx < n_outputs, "label out of bounds");

                // Get predicted probability for the true class
                let prob = predictions[i * n_outputs + class_idx] as f64;
                let prob = prob.clamp(eps, 1.0 - eps);
                -prob.ln()
            })
            .sum();

        total_loss / n_samples as f64
    }

    fn higher_is_better(&self) -> bool {
        false
    }

    fn name(&self) -> &str {
        "mlogloss"
    }
}

// =============================================================================
// Quantile Loss (Pinball Loss)
// =============================================================================

/// Quantile loss (pinball loss) for quantile regression.
///
/// L_tau(y, q) = tau * max(y - q, 0) + (1 - tau) * max(q - y, 0)
///
/// Lower is better. For multi-quantile models, provide quantile alphas
/// and the predictions buffer has shape `[n_samples, n_quantiles]`.
#[derive(Debug, Clone)]
pub struct QuantileLoss {
    /// Quantile levels (e.g., [0.1, 0.5, 0.9])
    pub alphas: Vec<f32>,
}

impl QuantileLoss {
    /// Create quantile loss with specified quantile levels.
    pub fn new(alphas: Vec<f32>) -> Self {
        debug_assert!(alphas.iter().all(|&a| (0.0..=1.0).contains(&a)));
        Self { alphas }
    }

    /// Create for median prediction (alpha = 0.5).
    pub fn median() -> Self {
        Self { alphas: vec![0.5] }
    }
}

impl Default for QuantileLoss {
    fn default() -> Self {
        Self::median()
    }
}

impl Metric for QuantileLoss {
    fn evaluate(&self, predictions: &[f32], labels: &[f32], n_outputs: usize) -> f64 {
        if labels.is_empty() || n_outputs == 0 {
            return 0.0;
        }

        let n_samples = labels.len();
        let n_quantiles = self.alphas.len().max(n_outputs);
        debug_assert_eq!(predictions.len(), n_samples * n_quantiles);

        let total_loss: f64 = labels
            .iter()
            .enumerate()
            .flat_map(|(i, &label)| {
                self.alphas.iter().enumerate().map(move |(q, &alpha)| {
                    let pred = predictions[i * n_quantiles + q] as f64;
                    let y = label as f64;
                    let residual = y - pred;

                    if residual >= 0.0 {
                        alpha as f64 * residual
                    } else {
                        (1.0 - alpha as f64) * (-residual)
                    }
                })
            })
            .sum();

        total_loss / (n_samples * n_quantiles) as f64
    }

    fn higher_is_better(&self) -> bool {
        false
    }

    fn name(&self) -> &str {
        "quantile"
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to call evaluate for single-output metrics
    fn eval<M: Metric>(m: &M, preds: &[f32], labels: &[f32]) -> f64 {
        m.evaluate(preds, labels, 1)
    }

    #[test]
    fn rmse_perfect() {
        let preds = vec![1.0, 2.0, 3.0];
        let labels = vec![1.0, 2.0, 3.0];
        let rmse = eval(&Rmse, &preds, &labels);
        assert!(rmse.abs() < 1e-10);
    }

    #[test]
    fn rmse_known_value() {
        // RMSE of [1, 2] vs [0, 0] = sqrt((1 + 4) / 2) = sqrt(2.5)
        let preds = vec![1.0, 2.0];
        let labels = vec![0.0, 0.0];
        let rmse = eval(&Rmse, &preds, &labels);
        assert!((rmse - 2.5f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn mae_perfect() {
        let preds = vec![1.0, 2.0, 3.0];
        let labels = vec![1.0, 2.0, 3.0];
        let mae = eval(&Mae, &preds, &labels);
        assert!(mae.abs() < 1e-10);
    }

    #[test]
    fn mae_known_value() {
        // MAE of [1, 2] vs [0, 0] = (1 + 2) / 2 = 1.5
        let preds = vec![1.0, 2.0];
        let labels = vec![0.0, 0.0];
        let mae = eval(&Mae, &preds, &labels);
        assert!((mae - 1.5).abs() < 1e-10);
    }

    #[test]
    fn logloss_perfect() {
        // Perfect predictions: p=1 for y=1, p=0 for y=0
        let preds = vec![0.9999, 0.0001];
        let labels = vec![1.0, 0.0];
        let ll = eval(&LogLoss, &preds, &labels);
        assert!(ll < 0.01); // Should be very small
    }

    #[test]
    fn logloss_worst() {
        // Worst predictions: p=0 for y=1, p=1 for y=0
        let preds = vec![0.0001, 0.9999];
        let labels = vec![1.0, 0.0];
        let ll = eval(&LogLoss, &preds, &labels);
        assert!(ll > 5.0); // Should be large
    }

    #[test]
    fn logloss_random() {
        // Random predictions (p=0.5)
        let preds = vec![0.5, 0.5];
        let labels = vec![1.0, 0.0];
        let ll = eval(&LogLoss, &preds, &labels);
        // -log(0.5) ≈ 0.693
        assert!((ll - 0.693).abs() < 0.01);
    }

    #[test]
    fn accuracy_perfect() {
        let preds = vec![0.9, 0.1, 0.8, 0.2];
        let labels = vec![1.0, 0.0, 1.0, 0.0];
        let acc = eval(&Accuracy::default(), &preds, &labels);
        assert!((acc - 1.0).abs() < 1e-10);
    }

    #[test]
    fn accuracy_half() {
        let preds = vec![0.9, 0.9, 0.1, 0.1];
        let labels = vec![1.0, 0.0, 1.0, 0.0];
        let acc = eval(&Accuracy::default(), &preds, &labels);
        assert!((acc - 0.5).abs() < 1e-10);
    }

    #[test]
    fn accuracy_custom_threshold() {
        let preds = vec![0.3, 0.3, 0.3, 0.3];
        let labels = vec![1.0, 0.0, 1.0, 0.0];

        // With threshold 0.5, all predictions are 0
        let acc_05 = eval(&Accuracy::default(), &preds, &labels);
        assert!((acc_05 - 0.5).abs() < 1e-10);

        // With threshold 0.2, all predictions are 1
        let acc_02 = eval(&Accuracy::with_threshold(0.2), &preds, &labels);
        assert!((acc_02 - 0.5).abs() < 1e-10);
    }

    #[test]
    fn auc_perfect() {
        // Perfect separation: all positives have higher scores
        let preds = vec![0.9, 0.8, 0.3, 0.2];
        let labels = vec![1.0, 1.0, 0.0, 0.0];
        let auc = eval(&Auc, &preds, &labels);
        assert!((auc - 1.0).abs() < 1e-10);
    }

    #[test]
    fn auc_random() {
        // Random: mixed ordering
        let preds = vec![0.5, 0.5, 0.5, 0.5];
        let labels = vec![1.0, 0.0, 1.0, 0.0];
        let auc = eval(&Auc, &preds, &labels);
        assert!((auc - 0.5).abs() < 1e-10);
    }

    #[test]
    fn auc_worst() {
        // Worst: all negatives have higher scores
        let preds = vec![0.2, 0.3, 0.8, 0.9];
        let labels = vec![1.0, 1.0, 0.0, 0.0];
        let auc = eval(&Auc, &preds, &labels);
        assert!(auc.abs() < 1e-10);
    }

    #[test]
    fn metric_higher_is_better() {
        assert!(!Metric::higher_is_better(&Rmse));
        assert!(!Metric::higher_is_better(&Mae));
        assert!(!Metric::higher_is_better(&LogLoss));
        assert!(Metric::higher_is_better(&Accuracy::default()));
        assert!(Metric::higher_is_better(&Auc));
    }

    #[test]
    fn metric_names() {
        assert_eq!(Metric::name(&Rmse), "rmse");
        assert_eq!(Metric::name(&Mae), "mae");
        assert_eq!(Metric::name(&LogLoss), "logloss");
        assert_eq!(Metric::name(&Accuracy::default()), "accuracy");
        assert_eq!(Metric::name(&Auc), "auc");
    }

    #[test]
    fn multiclass_accuracy() {
        let preds = vec![0.0, 1.0, 2.0, 1.0]; // Predicted classes
        let labels = vec![0.0, 1.0, 2.0, 0.0]; // True classes
        let acc = eval(&MulticlassAccuracy, &preds, &labels);
        assert!((acc - 0.75).abs() < 1e-10); // 3/4 correct
    }

    #[test]
    fn mape_perfect() {
        let preds = vec![1.0, 2.0, 3.0];
        let labels = vec![1.0, 2.0, 3.0];
        let mape = eval(&Mape, &preds, &labels);
        assert!(mape.abs() < 1e-10);
    }

    #[test]
    fn mape_known_value() {
        // MAPE: mean(|pred - label| / |label|) * 100
        // |1-2|/2 = 0.5, |3-4|/4 = 0.25 → mean = 0.375 → 37.5%
        let preds = vec![1.0, 3.0];
        let labels = vec![2.0, 4.0];
        let mape = eval(&Mape, &preds, &labels);
        assert!((mape - 37.5).abs() < 1e-10);
    }

    #[test]
    fn mlogloss_perfect() {
        // 3-class classification, predictions are probabilities
        // Sample 0: true class 0, predictions [0.99, 0.005, 0.005]
        // Sample 1: true class 1, predictions [0.005, 0.99, 0.005]
        let preds = vec![
            0.99, 0.005, 0.005, // sample 0
            0.005, 0.99, 0.005, // sample 1
        ];
        let labels = vec![0.0, 1.0];
        let mlogloss = MulticlassLogLoss.evaluate(&preds, &labels, 3);
        assert!(mlogloss < 0.02); // Should be very small
    }

    #[test]
    fn mlogloss_uniform() {
        // Uniform predictions for 3 classes: -log(1/3) ≈ 1.099
        let preds = vec![
            0.333, 0.333, 0.334, // sample 0
            0.333, 0.333, 0.334, // sample 1
            0.333, 0.333, 0.334, // sample 2
        ];
        let labels = vec![0.0, 1.0, 2.0];
        let mlogloss = MulticlassLogLoss.evaluate(&preds, &labels, 3);
        assert!((mlogloss - 1.099).abs() < 0.01);
    }

    #[test]
    fn quantile_median() {
        // Median (alpha=0.5): pinball loss = 0.5 * |y - q|
        // Predictions exactly match labels
        let preds = vec![1.0, 2.0, 3.0];
        let labels = vec![1.0, 2.0, 3.0];
        let loss = QuantileLoss::median().evaluate(&preds, &labels, 1);
        assert!(loss.abs() < 1e-10);
    }

    #[test]
    fn quantile_median_error() {
        // Median (alpha=0.5): pinball loss = 0.5 * |y - q|
        // |1-2| = 1, |3-2| = 1 → pinball each = 0.5 → mean = 0.5
        let preds = vec![2.0, 2.0];
        let labels = vec![1.0, 3.0];
        let loss = QuantileLoss::median().evaluate(&preds, &labels, 1);
        assert!((loss - 0.5).abs() < 1e-10);
    }

    #[test]
    fn quantile_asymmetric() {
        // Alpha = 0.1: penalize over-prediction more
        // y=5, q=3: residual=2 (under-predict) → 0.1 * 2 = 0.2
        // y=5, q=7: residual=-2 (over-predict) → 0.9 * 2 = 1.8
        let alphas = vec![0.1];
        let metric = QuantileLoss::new(alphas);

        let loss_under = metric.evaluate(&[3.0], &[5.0], 1);
        assert!((loss_under - 0.2).abs() < 1e-6, "got {}", loss_under);

        let loss_over = metric.evaluate(&[7.0], &[5.0], 1);
        assert!((loss_over - 1.8).abs() < 1e-6, "got {}", loss_over);
    }

    #[test]
    fn new_metric_properties() {
        // Test higher_is_better and names for new metrics
        assert!(!Metric::higher_is_better(&Mape));
        assert!(!Metric::higher_is_better(&MulticlassLogLoss));
        assert!(!Metric::higher_is_better(&QuantileLoss::median()));

        assert_eq!(Metric::name(&Mape), "mape");
        assert_eq!(Metric::name(&MulticlassLogLoss), "mlogloss");
        assert_eq!(Metric::name(&QuantileLoss::median()), "quantile");
    }
}

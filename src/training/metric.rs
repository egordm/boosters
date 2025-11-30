//! Evaluation metrics for model quality.
//!
//! Metrics are separate from loss functions — a model might be trained with
//! one loss but evaluated with different metrics.

/// A metric for evaluating model quality.
///
/// Unlike [`super::Loss`] which computes gradients for optimization,
/// metrics compute scalar values for model evaluation and monitoring.
///
/// # Implementation Notes
///
/// - `compute`: Called with predictions and labels, returns a scalar score
/// - Higher is better for some metrics (accuracy, AUC), lower for others (RMSE, logloss)
/// - Use `higher_is_better()` to determine the direction
pub trait Metric: Send + Sync {
    /// Compute the metric value.
    ///
    /// # Arguments
    ///
    /// * `preds` - Model predictions
    /// * `labels` - Ground truth labels
    ///
    /// # Returns
    ///
    /// Scalar metric value.
    fn compute(&self, preds: &[f32], labels: &[f32]) -> f64;

    /// Whether higher values indicate better performance.
    ///
    /// - `true`: Higher is better (accuracy, AUC)
    /// - `false`: Lower is better (RMSE, MAE, logloss)
    fn higher_is_better(&self) -> bool;

    /// Name of the metric (for logging).
    fn name(&self) -> &'static str;
}

// =============================================================================
// RMSE (Root Mean Squared Error)
// =============================================================================

/// Root Mean Squared Error: sqrt(mean((pred - label)²))
///
/// Lower is better. Used for regression tasks.
#[derive(Debug, Clone, Copy, Default)]
pub struct Rmse;

impl Metric for Rmse {
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

    fn name(&self) -> &'static str {
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

impl Metric for Mae {
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

    fn name(&self) -> &'static str {
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

impl Metric for LogLoss {
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

    fn name(&self) -> &'static str {
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

impl Metric for Accuracy {
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

    fn name(&self) -> &'static str {
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

impl Metric for MulticlassAccuracy {
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

    fn name(&self) -> &'static str {
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

impl Metric for Auc {
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

    fn name(&self) -> &'static str {
        "auc"
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rmse_perfect() {
        let preds = vec![1.0, 2.0, 3.0];
        let labels = vec![1.0, 2.0, 3.0];
        let rmse = Rmse.compute(&preds, &labels);
        assert!(rmse.abs() < 1e-10);
    }

    #[test]
    fn rmse_known_value() {
        // RMSE of [1, 2] vs [0, 0] = sqrt((1 + 4) / 2) = sqrt(2.5)
        let preds = vec![1.0, 2.0];
        let labels = vec![0.0, 0.0];
        let rmse = Rmse.compute(&preds, &labels);
        assert!((rmse - 2.5f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn mae_perfect() {
        let preds = vec![1.0, 2.0, 3.0];
        let labels = vec![1.0, 2.0, 3.0];
        let mae = Mae.compute(&preds, &labels);
        assert!(mae.abs() < 1e-10);
    }

    #[test]
    fn mae_known_value() {
        // MAE of [1, 2] vs [0, 0] = (1 + 2) / 2 = 1.5
        let preds = vec![1.0, 2.0];
        let labels = vec![0.0, 0.0];
        let mae = Mae.compute(&preds, &labels);
        assert!((mae - 1.5).abs() < 1e-10);
    }

    #[test]
    fn logloss_perfect() {
        // Perfect predictions: p=1 for y=1, p=0 for y=0
        let preds = vec![0.9999, 0.0001];
        let labels = vec![1.0, 0.0];
        let ll = LogLoss.compute(&preds, &labels);
        assert!(ll < 0.01); // Should be very small
    }

    #[test]
    fn logloss_worst() {
        // Worst predictions: p=0 for y=1, p=1 for y=0
        let preds = vec![0.0001, 0.9999];
        let labels = vec![1.0, 0.0];
        let ll = LogLoss.compute(&preds, &labels);
        assert!(ll > 5.0); // Should be large
    }

    #[test]
    fn logloss_random() {
        // Random predictions (p=0.5)
        let preds = vec![0.5, 0.5];
        let labels = vec![1.0, 0.0];
        let ll = LogLoss.compute(&preds, &labels);
        // -log(0.5) ≈ 0.693
        assert!((ll - 0.693).abs() < 0.01);
    }

    #[test]
    fn accuracy_perfect() {
        let preds = vec![0.9, 0.1, 0.8, 0.2];
        let labels = vec![1.0, 0.0, 1.0, 0.0];
        let acc = Accuracy::default().compute(&preds, &labels);
        assert!((acc - 1.0).abs() < 1e-10);
    }

    #[test]
    fn accuracy_half() {
        let preds = vec![0.9, 0.9, 0.1, 0.1];
        let labels = vec![1.0, 0.0, 1.0, 0.0];
        let acc = Accuracy::default().compute(&preds, &labels);
        assert!((acc - 0.5).abs() < 1e-10);
    }

    #[test]
    fn accuracy_custom_threshold() {
        let preds = vec![0.3, 0.3, 0.3, 0.3];
        let labels = vec![1.0, 0.0, 1.0, 0.0];

        // With threshold 0.5, all predictions are 0
        let acc_05 = Accuracy::default().compute(&preds, &labels);
        assert!((acc_05 - 0.5).abs() < 1e-10);

        // With threshold 0.2, all predictions are 1
        let acc_02 = Accuracy::with_threshold(0.2).compute(&preds, &labels);
        assert!((acc_02 - 0.5).abs() < 1e-10);
    }

    #[test]
    fn auc_perfect() {
        // Perfect separation: all positives have higher scores
        let preds = vec![0.9, 0.8, 0.3, 0.2];
        let labels = vec![1.0, 1.0, 0.0, 0.0];
        let auc = Auc.compute(&preds, &labels);
        assert!((auc - 1.0).abs() < 1e-10);
    }

    #[test]
    fn auc_random() {
        // Random: mixed ordering
        let preds = vec![0.5, 0.5, 0.5, 0.5];
        let labels = vec![1.0, 0.0, 1.0, 0.0];
        let auc = Auc.compute(&preds, &labels);
        assert!((auc - 0.5).abs() < 1e-10);
    }

    #[test]
    fn auc_worst() {
        // Worst: all negatives have higher scores
        let preds = vec![0.2, 0.3, 0.8, 0.9];
        let labels = vec![1.0, 1.0, 0.0, 0.0];
        let auc = Auc.compute(&preds, &labels);
        assert!(auc.abs() < 1e-10);
    }

    #[test]
    fn metric_higher_is_better() {
        assert!(!Rmse.higher_is_better());
        assert!(!Mae.higher_is_better());
        assert!(!LogLoss.higher_is_better());
        assert!(Accuracy::default().higher_is_better());
        assert!(Auc.higher_is_better());
    }

    #[test]
    fn metric_names() {
        assert_eq!(Rmse.name(), "rmse");
        assert_eq!(Mae.name(), "mae");
        assert_eq!(LogLoss.name(), "logloss");
        assert_eq!(Accuracy::default().name(), "accuracy");
        assert_eq!(Auc.name(), "auc");
    }

    #[test]
    fn multiclass_accuracy() {
        let preds = vec![0.0, 1.0, 2.0, 1.0]; // Predicted classes
        let labels = vec![0.0, 1.0, 2.0, 0.0]; // True classes
        let acc = MulticlassAccuracy.compute(&preds, &labels);
        assert!((acc - 0.75).abs() < 1e-10); // 3/4 correct
    }
}

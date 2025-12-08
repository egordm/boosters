//! Regression metrics.
//!
//! Metrics for evaluating regression model quality.

use super::Metric;

// =============================================================================
// RMSE (Root Mean Squared Error)
// =============================================================================

/// Root Mean Squared Error: sqrt(mean((pred - label)²))
///
/// Lower is better. Used for regression tasks.
///
/// # Weighted Computation
///
/// When weights are provided, computes weighted RMSE:
/// ```text
/// sqrt(sum(w * (p - l)²) / sum(w))
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct Rmse;

impl Metric for Rmse {
    fn evaluate(
        &self,
        predictions: &[f32],
        labels: &[f32],
        weights: Option<&[f32]>,
        _n_outputs: usize,
    ) -> f64 {
        debug_assert_eq!(predictions.len(), labels.len());

        if predictions.is_empty() {
            return 0.0;
        }

        match weights {
            None => {
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
            Some(w) => {
                debug_assert_eq!(predictions.len(), w.len());

                let (sum_sq, sum_w) = predictions
                    .iter()
                    .zip(labels.iter())
                    .zip(w.iter())
                    .fold((0.0f64, 0.0f64), |(ss, sw), ((&p, &l), &wt)| {
                        let diff = (p as f64) - (l as f64);
                        (ss + (wt as f64) * diff * diff, sw + wt as f64)
                    });

                if sum_w > 0.0 {
                    (sum_sq / sum_w).sqrt()
                } else {
                    0.0
                }
            }
        }
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
///
/// # Weighted Computation
///
/// When weights are provided, computes weighted MAE:
/// ```text
/// sum(w * |p - l|) / sum(w)
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct Mae;

impl Metric for Mae {
    fn evaluate(
        &self,
        predictions: &[f32],
        labels: &[f32],
        weights: Option<&[f32]>,
        _n_outputs: usize,
    ) -> f64 {
        debug_assert_eq!(predictions.len(), labels.len());

        if predictions.is_empty() {
            return 0.0;
        }

        match weights {
            None => {
                predictions
                    .iter()
                    .zip(labels.iter())
                    .map(|(p, l)| ((*p as f64) - (*l as f64)).abs())
                    .sum::<f64>()
                    / predictions.len() as f64
            }
            Some(w) => {
                debug_assert_eq!(predictions.len(), w.len());

                let (weighted_sum, weight_sum) = predictions
                    .iter()
                    .zip(labels.iter())
                    .zip(w.iter())
                    .fold((0.0f64, 0.0f64), |(acc_err, acc_w), ((p, l), wt)| {
                        let wt = *wt as f64;
                        let abs_err = ((*p as f64) - (*l as f64)).abs();
                        (acc_err + wt * abs_err, acc_w + wt)
                    });

                if weight_sum == 0.0 {
                    return 0.0;
                }

                weighted_sum / weight_sum
            }
        }
    }

    fn higher_is_better(&self) -> bool {
        false
    }

    fn name(&self) -> &str {
        "mae"
    }
}

// =============================================================================
// MAPE (Mean Absolute Percentage Error)
// =============================================================================

/// Mean Absolute Percentage Error: mean(|pred - label| / |label|) * 100
///
/// Lower is better. Used for regression tasks when relative error matters.
/// Undefined when label is 0; uses epsilon to avoid division by zero.
///
/// # Weighted Computation
///
/// When weights are provided, computes weighted MAPE:
/// ```text
/// sum(w * |p - l| / |l|) / sum(w) * 100
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct Mape;

impl Metric for Mape {
    fn evaluate(
        &self,
        predictions: &[f32],
        labels: &[f32],
        weights: Option<&[f32]>,
        _n_outputs: usize,
    ) -> f64 {
        debug_assert_eq!(predictions.len(), labels.len());

        if predictions.is_empty() {
            return 0.0;
        }

        let eps = 1e-15f64; // Avoid division by zero

        match weights {
            None => {
                let mape: f64 = predictions
                    .iter()
                    .zip(labels.iter())
                    .map(|(p, l)| {
                        let p = *p as f64;
                        let l = *l as f64;
                        (p - l).abs() / l.abs().max(eps)
                    })
                    .sum::<f64>()
                    / predictions.len() as f64;

                mape * 100.0 // Return as percentage
            }
            Some(w) => {
                debug_assert_eq!(predictions.len(), w.len());

                let (weighted_sum, weight_sum) = predictions
                    .iter()
                    .zip(labels.iter())
                    .zip(w.iter())
                    .fold((0.0f64, 0.0f64), |(acc_err, acc_w), ((p, l), wt)| {
                        let wt = *wt as f64;
                        let p = *p as f64;
                        let l = *l as f64;
                        let ape = (p - l).abs() / l.abs().max(eps);
                        (acc_err + wt * ape, acc_w + wt)
                    });

                if weight_sum == 0.0 {
                    return 0.0;
                }

                (weighted_sum / weight_sum) * 100.0 // Return as percentage
            }
        }
    }

    fn higher_is_better(&self) -> bool {
        false
    }

    fn name(&self) -> &str {
        "mape"
    }
}

// =============================================================================
// Quantile Metric (Pinball Loss)
// =============================================================================

/// Quantile metric (pinball loss) for quantile regression.
///
/// L_tau(y, q) = tau * max(y - q, 0) + (1 - tau) * max(q - y, 0)
///
/// Lower is better. For multi-quantile models, provide quantile alphas
/// and the predictions buffer has shape `[n_samples, n_quantiles]`.
///
/// # Weighted Computation
///
/// When weights are provided, computes weighted quantile loss:
/// ```text
/// sum(w * pinball_loss) / sum(w)
/// ```
#[derive(Debug, Clone)]
pub struct QuantileMetric {
    /// Quantile levels (e.g., [0.1, 0.5, 0.9])
    pub alphas: Vec<f32>,
}

impl QuantileMetric {
    /// Create quantile metric with specified quantile levels.
    pub fn new(alphas: Vec<f32>) -> Self {
        debug_assert!(alphas.iter().all(|&a| (0.0..=1.0).contains(&a)));
        Self { alphas }
    }

    /// Create for median prediction (alpha = 0.5).
    pub fn median() -> Self {
        Self { alphas: vec![0.5] }
    }
}

impl Default for QuantileMetric {
    fn default() -> Self {
        Self::median()
    }
}

impl Metric for QuantileMetric {
    fn evaluate(
        &self,
        predictions: &[f32],
        labels: &[f32],
        weights: Option<&[f32]>,
        n_outputs: usize,
    ) -> f64 {
        if labels.is_empty() || n_outputs == 0 {
            return 0.0;
        }

        let n_samples = labels.len();
        let n_quantiles = self.alphas.len().max(n_outputs);
        debug_assert_eq!(predictions.len(), n_samples * n_quantiles);

        match weights {
            None => {
                // Column-major: predictions[q * n_samples + i]
                let total_loss: f64 = self
                    .alphas
                    .iter()
                    .enumerate()
                    .flat_map(|(q, &alpha)| {
                        labels.iter().enumerate().map(move |(i, &label)| {
                            let pred = predictions[q * n_samples + i] as f64;
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
            Some(w) => {
                debug_assert_eq!(w.len(), n_samples);

                // Weighted quantile loss: sum over quantiles of sum(w * loss) / sum(w)
                let (weighted_loss, weight_sum): (f64, f64) = self
                    .alphas
                    .iter()
                    .enumerate()
                    .map(|(q, &alpha)| {
                        labels
                            .iter()
                            .enumerate()
                            .fold((0.0f64, 0.0f64), |(acc_loss, acc_w), (i, &label)| {
                                let pred = predictions[q * n_samples + i] as f64;
                                let y = label as f64;
                                let residual = y - pred;
                                let wt = w[i] as f64;

                                let loss = if residual >= 0.0 {
                                    alpha as f64 * residual
                                } else {
                                    (1.0 - alpha as f64) * (-residual)
                                };

                                (acc_loss + wt * loss, acc_w + wt)
                            })
                    })
                    .fold((0.0, 0.0), |(tl, tw), (l, wsum)| (tl + l, tw + wsum));

                if weight_sum == 0.0 {
                    return 0.0;
                }

                weighted_loss / weight_sum
            }
        }
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

    // =========================================================================
    // RMSE tests
    // =========================================================================

    #[test]
    fn rmse_perfect() {
        let preds = vec![1.0, 2.0, 3.0];
        let labels = vec![1.0, 2.0, 3.0];
        let rmse = Rmse.evaluate(&preds, &labels, None, 1);
        assert!(rmse.abs() < 1e-10);
    }

    #[test]
    fn rmse_known_value() {
        // RMSE of [1, 2] vs [0, 0] = sqrt((1 + 4) / 2) = sqrt(2.5)
        let preds = vec![1.0, 2.0];
        let labels = vec![0.0, 0.0];
        let rmse = Rmse.evaluate(&preds, &labels, None, 1);
        assert!((rmse - 2.5f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn weighted_rmse_uniform_weights_equals_unweighted() {
        let preds = vec![1.0, 2.0, 3.0, 4.0];
        let labels = vec![1.5, 2.5, 2.5, 4.5];
        let weights = vec![1.0, 1.0, 1.0, 1.0];

        let unweighted = Rmse.evaluate(&preds, &labels, None, 1);
        let weighted = Rmse.evaluate(&preds, &labels, Some(&weights), 1);

        assert!((weighted - unweighted).abs() < 1e-10);
    }

    #[test]
    fn weighted_rmse_emphasizes_high_weight_samples() {
        // Sample 0 has error 1.0, sample 1 has error 0.0
        let preds = vec![1.0, 2.0];
        let labels = vec![2.0, 2.0];

        // With equal weights: RMSE = sqrt((1 + 0) / 2) = sqrt(0.5)
        let equal_weights = vec![1.0, 1.0];
        let rmse_equal = Rmse.evaluate(&preds, &labels, Some(&equal_weights), 1);
        assert!((rmse_equal - 0.5f64.sqrt()).abs() < 1e-10);

        // With high weight on sample 0: RMSE moves toward error of sample 0
        // w * (err)^2 / w = 10 * 1 + 1 * 0 / 11 = 10/11
        let unequal_weights = vec![10.0, 1.0];
        let rmse_unequal = Rmse.evaluate(&preds, &labels, Some(&unequal_weights), 1);
        let expected = (10.0 / 11.0f64).sqrt();
        assert!(
            (rmse_unequal - expected).abs() < 1e-10,
            "got {} expected {}",
            rmse_unequal,
            expected
        );
    }

    // =========================================================================
    // MAE tests
    // =========================================================================

    #[test]
    fn mae_perfect() {
        let preds = vec![1.0, 2.0, 3.0];
        let labels = vec![1.0, 2.0, 3.0];
        let mae = Mae.evaluate(&preds, &labels, None, 1);
        assert!(mae.abs() < 1e-10);
    }

    #[test]
    fn mae_known_value() {
        // MAE of [1, 2] vs [0, 0] = (1 + 2) / 2 = 1.5
        let preds = vec![1.0, 2.0];
        let labels = vec![0.0, 0.0];
        let mae = Mae.evaluate(&preds, &labels, None, 1);
        assert!((mae - 1.5).abs() < 1e-10);
    }

    #[test]
    fn weighted_mae_emphasizes_high_weight_samples() {
        // Sample 0 has error 2.0, sample 1 has error 1.0
        let preds = vec![0.0, 2.0];
        let labels = vec![2.0, 3.0];

        // Unweighted MAE = (2 + 1) / 2 = 1.5
        let unweighted = Mae.evaluate(&preds, &labels, None, 1);
        assert!((unweighted - 1.5).abs() < 1e-10);

        // With high weight on sample 0: MAE = (10*2 + 1*1) / 11 = 21/11
        let weights = vec![10.0, 1.0];
        let weighted = Mae.evaluate(&preds, &labels, Some(&weights), 1);
        let expected = 21.0 / 11.0;
        assert!(
            (weighted - expected).abs() < 1e-10,
            "got {} expected {}",
            weighted,
            expected
        );
    }

    // =========================================================================
    // MAPE tests
    // =========================================================================

    #[test]
    fn mape_perfect() {
        let preds = vec![1.0, 2.0, 3.0];
        let labels = vec![1.0, 2.0, 3.0];
        let mape = Mape.evaluate(&preds, &labels, None, 1);
        assert!(mape.abs() < 1e-10);
    }

    #[test]
    fn mape_known_value() {
        // MAPE: mean(|pred - label| / |label|) * 100
        // |1-2|/2 = 0.5, |3-4|/4 = 0.25 → mean = 0.375 → 37.5%
        let preds = vec![1.0, 3.0];
        let labels = vec![2.0, 4.0];
        let mape = Mape.evaluate(&preds, &labels, None, 1);
        assert!((mape - 37.5).abs() < 1e-10);
    }

    #[test]
    fn weighted_mape_emphasizes_high_weight_samples() {
        // Sample 0: pred=1, label=2 → APE = 0.5 (50%)
        // Sample 1: pred=3, label=4 → APE = 0.25 (25%)
        let preds = vec![1.0, 3.0];
        let labels = vec![2.0, 4.0];

        // Unweighted MAPE = (50 + 25) / 2 = 37.5%
        let unweighted = Mape.evaluate(&preds, &labels, None, 1);
        assert!((unweighted - 37.5).abs() < 1e-10);

        // High weight on sample 0 (higher error): MAPE increases
        let weights = vec![10.0, 1.0];
        let weighted = Mape.evaluate(&preds, &labels, Some(&weights), 1);
        // (10 * 0.5 + 1 * 0.25) / 11 * 100 = (5.25 / 11) * 100 ≈ 47.73%
        let expected = (10.0 * 0.5 + 1.0 * 0.25) / 11.0 * 100.0;
        assert!(
            (weighted - expected).abs() < 1e-6,
            "got {} expected {}",
            weighted,
            expected
        );
    }

    // =========================================================================
    // Quantile Metric tests
    // =========================================================================

    #[test]
    fn quantile_median() {
        // Median (alpha=0.5): pinball loss = 0.5 * |y - q|
        // Predictions exactly match labels
        let preds = vec![1.0, 2.0, 3.0];
        let labels = vec![1.0, 2.0, 3.0];
        let loss = QuantileMetric::median().evaluate(&preds, &labels, None, 1);
        assert!(loss.abs() < 1e-10);
    }

    #[test]
    fn quantile_median_error() {
        // Median (alpha=0.5): pinball loss = 0.5 * |y - q|
        // |1-2| = 1, |3-2| = 1 → pinball each = 0.5 → mean = 0.5
        let preds = vec![2.0, 2.0];
        let labels = vec![1.0, 3.0];
        let loss = QuantileMetric::median().evaluate(&preds, &labels, None, 1);
        assert!((loss - 0.5).abs() < 1e-10);
    }

    #[test]
    fn quantile_asymmetric() {
        // Alpha = 0.1: penalize over-prediction more
        // y=5, q=3: residual=2 (under-predict) → 0.1 * 2 = 0.2
        // y=5, q=7: residual=-2 (over-predict) → 0.9 * 2 = 1.8
        let alphas = vec![0.1];
        let metric = QuantileMetric::new(alphas);

        let loss_under = metric.evaluate(&[3.0], &[5.0], None, 1);
        assert!((loss_under - 0.2).abs() < 1e-6, "got {}", loss_under);

        let loss_over = metric.evaluate(&[7.0], &[5.0], None, 1);
        assert!((loss_over - 1.8).abs() < 1e-6, "got {}", loss_over);
    }

    #[test]
    fn weighted_quantile_loss() {
        // Median (alpha=0.5): pinball loss = 0.5 * |y - q|
        let preds = vec![2.0, 2.0];
        let labels = vec![1.0, 3.0]; // errors: 1, 1

        // Unweighted: 0.5 * (1 + 1) / 2 = 0.5
        let unweighted = QuantileMetric::median().evaluate(&preds, &labels, None, 1);
        assert!((unweighted - 0.5).abs() < 1e-10);

        // Weighted with equal weights should match
        let weights = vec![1.0, 1.0];
        let weighted = QuantileMetric::median().evaluate(&preds, &labels, Some(&weights), 1);
        assert!((weighted - unweighted).abs() < 1e-10);

        // Weighted emphasizing sample 0
        let weights2 = vec![10.0, 1.0];
        let weighted2 = QuantileMetric::median().evaluate(&preds, &labels, Some(&weights2), 1);
        // Both samples have same error, so result should be same
        assert!((weighted2 - 0.5).abs() < 1e-10);
    }

    #[test]
    fn metric_properties() {
        assert!(!Rmse.higher_is_better());
        assert!(!Mae.higher_is_better());
        assert!(!Mape.higher_is_better());
        assert!(!QuantileMetric::median().higher_is_better());

        assert_eq!(Rmse.name(), "rmse");
        assert_eq!(Mae.name(), "mae");
        assert_eq!(Mape.name(), "mape");
        assert_eq!(QuantileMetric::median().name(), "quantile");
    }
}

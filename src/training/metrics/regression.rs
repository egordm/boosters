//! Regression metrics.
//!
//! Metrics for evaluating regression model quality.
//!
//! # Multi-Output Support
//!
//! For multi-output models, metrics are computed per-output and then averaged.
//! This provides an honest aggregate measure across all outputs.

use super::{weight_iter, Metric};
use crate::inference::common::PredictionKind;

// =============================================================================
// RMSE (Root Mean Squared Error)
// =============================================================================

/// Root Mean Squared Error: sqrt(mean((pred - label)²))
///
/// Lower is better. Used for regression tasks.
///
/// # Multi-Output
///
/// For multi-output models, computes RMSE per output and returns the average.
///
/// # Weighted Computation
///
/// When weights are provided, computes weighted RMSE:
/// ```text
/// sqrt(sum(w * (p - l)²) / sum(w))
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct Rmse;

impl Rmse {
    /// Compute RMSE for a single output.
    fn compute_single(preds: &[f32], labels: &[f32], weights: &[f32]) -> f64 {
        if preds.is_empty() {
            return 0.0;
        }

        if weights.is_empty() {
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
        } else {
            let (sum_sq, sum_w) = preds
                .iter()
                .zip(labels.iter())
                .zip(weights.iter())
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

impl Metric for Rmse {
    fn compute(
        &self,
        n_rows: usize,
        n_outputs: usize,
        predictions: &[f32],
        targets: &[f32],
        weights: &[f32],
    ) -> f64 {
        if n_rows == 0 {
            return 0.0;
        }
        debug_assert!(predictions.len() >= n_rows * n_outputs);

        // Determine targets layout: could be [n_outputs * n_rows] or [n_rows] (shared)
        let targets_per_output = targets.len() >= n_outputs * n_rows;

        let mut sum_rmse = 0.0f64;
        for out_idx in 0..n_outputs {
            let preds = &predictions[out_idx * n_rows..(out_idx + 1) * n_rows];
            let labels = if targets_per_output {
                &targets[out_idx * n_rows..(out_idx + 1) * n_rows]
            } else {
                &targets[..n_rows]
            };
            sum_rmse += Self::compute_single(preds, labels, weights);
        }

        sum_rmse / n_outputs as f64
    }

    fn higher_is_better(&self) -> bool {
        false
    }

    fn expected_prediction_kind(&self) -> PredictionKind {
        PredictionKind::Value
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
///
/// # Multi-Output
///
/// For multi-output models, computes MAE per output and returns the average.
///
/// # Weighted Computation
///
/// When weights are provided, computes weighted MAE:
/// ```text
/// sum(w * |p - l|) / sum(w)
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct Mae;

impl Mae {
    /// Compute MAE for a single output.
    fn compute_single(preds: &[f32], labels: &[f32], weights: &[f32]) -> f64 {
        if preds.is_empty() {
            return 0.0;
        }

        if weights.is_empty() {
            let sum: f64 = preds
                .iter()
                .zip(labels.iter())
                .map(|(p, l)| ((*p as f64) - (*l as f64)).abs())
                .sum();
            sum / preds.len() as f64
        } else {
            let (sum_ae, sum_w) = preds
                .iter()
                .zip(labels.iter())
                .zip(weights.iter())
                .fold((0.0f64, 0.0f64), |(sa, sw), ((&p, &l), &wt)| {
                    let ae = ((p as f64) - (l as f64)).abs();
                    (sa + (wt as f64) * ae, sw + wt as f64)
                });
            if sum_w > 0.0 {
                sum_ae / sum_w
            } else {
                0.0
            }
        }
    }
}

impl Metric for Mae {
    fn compute(
        &self,
        n_rows: usize,
        n_outputs: usize,
        predictions: &[f32],
        targets: &[f32],
        weights: &[f32],
    ) -> f64 {
        if n_rows == 0 {
            return 0.0;
        }
        debug_assert!(predictions.len() >= n_rows * n_outputs);

        // Determine targets layout: could be [n_outputs * n_rows] or [n_rows] (shared)
        let targets_per_output = targets.len() >= n_outputs * n_rows;

        let mut sum_mae = 0.0f64;
        for out_idx in 0..n_outputs {
            let preds = &predictions[out_idx * n_rows..(out_idx + 1) * n_rows];
            let labels = if targets_per_output {
                &targets[out_idx * n_rows..(out_idx + 1) * n_rows]
            } else {
                &targets[..n_rows]
            };
            sum_mae += Self::compute_single(preds, labels, weights);
        }

        sum_mae / n_outputs as f64
    }

    fn higher_is_better(&self) -> bool {
        false
    }

    fn expected_prediction_kind(&self) -> PredictionKind {
        PredictionKind::Value
    }

    fn name(&self) -> &'static str {
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
    fn compute(
        &self,
        n_rows: usize,
        n_outputs: usize,
        predictions: &[f32],
        targets: &[f32],
        weights: &[f32],
    ) -> f64 {
        let _ = n_outputs; // May be > 1 for multi-output, but MAPE uses first output
        if n_rows == 0 {
            return 0.0;
        }

        let predictions = &predictions[..n_rows];
        let labels = &targets[..n_rows];
        debug_assert!(targets.len() >= n_rows);

        let eps = 1e-15f64; // Avoid division by zero

        if weights.is_empty() {
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
        } else {
                debug_assert_eq!(predictions.len(), weights.len());

                let (weighted_sum, weight_sum) = predictions
                    .iter()
                    .zip(labels.iter())
                    .zip(weights.iter())
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

    fn higher_is_better(&self) -> bool {
        false
    }

    fn expected_prediction_kind(&self) -> PredictionKind {
        PredictionKind::Value
    }

    fn name(&self) -> &'static str {
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

    /// Compute pinball loss for a single quantile level without allocation.
    /// 
    /// This avoids the Vec allocation in [`Self::new`] for the common single-quantile case.
    pub fn compute_single(
        alpha: f32,
        n_rows: usize,
        predictions: &[f32],
        targets: &[f32],
        weights: &[f32],
    ) -> f64 {
        if n_rows == 0 {
            return 0.0;
        }

        let labels = &targets[..n_rows];
        let alpha_f64 = alpha as f64;

        let (weighted_loss, weight_sum): (f64, f64) = labels
            .iter()
            .zip(predictions.iter())
            .zip(weight_iter(weights, n_rows))
            .fold((0.0, 0.0), |(acc_loss, acc_w), ((&y, &pred), wt)| {
                let residual = y as f64 - pred as f64;
                let wt = wt as f64;
                let loss = if residual >= 0.0 {
                    alpha_f64 * residual
                } else {
                    (1.0 - alpha_f64) * (-residual)
                };
                (acc_loss + wt * loss, acc_w + wt)
            });

        if weight_sum == 0.0 {
            return 0.0;
        }

        weighted_loss / weight_sum
    }
}

impl Default for QuantileMetric {
    fn default() -> Self {
        Self::median()
    }
}

impl Metric for QuantileMetric {
    fn compute(
        &self,
        n_rows: usize,
        n_outputs: usize,
        predictions: &[f32],
        targets: &[f32],
        weights: &[f32],
    ) -> f64 {
        if n_rows == 0 || n_outputs == 0 {
            return 0.0;
        }

        let n_quantiles = self.alphas.len();
        debug_assert_eq!(n_outputs, n_quantiles);
        debug_assert_eq!(predictions.len(), n_rows * n_quantiles);

        // Single quantile: delegate to optimized compute_single (no multi-quantile overhead)
        if n_quantiles == 1 {
            return Self::compute_single(
                self.alphas[0],
                n_rows,
                predictions,
                targets,
                weights,
            );
        }

        // Multi-quantile: compute weighted loss across all quantiles
        // Uses weight_iter for unified weighted/unweighted handling
        let labels = &targets[..n_rows];

        let (weighted_loss, weight_sum): (f64, f64) = self
            .alphas
            .iter()
            .enumerate()
            .map(|(q, &alpha)| {
                let alpha_f64 = alpha as f64;
                labels
                    .iter()
                    .enumerate()
                    .zip(weight_iter(weights, n_rows))
                    .fold((0.0f64, 0.0f64), |(acc_loss, acc_w), ((row, &label), wt)| {
                        // Column-major: predictions[q * n_rows + row]
                        let pred = predictions[q * n_rows + row] as f64;
                        let residual = label as f64 - pred;
                        let wt = wt as f64;
                        let loss = if residual >= 0.0 {
                            alpha_f64 * residual
                        } else {
                            (1.0 - alpha_f64) * (-residual)
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

    fn higher_is_better(&self) -> bool {
        false
    }

    fn expected_prediction_kind(&self) -> PredictionKind {
        PredictionKind::Value
    }

    fn name(&self) -> &'static str {
        "quantile"
    }
}

// =============================================================================
// Poisson Deviance
// =============================================================================

/// Poisson deviance metric for count data.
///
/// Deviance = 2 * mean(y * log(y/pred) - (y - pred))
///
/// This is the natural metric for Poisson regression models.
/// Predictions should be positive (typically from exp transform).
///
/// Lower is better.
///
/// # Weighted Computation
///
/// When weights are provided, computes weighted deviance.
#[derive(Debug, Clone, Copy, Default)]
pub struct PoissonDeviance;

impl Metric for PoissonDeviance {
    fn compute(
        &self,
        n_rows: usize,
        n_outputs: usize,
        predictions: &[f32],
        targets: &[f32],
        weights: &[f32],
    ) -> f64 {
        let _ = n_outputs; // May be > 1 for multi-output, but Poisson uses first output
        if n_rows == 0 {
            return 0.0;
        }

        let predictions = &predictions[..n_rows];
        let labels = &targets[..n_rows];
        debug_assert!(targets.len() >= n_rows);

        const EPSILON: f64 = 1e-9;

        if weights.is_empty() {
                let deviance: f64 = predictions
                    .iter()
                    .zip(labels.iter())
                    .map(|(&p, &l)| {
                        let y = l as f64;
                        let mu = (p as f64).max(EPSILON);

                        // Deviance contribution:
                        // 2 * (y*log(y/mu) - (y - mu))
                        // When y=0: 2 * (0 - (-mu)) = 2*mu
                        if y > EPSILON {
                            2.0 * (y * (y / mu).ln() - (y - mu))
                        } else {
                            2.0 * mu
                        }
                    })
                    .sum::<f64>();

                deviance / predictions.len() as f64
        } else {
                debug_assert_eq!(weights.len(), predictions.len());

                let (sum_wdev, sum_w) = predictions
                    .iter()
                    .zip(labels.iter())
                    .zip(weights.iter())
                    .fold((0.0f64, 0.0f64), |(swd, sw), ((&p, &l), &wt)| {
                        let y = l as f64;
                        let mu = (p as f64).max(EPSILON);
                        let wt = wt as f64;

                        let dev = if y > EPSILON {
                            2.0 * (y * (y / mu).ln() - (y - mu))
                        } else {
                            2.0 * mu
                        };

                        (swd + wt * dev, sw + wt)
                    });

                if sum_w > 0.0 {
                    sum_wdev / sum_w
                } else {
                    0.0
                }
        }
    }

    fn higher_is_better(&self) -> bool {
        false
    }

    fn expected_prediction_kind(&self) -> PredictionKind {
        PredictionKind::Value
    }

    fn name(&self) -> &'static str {
        "poisson"
    }
}

// =============================================================================
// Huber Metric
// =============================================================================

/// Huber loss metric for robust regression.
///
/// The Huber loss is quadratic for small residuals and linear for large
/// residuals, providing robustness to outliers.
///
/// For residual r = |pred - label|:
/// - Loss = 0.5 * r² if r ≤ delta
/// - Loss = delta * (r - 0.5 * delta) otherwise
///
/// Lower is better.
#[derive(Debug, Clone, Copy)]
pub struct HuberMetric {
    /// Threshold at which the loss transitions from quadratic to linear.
    pub delta: f64,
}

impl HuberMetric {
    /// Create a new Huber metric with the given delta threshold.
    pub fn new(delta: f64) -> Self {
        debug_assert!(delta > 0.0, "delta must be positive, got {}", delta);
        Self { delta }
    }
}

impl Default for HuberMetric {
    fn default() -> Self {
        Self { delta: 1.0 }
    }
}

impl Metric for HuberMetric {
    fn compute(
        &self,
        n_rows: usize,
        n_outputs: usize,
        predictions: &[f32],
        targets: &[f32],
        weights: &[f32],
    ) -> f64 {
        let _ = n_outputs; // May be > 1 for multi-output, but Huber uses first output
        if n_rows == 0 {
            return 0.0;
        }

        let predictions = &predictions[..n_rows];
        let labels = &targets[..n_rows];
        debug_assert!(targets.len() >= n_rows);

        let delta = self.delta;

        if weights.is_empty() {
                let loss: f64 = predictions
                    .iter()
                    .zip(labels.iter())
                    .map(|(&p, &l)| {
                        let r = ((p as f64) - (l as f64)).abs();
                        if r <= delta {
                            0.5 * r * r
                        } else {
                            delta * (r - 0.5 * delta)
                        }
                    })
                    .sum::<f64>();

                loss / predictions.len() as f64
        } else {
                debug_assert_eq!(weights.len(), predictions.len());

                let (sum_wloss, sum_w) = predictions
                    .iter()
                    .zip(labels.iter())
                    .zip(weights.iter())
                    .fold((0.0f64, 0.0f64), |(swl, sw), ((&p, &l), &wt)| {
                        let r = ((p as f64) - (l as f64)).abs();
                        let wt = wt as f64;
                        let loss = if r <= delta {
                            0.5 * r * r
                        } else {
                            delta * (r - 0.5 * delta)
                        };
                        (swl + wt * loss, sw + wt)
                    });

                if sum_w > 0.0 {
                    sum_wloss / sum_w
                } else {
                    0.0
                }
        }
    }

    fn higher_is_better(&self) -> bool {
        false
    }

    fn expected_prediction_kind(&self) -> PredictionKind {
        PredictionKind::Value
    }

    fn name(&self) -> &'static str {
        "huber"
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
        let rmse = Rmse.compute(3, 1, &preds, &labels, &[]);
        assert!(rmse.abs() < 1e-10);
    }

    #[test]
    fn rmse_known_value() {
        // RMSE of [1, 2] vs [0, 0] = sqrt((1 + 4) / 2) = sqrt(2.5)
        let preds = vec![1.0, 2.0];
        let labels = vec![0.0, 0.0];
        let rmse = Rmse.compute(2, 1, &preds, &labels, &[]);
        assert!((rmse - 2.5f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn weighted_rmse_uniform_weights_equals_unweighted() {
        let preds = vec![1.0, 2.0, 3.0, 4.0];
        let labels = vec![1.5, 2.5, 2.5, 4.5];
        let weights = vec![1.0, 1.0, 1.0, 1.0];

        let unweighted = Rmse.compute(4, 1, &preds, &labels, &[]);
        let weighted = Rmse.compute(4, 1, &preds, &labels, &weights);

        assert!((weighted - unweighted).abs() < 1e-10);
    }

    #[test]
    fn weighted_rmse_emphasizes_high_weight_samples() {
        // Sample 0 has error 1.0, sample 1 has error 0.0
        let preds = vec![1.0, 2.0];
        let labels = vec![2.0, 2.0];

        // With equal weights: RMSE = sqrt((1 + 0) / 2) = sqrt(0.5)
        let equal_weights = vec![1.0, 1.0];
        let rmse_equal = Rmse.compute(2, 1, &preds, &labels, &equal_weights);
        assert!((rmse_equal - 0.5f64.sqrt()).abs() < 1e-10);

        // With high weight on sample 0: RMSE moves toward error of sample 0
        // w * (err)^2 / w = 10 * 1 + 1 * 0 / 11 = 10/11
        let unequal_weights = vec![10.0, 1.0];
        let rmse_unequal = Rmse.compute(2, 1, &preds, &labels, &unequal_weights);
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
        let mae = Mae.compute(3, 1, &preds, &labels, &[]);
        assert!(mae.abs() < 1e-10);
    }

    #[test]
    fn mae_known_value() {
        // MAE of [1, 2] vs [0, 0] = (1 + 2) / 2 = 1.5
        let preds = vec![1.0, 2.0];
        let labels = vec![0.0, 0.0];
        let mae = Mae.compute(2, 1, &preds, &labels, &[]);
        assert!((mae - 1.5).abs() < 1e-10);
    }

    #[test]
    fn weighted_mae_emphasizes_high_weight_samples() {
        // Sample 0 has error 2.0, sample 1 has error 1.0
        let preds = vec![0.0, 2.0];
        let labels = vec![2.0, 3.0];

        // Unweighted MAE = (2 + 1) / 2 = 1.5
        let unweighted = Mae.compute(2, 1, &preds, &labels, &[]);
        assert!((unweighted - 1.5).abs() < 1e-10);

        // With high weight on sample 0: MAE = (10*2 + 1*1) / 11 = 21/11
        let weights = vec![10.0, 1.0];
        let weighted = Mae.compute(2, 1, &preds, &labels, &weights);
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
        let mape = Mape.compute(3, 1, &preds, &labels, &[]);
        assert!(mape.abs() < 1e-10);
    }

    #[test]
    fn mape_known_value() {
        // MAPE: mean(|pred - label| / |label|) * 100
        // |1-2|/2 = 0.5, |3-4|/4 = 0.25 → mean = 0.375 → 37.5%
        let preds = vec![1.0, 3.0];
        let labels = vec![2.0, 4.0];
        let mape = Mape.compute(2, 1, &preds, &labels, &[]);
        assert!((mape - 37.5).abs() < 1e-10);
    }

    #[test]
    fn weighted_mape_emphasizes_high_weight_samples() {
        // Sample 0: pred=1, label=2 → APE = 0.5 (50%)
        // Sample 1: pred=3, label=4 → APE = 0.25 (25%)
        let preds = vec![1.0, 3.0];
        let labels = vec![2.0, 4.0];

        // Unweighted MAPE = (50 + 25) / 2 = 37.5%
        let unweighted = Mape.compute(2, 1, &preds, &labels, &[]);
        assert!((unweighted - 37.5).abs() < 1e-10);

        // High weight on sample 0 (higher error): MAPE increases
        let weights = vec![10.0, 1.0];
        let weighted = Mape.compute(2, 1, &preds, &labels, &weights);
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
        let loss = QuantileMetric::median().compute(3, 1, &preds, &labels, &[]);
        assert!(loss.abs() < 1e-10);
    }

    #[test]
    fn quantile_median_error() {
        // Median (alpha=0.5): pinball loss = 0.5 * |y - q|
        // |1-2| = 1, |3-2| = 1 → pinball each = 0.5 → mean = 0.5
        let preds = vec![2.0, 2.0];
        let labels = vec![1.0, 3.0];
        let loss = QuantileMetric::median().compute(2, 1, &preds, &labels, &[]);
        assert!((loss - 0.5).abs() < 1e-10);
    }

    #[test]
    fn quantile_asymmetric() {
        // Alpha = 0.1: penalize over-prediction more
        // y=5, q=3: residual=2 (under-predict) → 0.1 * 2 = 0.2
        // y=5, q=7: residual=-2 (over-predict) → 0.9 * 2 = 1.8
        let alphas = vec![0.1];
        let metric = QuantileMetric::new(alphas);

        let loss_under = metric.compute(1, 1, &[3.0], &[5.0], &[]);
        assert!((loss_under - 0.2).abs() < 1e-6, "got {}", loss_under);

        let loss_over = metric.compute(1, 1, &[7.0], &[5.0], &[]);
        assert!((loss_over - 1.8).abs() < 1e-6, "got {}", loss_over);
    }

    #[test]
    fn weighted_quantile_loss() {
        // Median (alpha=0.5): pinball loss = 0.5 * |y - q|
        let preds = vec![2.0, 2.0];
        let labels = vec![1.0, 3.0]; // errors: 1, 1

        // Unweighted: 0.5 * (1 + 1) / 2 = 0.5
        let unweighted = QuantileMetric::median().compute(2, 1, &preds, &labels, &[]);
        assert!((unweighted - 0.5).abs() < 1e-10);

        // Weighted with equal weights should match
        let weights = vec![1.0, 1.0];
        let weighted = QuantileMetric::median().compute(2, 1, &preds, &labels, &weights);
        assert!((weighted - unweighted).abs() < 1e-10);

        // Weighted emphasizing sample 0
        let weights2 = vec![10.0, 1.0];
        let weighted2 = QuantileMetric::median().compute(2, 1, &preds, &labels, &weights2);
        // Both samples have same error, so result should be same
        assert!((weighted2 - 0.5).abs() < 1e-10);
    }

    #[test]
    fn metric_properties() {
        assert!(!Rmse.higher_is_better());
        assert!(!Mae.higher_is_better());
        assert!(!Mape.higher_is_better());
        assert!(!QuantileMetric::median().higher_is_better());
        assert!(!PoissonDeviance.higher_is_better());
        assert!(!HuberMetric::default().higher_is_better());

        assert_eq!(Rmse.name(), "rmse");
        assert_eq!(Mae.name(), "mae");
        assert_eq!(Mape.name(), "mape");
        assert_eq!(QuantileMetric::median().name(), "quantile");
        assert_eq!(PoissonDeviance.name(), "poisson");
        assert_eq!(HuberMetric::default().name(), "huber");
    }

    // =========================================================================
    // Poisson Deviance tests
    // =========================================================================

    #[test]
    fn poisson_deviance_perfect() {
        // When predictions exactly match labels, deviance should be minimal
        let preds = vec![1.0, 2.0, 3.0];
        let labels = vec![1.0, 2.0, 3.0];
        let dev = PoissonDeviance.compute(3, 1, &preds, &labels, &[]);
        assert!(dev.abs() < 1e-8);
    }

    #[test]
    fn poisson_deviance_zero_labels() {
        // When y=0: deviance contribution = 2 * mu
        // preds = [1.0, 2.0], labels = [0.0, 0.0]
        // deviance = mean(2*1 + 2*2) = mean(2 + 4) = 3
        let preds = vec![1.0, 2.0];
        let labels = vec![0.0, 0.0];
        let dev = PoissonDeviance.compute(2, 1, &preds, &labels, &[]);
        assert!((dev - 3.0).abs() < 1e-6, "got {} expected 3.0", dev);
    }

    #[test]
    fn poisson_deviance_known_value() {
        // Single sample: y=4, mu=2
        // deviance = 2 * (4 * ln(4/2) - (4 - 2)) = 2 * (4 * ln(2) - 2)
        //          = 2 * (4 * 0.693... - 2) = 2 * (2.773 - 2) = 2 * 0.773 ≈ 1.545
        let preds = vec![2.0];
        let labels = vec![4.0];
        let dev = PoissonDeviance.compute(1, 1, &preds, &labels, &[]);
        let expected = 2.0 * (4.0 * 2.0f64.ln() - 2.0);
        assert!((dev - expected).abs() < 1e-6, "got {} expected {}", dev, expected);
    }

    #[test]
    fn weighted_poisson_deviance() {
        let preds = vec![1.0, 2.0];
        let labels = vec![0.0, 0.0];
        
        // Unweighted: mean(2*1 + 2*2) = 3
        let unweighted = PoissonDeviance.compute(2, 1, &preds, &labels, &[]);
        
        // Uniform weights should match
        let weights = vec![1.0, 1.0];
        let weighted = PoissonDeviance.compute(2, 1, &preds, &labels, &weights);
        assert!((weighted - unweighted).abs() < 1e-10);
        
        // Heavy weight on sample 0: (10*2 + 1*4) / 11 = 24/11 ≈ 2.18
        let weights2 = vec![10.0, 1.0];
        let weighted2 = PoissonDeviance.compute(2, 1, &preds, &labels, &weights2);
        let expected = (10.0 * 2.0 + 1.0 * 4.0) / 11.0;
        assert!((weighted2 - expected).abs() < 1e-6, "got {} expected {}", weighted2, expected);
    }

    // =========================================================================
    // Huber Metric tests
    // =========================================================================

    #[test]
    fn huber_perfect() {
        let preds = vec![1.0, 2.0, 3.0];
        let labels = vec![1.0, 2.0, 3.0];
        let loss = HuberMetric::default().compute(3, 1, &preds, &labels, &[]);
        assert!(loss.abs() < 1e-10);
    }

    #[test]
    fn huber_quadratic_region() {
        // For residual r <= delta, loss = 0.5 * r^2
        // delta=1.0, residual=0.5 → loss = 0.5 * 0.25 = 0.125
        let preds = vec![1.5];
        let labels = vec![1.0];
        let loss = HuberMetric::new(1.0).compute(1, 1, &preds, &labels, &[]);
        assert!((loss - 0.125).abs() < 1e-10, "got {} expected 0.125", loss);
    }

    #[test]
    fn huber_linear_region() {
        // For residual r > delta, loss = delta * (r - 0.5 * delta)
        // delta=1.0, residual=2.0 → loss = 1.0 * (2.0 - 0.5) = 1.5
        let preds = vec![3.0];
        let labels = vec![1.0];
        let loss = HuberMetric::new(1.0).compute(1, 1, &preds, &labels, &[]);
        assert!((loss - 1.5).abs() < 1e-10, "got {} expected 1.5", loss);
    }

    #[test]
    fn huber_boundary() {
        // At boundary r = delta, both formulas give same result
        // delta=1.0, residual=1.0
        // quadratic: 0.5 * 1.0^2 = 0.5
        // linear: 1.0 * (1.0 - 0.5) = 0.5
        let preds = vec![2.0];
        let labels = vec![1.0];
        let loss = HuberMetric::new(1.0).compute(1, 1, &preds, &labels, &[]);
        assert!((loss - 0.5).abs() < 1e-10, "got {} expected 0.5", loss);
    }

    #[test]
    fn huber_custom_delta() {
        // delta=2.0, residual=1.0 (in quadratic region)
        // loss = 0.5 * 1.0^2 = 0.5
        let preds = vec![2.0];
        let labels = vec![1.0];
        let loss = HuberMetric::new(2.0).compute(1, 1, &preds, &labels, &[]);
        assert!((loss - 0.5).abs() < 1e-10);

        // delta=2.0, residual=3.0 (in linear region)
        // loss = 2.0 * (3.0 - 1.0) = 4.0
        let preds2 = vec![4.0];
        let labels2 = vec![1.0];
        let loss2 = HuberMetric::new(2.0).compute(1, 1, &preds2, &labels2, &[]);
        assert!((loss2 - 4.0).abs() < 1e-10, "got {} expected 4.0", loss2);
    }

    #[test]
    fn weighted_huber() {
        // Sample 0: residual=0.5 (quadratic), loss=0.125
        // Sample 1: residual=2.0 (linear, delta=1), loss=1.5
        let preds = vec![1.5, 3.0];
        let labels = vec![1.0, 1.0];
        
        // Unweighted: (0.125 + 1.5) / 2 = 0.8125
        let unweighted = HuberMetric::new(1.0).compute(2, 1, &preds, &labels, &[]);
        assert!((unweighted - 0.8125).abs() < 1e-10);
        
        // Heavy weight on sample 0 (smaller loss):
        // (10 * 0.125 + 1 * 1.5) / 11 = (1.25 + 1.5) / 11 = 2.75 / 11 ≈ 0.25
        let weights = vec![10.0, 1.0];
        let weighted = HuberMetric::new(1.0).compute(2, 1, &preds, &labels, &weights);
        let expected = 2.75 / 11.0;
        assert!((weighted - expected).abs() < 1e-6, "got {} expected {}", weighted, expected);
    }
}

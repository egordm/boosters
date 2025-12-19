//! Regression objective functions.

use super::{validate_objective_inputs, Objective, TargetSchema, TaskKind};
use crate::inference::common::{PredictionKind, PredictionOutput};
use crate::training::GradsTuple;
use crate::training::metrics::MetricKind;
use crate::utils::weight_iter;

// =============================================================================
// Squared Loss
// =============================================================================

/// Squared error loss (L2 loss) for regression.
///
/// Supports multi-output regression where each output has its own target.
///
/// - Loss: `0.5 * (pred - target)²`
/// - Gradient: `pred - target`
/// - Hessian: `1.0`
///
/// # Multi-Output
///
/// For `n_outputs` outputs, expects `n_outputs` targets (1:1 mapping).
/// Each output independently computes gradients against its corresponding target.
#[derive(Debug, Clone, Copy, Default)]
pub struct SquaredLoss;

impl Objective for SquaredLoss {
    fn compute_gradients(
        &self,
        n_rows: usize,
        n_outputs: usize,
        predictions: &[f32],
        targets: &[f32],
        weights: &[f32],
        grad_hess: &mut [GradsTuple],
    ) {
        validate_objective_inputs(
            n_rows,
            n_outputs,
            predictions.len(),
            grad_hess.len(),
            weights,
        );
        debug_assert!(targets.len() >= n_outputs * n_rows);

        // Process each output
        for out_idx in 0..n_outputs {
            let offset = out_idx * n_rows;
            let pred_slice = &predictions[offset..offset + n_rows];
            let target_slice = &targets[offset..offset + n_rows];
            let pair_slice = &mut grad_hess[offset..offset + n_rows];

            for (i, w) in weight_iter(weights, n_rows).enumerate() {
                pair_slice[i].grad = w * (pred_slice[i] - target_slice[i]);
                pair_slice[i].hess = w;
            }
        }
    }

    fn compute_base_score(
        &self,
        n_rows: usize,
        n_outputs: usize,
        targets: &[f32],
        weights: &[f32],
        outputs: &mut [f32],
    ) {
        debug_assert!(targets.len() >= n_outputs * n_rows);
        debug_assert!(outputs.len() >= n_outputs);
        debug_assert!(weights.is_empty() || weights.len() >= n_rows);

        if n_rows == 0 {
            outputs[..n_outputs].fill(0.0);
            return;
        }

        // Compute weighted mean for each output
        for out_idx in 0..n_outputs {
            let offset = out_idx * n_rows;
            let target_slice = &targets[offset..offset + n_rows];

            let (sum_w, sum_wy) = target_slice
                .iter()
                .zip(weight_iter(weights, n_rows))
                .fold((0.0f64, 0.0f64), |(sw, swy), (&y, w)| {
                    (sw + w as f64, swy + w as f64 * y as f64)
                });

            outputs[out_idx] = if sum_w > 0.0 {
                (sum_wy / sum_w) as f32
            } else {
                0.0
            };
        }
    }

    fn name(&self) -> &'static str {
        "squared"
    }

    fn task_kind(&self) -> TaskKind {
        TaskKind::Regression
    }

    fn target_schema(&self) -> TargetSchema {
        TargetSchema::Continuous
    }

    fn default_metric(&self) -> MetricKind {
        MetricKind::Rmse
    }

    fn transform_prediction_inplace(&self, _raw: &mut PredictionOutput) -> PredictionKind {
        PredictionKind::Value
    }
}

// =============================================================================
// Pinball Loss (Quantile Regression)
// =============================================================================

/// Pinball loss for quantile regression.
///
/// Also known as quantile loss. For quantile `α`:
/// - Loss: `α * (target - pred)` if `target > pred`, else `(1-α) * (pred - target)`
/// - Gradient: `α - 1` if `pred < target`, else `α`
/// - Hessian: `1.0` (constant)
///
/// # Multi-Output (Multiple Quantiles)
///
/// When predicting multiple quantiles (e.g., [0.1, 0.5, 0.9]):
/// - `n_outputs` = number of quantiles
/// - `alphas` = quantile levels for each output
/// - Targets can be:
///   - **n_targets = 1**: Single target shared across all quantiles
///   - **n_targets = n_outputs**: Each quantile has its own target
///
/// Common quantiles:
/// - `α = 0.5`: Median regression (MAE equivalent)
/// - `α = 0.1`: 10th percentile (lower bound)
/// - `α = 0.9`: 90th percentile (upper bound)
#[derive(Debug, Clone)]
pub struct PinballLoss {
    /// Quantile levels for each output, each in (0, 1).
    pub alphas: Vec<f32>,
}

impl PinballLoss {
    /// Create a pinball loss for a single quantile.
    ///
    /// # Arguments
    ///
    /// * `alpha` - Quantile level in (0, 1). E.g., 0.5 for median.
    pub fn new(alpha: f32) -> Self {
        debug_assert!(
            alpha > 0.0 && alpha < 1.0,
            "alpha must be in (0, 1), got {}",
            alpha
        );
        Self {
            alphas: vec![alpha],
        }
    }

    /// Create a pinball loss for multiple quantiles.
    ///
    /// # Arguments
    ///
    /// * `alphas` - Quantile levels, each in (0, 1).
    pub fn with_quantiles(alphas: Vec<f32>) -> Self {
        debug_assert!(!alphas.is_empty(), "alphas must not be empty");
        for &a in &alphas {
            debug_assert!(a > 0.0 && a < 1.0, "alpha must be in (0, 1), got {}", a);
        }
        Self { alphas }
    }

    /// Number of quantiles (outputs).
    #[inline]
    pub fn n_quantiles(&self) -> usize {
        self.alphas.len()
    }
}

impl Objective for PinballLoss {
    fn n_outputs(&self) -> usize {
        self.alphas.len()
    }

    fn compute_gradients(
        &self,
        n_rows: usize,
        n_outputs: usize,
        predictions: &[f32],
        targets: &[f32],
        weights: &[f32],
        grad_hess: &mut [GradsTuple],
    ) {
        debug_assert_eq!(n_outputs, self.alphas.len());
        validate_objective_inputs(
            n_rows,
            n_outputs,
            predictions.len(),
            grad_hess.len(),
            weights,
        );

        // Determine if targets are shared (n_targets=1) or per-output (n_targets=n_outputs)
        let n_targets = targets.len() / n_rows;
        let shared_target = n_targets == 1;
        debug_assert!(n_targets == 1 || n_targets == n_outputs);

        for (out_idx, &alpha) in self.alphas.iter().enumerate() {
            let pred_offset = out_idx * n_rows;
            let target_offset = if shared_target { 0 } else { out_idx * n_rows };

            let pred_slice = &predictions[pred_offset..pred_offset + n_rows];
            let target_slice = &targets[target_offset..target_offset + n_rows];
            let pair_slice = &mut grad_hess[pred_offset..pred_offset + n_rows];

            for (i, w) in weight_iter(weights, n_rows).enumerate() {
                let diff = pred_slice[i] - target_slice[i];
                let g = if diff < 0.0 { alpha - 1.0 } else { alpha };
                pair_slice[i].grad = w * g;
                pair_slice[i].hess = w;
            }
        }
    }

    fn compute_base_score(
        &self,
        n_rows: usize,
        n_outputs: usize,
        targets: &[f32],
        weights: &[f32],
        outputs: &mut [f32],
    ) {
        debug_assert_eq!(n_outputs, self.alphas.len());
        debug_assert!(outputs.len() >= n_outputs);
        debug_assert!(weights.is_empty() || weights.len() >= n_rows);

        if n_rows == 0 {
            outputs[..n_outputs].fill(0.0);
            return;
        }

        let n_targets = targets.len() / n_rows;
        let shared_target = n_targets == 1;

        for (out_idx, &alpha) in self.alphas.iter().enumerate() {
            let target_offset = if shared_target { 0 } else { out_idx * n_rows };
            let target_slice = &targets[target_offset..target_offset + n_rows];

            // Compute weighted quantile
            let mut sorted: Vec<(f32, f32)> = target_slice
                .iter()
                .zip(weight_iter(weights, n_rows))
                .map(|(&t, w)| (t, w))
                .collect();
            sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            let total_weight: f32 = sorted.iter().map(|(_, w)| w).sum();
            let target_weight = alpha * total_weight;

            let mut cumulative = 0.0f32;
            outputs[out_idx] = sorted.last().map(|(v, _)| *v).unwrap_or(0.0);
            for (value, w) in &sorted {
                cumulative += w;
                if cumulative >= target_weight {
                    outputs[out_idx] = *value;
                    break;
                }
            }
        }
    }

    fn task_kind(&self) -> TaskKind {
        TaskKind::Regression
    }

    fn target_schema(&self) -> TargetSchema {
        TargetSchema::Continuous
    }

    fn default_metric(&self) -> MetricKind {
        MetricKind::Quantile
    }

    fn transform_prediction_inplace(&self, _raw: &mut PredictionOutput) -> PredictionKind {
        PredictionKind::Value
    }

    fn name(&self) -> &'static str {
        "pinball"
    }
}

// =============================================================================
// Pseudo-Huber Loss
// =============================================================================

/// Pseudo-Huber loss for robust regression.
///
/// A smooth approximation to Huber loss that transitions from quadratic
/// near zero to linear for large residuals:
/// - Loss: `delta² * (sqrt(1 + (residual/delta)²) - 1)`
/// - Gradient: `residual / sqrt(1 + (residual/delta)²)`
/// - Hessian: `1 / (1 + (residual/delta)²)^1.5`
///
/// # Multi-Output
///
/// Supports multi-output regression with 1:1 output-target mapping.
///
/// The `delta` parameter controls the transition point:
/// - Large delta: Behaves like squared loss
/// - Small delta: Behaves like absolute loss (more robust to outliers)
#[derive(Debug, Clone, Copy)]
pub struct PseudoHuberLoss {
    /// Transition parameter.
    pub delta: f32,
}

impl PseudoHuberLoss {
    /// Create a new Pseudo-Huber loss with the given delta.
    ///
    /// # Arguments
    ///
    /// * `delta` - Transition parameter. Larger values make the loss more quadratic.
    pub fn new(delta: f32) -> Self {
        debug_assert!(delta > 0.0, "delta must be positive, got {}", delta);
        Self { delta }
    }
}

impl Objective for PseudoHuberLoss {
    fn compute_gradients(
        &self,
        n_rows: usize,
        n_outputs: usize,
        predictions: &[f32],
        targets: &[f32],
        weights: &[f32],
        grad_hess: &mut [GradsTuple],
    ) {
        validate_objective_inputs(
            n_rows,
            n_outputs,
            predictions.len(),
            grad_hess.len(),
            weights,
        );
        debug_assert!(targets.len() >= n_outputs * n_rows);

        let delta_sq = self.delta * self.delta;
        let inv_delta_sq = 1.0 / delta_sq;

        for out_idx in 0..n_outputs {
            let offset = out_idx * n_rows;
            let pred_slice = &predictions[offset..offset + n_rows];
            let target_slice = &targets[offset..offset + n_rows];
            let pair_slice = &mut grad_hess[offset..offset + n_rows];

            for (i, w) in weight_iter(weights, n_rows).enumerate() {
                let residual = pred_slice[i] - target_slice[i];
                let r_sq = residual * residual;
                let factor = 1.0 + r_sq * inv_delta_sq;
                let sqrt_factor = factor.sqrt();

                pair_slice[i].grad = w * residual / sqrt_factor;
                pair_slice[i].hess = w / (factor * sqrt_factor);
            }
        }
    }

    fn compute_base_score(
        &self,
        n_rows: usize,
        n_outputs: usize,
        targets: &[f32],
        weights: &[f32],
        outputs: &mut [f32],
    ) {
        debug_assert!(targets.len() >= n_outputs * n_rows);
        debug_assert!(outputs.len() >= n_outputs);
        debug_assert!(weights.is_empty() || weights.len() >= n_rows);

        if n_rows == 0 {
            outputs[..n_outputs].fill(0.0);
            return;
        }

        // Use median as robust base score for each output
        for out_idx in 0..n_outputs {
            let offset = out_idx * n_rows;
            let target_slice = &targets[offset..offset + n_rows];

            let mut sorted: Vec<(f32, f32)> = target_slice
                .iter()
                .zip(weight_iter(weights, n_rows))
                .map(|(&t, w)| (t, w))
                .collect();
            sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            let total_weight: f32 = sorted.iter().map(|(_, w)| w).sum();
            let target_weight = 0.5 * total_weight;

            let mut cumulative = 0.0f32;
            outputs[out_idx] = sorted.last().map(|(v, _)| *v).unwrap_or(0.0);
            for (value, w) in &sorted {
                cumulative += w;
                if cumulative >= target_weight {
                    outputs[out_idx] = *value;
                    break;
                }
            }
        }
    }

    fn name(&self) -> &'static str {
        "pseudo_huber"
    }

    fn task_kind(&self) -> TaskKind {
        TaskKind::Regression
    }

    fn target_schema(&self) -> TargetSchema {
        TargetSchema::Continuous
    }

    fn default_metric(&self) -> MetricKind {
        MetricKind::Huber
    }

    fn transform_prediction_inplace(&self, _raw: &mut PredictionOutput) -> PredictionKind {
        PredictionKind::Value
    }
}

// =============================================================================
// Absolute Loss (MAE)
// =============================================================================

/// Absolute error loss (L1 loss) for robust regression.
///
/// Minimizes the mean absolute error. More robust to outliers than squared loss.
///
/// - Loss: `|pred - target|`
/// - Gradient: `sign(pred - target)`
/// - Hessian: `1.0` (constant for Newton step stability)
///
/// Note: Also available as `PinballLoss::new(0.5)` which is mathematically equivalent.
#[derive(Debug, Clone, Copy, Default)]
pub struct AbsoluteLoss;

impl Objective for AbsoluteLoss {
    fn compute_gradients(
        &self,
        n_rows: usize,
        n_outputs: usize,
        predictions: &[f32],
        targets: &[f32],
        weights: &[f32],
        grad_hess: &mut [GradsTuple],
    ) {
        validate_objective_inputs(
            n_rows,
            n_outputs,
            predictions.len(),
            grad_hess.len(),
            weights,
        );
        debug_assert!(targets.len() >= n_outputs * n_rows);

        for out_idx in 0..n_outputs {
            let offset = out_idx * n_rows;
            let pred_slice = &predictions[offset..offset + n_rows];
            let target_slice = &targets[offset..offset + n_rows];
            let pair_slice = &mut grad_hess[offset..offset + n_rows];

            for (i, w) in weight_iter(weights, n_rows).enumerate() {
                let diff = pred_slice[i] - target_slice[i];
                pair_slice[i].grad = w * diff.signum();
                pair_slice[i].hess = w;
            }
        }
    }

    fn compute_base_score(
        &self,
        n_rows: usize,
        n_outputs: usize,
        targets: &[f32],
        weights: &[f32],
        outputs: &mut [f32],
    ) {
        debug_assert!(targets.len() >= n_outputs * n_rows);
        debug_assert!(outputs.len() >= n_outputs);
        debug_assert!(weights.is_empty() || weights.len() >= n_rows);

        if n_rows == 0 {
            outputs[..n_outputs].fill(0.0);
            return;
        }

        // Use median as base score (optimal for L1 loss)
        for out_idx in 0..n_outputs {
            let offset = out_idx * n_rows;
            let target_slice = &targets[offset..offset + n_rows];

            let mut sorted: Vec<(f32, f32)> = target_slice
                .iter()
                .zip(weight_iter(weights, n_rows))
                .map(|(&t, w)| (t, w))
                .collect();
            sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            let total_weight: f32 = sorted.iter().map(|(_, w)| w).sum();
            let target_weight = 0.5 * total_weight;

            let mut cumulative = 0.0f32;
            outputs[out_idx] = sorted.last().map(|(v, _)| *v).unwrap_or(0.0);
            for (value, w) in &sorted {
                cumulative += w;
                if cumulative >= target_weight {
                    outputs[out_idx] = *value;
                    break;
                }
            }
        }
    }

    fn name(&self) -> &'static str {
        "absolute"
    }

    fn task_kind(&self) -> TaskKind {
        TaskKind::Regression
    }

    fn target_schema(&self) -> TargetSchema {
        TargetSchema::Continuous
    }

    fn default_metric(&self) -> MetricKind {
        MetricKind::Mae
    }

    fn transform_prediction_inplace(&self, _raw: &mut PredictionOutput) -> PredictionKind {
        PredictionKind::Value
    }
}

// =============================================================================
// Poisson Loss
// =============================================================================

/// Poisson regression loss for count data.
///
/// Assumes the target follows a Poisson distribution with rate exp(pred).
/// Predictions are in log-space (raw scores, not exponentiated).
///
/// - Loss: `exp(pred) - target * pred`
/// - Gradient: `exp(pred) - target`
/// - Hessian: `exp(pred)` (always positive)
///
/// # Notes
///
/// - Targets should be non-negative counts (or rates)
/// - Predictions are log(expected count)
/// - Use `exp(prediction)` to get the actual count prediction
#[derive(Debug, Clone, Copy, Default)]
pub struct PoissonLoss;

impl Objective for PoissonLoss {
    fn compute_gradients(
        &self,
        n_rows: usize,
        n_outputs: usize,
        predictions: &[f32],
        targets: &[f32],
        weights: &[f32],
        grad_hess: &mut [GradsTuple],
    ) {
        validate_objective_inputs(
            n_rows,
            n_outputs,
            predictions.len(),
            grad_hess.len(),
            weights,
        );
        debug_assert!(targets.len() >= n_outputs * n_rows);

        const MAX_EXP: f32 = 30.0; // Prevent overflow
        const HESS_MIN: f32 = 1e-6;

        for out_idx in 0..n_outputs {
            let offset = out_idx * n_rows;
            let pred_slice = &predictions[offset..offset + n_rows];
            let target_slice = &targets[offset..offset + n_rows];
            let pair_slice = &mut grad_hess[offset..offset + n_rows];

            for (i, w) in weight_iter(weights, n_rows).enumerate() {
                let pred = pred_slice[i].clamp(-MAX_EXP, MAX_EXP);
                let exp_pred = pred.exp();
                pair_slice[i].grad = w * (exp_pred - target_slice[i]);
                pair_slice[i].hess = (w * exp_pred).max(HESS_MIN);
            }
        }
    }

    fn compute_base_score(
        &self,
        n_rows: usize,
        n_outputs: usize,
        targets: &[f32],
        weights: &[f32],
        outputs: &mut [f32],
    ) {
        debug_assert!(targets.len() >= n_outputs * n_rows);
        debug_assert!(outputs.len() >= n_outputs);
        debug_assert!(weights.is_empty() || weights.len() >= n_rows);

        if n_rows == 0 {
            outputs[..n_outputs].fill(0.0);
            return;
        }

        // Base score is log of weighted mean target
        for out_idx in 0..n_outputs {
            let offset = out_idx * n_rows;
            let target_slice = &targets[offset..offset + n_rows];

            let (sum_w, sum_wy) = target_slice
                .iter()
                .zip(weight_iter(weights, n_rows))
                .fold((0.0f64, 0.0f64), |(sw, swy), (&y, w)| {
                    (sw + w as f64, swy + w as f64 * y as f64)
                });

            let mean = if sum_w > 0.0 {
                (sum_wy / sum_w).max(1e-7)
            } else {
                1.0
            };

            outputs[out_idx] = mean.ln() as f32;
        }
    }

    fn name(&self) -> &'static str {
        "poisson"
    }

    fn task_kind(&self) -> TaskKind {
        TaskKind::Regression
    }

    fn target_schema(&self) -> TargetSchema {
        TargetSchema::CountNonNegative
    }

    fn default_metric(&self) -> MetricKind {
        MetricKind::PoissonDeviance
    }

    fn transform_prediction_inplace(&self, raw: &mut PredictionOutput) -> PredictionKind {
        // Poisson mean parameter is exp(margin).
        for v in raw.as_mut_slice().iter_mut() {
            *v = (*v).exp();
        }
        PredictionKind::Value
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn squared_loss_single_output() {
        let obj = SquaredLoss;
        let preds = [1.0f32, 2.0, 3.0];
        let targets = [0.5f32, 2.5, 2.5];
        let mut grad_hess = [GradsTuple { grad: 0.0, hess: 0.0 }; 3];

        obj.compute_gradients(3, 1, &preds, &targets, &[], &mut grad_hess);

        assert!((grad_hess[0].grad - 0.5).abs() < 1e-6);
        assert!((grad_hess[1].grad - -0.5).abs() < 1e-6);
        assert!((grad_hess[2].grad - 0.5).abs() < 1e-6);
        assert!((grad_hess[0].hess - 1.0).abs() < 1e-6);
    }

    #[test]
    fn squared_loss_multi_output() {
        let obj = SquaredLoss;
        // 2 rows, 2 outputs - column major
        let preds = [1.0f32, 2.0, 3.0, 4.0]; // out0=[1,2], out1=[3,4]
        let targets = [0.0f32, 1.0, 2.0, 3.0]; // out0=[0,1], out1=[2,3]
        let mut grad_hess = [GradsTuple { grad: 0.0, hess: 0.0 }; 4];

        obj.compute_gradients(2, 2, &preds, &targets, &[], &mut grad_hess);

        // out0: grads = [1-0, 2-1] = [1, 1]
        assert!((grad_hess[0].grad - 1.0).abs() < 1e-6);
        assert!((grad_hess[1].grad - 1.0).abs() < 1e-6);
        // out1: grads = [3-2, 4-3] = [1, 1]
        assert!((grad_hess[2].grad - 1.0).abs() < 1e-6);
        assert!((grad_hess[3].grad - 1.0).abs() < 1e-6);
    }

    #[test]
    fn squared_loss_base_score() {
        let obj = SquaredLoss;
        let targets = [1.0f32, 2.0, 3.0, 4.0];
        let mut output = [0.0f32];

        obj.compute_base_score(4, 1, &targets, &[], &mut output);
        assert!((output[0] - 2.5).abs() < 1e-6);
    }

    #[test]
    fn squared_loss_weighted() {
        let obj = SquaredLoss;
        let preds = [1.0f32, 2.0];
        let targets = [0.5f32, 2.5];
        let weights = [2.0f32, 0.5];
        let mut grad_hess = [GradsTuple { grad: 0.0, hess: 0.0 }; 2];

        obj.compute_gradients(2, 1, &preds, &targets, &weights, &mut grad_hess);

        assert!((grad_hess[0].grad - 1.0).abs() < 1e-6); // 2.0 * 0.5
        assert!((grad_hess[1].grad - -0.25).abs() < 1e-6); // 0.5 * -0.5
        assert!((grad_hess[0].hess - 2.0).abs() < 1e-6);
        assert!((grad_hess[1].hess - 0.5).abs() < 1e-6);
    }

    #[test]
    fn pinball_loss_median() {
        let obj = PinballLoss::new(0.5);
        let preds = [1.0f32, 2.0, 3.0];
        let targets = [0.5f32, 2.5, 2.5];
        let mut grad_hess = [GradsTuple { grad: 0.0, hess: 0.0 }; 3];

        obj.compute_gradients(3, 1, &preds, &targets, &[], &mut grad_hess);

        assert!((grad_hess[0].grad - 0.5).abs() < 1e-6); // pred > target
        assert!((grad_hess[1].grad - -0.5).abs() < 1e-6); // pred < target
        assert!((grad_hess[2].grad - 0.5).abs() < 1e-6); // pred > target
    }

    #[test]
    fn pinball_loss_multi_quantile_shared_target() {
        let obj = PinballLoss::with_quantiles(vec![0.1, 0.9]);
        // 2 rows, 2 quantiles, 1 shared target
        let preds = [5.0f32, 5.0, 5.0, 5.0]; // q0.1=[5,5], q0.9=[5,5]
        let targets = [10.0f32, 0.0]; // shared target for all quantiles
        let mut grad_hess = [GradsTuple { grad: 0.0, hess: 0.0 }; 4];

        obj.compute_gradients(2, 2, &preds, &targets, &[], &mut grad_hess);

        // For alpha=0.1: pred[0]=5 < target[0]=10 → grad = alpha-1 = -0.9
        assert!((grad_hess[0].grad - -0.9).abs() < 1e-6);
        // For alpha=0.1: pred[1]=5 > target[1]=0 → grad = alpha = 0.1
        assert!((grad_hess[1].grad - 0.1).abs() < 1e-6);
        // For alpha=0.9: same logic
        assert!((grad_hess[2].grad - -0.1).abs() < 1e-6); // 5 < 10
        assert!((grad_hess[3].grad - 0.9).abs() < 1e-6); // 5 > 0
    }

    #[test]
    fn pinball_loss_quantiles() {
        let obj = PinballLoss::new(0.1);
        let preds = [5.0f32];
        let targets = [10.0f32];
        let mut grad_hess = [GradsTuple { grad: 0.0, hess: 0.0 }];

        obj.compute_gradients(1, 1, &preds, &targets, &[], &mut grad_hess);
        assert!((grad_hess[0].grad - -0.9).abs() < 1e-6);

        let obj = PinballLoss::new(0.9);
        obj.compute_gradients(1, 1, &preds, &targets, &[], &mut grad_hess);
        assert!((grad_hess[0].grad - -0.1).abs() < 1e-6);
    }

    #[test]
    fn pseudo_huber_gradient_near_zero() {
        let obj = PseudoHuberLoss::new(1.0);
        let preds = [0.01f32];
        let targets = [0.0f32];
        let mut grad_hess = [GradsTuple { grad: 0.0, hess: 0.0 }];

        obj.compute_gradients(1, 1, &preds, &targets, &[], &mut grad_hess);

        assert!((grad_hess[0].grad - 0.01).abs() < 0.001);
        assert!((grad_hess[0].hess - 1.0).abs() < 0.01);
    }

    #[test]
    fn pseudo_huber_gradient_large_residual() {
        let obj = PseudoHuberLoss::new(1.0);
        let preds = [100.0f32];
        let targets = [0.0f32];
        let mut grad_hess = [GradsTuple { grad: 0.0, hess: 0.0 }];

        obj.compute_gradients(1, 1, &preds, &targets, &[], &mut grad_hess);

        assert!(grad_hess[0].grad > 0.9 && grad_hess[0].grad < 1.1);
        assert!(grad_hess[0].hess < 0.01);
    }

    #[test]
    fn pseudo_huber_multi_output() {
        let obj = PseudoHuberLoss::new(1.0);
        // 2 rows, 2 outputs
        let preds = [0.01f32, 0.02, 100.0, 100.0];
        let targets = [0.0f32, 0.0, 0.0, 0.0];
        let mut grad_hess = [GradsTuple { grad: 0.0, hess: 0.0 }; 4];

        obj.compute_gradients(2, 2, &preds, &targets, &[], &mut grad_hess);

        // Near-zero residuals
        assert!((grad_hess[0].grad - 0.01).abs() < 0.001);
        assert!((grad_hess[1].grad - 0.02).abs() < 0.001);
        // Large residuals
        assert!(grad_hess[2].grad > 0.9 && grad_hess[2].grad < 1.1);
        assert!(grad_hess[3].grad > 0.9 && grad_hess[3].grad < 1.1);
    }

    #[test]
    fn absolute_loss_basic() {
        let obj = AbsoluteLoss;
        let preds = [1.0f32, 2.0, 3.0];
        let targets = [0.5f32, 2.5, 2.5];
        let mut grad_hess = [GradsTuple { grad: 0.0, hess: 0.0 }; 3];

        obj.compute_gradients(3, 1, &preds, &targets, &[], &mut grad_hess);

        // grad = sign(pred - target)
        assert!((grad_hess[0].grad - 1.0).abs() < 1e-6); // pred > target
        assert!((grad_hess[1].grad - -1.0).abs() < 1e-6); // pred < target
        assert!((grad_hess[2].grad - 1.0).abs() < 1e-6); // pred > target
        // hess = 1.0
        assert!((grad_hess[0].hess - 1.0).abs() < 1e-6);
    }

    #[test]
    fn absolute_loss_base_score() {
        let obj = AbsoluteLoss;
        // Median of [1, 2, 3, 4] is 2.5
        let targets = [1.0f32, 2.0, 3.0, 4.0];
        let mut output = [0.0f32];

        obj.compute_base_score(4, 1, &targets, &[], &mut output);
        // Median should be around 2 or 3
        assert!(output[0] >= 2.0 && output[0] <= 3.0);
    }

    #[test]
    fn poisson_loss_basic() {
        let obj = PoissonLoss;
        // pred=0 means expected count = exp(0) = 1
        let preds = [0.0f32];
        let targets = [2.0f32]; // actual count
        let mut grad_hess = [GradsTuple { grad: 0.0, hess: 0.0 }];

        obj.compute_gradients(1, 1, &preds, &targets, &[], &mut grad_hess);

        // grad = exp(pred) - target = 1 - 2 = -1
        assert!((grad_hess[0].grad - -1.0).abs() < 1e-6);
        // hess = exp(pred) = 1
        assert!((grad_hess[0].hess - 1.0).abs() < 1e-6);
    }

    #[test]
    fn poisson_loss_base_score() {
        let obj = PoissonLoss;
        let targets = [1.0f32, 2.0, 3.0, 4.0];
        let mut output = [0.0f32];

        obj.compute_base_score(4, 1, &targets, &[], &mut output);

        // Base score should be log(mean(targets)) = log(2.5)
        let expected = (2.5f32).ln();
        assert!((output[0] - expected).abs() < 1e-6);
    }

    #[test]
    fn poisson_loss_positive_pred() {
        let obj = PoissonLoss;
        let preds = [1.0f32]; // exp(1) ≈ 2.718
        let targets = [3.0f32];
        let mut grad_hess = [GradsTuple { grad: 0.0, hess: 0.0 }];

        obj.compute_gradients(1, 1, &preds, &targets, &[], &mut grad_hess);

        let exp_1 = 1.0f32.exp();
        assert!((grad_hess[0].grad - (exp_1 - 3.0)).abs() < 1e-5);
        assert!((grad_hess[0].hess - exp_1).abs() < 1e-5);
    }
}

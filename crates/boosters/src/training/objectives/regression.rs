//! Regression objective functions.
//!
//! Performance-focused implementations using iterators and slices.
//! Data layout: predictions/gradients are `[n_outputs, n_samples]` (row-major within ndarray).
//! Targets are TargetsView with shape `[n_outputs, n_samples]`, weights are WeightsView.

use ndarray::{ArrayView1, ArrayView2, ArrayViewMut2};

use super::{ObjectiveFn, TaskKind};
use crate::data::{TargetsView, WeightsView};
use crate::inference::PredictionKind;
use crate::training::GradsTuple;

// =============================================================================
// Squared Loss
// =============================================================================

/// Squared error loss (L2 loss) for regression.
///
/// - Loss: `0.5 * (pred - target)²`
/// - Gradient: `pred - target`
/// - Hessian: `1.0` (or weight if weighted)
#[derive(Debug, Clone, Copy, Default)]
pub struct SquaredLoss;

impl ObjectiveFn for SquaredLoss {
    fn n_outputs(&self) -> usize {
        1
    }

    fn compute_gradients_into(
        &self,
        predictions: ArrayView2<f32>,
        targets: TargetsView<'_>,
        weights: WeightsView<'_>,
        mut grad_hess: ArrayViewMut2<GradsTuple>,
    ) {
        let (n_outputs, n_rows) = predictions.dim();
        let targets = targets.output(0);

        for out_idx in 0..n_outputs {
            let preds_row = predictions.row(out_idx);
            let mut gh_row = grad_hess.row_mut(out_idx);

            for (i, ((&pred, &target), w)) in preds_row
                .iter()
                .zip(targets.iter())
                .zip(weights.iter(n_rows))
                .enumerate()
            {
                gh_row[i].grad = w * (pred - target);
                gh_row[i].hess = w;
            }
        }
    }

    fn compute_base_score(&self, targets: TargetsView<'_>, weights: WeightsView<'_>) -> Vec<f32> {
        let targets = targets.output(0);
        let n_rows = targets.len();
        if n_rows == 0 {
            return vec![0.0];
        }

        // Compute weighted mean
        let (sum_w, sum_wy) = targets
            .iter()
            .zip(weights.iter(n_rows))
            .fold((0.0f64, 0.0f64), |(sw, swy), (&y, w)| {
                (sw + w as f64, swy + w as f64 * y as f64)
            });

        let base = if sum_w > 0.0 {
            (sum_wy / sum_w) as f32
        } else {
            0.0
        };
        vec![base]
    }

    fn name(&self) -> &'static str {
        "squared"
    }

    fn task_kind(&self) -> TaskKind {
        TaskKind::Regression
    }

    fn transform_predictions_inplace(&self, _predictions: ArrayViewMut2<f32>) -> PredictionKind {
        PredictionKind::Value
    }
}

// =============================================================================
// Absolute Loss (MAE / L1)
// =============================================================================

/// Absolute error loss (L1 loss) for robust regression.
///
/// - Loss: `|pred - target|`
/// - Gradient: `sign(pred - target)`
/// - Hessian: `1.0` (constant for Newton step stability)
#[derive(Debug, Clone, Copy, Default)]
pub struct AbsoluteLoss;

impl ObjectiveFn for AbsoluteLoss {
    fn n_outputs(&self) -> usize {
        1
    }

    fn compute_gradients_into(
        &self,
        predictions: ArrayView2<f32>,
        targets: TargetsView<'_>,
        weights: WeightsView<'_>,
        mut grad_hess: ArrayViewMut2<GradsTuple>,
    ) {
        let (n_outputs, n_rows) = predictions.dim();
        let targets = targets.output(0);

        for out_idx in 0..n_outputs {
            let preds_row = predictions.row(out_idx);
            let mut gh_row = grad_hess.row_mut(out_idx);

            for (i, ((&pred, &target), w)) in preds_row
                .iter()
                .zip(targets.iter())
                .zip(weights.iter(n_rows))
                .enumerate()
            {
                let diff = pred - target;
                gh_row[i].grad = w * diff.signum();
                gh_row[i].hess = w;
            }
        }
    }

    fn compute_base_score(&self, targets: TargetsView<'_>, weights: WeightsView<'_>) -> Vec<f32> {
        let targets = targets.output(0);
        // Use weighted median as base score (optimal for L1 loss)
        if targets.is_empty() {
            return vec![0.0];
        }

        let median = compute_weighted_quantile(targets, weights, 0.5);
        vec![median]
    }

    fn name(&self) -> &'static str {
        "absolute"
    }

    fn task_kind(&self) -> TaskKind {
        TaskKind::Regression
    }

    fn transform_predictions_inplace(&self, _predictions: ArrayViewMut2<f32>) -> PredictionKind {
        PredictionKind::Value
    }
}

// =============================================================================
// Pinball Loss (Quantile Regression)
// =============================================================================

/// Pinball loss for quantile regression.
///
/// - Loss: `α * (target - pred)` if `target > pred`, else `(1-α) * (pred - target)`
/// - Gradient: `α - 1` if `pred < target`, else `α`
/// - Hessian: `1.0`
#[derive(Debug, Clone)]
pub struct PinballLoss {
    /// Quantile levels for each output, each in (0, 1).
    pub alphas: Vec<f32>,
}

impl PinballLoss {
    /// Create a pinball loss for a single quantile.
    pub fn new(alpha: f32) -> Self {
        debug_assert!(alpha > 0.0 && alpha < 1.0, "alpha must be in (0, 1)");
        Self {
            alphas: vec![alpha],
        }
    }

    /// Create a pinball loss for multiple quantiles.
    pub fn with_quantiles(alphas: Vec<f32>) -> Self {
        debug_assert!(!alphas.is_empty());
        for &a in &alphas {
            debug_assert!(a > 0.0 && a < 1.0, "alpha must be in (0, 1)");
        }
        Self { alphas }
    }

    /// Number of quantiles (outputs).
    #[inline]
    pub fn n_quantiles(&self) -> usize {
        self.alphas.len()
    }
}

impl ObjectiveFn for PinballLoss {
    fn n_outputs(&self) -> usize {
        self.alphas.len()
    }

    fn compute_gradients_into(
        &self,
        predictions: ArrayView2<f32>,
        targets: TargetsView<'_>,
        weights: WeightsView<'_>,
        mut grad_hess: ArrayViewMut2<GradsTuple>,
    ) {
        let targets = targets.output(0);
        let n_rows = targets.len();

        for (out_idx, &alpha) in self.alphas.iter().enumerate() {
            let preds_row = predictions.row(out_idx);
            let mut gh_row = grad_hess.row_mut(out_idx);

            for (i, ((&pred, &target), w)) in preds_row
                .iter()
                .zip(targets.iter())
                .zip(weights.iter(n_rows))
                .enumerate()
            {
                let diff = pred - target;
                let g = if diff < 0.0 { alpha - 1.0 } else { alpha };
                gh_row[i].grad = w * g;
                gh_row[i].hess = w;
            }
        }
    }

    fn compute_base_score(&self, targets: TargetsView<'_>, weights: WeightsView<'_>) -> Vec<f32> {
        let targets = targets.output(0);
        if targets.is_empty() {
            return vec![0.0; self.alphas.len()];
        }

        self.alphas
            .iter()
            .map(|&alpha| compute_weighted_quantile(targets, weights, alpha))
            .collect()
    }

    fn task_kind(&self) -> TaskKind {
        TaskKind::Regression
    }

    fn transform_predictions_inplace(&self, _predictions: ArrayViewMut2<f32>) -> PredictionKind {
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
/// A smooth approximation to Huber loss:
/// - Loss: `delta² * (sqrt(1 + (residual/delta)²) - 1)`
/// - Gradient: `residual / sqrt(1 + (residual/delta)²)`
/// - Hessian: `1 / (1 + (residual/delta)²)^1.5`
#[derive(Debug, Clone, Copy)]
pub struct PseudoHuberLoss {
    pub delta: f32,
}

impl PseudoHuberLoss {
    pub fn new(delta: f32) -> Self {
        debug_assert!(delta > 0.0);
        Self { delta }
    }
}

impl ObjectiveFn for PseudoHuberLoss {
    fn n_outputs(&self) -> usize {
        1
    }

    fn compute_gradients_into(
        &self,
        predictions: ArrayView2<f32>,
        targets: TargetsView<'_>,
        weights: WeightsView<'_>,
        mut grad_hess: ArrayViewMut2<GradsTuple>,
    ) {
        let (n_outputs, n_rows) = predictions.dim();
        let targets = targets.output(0);
        let inv_delta_sq = 1.0 / (self.delta * self.delta);

        for out_idx in 0..n_outputs {
            let preds_row = predictions.row(out_idx);
            let mut gh_row = grad_hess.row_mut(out_idx);

            for (i, ((&pred, &target), w)) in preds_row
                .iter()
                .zip(targets.iter())
                .zip(weights.iter(n_rows))
                .enumerate()
            {
                let residual = pred - target;
                let r_sq = residual * residual;
                let factor = 1.0 + r_sq * inv_delta_sq;
                let sqrt_factor = factor.sqrt();

                gh_row[i].grad = w * residual / sqrt_factor;
                gh_row[i].hess = w / (factor * sqrt_factor);
            }
        }
    }

    fn compute_base_score(&self, targets: TargetsView<'_>, weights: WeightsView<'_>) -> Vec<f32> {
        let targets = targets.output(0);
        // Use median as robust base score
        if targets.is_empty() {
            return vec![0.0];
        }

        let median = compute_weighted_quantile(targets, weights, 0.5);
        vec![median]
    }

    fn name(&self) -> &'static str {
        "pseudo_huber"
    }

    fn task_kind(&self) -> TaskKind {
        TaskKind::Regression
    }

    fn transform_predictions_inplace(&self, _predictions: ArrayViewMut2<f32>) -> PredictionKind {
        PredictionKind::Value
    }
}

// =============================================================================
// Poisson Loss
// =============================================================================

/// Poisson regression loss for count data.
///
/// Predictions are in log-space (raw scores, not exponentiated).
///
/// - Loss: `exp(pred) - target * pred`
/// - Gradient: `exp(pred) - target`
/// - Hessian: `exp(pred)`
#[derive(Debug, Clone, Copy, Default)]
pub struct PoissonLoss;

impl ObjectiveFn for PoissonLoss {
    fn n_outputs(&self) -> usize {
        1
    }

    fn compute_gradients_into(
        &self,
        predictions: ArrayView2<f32>,
        targets: TargetsView<'_>,
        weights: WeightsView<'_>,
        mut grad_hess: ArrayViewMut2<GradsTuple>,
    ) {
        const MAX_EXP: f32 = 30.0;
        const HESS_MIN: f32 = 1e-6;

        let (n_outputs, n_rows) = predictions.dim();
        let targets = targets.output(0);

        for out_idx in 0..n_outputs {
            let preds_row = predictions.row(out_idx);
            let mut gh_row = grad_hess.row_mut(out_idx);

            for (i, ((&pred, &target), w)) in preds_row
                .iter()
                .zip(targets.iter())
                .zip(weights.iter(n_rows))
                .enumerate()
            {
                let pred_clamped = pred.clamp(-MAX_EXP, MAX_EXP);
                let exp_pred = pred_clamped.exp();
                gh_row[i].grad = w * (exp_pred - target);
                gh_row[i].hess = (w * exp_pred).max(HESS_MIN);
            }
        }
    }

    fn compute_base_score(&self, targets: TargetsView<'_>, weights: WeightsView<'_>) -> Vec<f32> {
        let targets = targets.output(0);
        let n_rows = targets.len();
        if n_rows == 0 {
            return vec![0.0];
        }

        // Base score is log of weighted mean target
        let (sum_w, sum_wy) = targets
            .iter()
            .zip(weights.iter(n_rows))
            .fold((0.0f64, 0.0f64), |(sw, swy), (&y, w)| {
                (sw + w as f64, swy + w as f64 * y as f64)
            });

        let mean = if sum_w > 0.0 {
            (sum_wy / sum_w).max(1e-7)
        } else {
            1.0
        };
        vec![mean.ln() as f32]
    }

    fn name(&self) -> &'static str {
        "poisson"
    }

    fn task_kind(&self) -> TaskKind {
        TaskKind::Regression
    }

    fn transform_predictions_inplace(&self, _predictions: ArrayViewMut2<f32>) -> PredictionKind {
        PredictionKind::Value
    }
}

// =============================================================================
// Helpers
// =============================================================================

/// Compute weighted quantile (used for base scores in quantile/L1 objectives).
///
/// Note: This requires sorting which is O(n log n). For `compute_base_score` this
/// is acceptable as it's called once per training run, not in the hot path.
fn compute_weighted_quantile(values: ArrayView1<f32>, weights: WeightsView<'_>, alpha: f32) -> f32 {
    let n_rows = values.len();
    if n_rows == 0 {
        return 0.0;
    }

    // Collect (value, weight) pairs and sort by value
    let mut sorted: Vec<(f32, f32)> = values
        .iter()
        .zip(weights.iter(n_rows))
        .map(|(&v, w)| (v, w))
        .collect();
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let total_weight: f32 = sorted.iter().map(|(_, w)| w).sum();
    let target_weight = alpha * total_weight;

    let mut cumulative = 0.0f32;
    for (value, w) in &sorted {
        cumulative += w;
        if cumulative >= target_weight {
            return *value;
        }
    }

    sorted.last().map(|(v, _)| *v).unwrap_or(0.0)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{TargetsView, WeightsView};
    use ndarray::Array2;

    fn make_preds(n_outputs: usize, n_samples: usize, data: &[f32]) -> Array2<f32> {
        Array2::from_shape_vec((n_outputs, n_samples), data.to_vec()).unwrap()
    }

    fn make_targets_array(data: &[f32]) -> Array2<f32> {
        Array2::from_shape_vec((1, data.len()), data.to_vec()).unwrap()
    }

    fn make_grad_hess(n_outputs: usize, n_samples: usize) -> Array2<GradsTuple> {
        Array2::from_elem((n_outputs, n_samples), GradsTuple::default())
    }

    #[test]
    fn squared_loss_single_output() {
        let obj = SquaredLoss;
        let preds = make_preds(1, 3, &[1.0, 2.0, 3.0]);
        let targets_arr = make_targets_array(&[0.5, 2.5, 2.5]);
        let targets = TargetsView::new(targets_arr.view());
        let mut gh = make_grad_hess(1, 3);

        obj.compute_gradients_into(preds.view(), targets, WeightsView::None, gh.view_mut());

        assert!((gh[[0, 0]].grad - 0.5).abs() < 1e-6); // 1.0 - 0.5
        assert!((gh[[0, 1]].grad - -0.5).abs() < 1e-6); // 2.0 - 2.5
        assert!((gh[[0, 2]].grad - 0.5).abs() < 1e-6); // 3.0 - 2.5
        assert!((gh[[0, 0]].hess - 1.0).abs() < 1e-6);
    }

    #[test]
    fn squared_loss_weighted() {
        let obj = SquaredLoss;
        let preds = make_preds(1, 2, &[1.0, 2.0]);
        let targets_arr = make_targets_array(&[0.5, 2.5]);
        let targets = TargetsView::new(targets_arr.view());
        let weights = ndarray::array![2.0f32, 0.5];
        let mut gh = make_grad_hess(1, 2);

        obj.compute_gradients_into(
            preds.view(),
            targets,
            WeightsView::from_array(weights.view()),
            gh.view_mut(),
        );

        assert!((gh[[0, 0]].grad - 1.0).abs() < 1e-6); // 2.0 * 0.5
        assert!((gh[[0, 1]].grad - -0.25).abs() < 1e-6); // 0.5 * -0.5
        assert!((gh[[0, 0]].hess - 2.0).abs() < 1e-6);
        assert!((gh[[0, 1]].hess - 0.5).abs() < 1e-6);
    }

    #[test]
    fn squared_loss_base_score() {
        let obj = SquaredLoss;
        let targets_arr = make_targets_array(&[1.0, 2.0, 3.0, 4.0]);
        let targets = TargetsView::new(targets_arr.view());

        let output = obj.compute_base_score(targets, WeightsView::None);
        assert_eq!(output.len(), 1);
        assert!((output[0] - 2.5).abs() < 1e-6);
    }

    #[test]
    fn pinball_loss_median() {
        let obj = PinballLoss::new(0.5);
        let preds = make_preds(1, 3, &[1.0, 2.0, 3.0]);
        let targets_arr = make_targets_array(&[0.5, 2.5, 2.5]);
        let targets = TargetsView::new(targets_arr.view());
        let mut gh = make_grad_hess(1, 3);

        obj.compute_gradients_into(preds.view(), targets, WeightsView::None, gh.view_mut());

        assert!((gh[[0, 0]].grad - 0.5).abs() < 1e-6); // pred > target
        assert!((gh[[0, 1]].grad - -0.5).abs() < 1e-6); // pred < target
        assert!((gh[[0, 2]].grad - 0.5).abs() < 1e-6); // pred > target
    }

    #[test]
    fn absolute_loss_basic() {
        let obj = AbsoluteLoss;
        let preds = make_preds(1, 3, &[1.0, 2.0, 3.0]);
        let targets_arr = make_targets_array(&[0.5, 2.5, 2.5]);
        let targets = TargetsView::new(targets_arr.view());
        let mut gh = make_grad_hess(1, 3);

        obj.compute_gradients_into(preds.view(), targets, WeightsView::None, gh.view_mut());

        assert!((gh[[0, 0]].grad - 1.0).abs() < 1e-6); // sign(0.5) = 1
        assert!((gh[[0, 1]].grad - -1.0).abs() < 1e-6); // sign(-0.5) = -1
        assert!((gh[[0, 2]].grad - 1.0).abs() < 1e-6); // sign(0.5) = 1
    }

    #[test]
    fn pseudo_huber_near_zero() {
        let obj = PseudoHuberLoss::new(1.0);
        let preds = make_preds(1, 1, &[0.01]);
        let targets_arr = make_targets_array(&[0.0]);
        let targets = TargetsView::new(targets_arr.view());
        let mut gh = make_grad_hess(1, 1);

        obj.compute_gradients_into(preds.view(), targets, WeightsView::None, gh.view_mut());

        // Near zero, should be approximately linear (grad ≈ residual)
        assert!((gh[[0, 0]].grad - 0.01).abs() < 0.001);
        assert!((gh[[0, 0]].hess - 1.0).abs() < 0.01);
    }

    #[test]
    fn poisson_loss_basic() {
        let obj = PoissonLoss;
        let preds = make_preds(1, 1, &[0.0]); // exp(0) = 1
        let targets_arr = make_targets_array(&[2.0]);
        let targets = TargetsView::new(targets_arr.view());
        let mut gh = make_grad_hess(1, 1);

        obj.compute_gradients_into(preds.view(), targets, WeightsView::None, gh.view_mut());

        // grad = exp(0) - 2 = 1 - 2 = -1
        assert!((gh[[0, 0]].grad - -1.0).abs() < 1e-6);
        // hess = exp(0) = 1
        assert!((gh[[0, 0]].hess - 1.0).abs() < 1e-6);
    }

    #[test]
    fn poisson_loss_base_score() {
        let obj = PoissonLoss;
        let targets_arr = make_targets_array(&[1.0, 2.0, 3.0, 4.0]);
        let targets = TargetsView::new(targets_arr.view());

        let output = obj.compute_base_score(targets, WeightsView::None);

        // Base score = log(mean) = log(2.5)
        let expected = (2.5f32).ln();
        assert_eq!(output.len(), 1);
        assert!((output[0] - expected).abs() < 1e-6);
    }
}

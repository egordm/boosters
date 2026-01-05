//! Regression objective functions.
//!
//! Performance-focused implementations using iterators and slices.
//! Data layout: predictions/gradients are `[n_outputs, n_samples]` (row-major within ndarray).
//! Targets are TargetsView with shape `[n_outputs, n_samples]`, weights are WeightsView.

use crate::data::{TargetsView, WeightsView};
use crate::training::GradsTuple;
use ndarray::{ArrayView1, ArrayView2, ArrayViewMut2};

// =============================================================================
// Squared Loss
// =============================================================================

/// Squared error loss (L2 loss) for regression.
///
/// - Loss: `0.5 * (pred - target)²`
/// - Gradient: `pred - target`
/// - Hessian: `1.0` (or weight if weighted)
pub(super) fn compute_squared_gradients_into(
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

pub(super) fn compute_squared_base_score(
    targets: TargetsView<'_>,
    weights: WeightsView<'_>,
) -> Vec<f32> {
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

// =============================================================================
// Absolute Loss (MAE / L1)
// =============================================================================

/// Absolute error loss (L1 loss) for robust regression.
///
/// - Loss: `|pred - target|`
/// - Gradient: `sign(pred - target)`
/// - Hessian: `1.0` (constant for Newton step stability)
pub(super) fn compute_absolute_gradients_into(
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

pub(super) fn compute_absolute_base_score(
    targets: TargetsView<'_>,
    weights: WeightsView<'_>,
) -> Vec<f32> {
    let targets = targets.output(0);
    // Use weighted median as base score (optimal for L1 loss)
    if targets.is_empty() {
        return vec![0.0];
    }

    let median = compute_weighted_quantile(targets, weights, 0.5);
    vec![median]
}

// =============================================================================
// Pinball Loss (Quantile Regression)
// =============================================================================

/// Pinball loss helpers for quantile regression.
///
/// - Loss: `α * (target - pred)` if `target > pred`, else `(1-α) * (pred - target)`
/// - Gradient: `-α` if `pred < target`, else `1-α`
/// - Hessian: `1.0`
pub(super) fn compute_pinball_gradients_into(
    predictions: ArrayView2<f32>,
    targets: TargetsView<'_>,
    weights: WeightsView<'_>,
    alphas: &[f32],
    mut grad_hess: ArrayViewMut2<GradsTuple>,
) {
    let targets = targets.output(0);
    let n_rows = targets.len();

    for (out_idx, &alpha) in alphas.iter().enumerate() {
        let preds_row = predictions.row(out_idx);
        let mut gh_row = grad_hess.row_mut(out_idx);

        for (i, ((&pred, &target), w)) in preds_row
            .iter()
            .zip(targets.iter())
            .zip(weights.iter(n_rows))
            .enumerate()
        {
            let diff = pred - target;
            let g = if diff < 0.0 { -alpha } else { 1.0 - alpha };
            gh_row[i].grad = w * g;
            gh_row[i].hess = w;
        }
    }
}

pub(super) fn compute_pinball_base_score(
    targets: TargetsView<'_>,
    weights: WeightsView<'_>,
    alphas: &[f32],
) -> Vec<f32> {
    let targets = targets.output(0);
    if targets.is_empty() {
        return vec![0.0; alphas.len()];
    }

    alphas
        .iter()
        .map(|&alpha| compute_weighted_quantile(targets, weights, alpha))
        .collect()
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
pub(super) fn compute_pseudo_huber_gradients_into(
    predictions: ArrayView2<f32>,
    targets: TargetsView<'_>,
    weights: WeightsView<'_>,
    delta: f32,
    mut grad_hess: ArrayViewMut2<GradsTuple>,
) {
    debug_assert!(delta > 0.0);

    let (n_outputs, n_rows) = predictions.dim();
    let targets = targets.output(0);
    let inv_delta_sq = 1.0 / (delta * delta);

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

pub(super) fn compute_pseudo_huber_base_score(
    targets: TargetsView<'_>,
    weights: WeightsView<'_>,
    _delta: f32,
) -> Vec<f32> {
    let targets = targets.output(0);
    // Use median as robust base score
    if targets.is_empty() {
        return vec![0.0];
    }

    let median = compute_weighted_quantile(targets, weights, 0.5);
    vec![median]
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
pub(super) fn compute_poisson_gradients_into(
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

pub(super) fn compute_poisson_base_score(
    targets: TargetsView<'_>,
    weights: WeightsView<'_>,
) -> Vec<f32> {
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
        let preds = make_preds(1, 3, &[1.0, 2.0, 3.0]);
        let targets_arr = make_targets_array(&[0.5, 2.5, 2.5]);
        let targets = TargetsView::new(targets_arr.view());
        let mut gh = make_grad_hess(1, 3);

        compute_squared_gradients_into(preds.view(), targets, WeightsView::None, gh.view_mut());

        assert!((gh[[0, 0]].grad - 0.5).abs() < 1e-6); // 1.0 - 0.5
        assert!((gh[[0, 1]].grad - -0.5).abs() < 1e-6); // 2.0 - 2.5
        assert!((gh[[0, 2]].grad - 0.5).abs() < 1e-6); // 3.0 - 2.5
        assert!((gh[[0, 0]].hess - 1.0).abs() < 1e-6);
    }

    #[test]
    fn pinball_gradients_match_definition() {
        // Targets chosen so we hit both sides of the kink for each sample.
        // y: [2.0, 1.0]
        // pred: [1.0, 2.0]
        // residual (pred - y): [-1.0, +1.0]
        let preds = make_preds(1, 2, &[1.0, 2.0]);
        let targets_arr = make_targets_array(&[2.0, 1.0]);
        let targets = TargetsView::new(targets_arr.view());

        // alpha = 0.1: grad = -alpha when pred < y, else 1 - alpha
        let mut gh = make_grad_hess(1, 2);
        compute_pinball_gradients_into(
            preds.view(),
            targets,
            WeightsView::None,
            &[0.1],
            gh.view_mut(),
        );
        assert!((gh[[0, 0]].grad - -0.1).abs() < 1e-6);
        assert!((gh[[0, 1]].grad - 0.9).abs() < 1e-6);
        assert!((gh[[0, 0]].hess - 1.0).abs() < 1e-6);

        // alpha = 0.9: grad = -alpha when pred < y, else 1 - alpha
        let mut gh = make_grad_hess(1, 2);
        compute_pinball_gradients_into(
            preds.view(),
            targets,
            WeightsView::None,
            &[0.9],
            gh.view_mut(),
        );
        assert!((gh[[0, 0]].grad - -0.9).abs() < 1e-6);
        assert!((gh[[0, 1]].grad - 0.1).abs() < 1e-6);
        assert!((gh[[0, 1]].hess - 1.0).abs() < 1e-6);
    }

    #[test]
    fn squared_loss_weighted() {
        let preds = make_preds(1, 2, &[1.0, 2.0]);
        let targets_arr = make_targets_array(&[0.5, 2.5]);
        let targets = TargetsView::new(targets_arr.view());
        let weights = ndarray::array![2.0f32, 0.5];
        let mut gh = make_grad_hess(1, 2);

        compute_squared_gradients_into(
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
        let targets_arr = make_targets_array(&[1.0, 2.0, 3.0, 4.0]);
        let targets = TargetsView::new(targets_arr.view());

        let output = compute_squared_base_score(targets, WeightsView::None);
        assert_eq!(output.len(), 1);
        assert!((output[0] - 2.5).abs() < 1e-6);
    }

    #[test]
    fn pinball_loss_median() {
        let alphas = [0.5f32];
        let preds = make_preds(1, 3, &[1.0, 2.0, 3.0]);
        let targets_arr = make_targets_array(&[0.5, 2.5, 2.5]);
        let targets = TargetsView::new(targets_arr.view());
        let mut gh = make_grad_hess(1, 3);

        compute_pinball_gradients_into(
            preds.view(),
            targets,
            WeightsView::None,
            &alphas,
            gh.view_mut(),
        );

        assert!((gh[[0, 0]].grad - 0.5).abs() < 1e-6); // pred > target
        assert!((gh[[0, 1]].grad - -0.5).abs() < 1e-6); // pred < target
        assert!((gh[[0, 2]].grad - 0.5).abs() < 1e-6); // pred > target
    }

    #[test]
    fn absolute_loss_basic() {
        let preds = make_preds(1, 3, &[1.0, 2.0, 3.0]);
        let targets_arr = make_targets_array(&[0.5, 2.5, 2.5]);
        let targets = TargetsView::new(targets_arr.view());
        let mut gh = make_grad_hess(1, 3);

        compute_absolute_gradients_into(preds.view(), targets, WeightsView::None, gh.view_mut());

        assert!((gh[[0, 0]].grad - 1.0).abs() < 1e-6); // sign(0.5) = 1
        assert!((gh[[0, 1]].grad - -1.0).abs() < 1e-6); // sign(-0.5) = -1
        assert!((gh[[0, 2]].grad - 1.0).abs() < 1e-6); // sign(0.5) = 1
    }

    #[test]
    fn pseudo_huber_near_zero() {
        let delta = 1.0;
        let preds = make_preds(1, 1, &[0.01]);
        let targets_arr = make_targets_array(&[0.0]);
        let targets = TargetsView::new(targets_arr.view());
        let mut gh = make_grad_hess(1, 1);

        compute_pseudo_huber_gradients_into(
            preds.view(),
            targets,
            WeightsView::None,
            delta,
            gh.view_mut(),
        );

        // Near zero, should be approximately linear (grad ≈ residual)
        assert!((gh[[0, 0]].grad - 0.01).abs() < 0.001);
        assert!((gh[[0, 0]].hess - 1.0).abs() < 0.01);
    }

    #[test]
    fn poisson_loss_basic() {
        let preds = make_preds(1, 1, &[0.0]); // exp(0) = 1
        let targets_arr = make_targets_array(&[2.0]);
        let targets = TargetsView::new(targets_arr.view());
        let mut gh = make_grad_hess(1, 1);

        compute_poisson_gradients_into(preds.view(), targets, WeightsView::None, gh.view_mut());

        // grad = exp(0) - 2 = 1 - 2 = -1
        assert!((gh[[0, 0]].grad - -1.0).abs() < 1e-6);
        // hess = exp(0) = 1
        assert!((gh[[0, 0]].hess - 1.0).abs() < 1e-6);
    }

    #[test]
    fn poisson_loss_base_score() {
        let targets_arr = make_targets_array(&[1.0, 2.0, 3.0, 4.0]);
        let targets = TargetsView::new(targets_arr.view());

        let output = compute_poisson_base_score(targets, WeightsView::None);

        // Base score = log(mean) = log(2.5)
        let expected = (2.5f32).ln();
        assert_eq!(output.len(), 1);
        assert!((output[0] - expected).abs() < 1e-6);
    }
}

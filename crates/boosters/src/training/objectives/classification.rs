//! Classification objective functions.

use ndarray::{ArrayView2, ArrayViewMut2};

use crate::data::TargetsView;
use crate::data::WeightsView;
use crate::training::GradsTuple;

// =============================================================================
// Logistic Loss
// =============================================================================

/// Logistic loss (log loss / binary cross-entropy) for binary classification.
///
/// Expects labels in {0, 1} and outputs log-odds.
/// - Loss: `-y*log(σ(pred)) - (1-y)*log(1-σ(pred))` where σ is sigmoid
/// - Gradient: `σ(pred) - y`
/// - Hessian: `σ(pred) * (1 - σ(pred))`
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub(super) fn compute_logistic_gradients_into(
    predictions: ArrayView2<f32>,
    targets: TargetsView<'_>,
    weights: WeightsView<'_>,
    mut grad_hess: ArrayViewMut2<GradsTuple>,
) {
    const HESS_MIN: f32 = 1e-6;
    let (n_outputs, n_rows) = predictions.dim();
    debug_assert_eq!(grad_hess.dim(), (n_outputs, n_rows));
    let targets = targets.output(0);
    debug_assert_eq!(targets.len(), n_rows);

    // Process each output independently
    for out_idx in 0..n_outputs {
        let preds_row = predictions.row(out_idx);
        let mut gh_row = grad_hess.row_mut(out_idx);

        for (i, ((&pred, &target), w)) in preds_row
            .iter()
            .zip(targets.iter())
            .zip(weights.iter(n_rows))
            .enumerate()
        {
            let p = sigmoid(pred);
            gh_row[i].grad = w * (p - target);
            gh_row[i].hess = (w * p * (1.0 - p)).max(HESS_MIN);
        }
    }
}

pub(super) fn compute_logistic_base_score(
    targets: TargetsView<'_>,
    weights: WeightsView<'_>,
) -> Vec<f32> {
    let targets = targets.output(0);
    let n_rows = targets.len();

    if n_rows == 0 {
        return vec![0.0];
    }

    // Single output: compute weighted mean of targets, convert to log-odds
    let (pos_weight, total_weight) = targets
        .iter()
        .zip(weights.iter(n_rows))
        .map(|(&t, w)| (w as f64 * t as f64, w as f64))
        .fold((0.0f64, 0.0f64), |(pos, total), (p, w)| {
            (pos + p, total + w)
        });

    let p = (pos_weight / total_weight).clamp(1e-7, 1.0 - 1e-7);
    vec![(p / (1.0 - p)).ln() as f32]
}

// =============================================================================
// Hinge Loss
// =============================================================================

/// Hinge loss for SVM-style binary classification.
///
/// Expects labels in {-1, +1} (or {0, 1} which will be converted).
/// - Loss: `max(0, 1 - y * pred)`
/// - Gradient: `-y` if `y * pred < 1`, else `0`
/// - Hessian: `1` (constant for Newton step stability)
pub(super) fn compute_hinge_gradients_into(
    predictions: ArrayView2<f32>,
    targets: TargetsView<'_>,
    weights: WeightsView<'_>,
    mut grad_hess: ArrayViewMut2<GradsTuple>,
) {
    let (n_outputs, n_rows) = predictions.dim();
    debug_assert_eq!(grad_hess.dim(), (n_outputs, n_rows));
    let targets = targets.output(0);
    debug_assert_eq!(targets.len(), n_rows);

    for out_idx in 0..n_outputs {
        let preds_row = predictions.row(out_idx);
        let mut gh_row = grad_hess.row_mut(out_idx);

        for (i, ((&pred, &target), w)) in preds_row
            .iter()
            .zip(targets.iter())
            .zip(weights.iter(n_rows))
            .enumerate()
        {
            // Convert {0, 1} to {-1, +1}
            let y = if target > 0.5 { 1.0 } else { -1.0 };
            let margin = y * pred;

            gh_row[i].grad = if margin < 1.0 { -w * y } else { 0.0 };
            gh_row[i].hess = w;
        }
    }
}

pub(super) fn compute_hinge_base_score(
    _targets: TargetsView<'_>,
    _weights: WeightsView<'_>,
) -> Vec<f32> {
    vec![0.0]
}

// =============================================================================
// Softmax Loss
// =============================================================================

/// Softmax cross-entropy loss for multiclass classification.
///
/// Expects labels as class indices (0 to n_classes-1) stored as f32.
/// Predictions are K raw logits per sample.
///
/// # Layout
///
/// - `n_outputs` = n_classes
/// - `predictions`: `[n_classes, n_samples]`
/// - `targets`: class indices `[n_samples]` (single target column)
/// - `gradients/hessians`: `[n_classes, n_samples]`
pub(super) fn compute_softmax_gradients_into(
    predictions: ArrayView2<f32>,
    targets: TargetsView<'_>,
    weights: WeightsView<'_>,
    n_classes: usize,
    mut grad_hess: ArrayViewMut2<GradsTuple>,
) {
    const HESS_MIN: f32 = 1e-6;
    debug_assert!(n_classes >= 2, "n_classes must be >= 2");

    let (n_outputs, n_rows) = predictions.dim();
    debug_assert_eq!(n_outputs, n_classes);
    debug_assert_eq!(grad_hess.dim(), (n_outputs, n_rows));
    let targets = targets.output(0);
    debug_assert_eq!(targets.len(), n_rows);

    // Process sample by sample (need to compute softmax across classes)
    // Use column views to avoid repeated double-indexing
    for (i, (&target, w)) in targets.iter().zip(weights.iter(n_rows)).enumerate() {
        let label = target as usize;
        debug_assert!(
            label < n_outputs,
            "label {} >= n_classes {}",
            label,
            n_outputs
        );

        // Get column views for this sample
        let pred_col = predictions.column(i);
        let mut gh_col = grad_hess.column_mut(i);

        // Find max logit for numerical stability
        let max_logit = pred_col.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // Compute sum of exp(logit - max)
        let exp_sum: f32 = pred_col.iter().map(|&x| (x - max_logit).exp()).sum();

        // Compute gradients and hessians for each class
        for (c, (&pred, gh)) in pred_col.iter().zip(gh_col.iter_mut()).enumerate() {
            let p = (pred - max_logit).exp() / exp_sum;
            let target_indicator = if c == label { 1.0 } else { 0.0 };

            gh.grad = w * (p - target_indicator);
            gh.hess = (w * p * (1.0 - p)).max(HESS_MIN);
        }
    }
}

pub(super) fn compute_softmax_base_score(
    targets: TargetsView<'_>,
    weights: WeightsView<'_>,
    n_classes: usize,
) -> Vec<f32> {
    let targets = targets.output(0);
    let n_rows = targets.len();
    let n_outputs = n_classes;

    if n_rows == 0 {
        return vec![0.0; n_outputs];
    }

    // Count class frequencies
    let mut class_weights = vec![0.0f64; n_outputs];
    let mut total_weight = 0.0f64;

    for (&target, w) in targets.iter().zip(weights.iter(n_rows)) {
        let label = target as usize;
        if label < n_outputs {
            class_weights[label] += w as f64;
        }
        total_weight += w as f64;
    }

    // Convert to log-probabilities
    class_weights
        .into_iter()
        .map(|cw| {
            let p = (cw / total_weight).clamp(1e-7, 1.0 - 1e-7);
            p.ln() as f32
        })
        .collect()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::WeightsView;
    use crate::testing::DEFAULT_TOLERANCE;
    use approx::assert_abs_diff_eq;
    use ndarray::Array2;

    fn make_preds(n_outputs: usize, n_samples: usize, data: &[f32]) -> Array2<f32> {
        Array2::from_shape_vec((n_outputs, n_samples), data.to_vec()).unwrap()
    }

    fn make_targets(data: &[f32]) -> Array2<f32> {
        Array2::from_shape_vec((1, data.len()), data.to_vec()).unwrap()
    }

    fn make_grad_hess(n_outputs: usize, n_samples: usize) -> Array2<GradsTuple> {
        Array2::from_elem((n_outputs, n_samples), GradsTuple::default())
    }

    #[test]
    fn logistic_gradient_at_zero() {
        let preds = make_preds(1, 1, &[0.0]);
        let targets = make_targets(&[1.0]);
        let mut gh = make_grad_hess(1, 1);

        compute_logistic_gradients_into(
            preds.view(),
            TargetsView::new(targets.view()),
            WeightsView::None,
            gh.view_mut(),
        );

        // sigmoid(0) = 0.5, grad = 0.5 - 1 = -0.5
        assert_abs_diff_eq!(gh[[0, 0]].grad, -0.5, epsilon = DEFAULT_TOLERANCE);
        // hess = 0.5 * 0.5 = 0.25
        assert_abs_diff_eq!(gh[[0, 0]].hess, 0.25, epsilon = DEFAULT_TOLERANCE);
    }

    #[test]
    fn logistic_base_score() {
        let targets = make_targets(&[0.0, 0.0, 1.0, 1.0]);

        let output =
            compute_logistic_base_score(TargetsView::new(targets.view()), WeightsView::None);

        // 50% positive: log-odds = log(0.5/0.5) = 0
        assert_eq!(output.len(), 1);
        assert_abs_diff_eq!(output[0], 0.0, epsilon = DEFAULT_TOLERANCE);
    }

    #[test]
    fn logistic_weighted() {
        let preds = make_preds(1, 2, &[0.0, 0.0]);
        let targets = make_targets(&[1.0, 0.0]);
        let weights = ndarray::array![2.0f32, 0.5];
        let mut gh = make_grad_hess(1, 2);

        compute_logistic_gradients_into(
            preds.view(),
            TargetsView::new(targets.view()),
            WeightsView::from_array(weights.view()),
            gh.view_mut(),
        );

        // sigmoid(0) = 0.5
        // grad[0] = 2.0 * (0.5 - 1) = -1.0
        assert_abs_diff_eq!(gh[[0, 0]].grad, -1.0, epsilon = DEFAULT_TOLERANCE);
        // grad[1] = 0.5 * (0.5 - 0) = 0.25
        assert_abs_diff_eq!(gh[[0, 1]].grad, 0.25, epsilon = DEFAULT_TOLERANCE);
    }

    #[test]
    fn hinge_loss_margin() {
        // Correctly classified with margin
        let preds = make_preds(1, 1, &[2.0]);
        let targets = make_targets(&[1.0]);
        let mut gh = make_grad_hess(1, 1);

        compute_hinge_gradients_into(
            preds.view(),
            TargetsView::new(targets.view()),
            WeightsView::None,
            gh.view_mut(),
        );
        // margin = 1 * 2 = 2 >= 1, so grad = 0
        assert_abs_diff_eq!(gh[[0, 0]].grad, 0.0, epsilon = DEFAULT_TOLERANCE);

        // Misclassified
        let preds = make_preds(1, 1, &[-0.5]);
        compute_hinge_gradients_into(
            preds.view(),
            TargetsView::new(targets.view()),
            WeightsView::None,
            gh.view_mut(),
        );
        // margin = 1 * -0.5 < 1, so grad = -y = -1
        assert_abs_diff_eq!(gh[[0, 0]].grad, -1.0, epsilon = DEFAULT_TOLERANCE);
    }

    #[test]
    fn softmax_gradient() {
        let n_classes = 3;
        // 3 classes, 2 samples
        let preds = make_preds(
            3,
            2,
            &[
                1.0, 0.0, // class 0
                0.0, 1.0, // class 1
                0.0, 0.0, // class 2
            ],
        );
        // Class indices: sample 0 -> class 0, sample 1 -> class 1
        let targets = make_targets(&[0.0, 1.0]);
        let mut gh = make_grad_hess(3, 2);

        compute_softmax_gradients_into(
            preds.view(),
            TargetsView::new(targets.view()),
            WeightsView::None,
            n_classes,
            gh.view_mut(),
        );

        // For sample 0 (label=0): grad for class 0 should be negative (correct class)
        assert!(gh[[0, 0]].grad < 0.0);
        // Wrong classes should have positive gradient
        assert!(gh[[1, 0]].grad > 0.0);
    }

    #[test]
    fn softmax_base_score() {
        let n_classes = 3;
        // Class indices
        let targets = make_targets(&[0.0, 0.0, 1.0, 2.0]);

        let outputs = compute_softmax_base_score(
            TargetsView::new(targets.view()),
            WeightsView::None,
            n_classes,
        );

        // Class 0: 2/4, Class 1: 1/4, Class 2: 1/4
        assert_eq!(outputs.len(), 3);
        assert!(outputs[0] > outputs[1]); // Class 0 more frequent
        assert_abs_diff_eq!(outputs[1], outputs[2], epsilon = DEFAULT_TOLERANCE);
    }
}

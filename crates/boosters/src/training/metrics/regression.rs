//! Regression metrics.
//!
//! Metrics for evaluating regression model quality.
//!
//! # Multi-Output Support
//!
//! For multi-output models, metrics are computed per-output and then averaged.

use crate::data::TargetsView;
use crate::data::WeightsView;
use ndarray::ArrayView2;

// =============================================================================
// RMSE (Root Mean Squared Error)
// =============================================================================

pub(super) fn compute_rmse(
    predictions: ArrayView2<f32>,
    targets: TargetsView<'_>,
    weights: WeightsView<'_>,
) -> f64 {
    let targets = targets.output(0);
    let (n_outputs, n_rows) = predictions.dim();
    if n_rows == 0 {
        return 0.0;
    }

    let mut sum_rmse = 0.0f64;
    for out_idx in 0..n_outputs {
        let preds_row = predictions.row(out_idx);

        let (sum_sq, sum_w) = preds_row
            .iter()
            .zip(targets.iter())
            .zip(weights.iter(n_rows))
            .fold((0.0f64, 0.0f64), |(ss, sw), ((&p, &l), w)| {
                let diff = (p as f64) - (l as f64);
                (ss + (w as f64) * diff * diff, sw + w as f64)
            });

        sum_rmse += if sum_w > 0.0 {
            (sum_sq / sum_w).sqrt()
        } else {
            0.0
        };
    }

    sum_rmse / n_outputs as f64
}

// =============================================================================
// MAE (Mean Absolute Error)
// =============================================================================

pub(super) fn compute_mae(
    predictions: ArrayView2<f32>,
    targets: TargetsView<'_>,
    weights: WeightsView<'_>,
) -> f64 {
    let targets = targets.output(0);
    let (n_outputs, n_rows) = predictions.dim();
    if n_rows == 0 {
        return 0.0;
    }

    let mut sum_mae = 0.0f64;
    for out_idx in 0..n_outputs {
        let preds_row = predictions.row(out_idx);

        let (sum_ae, sum_w) = preds_row
            .iter()
            .zip(targets.iter())
            .zip(weights.iter(n_rows))
            .fold((0.0f64, 0.0f64), |(sa, sw), ((&p, &l), w)| {
                let ae = ((p as f64) - (l as f64)).abs();
                (sa + (w as f64) * ae, sw + w as f64)
            });

        sum_mae += if sum_w > 0.0 { sum_ae / sum_w } else { 0.0 };
    }

    sum_mae / n_outputs as f64
}

// =============================================================================
// MAPE (Mean Absolute Percentage Error)
// =============================================================================

pub(super) fn compute_mape(
    predictions: ArrayView2<f32>,
    targets: TargetsView<'_>,
    weights: WeightsView<'_>,
) -> f64 {
    let targets = targets.output(0);
    let (_, n_rows) = predictions.dim();
    if n_rows == 0 {
        return 0.0;
    }

    const EPS: f64 = 1e-15;

    let preds_row = predictions.row(0);

    let (sum_ape, sum_w) = preds_row
        .iter()
        .zip(targets.iter())
        .zip(weights.iter(n_rows))
        .fold((0.0f64, 0.0f64), |(sa, sw), ((&p, &l), w)| {
            let p = p as f64;
            let l = l as f64;
            let ape = (p - l).abs() / l.abs().max(EPS);
            (sa + (w as f64) * ape, sw + w as f64)
        });

    if sum_w > 0.0 {
        (sum_ape / sum_w) * 100.0
    } else {
        0.0
    }
}

// =============================================================================
// Quantile Metric (Pinball Loss)
// =============================================================================

pub(super) fn compute_quantile_pinball(
    predictions: ArrayView2<f32>,
    targets: TargetsView<'_>,
    weights: WeightsView<'_>,
    alphas: &[f32],
) -> f64 {
    let targets = targets.output(0);
    let (n_outputs, n_rows) = predictions.dim();
    if n_rows == 0 || n_outputs == 0 {
        return 0.0;
    }

    let n_quantiles = alphas.len();
    debug_assert_eq!(n_outputs, n_quantiles);

    let (total_loss, total_weight) = alphas
        .iter()
        .enumerate()
        .map(|(q, &alpha)| {
            let alpha_f64 = alpha as f64;
            let preds_row = predictions.row(q);

            preds_row
                .iter()
                .zip(targets.iter())
                .zip(weights.iter(n_rows))
                .fold((0.0f64, 0.0f64), |(acc_loss, acc_w), ((&pred, &y), w)| {
                    let residual = y as f64 - pred as f64;
                    let w = w as f64;
                    let loss = if residual >= 0.0 {
                        alpha_f64 * residual
                    } else {
                        (1.0 - alpha_f64) * (-residual)
                    };
                    (acc_loss + w * loss, acc_w + w)
                })
        })
        .fold((0.0, 0.0), |(tl, tw), (l, w)| (tl + l, tw + w));

    if total_weight > 0.0 {
        total_loss / total_weight
    } else {
        0.0
    }
}

// =============================================================================
// Poisson Deviance
// =============================================================================

pub(super) fn compute_poisson_deviance(
    predictions: ArrayView2<f32>,
    targets: TargetsView<'_>,
    weights: WeightsView<'_>,
) -> f64 {
    let targets = targets.output(0);
    let (_, n_rows) = predictions.dim();
    if n_rows == 0 {
        return 0.0;
    }

    const EPSILON: f64 = 1e-9;

    let preds_row = predictions.row(0);

    let (sum_wdev, sum_w) = preds_row
        .iter()
        .zip(targets.iter())
        .zip(weights.iter(n_rows))
        .fold((0.0f64, 0.0f64), |(swd, sw), ((&p, &l), w)| {
            let y = l as f64;
            let mu = (p as f64).max(EPSILON);
            let w = w as f64;

            let dev = if y > EPSILON {
                2.0 * (y * (y / mu).ln() - (y - mu))
            } else {
                2.0 * mu
            };

            (swd + w * dev, sw + w)
        });

    if sum_w > 0.0 { sum_wdev / sum_w } else { 0.0 }
}

// =============================================================================
// Huber Metric
// =============================================================================

pub(super) fn compute_huber(
    predictions: ArrayView2<f32>,
    targets: TargetsView<'_>,
    weights: WeightsView<'_>,
    delta: f64,
) -> f64 {
    debug_assert!(delta > 0.0, "delta must be positive, got {}", delta);
    let targets = targets.output(0);
    let (_, n_rows) = predictions.dim();
    if n_rows == 0 {
        return 0.0;
    }

    let preds_row = predictions.row(0);

    let (sum_wloss, sum_w) = preds_row
        .iter()
        .zip(targets.iter())
        .zip(weights.iter(n_rows))
        .fold((0.0f64, 0.0f64), |(swl, sw), ((&p, &l), w)| {
            let r = ((p as f64) - (l as f64)).abs();
            let w = w as f64;
            let loss = if r <= delta {
                0.5 * r * r
            } else {
                delta * (r - 0.5 * delta)
            };
            (swl + w * loss, sw + w)
        });

    if sum_w > 0.0 { sum_wloss / sum_w } else { 0.0 }
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

    // =========================================================================
    // RMSE tests
    // =========================================================================

    #[test]
    fn rmse_perfect() {
        let preds = make_preds(1, 3, &[1.0, 2.0, 3.0]);
        let labels = make_targets(&[1.0, 2.0, 3.0]);
        let rmse = compute_rmse(
            preds.view(),
            TargetsView::new(labels.view()),
            WeightsView::None,
        );
        assert!(rmse.abs() < 1e-10);
    }

    #[test]
    fn rmse_known_value() {
        // RMSE of [1, 2] vs [0, 0] = sqrt((1 + 4) / 2) = sqrt(2.5)
        let preds = make_preds(1, 2, &[1.0, 2.0]);
        let labels = make_targets(&[0.0, 0.0]);
        let rmse = compute_rmse(
            preds.view(),
            TargetsView::new(labels.view()),
            WeightsView::None,
        );
        assert!((rmse - 2.5f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn weighted_rmse() {
        let preds = make_preds(1, 2, &[1.0, 2.0]);
        let labels = make_targets(&[2.0, 2.0]);
        // Sample 0: error=1, sample 1: error=0
        // High weight on sample 0: (10*1 + 1*0) / 11 = 10/11
        let weights = ndarray::array![10.0f32, 1.0];
        let rmse = compute_rmse(
            preds.view(),
            TargetsView::new(labels.view()),
            WeightsView::from_array(weights.view()),
        );
        let expected = (10.0 / 11.0f64).sqrt();
        assert_abs_diff_eq!(rmse as f32, expected as f32, epsilon = DEFAULT_TOLERANCE);
    }

    // =========================================================================
    // MAE tests
    // =========================================================================

    #[test]
    fn mae_perfect() {
        let preds = make_preds(1, 3, &[1.0, 2.0, 3.0]);
        let labels = make_targets(&[1.0, 2.0, 3.0]);
        let mae = compute_mae(
            preds.view(),
            TargetsView::new(labels.view()),
            WeightsView::None,
        );
        assert!(mae.abs() < 1e-10);
    }

    #[test]
    fn mae_known_value() {
        // MAE of [1, 2] vs [0, 0] = (1 + 2) / 2 = 1.5
        let preds = make_preds(1, 2, &[1.0, 2.0]);
        let labels = make_targets(&[0.0, 0.0]);
        let mae = compute_mae(
            preds.view(),
            TargetsView::new(labels.view()),
            WeightsView::None,
        );
        assert_abs_diff_eq!(mae as f32, 1.5, epsilon = DEFAULT_TOLERANCE);
    }

    #[test]
    fn weighted_mae() {
        let preds = make_preds(1, 2, &[0.0, 2.0]);
        let labels = make_targets(&[2.0, 3.0]);
        // Errors: 2, 1
        // (10*2 + 1*1) / 11 = 21/11
        let weights = ndarray::array![10.0f32, 1.0];
        let mae = compute_mae(
            preds.view(),
            TargetsView::new(labels.view()),
            WeightsView::from_array(weights.view()),
        );
        let expected = 21.0 / 11.0;
        assert_abs_diff_eq!(mae as f32, expected as f32, epsilon = DEFAULT_TOLERANCE);
    }

    // =========================================================================
    // MAPE tests
    // =========================================================================

    #[test]
    fn mape_perfect() {
        let preds = make_preds(1, 3, &[1.0, 2.0, 3.0]);
        let labels = make_targets(&[1.0, 2.0, 3.0]);
        let mape = compute_mape(
            preds.view(),
            TargetsView::new(labels.view()),
            WeightsView::None,
        );
        assert!(mape.abs() < 1e-10);
    }

    #[test]
    fn mape_known_value() {
        // |1-2|/2 = 0.5, |3-4|/4 = 0.25 → mean = 0.375 → 37.5%
        let preds = make_preds(1, 2, &[1.0, 3.0]);
        let labels = make_targets(&[2.0, 4.0]);
        let mape = compute_mape(
            preds.view(),
            TargetsView::new(labels.view()),
            WeightsView::None,
        );
        assert_abs_diff_eq!(mape as f32, 37.5, epsilon = DEFAULT_TOLERANCE);
    }

    // =========================================================================
    // Quantile Metric tests
    // =========================================================================

    #[test]
    fn quantile_median_perfect() {
        let preds = make_preds(1, 3, &[1.0, 2.0, 3.0]);
        let labels = make_targets(&[1.0, 2.0, 3.0]);
        let loss = compute_quantile_pinball(
            preds.view(),
            TargetsView::new(labels.view()),
            WeightsView::None,
            &[0.5],
        );
        assert!(loss.abs() < 1e-10);
    }

    #[test]
    fn quantile_median_error() {
        // |1-2| = 1, |3-2| = 1 → pinball each = 0.5 → mean = 0.5
        let preds = make_preds(1, 2, &[2.0, 2.0]);
        let labels = make_targets(&[1.0, 3.0]);
        let loss = compute_quantile_pinball(
            preds.view(),
            TargetsView::new(labels.view()),
            WeightsView::None,
            &[0.5],
        );
        assert_abs_diff_eq!(loss as f32, 0.5, epsilon = DEFAULT_TOLERANCE);
    }

    // =========================================================================
    // Poisson Deviance tests
    // =========================================================================

    #[test]
    fn poisson_deviance_perfect() {
        let preds = make_preds(1, 3, &[1.0, 2.0, 3.0]);
        let labels = make_targets(&[1.0, 2.0, 3.0]);
        let dev = compute_poisson_deviance(
            preds.view(),
            TargetsView::new(labels.view()),
            WeightsView::None,
        );
        assert!(dev.abs() < 1e-8);
    }

    #[test]
    fn poisson_deviance_zero_labels() {
        // When y=0: deviance = 2 * mu
        let preds = make_preds(1, 2, &[1.0, 2.0]);
        let labels = make_targets(&[0.0, 0.0]);
        let dev = compute_poisson_deviance(
            preds.view(),
            TargetsView::new(labels.view()),
            WeightsView::None,
        );
        assert_abs_diff_eq!(dev as f32, 3.0, epsilon = DEFAULT_TOLERANCE);
    }

    // =========================================================================
    // Huber Metric tests
    // =========================================================================

    #[test]
    fn huber_perfect() {
        let preds = make_preds(1, 3, &[1.0, 2.0, 3.0]);
        let labels = make_targets(&[1.0, 2.0, 3.0]);
        let loss = compute_huber(
            preds.view(),
            TargetsView::new(labels.view()),
            WeightsView::None,
            1.0,
        );
        assert!(loss.abs() < 1e-10);
    }

    #[test]
    fn huber_quadratic_region() {
        // delta=1.0, residual=0.5 → loss = 0.5 * 0.25 = 0.125
        let preds = make_preds(1, 1, &[1.5]);
        let labels = make_targets(&[1.0]);
        let loss = compute_huber(
            preds.view(),
            TargetsView::new(labels.view()),
            WeightsView::None,
            1.0,
        );
        assert_abs_diff_eq!(loss as f32, 0.125, epsilon = DEFAULT_TOLERANCE);
    }

    #[test]
    fn huber_linear_region() {
        // delta=1.0, residual=2.0 → loss = 1.0 * (2.0 - 0.5) = 1.5
        let preds = make_preds(1, 1, &[3.0]);
        let labels = make_targets(&[1.0]);
        let loss = compute_huber(
            preds.view(),
            TargetsView::new(labels.view()),
            WeightsView::None,
            1.0,
        );
        assert_abs_diff_eq!(loss as f32, 1.5, epsilon = DEFAULT_TOLERANCE);
    }
}

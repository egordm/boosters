//! Classification metrics.
//!
//! Metrics for evaluating classification model quality.

use crate::data::TargetsView;
use crate::data::WeightsView;
use ndarray::ArrayView2;

// =============================================================================
// LogLoss (Binary Cross-Entropy)
// =============================================================================

pub(super) fn compute_logloss(
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

    let (sum_loss, sum_w) = preds_row
        .iter()
        .zip(targets.iter())
        .zip(weights.iter(n_rows))
        .fold((0.0f64, 0.0f64), |(sl, sw), ((&p, &l), w)| {
            let p = (p as f64).clamp(EPS, 1.0 - EPS);
            let l = l as f64;
            let loss = -(l * p.ln() + (1.0 - l) * (1.0 - p).ln());
            (sl + (w as f64) * loss, sw + w as f64)
        });

    if sum_w > 0.0 { sum_loss / sum_w } else { 0.0 }
}

// =============================================================================
// Accuracy
// =============================================================================

pub(super) fn compute_accuracy(
    predictions: ArrayView2<f32>,
    targets: TargetsView<'_>,
    weights: WeightsView<'_>,
    threshold: f32,
) -> f64 {
    let targets = targets.output(0);
    let (_, n_rows) = predictions.dim();
    if n_rows == 0 {
        return 0.0;
    }

    let preds_row = predictions.row(0);

    let (sum_correct, sum_w) = preds_row
        .iter()
        .zip(targets.iter())
        .zip(weights.iter(n_rows))
        .fold((0.0f64, 0.0f64), |(sc, sw), ((&p, &l), w)| {
            let pred_class = if p >= threshold { 1.0 } else { 0.0 };
            let correct = if (pred_class - l).abs() < 0.5 {
                1.0
            } else {
                0.0
            };
            (sc + (w as f64) * correct, sw + w as f64)
        });

    if sum_w > 0.0 {
        sum_correct / sum_w
    } else {
        0.0
    }
}

// =============================================================================
// Margin Accuracy
// =============================================================================

pub(super) fn compute_margin_accuracy(
    predictions: ArrayView2<f32>,
    targets: TargetsView<'_>,
    weights: WeightsView<'_>,
    threshold: f32,
) -> f64 {
    let targets = targets.output(0);
    let (_, n_rows) = predictions.dim();
    if n_rows == 0 {
        return 0.0;
    }

    let preds_row = predictions.row(0);

    let (sum_correct, sum_w) = preds_row
        .iter()
        .zip(targets.iter())
        .zip(weights.iter(n_rows))
        .fold((0.0f64, 0.0f64), |(sc, sw), ((&p, &l), w)| {
            let pred_class = if p >= threshold { 1.0 } else { 0.0 };
            let correct = if (pred_class - l).abs() < 0.5 {
                1.0
            } else {
                0.0
            };
            (sc + (w as f64) * correct, sw + w as f64)
        });

    if sum_w > 0.0 {
        sum_correct / sum_w
    } else {
        0.0
    }
}

// =============================================================================
// Multiclass Accuracy
// =============================================================================

pub(super) fn compute_multiclass_accuracy(
    predictions: ArrayView2<f32>,
    targets: TargetsView<'_>,
    weights: WeightsView<'_>,
) -> f64 {
    let targets = targets.output(0);
    let (n_outputs, n_rows) = predictions.dim();
    if n_rows == 0 {
        return 0.0;
    }

    // Single output: predictions are class indices
    if n_outputs == 1 {
        let preds_row = predictions.row(0);

        let (sum_correct, sum_w) = preds_row
            .iter()
            .zip(targets.iter())
            .zip(weights.iter(n_rows))
            .fold((0.0f64, 0.0f64), |(sc, sw), ((&p, &l), w)| {
                let correct = if (p.round() - l).abs() < 0.5 {
                    1.0
                } else {
                    0.0
                };
                (sc + (w as f64) * correct, sw + w as f64)
            });

        return if sum_w > 0.0 {
            sum_correct / sum_w
        } else {
            0.0
        };
    }

    // Multi-output: find argmax for each sample
    let argmax = |sample: usize| -> usize {
        (0..n_outputs)
            .max_by(|&a, &b| {
                let va = predictions[[a, sample]];
                let vb = predictions[[b, sample]];
                va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Less)
            })
            .unwrap_or(0)
    };

    let (sum_correct, sum_w) = targets.iter().enumerate().zip(weights.iter(n_rows)).fold(
        (0.0f64, 0.0f64),
        |(sc, sw), ((i, &l), w)| {
            let pred_class = argmax(i) as f32;
            let correct = if (pred_class - l).abs() < 0.5 {
                1.0
            } else {
                0.0
            };
            (sc + (w as f64) * correct, sw + w as f64)
        },
    );

    if sum_w > 0.0 {
        sum_correct / sum_w
    } else {
        0.0
    }
}

// =============================================================================
// AUC (Area Under ROC Curve)
// =============================================================================

pub(super) fn compute_auc(
    predictions: ArrayView2<f32>,
    targets: TargetsView<'_>,
    weights: WeightsView<'_>,
) -> f64 {
    let targets = targets.output(0);
    let (_, n_rows) = predictions.dim();
    if n_rows == 0 {
        return 0.5;
    }

    let preds_row = predictions.row(0);

    match weights {
        WeightsView::None => {
            let items: Vec<(f32, f32)> = preds_row
                .iter()
                .zip(targets.iter())
                .map(|(&p, &l)| (p, l))
                .collect();
            compute_auc_unweighted(&items)
        }
        WeightsView::Some(w) => {
            let items: Vec<(f32, f32, f32)> = preds_row
                .iter()
                .zip(targets.iter())
                .zip(w.iter())
                .map(|((&p, &l), &weight)| (p, l, weight))
                .collect();
            compute_auc_weighted(&items)
        }
    }
}

fn compute_auc_unweighted(items: &[(f32, f32)]) -> f64 {
    let n = items.len();

    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        let pa = items[a].0;
        let pb = items[b].0;
        pb.partial_cmp(&pa).unwrap_or(std::cmp::Ordering::Equal)
    });

    let n_pos = items.iter().filter(|(_, l)| *l > 0.5).count();
    let n_neg = n - n_pos;

    if n_pos == 0 || n_neg == 0 {
        return 0.5;
    }

    let mut rank_sum_pos = 0.0f64;
    let mut i = 0;

    while i < n {
        let mut j = i + 1;
        while j < n && (items[indices[i]].0 - items[indices[j]].0).abs() < 1e-10 {
            j += 1;
        }

        let avg_rank = (i + 1 + j) as f64 / 2.0;

        for &idx in indices.iter().take(j).skip(i) {
            if items[idx].1 > 0.5 {
                rank_sum_pos += avg_rank;
            }
        }

        i = j;
    }

    let n_pos_f = n_pos as f64;
    let n_neg_f = n_neg as f64;
    let sum_ascending_ranks = n_pos_f * (n as f64 + 1.0) - rank_sum_pos;

    (sum_ascending_ranks - n_pos_f * (n_pos_f + 1.0) / 2.0) / (n_pos_f * n_neg_f)
}

fn compute_auc_weighted(items: &[(f32, f32, f32)]) -> f64 {
    let n = items.len();

    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        let pa = items[a].0;
        let pb = items[b].0;
        pa.partial_cmp(&pb).unwrap_or(std::cmp::Ordering::Equal)
    });

    let (sum_pos, sum_neg) =
        items
            .iter()
            .fold((0.0f64, 0.0f64), |(sp, sn), &(_, label, weight)| {
                if label > 0.5 {
                    (sp + weight as f64, sn)
                } else {
                    (sp, sn + weight as f64)
                }
            });

    if sum_pos == 0.0 || sum_neg == 0.0 {
        return 0.5;
    }

    let mut weighted_concordant = 0.0f64;
    let mut cumulative_neg_weight = 0.0f64;
    let mut i = 0;

    while i < n {
        let mut j = i + 1;
        while j < n && (items[indices[i]].0 - items[indices[j]].0).abs() < 1e-10 {
            j += 1;
        }

        let mut group_pos_weight = 0.0f64;
        let mut group_neg_weight = 0.0f64;

        for &idx in indices.iter().take(j).skip(i) {
            let (_, label, weight) = items[idx];
            if label > 0.5 {
                group_pos_weight += weight as f64;
            } else {
                group_neg_weight += weight as f64;
            }
        }

        weighted_concordant += group_pos_weight * (cumulative_neg_weight + 0.5 * group_neg_weight);
        cumulative_neg_weight += group_neg_weight;

        i = j;
    }

    weighted_concordant / (sum_pos * sum_neg)
}

// =============================================================================
// Multiclass LogLoss
// =============================================================================

pub(super) fn compute_multiclass_logloss(
    predictions: ArrayView2<f32>,
    targets: TargetsView<'_>,
    weights: WeightsView<'_>,
) -> f64 {
    let targets = targets.output(0);
    let (n_outputs, n_rows) = predictions.dim();
    if n_rows == 0 || n_outputs == 0 {
        return 0.0;
    }

    const EPS: f64 = 1e-15;

    let (sum_loss, sum_w) = targets.iter().enumerate().zip(weights.iter(n_rows)).fold(
        (0.0f64, 0.0f64),
        |(sl, sw), ((i, &label), w)| {
            let class_idx = label.round() as usize;
            debug_assert!(class_idx < n_outputs, "label out of bounds");

            let prob = predictions[[class_idx, i]] as f64;
            let prob = prob.clamp(EPS, 1.0 - EPS);
            let loss = -prob.ln();

            (sl + (w as f64) * loss, sw + w as f64)
        },
    );

    if sum_w > 0.0 { sum_loss / sum_w } else { 0.0 }
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
    // LogLoss tests
    // =========================================================================

    #[test]
    fn logloss_perfect() {
        let preds = make_preds(1, 2, &[0.9999, 0.0001]);
        let labels = make_targets(&[1.0, 0.0]);
        let ll = compute_logloss(
            preds.view(),
            TargetsView::new(labels.view()),
            WeightsView::None,
        );
        assert!(ll < 0.01);
    }

    #[test]
    fn logloss_random() {
        let preds = make_preds(1, 2, &[0.5, 0.5]);
        let labels = make_targets(&[1.0, 0.0]);
        let ll = compute_logloss(
            preds.view(),
            TargetsView::new(labels.view()),
            WeightsView::None,
        );
        assert_abs_diff_eq!(ll as f32, 0.693, epsilon = 0.01);
    }

    #[test]
    fn logloss_weighted() {
        let preds = make_preds(1, 2, &[0.9, 0.1]);
        let labels = make_targets(&[1.0, 1.0]);

        let unweighted = compute_logloss(
            preds.view(),
            TargetsView::new(labels.view()),
            WeightsView::None,
        );

        // High weight on good prediction → lower loss
        let weights = ndarray::array![10.0f32, 1.0];
        let weighted = compute_logloss(
            preds.view(),
            TargetsView::new(labels.view()),
            WeightsView::from_array(weights.view()),
        );
        assert!(weighted < unweighted);
    }

    // =========================================================================
    // Accuracy tests
    // =========================================================================

    #[test]
    fn accuracy_perfect() {
        let preds = make_preds(1, 4, &[0.9, 0.1, 0.8, 0.2]);
        let labels = make_targets(&[1.0, 0.0, 1.0, 0.0]);
        let acc = compute_accuracy(
            preds.view(),
            TargetsView::new(labels.view()),
            WeightsView::None,
            0.5,
        );
        assert_abs_diff_eq!(acc as f32, 1.0, epsilon = DEFAULT_TOLERANCE);
    }

    #[test]
    fn accuracy_half() {
        let preds = make_preds(1, 4, &[0.9, 0.9, 0.1, 0.1]);
        let labels = make_targets(&[1.0, 0.0, 1.0, 0.0]);
        let acc = compute_accuracy(
            preds.view(),
            TargetsView::new(labels.view()),
            WeightsView::None,
            0.5,
        );
        assert_abs_diff_eq!(acc as f32, 0.5, epsilon = DEFAULT_TOLERANCE);
    }

    #[test]
    fn accuracy_weighted() {
        let preds = make_preds(1, 2, &[0.9, 0.9]);
        let labels = make_targets(&[1.0, 0.0]);

        // High weight on correct sample
        let weights = ndarray::array![10.0f32, 1.0];
        let weighted = compute_accuracy(
            preds.view(),
            TargetsView::new(labels.view()),
            WeightsView::from_array(weights.view()),
            0.5,
        );
        assert_abs_diff_eq!(weighted as f32, 10.0 / 11.0, epsilon = DEFAULT_TOLERANCE);
    }

    // =========================================================================
    // Multiclass Accuracy tests
    // =========================================================================

    #[test]
    fn multiclass_accuracy_basic() {
        // 3 classes, 4 samples - predictions in [n_classes, n_samples] layout
        let preds = make_preds(
            3,
            4,
            &[
                3.0, 0.0, 1.0, 0.0, // class 0
                1.0, 3.0, 0.0, 3.0, // class 1
                0.0, 1.0, 3.0, 1.0, // class 2
            ],
        );
        let labels = make_targets(&[0.0, 1.0, 2.0, 0.0]); // Last one wrong
        let acc = compute_multiclass_accuracy(
            preds.view(),
            TargetsView::new(labels.view()),
            WeightsView::None,
        );
        assert_abs_diff_eq!(acc as f32, 0.75, epsilon = DEFAULT_TOLERANCE);
    }

    // =========================================================================
    // AUC tests
    // =========================================================================

    #[test]
    fn auc_perfect() {
        let preds = make_preds(1, 4, &[0.9, 0.8, 0.3, 0.2]);
        let labels = make_targets(&[1.0, 1.0, 0.0, 0.0]);
        let auc = compute_auc(
            preds.view(),
            TargetsView::new(labels.view()),
            WeightsView::None,
        );
        assert_abs_diff_eq!(auc as f32, 1.0, epsilon = DEFAULT_TOLERANCE);
    }

    #[test]
    fn auc_random() {
        let preds = make_preds(1, 4, &[0.5, 0.5, 0.5, 0.5]);
        let labels = make_targets(&[1.0, 0.0, 1.0, 0.0]);
        let auc = compute_auc(
            preds.view(),
            TargetsView::new(labels.view()),
            WeightsView::None,
        );
        assert_abs_diff_eq!(auc as f32, 0.5, epsilon = DEFAULT_TOLERANCE);
    }

    #[test]
    fn auc_worst() {
        let preds = make_preds(1, 4, &[0.2, 0.3, 0.8, 0.9]);
        let labels = make_targets(&[1.0, 1.0, 0.0, 0.0]);
        let auc = compute_auc(
            preds.view(),
            TargetsView::new(labels.view()),
            WeightsView::None,
        );
        assert!(auc < 0.01);
    }

    // =========================================================================
    // Multiclass LogLoss tests
    // =========================================================================

    #[test]
    fn mlogloss_perfect() {
        // 3 classes, 2 samples - almost perfect predictions
        let preds = make_preds(
            3,
            2,
            &[
                0.99, 0.005, // class 0
                0.005, 0.99, // class 1
                0.005, 0.005, // class 2
            ],
        );
        let labels = make_targets(&[0.0, 1.0]);
        let mlogloss = compute_multiclass_logloss(
            preds.view(),
            TargetsView::new(labels.view()),
            WeightsView::None,
        );
        assert!(mlogloss < 0.02);
    }

    #[test]
    fn mlogloss_uniform() {
        // Uniform predictions: -log(1/3) ≈ 1.099
        let preds = make_preds(
            3,
            3,
            &[
                0.333, 0.333, 0.333, // class 0
                0.333, 0.333, 0.333, // class 1
                0.334, 0.334, 0.334, // class 2
            ],
        );
        let labels = make_targets(&[0.0, 1.0, 2.0]);
        let mlogloss = compute_multiclass_logloss(
            preds.view(),
            TargetsView::new(labels.view()),
            WeightsView::None,
        );
        assert_abs_diff_eq!(mlogloss as f32, 1.099, epsilon = 0.01);
    }
}

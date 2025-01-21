//! Classification metrics.
//!
//! Metrics for evaluating classification model quality.

use ndarray::ArrayView2;

use super::MetricFn;
use crate::dataset::TargetsView;
use crate::inference::PredictionKind;
use crate::dataset::WeightsView;

// =============================================================================
// LogLoss (Binary Cross-Entropy)
// =============================================================================

/// Binary cross-entropy: -mean(y*log(p) + (1-y)*log(1-p))
///
/// Lower is better. Used for binary classification.
/// Expects predictions to be probabilities in (0, 1).
#[derive(Debug, Clone, Copy, Default)]
pub struct LogLoss;

impl MetricFn for LogLoss {
    fn compute(
        &self,
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

    fn higher_is_better(&self) -> bool {
        false
    }

    fn expected_prediction_kind(&self) -> PredictionKind {
        PredictionKind::Probability
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
/// Higher is better. For binary classification, uses configurable threshold.
#[derive(Debug, Clone, Copy)]
pub struct Accuracy {
    pub threshold: f32,
}

impl Default for Accuracy {
    fn default() -> Self {
        Self { threshold: 0.5 }
    }
}

impl Accuracy {
    pub fn with_threshold(threshold: f32) -> Self {
        Self { threshold }
    }
}

impl MetricFn for Accuracy {
    fn compute(
        &self,
        predictions: ArrayView2<f32>,
        targets: TargetsView<'_>,
        weights: WeightsView<'_>,
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
                let pred_class = if p >= self.threshold { 1.0 } else { 0.0 };
                let correct = if (pred_class - l).abs() < 0.5 { 1.0 } else { 0.0 };
                (sc + (w as f64) * correct, sw + w as f64)
            });

        if sum_w > 0.0 { sum_correct / sum_w } else { 0.0 }
    }

    fn higher_is_better(&self) -> bool {
        true
    }

    fn expected_prediction_kind(&self) -> PredictionKind {
        PredictionKind::Probability
    }

    fn name(&self) -> &'static str {
        "accuracy"
    }
}

// =============================================================================
// Margin Accuracy
// =============================================================================

/// Binary classification accuracy for **margin** predictions.
///
/// Higher is better. Uses a configurable threshold in margin space (default: 0.0).
#[derive(Debug, Clone, Copy)]
pub struct MarginAccuracy {
    pub threshold: f32,
}

impl Default for MarginAccuracy {
    fn default() -> Self {
        Self { threshold: 0.0 }
    }
}

impl MarginAccuracy {
    #[allow(dead_code)]
    pub fn with_threshold(threshold: f32) -> Self {
        Self { threshold }
    }
}

impl MetricFn for MarginAccuracy {
    fn compute(
        &self,
        predictions: ArrayView2<f32>,
        targets: TargetsView<'_>,
        weights: WeightsView<'_>,
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
                let pred_class = if p >= self.threshold { 1.0 } else { 0.0 };
                let correct = if (pred_class - l).abs() < 0.5 { 1.0 } else { 0.0 };
                (sc + (w as f64) * correct, sw + w as f64)
            });

        if sum_w > 0.0 { sum_correct / sum_w } else { 0.0 }
    }

    fn higher_is_better(&self) -> bool {
        true
    }

    fn expected_prediction_kind(&self) -> PredictionKind {
        PredictionKind::Margin
    }

    fn name(&self) -> &'static str {
        "margin_accuracy"
    }
}

// =============================================================================
// Multiclass Accuracy
// =============================================================================

/// Multiclass accuracy.
///
/// Expects predictions as per-class scores with shape `[n_classes, n_samples]`.
/// Labels are class indices in `0..n_classes`.
///
/// Higher is better.
#[derive(Debug, Clone, Copy, Default)]
pub struct MulticlassAccuracy;

impl MetricFn for MulticlassAccuracy {
    fn compute(
        &self,
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
                    let correct = if (p.round() - l).abs() < 0.5 { 1.0 } else { 0.0 };
                    (sc + (w as f64) * correct, sw + w as f64)
                });

            return if sum_w > 0.0 { sum_correct / sum_w } else { 0.0 };
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

        let (sum_correct, sum_w) = targets
            .iter()
            .enumerate()
            .zip(weights.iter(n_rows))
            .fold((0.0f64, 0.0f64), |(sc, sw), ((i, &l), w)| {
                let pred_class = argmax(i) as f32;
                let correct = if (pred_class - l).abs() < 0.5 { 1.0 } else { 0.0 };
                (sc + (w as f64) * correct, sw + w as f64)
            });

        if sum_w > 0.0 { sum_correct / sum_w } else { 0.0 }
    }

    fn higher_is_better(&self) -> bool {
        true
    }

    fn expected_prediction_kind(&self) -> PredictionKind {
        PredictionKind::Probability
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
#[derive(Debug, Clone, Copy, Default)]
pub struct Auc;

impl MetricFn for Auc {
    fn compute(
        &self,
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
        let preds_slice = preds_row.as_slice().expect("predictions row should be contiguous");
        let targets_slice = targets.as_slice().expect("targets should be contiguous");

        match weights {
            WeightsView::None => compute_auc_unweighted(preds_slice, targets_slice),
            WeightsView::Some(w) => {
                let weights_slice = w.as_slice().expect("weights should be contiguous");
                compute_auc_weighted(preds_slice, targets_slice, weights_slice)
            }
        }
    }

    fn higher_is_better(&self) -> bool {
        true
    }

    fn expected_prediction_kind(&self) -> PredictionKind {
        PredictionKind::Probability
    }

    fn name(&self) -> &'static str {
        "auc"
    }
}

fn compute_auc_unweighted(predictions: &[f32], labels: &[f32]) -> f64 {
    let n = predictions.len();

    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        predictions[b]
            .partial_cmp(&predictions[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let n_pos = labels.iter().filter(|&&l| l > 0.5).count();
    let n_neg = n - n_pos;

    if n_pos == 0 || n_neg == 0 {
        return 0.5;
    }

    let mut rank_sum_pos = 0.0f64;
    let mut i = 0;

    while i < n {
        let mut j = i + 1;
        while j < n && (predictions[indices[i]] - predictions[indices[j]]).abs() < 1e-10 {
            j += 1;
        }

        let avg_rank = (i + 1 + j) as f64 / 2.0;

        for &idx in indices.iter().take(j).skip(i) {
            if labels[idx] > 0.5 {
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

fn compute_auc_weighted(predictions: &[f32], labels: &[f32], weights: &[f32]) -> f64 {
    let n = predictions.len();

    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        predictions[a]
            .partial_cmp(&predictions[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let (sum_pos, sum_neg) = labels
        .iter()
        .zip(weights.iter())
        .fold((0.0f64, 0.0f64), |(sp, sn), (&l, &w)| {
            if l > 0.5 {
                (sp + w as f64, sn)
            } else {
                (sp, sn + w as f64)
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
        while j < n && (predictions[indices[i]] - predictions[indices[j]]).abs() < 1e-10 {
            j += 1;
        }

        let mut group_pos_weight = 0.0f64;
        let mut group_neg_weight = 0.0f64;

        for &idx in indices.iter().take(j).skip(i) {
            if labels[idx] > 0.5 {
                group_pos_weight += weights[idx] as f64;
            } else {
                group_neg_weight += weights[idx] as f64;
            }
        }

        weighted_concordant +=
            group_pos_weight * (cumulative_neg_weight + 0.5 * group_neg_weight);
        cumulative_neg_weight += group_neg_weight;

        i = j;
    }

    weighted_concordant / (sum_pos * sum_neg)
}

// =============================================================================
// Multiclass LogLoss
// =============================================================================

/// Multiclass cross-entropy: -mean(log(p_true_class))
///
/// Lower is better. Labels are class indices in `0..n_classes`.
#[derive(Debug, Clone, Copy, Default)]
pub struct MulticlassLogLoss;

impl MetricFn for MulticlassLogLoss {
    fn compute(
        &self,
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

        let (sum_loss, sum_w) = targets
            .iter()
            .enumerate()
            .zip(weights.iter(n_rows))
            .fold((0.0f64, 0.0f64), |(sl, sw), ((i, &label), w)| {
                let class_idx = label.round() as usize;
                debug_assert!(class_idx < n_outputs, "label out of bounds");

                let prob = predictions[[class_idx, i]] as f64;
                let prob = prob.clamp(EPS, 1.0 - EPS);
                let loss = -prob.ln();

                (sl + (w as f64) * loss, sw + w as f64)
            });

        if sum_w > 0.0 { sum_loss / sum_w } else { 0.0 }
    }

    fn higher_is_better(&self) -> bool {
        false
    }

    fn expected_prediction_kind(&self) -> PredictionKind {
        PredictionKind::Probability
    }

    fn name(&self) -> &'static str {
        "mlogloss"
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use crate::testing::DEFAULT_TOLERANCE;
    use crate::dataset::WeightsView;
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
        let ll = LogLoss.compute(preds.view(), TargetsView::new(labels.view()), WeightsView::None);
        assert!(ll < 0.01);
    }

    #[test]
    fn logloss_random() {
        let preds = make_preds(1, 2, &[0.5, 0.5]);
        let labels = make_targets(&[1.0, 0.0]);
        let ll = LogLoss.compute(preds.view(), TargetsView::new(labels.view()), WeightsView::None);
        assert_abs_diff_eq!(ll as f32, 0.693, epsilon = 0.01);
    }

    #[test]
    fn logloss_weighted() {
        let preds = make_preds(1, 2, &[0.9, 0.1]);
        let labels = make_targets(&[1.0, 1.0]);
        
        let unweighted = LogLoss.compute(preds.view(), TargetsView::new(labels.view()), WeightsView::None);
        
        // High weight on good prediction → lower loss
        let weights = ndarray::array![10.0f32, 1.0];
        let weighted = LogLoss.compute(preds.view(), TargetsView::new(labels.view()), WeightsView::from_array(weights.view()));
        assert!(weighted < unweighted);
    }

    // =========================================================================
    // Accuracy tests
    // =========================================================================

    #[test]
    fn accuracy_perfect() {
        let preds = make_preds(1, 4, &[0.9, 0.1, 0.8, 0.2]);
        let labels = make_targets(&[1.0, 0.0, 1.0, 0.0]);
        let acc = Accuracy::default().compute(preds.view(), TargetsView::new(labels.view()), WeightsView::None);
        assert_abs_diff_eq!(acc as f32, 1.0, epsilon = DEFAULT_TOLERANCE);
    }

    #[test]
    fn accuracy_half() {
        let preds = make_preds(1, 4, &[0.9, 0.9, 0.1, 0.1]);
        let labels = make_targets(&[1.0, 0.0, 1.0, 0.0]);
        let acc = Accuracy::default().compute(preds.view(), TargetsView::new(labels.view()), WeightsView::None);
        assert_abs_diff_eq!(acc as f32, 0.5, epsilon = DEFAULT_TOLERANCE);
    }

    #[test]
    fn accuracy_weighted() {
        let preds = make_preds(1, 2, &[0.9, 0.9]);
        let labels = make_targets(&[1.0, 0.0]);
        
        // High weight on correct sample
        let weights = ndarray::array![10.0f32, 1.0];
        let weighted = Accuracy::default().compute(preds.view(), TargetsView::new(labels.view()), WeightsView::from_array(weights.view()));
        assert_abs_diff_eq!(weighted as f32, 10.0 / 11.0, epsilon = DEFAULT_TOLERANCE);
    }

    // =========================================================================
    // Multiclass Accuracy tests
    // =========================================================================

    #[test]
    fn multiclass_accuracy_basic() {
        // 3 classes, 4 samples - predictions in [n_classes, n_samples] layout
        let preds = make_preds(3, 4, &[
            3.0, 0.0, 1.0, 0.0,  // class 0
            1.0, 3.0, 0.0, 3.0,  // class 1
            0.0, 1.0, 3.0, 1.0,  // class 2
        ]);
        let labels = make_targets(&[0.0, 1.0, 2.0, 0.0]); // Last one wrong
        let acc = MulticlassAccuracy.compute(preds.view(), TargetsView::new(labels.view()), WeightsView::None);
        assert_abs_diff_eq!(acc as f32, 0.75, epsilon = DEFAULT_TOLERANCE);
    }

    // =========================================================================
    // AUC tests
    // =========================================================================

    #[test]
    fn auc_perfect() {
        let preds = make_preds(1, 4, &[0.9, 0.8, 0.3, 0.2]);
        let labels = make_targets(&[1.0, 1.0, 0.0, 0.0]);
        let auc = Auc.compute(preds.view(), TargetsView::new(labels.view()), WeightsView::None);
        assert_abs_diff_eq!(auc as f32, 1.0, epsilon = DEFAULT_TOLERANCE);
    }

    #[test]
    fn auc_random() {
        let preds = make_preds(1, 4, &[0.5, 0.5, 0.5, 0.5]);
        let labels = make_targets(&[1.0, 0.0, 1.0, 0.0]);
        let auc = Auc.compute(preds.view(), TargetsView::new(labels.view()), WeightsView::None);
        assert_abs_diff_eq!(auc as f32, 0.5, epsilon = DEFAULT_TOLERANCE);
    }

    #[test]
    fn auc_worst() {
        let preds = make_preds(1, 4, &[0.2, 0.3, 0.8, 0.9]);
        let labels = make_targets(&[1.0, 1.0, 0.0, 0.0]);
        let auc = Auc.compute(preds.view(), TargetsView::new(labels.view()), WeightsView::None);
        assert!(auc < 0.01);
    }

    // =========================================================================
    // Multiclass LogLoss tests
    // =========================================================================

    #[test]
    fn mlogloss_perfect() {
        // 3 classes, 2 samples - almost perfect predictions
        let preds = make_preds(3, 2, &[
            0.99, 0.005,   // class 0
            0.005, 0.99,   // class 1
            0.005, 0.005,  // class 2
        ]);
        let labels = make_targets(&[0.0, 1.0]);
        let mlogloss = MulticlassLogLoss.compute(preds.view(), TargetsView::new(labels.view()), WeightsView::None);
        assert!(mlogloss < 0.02);
    }

    #[test]
    fn mlogloss_uniform() {
        // Uniform predictions: -log(1/3) ≈ 1.099
        let preds = make_preds(3, 3, &[
            0.333, 0.333, 0.333,  // class 0
            0.333, 0.333, 0.333,  // class 1
            0.334, 0.334, 0.334,  // class 2
        ]);
        let labels = make_targets(&[0.0, 1.0, 2.0]);
        let mlogloss = MulticlassLogLoss.compute(preds.view(), TargetsView::new(labels.view()), WeightsView::None);
        assert_abs_diff_eq!(mlogloss as f32, 1.099, epsilon = 0.01);
    }

    // =========================================================================
    // Metric properties
    // =========================================================================

    #[test]
    fn metric_properties() {
        assert!(!LogLoss.higher_is_better());
        assert!(Accuracy::default().higher_is_better());
        assert!(MulticlassAccuracy.higher_is_better());
        assert!(Auc.higher_is_better());
        assert!(!MulticlassLogLoss.higher_is_better());

        assert_eq!(LogLoss.name(), "logloss");
        assert_eq!(Accuracy::default().name(), "accuracy");
        assert_eq!(MulticlassAccuracy.name(), "multiclass_accuracy");
        assert_eq!(Auc.name(), "auc");
        assert_eq!(MulticlassLogLoss.name(), "mlogloss");
    }
}

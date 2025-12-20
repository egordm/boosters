//! Classification metrics.
//!
//! Metrics for evaluating classification model quality.

use super::MetricFn;
use crate::inference::common::PredictionKind;

// =============================================================================
// LogLoss (Binary Cross-Entropy)
// =============================================================================

/// Binary cross-entropy: -mean(y*log(p) + (1-y)*log(1-p))
///
/// Lower is better. Used for binary classification.
/// Expects predictions to be probabilities in (0, 1).
///
/// # Weighted Computation
///
/// When weights are provided, computes weighted log loss:
/// ```text
/// sum(w * cross_entropy) / sum(w)
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct LogLoss;

impl MetricFn for LogLoss {
    fn compute(
        &self,
        n_rows: usize,
        n_outputs: usize,
        predictions: &[f32],
        targets: &[f32],
        weights: &[f32],
    ) -> f64 {
        debug_assert_eq!(n_outputs, 1);
        if n_rows == 0 {
            return 0.0;
        }

        let predictions = &predictions[..n_rows];
        let labels = &targets[..n_rows];
        debug_assert_eq!(predictions.len(), labels.len());

        let eps = 1e-15f64; // Clip to avoid log(0)

        if weights.is_empty() {
                predictions
                    .iter()
                    .zip(labels.iter())
                    .map(|(p, l)| {
                        let p = (*p as f64).clamp(eps, 1.0 - eps);
                        let l = *l as f64;
                        -(l * p.ln() + (1.0 - l) * (1.0 - p).ln())
                    })
                    .sum::<f64>()
                    / predictions.len() as f64
        } else {
                debug_assert_eq!(predictions.len(), weights.len());

                let (weighted_sum, weight_sum) = predictions
                    .iter()
                    .zip(labels.iter())
                    .zip(weights.iter())
                    .fold((0.0f64, 0.0f64), |(acc_loss, acc_w), ((p, l), wt)| {
                        let wt = *wt as f64;
                        let p = (*p as f64).clamp(eps, 1.0 - eps);
                        let l = *l as f64;
                        let loss = -(l * p.ln() + (1.0 - l) * (1.0 - p).ln());
                        (acc_loss + wt * loss, acc_w + wt)
                    });

                if weight_sum == 0.0 {
                    return 0.0;
                }

                weighted_sum / weight_sum
        }
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
///
/// # Weighted Computation
///
/// When weights are provided, computes weighted accuracy:
/// ```text
/// sum(w * correct) / sum(w)
/// ```
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

impl MetricFn for Accuracy {
    fn compute(
        &self,
        n_rows: usize,
        n_outputs: usize,
        predictions: &[f32],
        targets: &[f32],
        weights: &[f32],
    ) -> f64 {
        debug_assert_eq!(n_outputs, 1);
        if n_rows == 0 {
            return 0.0;
        }

        let predictions = &predictions[..n_rows];
        let labels = &targets[..n_rows];
        debug_assert_eq!(predictions.len(), labels.len());

        if weights.is_empty() {
                let correct = predictions
                    .iter()
                    .zip(labels.iter())
                    .filter(|(p, l)| {
                        let pred_class = if **p >= self.threshold { 1.0 } else { 0.0 };
                        (pred_class - **l).abs() < 0.5
                    })
                    .count();

                correct as f64 / predictions.len() as f64
        } else {
                debug_assert_eq!(predictions.len(), weights.len());

                let (weighted_correct, weight_sum) = predictions
                    .iter()
                    .zip(labels.iter())
                    .zip(weights.iter())
                    .fold((0.0f64, 0.0f64), |(acc_correct, acc_w), ((p, l), wt)| {
                        let wt = *wt as f64;
                        let pred_class = if *p >= self.threshold { 1.0 } else { 0.0 };
                        let is_correct = if (pred_class - *l).abs() < 0.5 {
                            1.0
                        } else {
                            0.0
                        };
                        (acc_correct + wt * is_correct, acc_w + wt)
                    });

                if weight_sum == 0.0 {
                    return 0.0;
                }

                weighted_correct / weight_sum
        }
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
#[allow(dead_code)]
pub struct MarginAccuracy {
    /// Threshold for margin classification (default: 0.0).
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
        n_rows: usize,
        n_outputs: usize,
        predictions: &[f32],
        targets: &[f32],
        weights: &[f32],
    ) -> f64 {
        debug_assert_eq!(n_outputs, 1);
        if n_rows == 0 {
            return 0.0;
        }

        let predictions = &predictions[..n_rows];
        let labels = &targets[..n_rows];
        debug_assert_eq!(predictions.len(), labels.len());

        if weights.is_empty() {
            let correct = predictions
                .iter()
                .zip(labels.iter())
                .filter(|(p, l)| {
                    let pred_class = if **p >= self.threshold { 1.0 } else { 0.0 };
                    (pred_class - **l).abs() < 0.5
                })
                .count();

            correct as f64 / predictions.len() as f64
        } else {
            debug_assert_eq!(predictions.len(), weights.len());

            let (weighted_correct, weight_sum) = predictions
                .iter()
                .zip(labels.iter())
                .zip(weights.iter())
                .fold((0.0f64, 0.0f64), |(acc_correct, acc_w), ((p, l), wt)| {
                    let wt = *wt as f64;
                    let pred_class = if *p >= self.threshold { 1.0 } else { 0.0 };
                    let is_correct = if (pred_class - *l).abs() < 0.5 { 1.0 } else { 0.0 };
                    (acc_correct + wt * is_correct, acc_w + wt)
                });

            if weight_sum == 0.0 {
                return 0.0;
            }

            weighted_correct / weight_sum
        }
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
/// Expects predictions as per-class scores (probabilities or logits) with shape
/// `(n_rows, n_classes)` in row-major order.
///
/// Higher is better.
///
/// # Weighted Computation
///
/// When weights are provided, computes weighted accuracy:
/// ```text
/// sum(w * correct) / sum(w)
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct MulticlassAccuracy;

impl MetricFn for MulticlassAccuracy {
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

        let labels = &targets[..n_rows];

        // Backward-compatible mode: predictions are class indices with shape (n_rows, 1).
        if n_outputs == 1 {
            let predictions = &predictions[..n_rows];

            if weights.is_empty() {
                let correct = predictions
                    .iter()
                    .zip(labels.iter())
                    .filter(|(p, l)| ((*p).round() - **l).abs() < 0.5)
                    .count();

                return correct as f64 / n_rows as f64;
            }

            debug_assert_eq!(weights.len(), n_rows);
            let mut weighted_correct = 0.0f64;
            let mut weight_sum = 0.0f64;
            for row in 0..n_rows {
                let wt = weights[row] as f64;
                weight_sum += wt;
                if (predictions[row].round() - labels[row]).abs() < 0.5 {
                    weighted_correct += wt;
                }
            }

            if weight_sum == 0.0 {
                return 0.0;
            }

            return weighted_correct / weight_sum;
        }

        debug_assert!(n_outputs >= 2, "multiclass requires n_outputs >= 2");

        let required = n_rows * n_outputs;
        debug_assert!(predictions.len() >= required);

        // Helper to find argmax for a sample in column-major layout
        let argmax = |row: usize| -> usize {
            (0..n_outputs)
                .max_by(|&a, &b| {
                    let va = predictions[a * n_rows + row];
                    let vb = predictions[b * n_rows + row];
                    va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Less)
                })
                .unwrap_or(0)
        };

        if weights.is_empty() {
            let mut correct = 0usize;
            for (row, &label) in labels.iter().enumerate() {
                let pred_class = argmax(row) as f32;
                if (pred_class - label).abs() < 0.5 {
                    correct += 1;
                }
            }

            correct as f64 / n_rows as f64
        } else {
            debug_assert_eq!(weights.len(), n_rows);

            let mut weighted_correct = 0.0f64;
            let mut weight_sum = 0.0f64;
            for row in 0..n_rows {
                let pred_class = argmax(row) as f32;
                let wt = weights[row] as f64;
                weight_sum += wt;
                if (pred_class - labels[row]).abs() < 0.5 {
                    weighted_correct += wt;
                }
            }

            if weight_sum == 0.0 {
                return 0.0;
            }

            weighted_correct / weight_sum
        }
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
/// Expects predictions to be scores (higher = more likely positive).
///
/// # Algorithm
///
/// Uses the efficient O(n log n) algorithm based on sorting and ranking,
/// rather than the naive O(n²) pairwise comparison. This is the standard
/// approach used in libraries like scikit-learn and LightGBM.
///
/// The algorithm:
/// 1. Sort samples by prediction score (descending)
/// 2. Assign ranks (handling ties by averaging)
/// 3. Compute AUC using the Wilcoxon-Mann-Whitney statistic:
///    AUC = (sum of positive ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
///
/// # Weighted Computation
///
/// When weights are provided, uses weighted rank sums.
#[derive(Debug, Clone, Copy, Default)]
pub struct Auc;

impl MetricFn for Auc {
    fn compute(
        &self,
        n_rows: usize,
        n_outputs: usize,
        predictions: &[f32],
        targets: &[f32],
        weights: &[f32],
    ) -> f64 {
        debug_assert_eq!(n_outputs, 1);
        if n_rows == 0 {
            return 0.5;
        }

        let predictions = &predictions[..n_rows];
        let labels = &targets[..n_rows];
        debug_assert_eq!(predictions.len(), labels.len());

        if weights.is_empty() {
            compute_auc_unweighted(predictions, labels)
        } else {
            debug_assert_eq!(predictions.len(), weights.len());
            compute_auc_weighted(predictions, labels, weights)
        }
    }

    fn higher_is_better(&self) -> bool {
        true
    }

    fn expected_prediction_kind(&self) -> PredictionKind {
        // AUC is invariant under any monotonic transform, but by convention
        // we treat it as probability-based for binary classification.
        PredictionKind::Probability
    }

    fn name(&self) -> &'static str {
        "auc"
    }
}

/// Compute AUC using efficient O(n log n) sorting-based algorithm.
fn compute_auc_unweighted(predictions: &[f32], labels: &[f32]) -> f64 {
    let n = predictions.len();

    // Create indices sorted by prediction (descending) for tie-breaking stability
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        predictions[b]
            .partial_cmp(&predictions[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Count positives and negatives
    let n_pos = labels.iter().filter(|&&l| l > 0.5).count();
    let n_neg = n - n_pos;

    if n_pos == 0 || n_neg == 0 {
        return 0.5; // Undefined, return random
    }

    // Compute rank sum for positives using the sorted order
    // Handle ties by averaging ranks
    let mut rank_sum_pos = 0.0f64;
    let mut i = 0;

    while i < n {
        // Find all samples with the same prediction (ties)
        let mut j = i + 1;
        while j < n && (predictions[indices[i]] - predictions[indices[j]]).abs() < 1e-10 {
            j += 1;
        }

        // Average rank for tied samples (1-indexed)
        // Ranks go from i+1 to j (inclusive), average = (i+1 + j) / 2
        let avg_rank = (i + 1 + j) as f64 / 2.0;

        // Add to rank sum if positive
        for &idx in indices.iter().take(j).skip(i) {
            if labels[idx] > 0.5 {
                rank_sum_pos += avg_rank;
            }
        }

        i = j;
    }

    // Wilcoxon-Mann-Whitney formula
    // Note: we sorted descending, so higher rank = lower prediction
    // We need to flip: use (n+1 - rank) or equivalently adjust the formula
    // AUC = (sum_pos_ranks - n_pos*(n_pos+1)/2) / (n_pos * n_neg)
    // But since we sorted descending, positives should have LOW ranks for good AUC
    // So we use: AUC = 1 - (sum_pos_ranks - n_pos*(n_pos+1)/2) / (n_pos * n_neg)
    // Which simplifies to: (n_pos*n_neg + n_pos*(n_pos+1)/2 - sum_pos_ranks) / (n_pos*n_neg)

    let n_pos_f = n_pos as f64;
    let n_neg_f = n_neg as f64;

    // Since we sorted descending, convert to ascending rank interpretation
    let sum_ascending_ranks = n_pos_f * (n as f64 + 1.0) - rank_sum_pos;

    (sum_ascending_ranks - n_pos_f * (n_pos_f + 1.0) / 2.0) / (n_pos_f * n_neg_f)
}

/// Compute weighted AUC using sorting-based algorithm.
fn compute_auc_weighted(predictions: &[f32], labels: &[f32], weights: &[f32]) -> f64 {
    let n = predictions.len();

    // Create indices sorted by prediction (ascending - lowest first)
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        predictions[a]
            .partial_cmp(&predictions[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Compute weighted sums of positives and negatives
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
        return 0.5; // Undefined, return random
    }

    // Compute weighted concordance by scanning from low to high prediction.
    // For each positive, we count how many negatives have LOWER scores (already seen).
    // This is the standard approach: positive P and negative N are concordant if P's score > N's score.

    let mut weighted_concordant = 0.0f64;
    let mut cumulative_neg_weight = 0.0f64;
    let mut i = 0;

    while i < n {
        // Find all samples with the same prediction (ties)
        let mut j = i + 1;
        while j < n && (predictions[indices[i]] - predictions[indices[j]]).abs() < 1e-10 {
            j += 1;
        }

        // For this group of tied samples, compute contribution
        let mut group_pos_weight = 0.0f64;
        let mut group_neg_weight = 0.0f64;

        for &idx in indices.iter().take(j).skip(i) {
            if labels[idx] > 0.5 {
                group_pos_weight += weights[idx] as f64;
            } else {
                group_neg_weight += weights[idx] as f64;
            }
        }

        // Positives in this group are concordant with all negatives that came before
        // (lower prediction), plus half the weight of tied negatives (tie = 0.5 concordance).
        weighted_concordant +=
            group_pos_weight * (cumulative_neg_weight + 0.5 * group_neg_weight);

        // Update cumulative negative weight after processing this group
        cumulative_neg_weight += group_neg_weight;

        i = j;
    }

    // Normalize by total possible weighted pairs
    weighted_concordant / (sum_pos * sum_neg)
}

// =============================================================================
// Multiclass LogLoss
// =============================================================================

/// Multiclass cross-entropy: -mean(sum_k y_k * log(p_k))
///
/// Lower is better. Used for multiclass classification.
/// Expects predictions to be probabilities with shape `(n_rows, n_classes)`
/// in row-major order, and labels to be class indices in `0..n_classes`.
///
/// # Weighted Computation
///
/// When weights are provided, computes weighted multiclass log loss:
/// ```text
/// sum(w * -log(p_true_class)) / sum(w)
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct MulticlassLogLoss;

impl MetricFn for MulticlassLogLoss {
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

        let labels = &targets[..n_rows];

        let n_samples = labels.len();
        debug_assert_eq!(predictions.len(), n_samples * n_outputs);

        let eps = 1e-15f64; // Clip to avoid log(0)

        if weights.is_empty() {
            let total_loss: f64 = labels
                .iter()
                .enumerate()
                .map(|(row, &label)| {
                    let class_idx = label.round() as usize;
                    debug_assert!(class_idx < n_outputs, "label out of bounds");

                    // Column-major: index = class * n_rows + row
                    let prob = predictions[class_idx * n_rows + row] as f64;
                    let prob = prob.clamp(eps, 1.0 - eps);
                    -prob.ln()
                })
                .sum();

            total_loss / n_samples as f64
        } else {
            debug_assert_eq!(weights.len(), n_samples);

            let (weighted_loss, weight_sum) = labels
                .iter()
                .enumerate()
                .fold((0.0f64, 0.0f64), |(acc_loss, acc_w), (row, &label)| {
                    let class_idx = label.round() as usize;
                    debug_assert!(class_idx < n_outputs, "label out of bounds");

                    // Column-major: index = class * n_rows + row
                    let prob = predictions[class_idx * n_rows + row] as f64;
                    let prob = prob.clamp(eps, 1.0 - eps);
                    let loss = -prob.ln();
                    let wt = weights[row] as f64;

                    (acc_loss + wt * loss, acc_w + wt)
                });

            if weight_sum == 0.0 {
                return 0.0;
            }

            weighted_loss / weight_sum
        }
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

    // =========================================================================
    // LogLoss tests
    // =========================================================================

    #[test]
    fn logloss_perfect() {
        // Perfect predictions: p=1 for y=1, p=0 for y=0
        let preds = vec![0.9999, 0.0001];
        let labels = vec![1.0, 0.0];
        let ll = LogLoss.compute(2, 1, &preds, &labels, &[]);
        assert!(ll < 0.01); // Should be very small
    }

    #[test]
    fn logloss_worst() {
        // Worst predictions: p=0 for y=1, p=1 for y=0
        let preds = vec![0.0001, 0.9999];
        let labels = vec![1.0, 0.0];
        let ll = LogLoss.compute(2, 1, &preds, &labels, &[]);
        assert!(ll > 5.0); // Should be large
    }

    #[test]
    fn logloss_random() {
        // Random predictions (p=0.5)
        let preds = vec![0.5, 0.5];
        let labels = vec![1.0, 0.0];
        let ll = LogLoss.compute(2, 1, &preds, &labels, &[]);
        // -log(0.5) ≈ 0.693
        assert!((ll - 0.693).abs() < 0.01);
    }

    #[test]
    fn weighted_logloss_emphasizes_high_weight_samples() {
        // Sample 0: p=0.9, y=1 → loss = -log(0.9) ≈ 0.105
        // Sample 1: p=0.1, y=1 → loss = -log(0.1) ≈ 2.303
        let preds = vec![0.9, 0.1];
        let labels = vec![1.0, 1.0];

        // Unweighted: mean of two losses
        let unweighted = LogLoss.compute(2, 1, &preds, &labels, &[]);

        // With high weight on sample 0 (good prediction): lower loss
        let weights_good = vec![10.0, 1.0];
        let weighted_good = LogLoss.compute(2, 1, &preds, &labels, &weights_good);

        // With high weight on sample 1 (bad prediction): higher loss
        let weights_bad = vec![1.0, 10.0];
        let weighted_bad = LogLoss.compute(2, 1, &preds, &labels, &weights_bad);

        assert!(weighted_good < unweighted);
        assert!(weighted_bad > unweighted);
    }

    // =========================================================================
    // Accuracy tests
    // =========================================================================

    #[test]
    fn accuracy_perfect() {
        let preds = vec![0.9, 0.1, 0.8, 0.2];
        let labels = vec![1.0, 0.0, 1.0, 0.0];
        let acc = Accuracy::default().compute(4, 1, &preds, &labels, &[]);
        assert!((acc - 1.0).abs() < 1e-10);
    }

    #[test]
    fn accuracy_half() {
        let preds = vec![0.9, 0.9, 0.1, 0.1];
        let labels = vec![1.0, 0.0, 1.0, 0.0];
        let acc = Accuracy::default().compute(4, 1, &preds, &labels, &[]);
        assert!((acc - 0.5).abs() < 1e-10);
    }

    #[test]
    fn accuracy_custom_threshold() {
        let preds = vec![0.3, 0.3, 0.3, 0.3];
        let labels = vec![1.0, 0.0, 1.0, 0.0];

        // With threshold 0.5, all predictions are 0
        let acc_05 = Accuracy::default().compute(4, 1, &preds, &labels, &[]);
        assert!((acc_05 - 0.5).abs() < 1e-10);

        // With threshold 0.2, all predictions are 1
        let acc_02 = Accuracy::with_threshold(0.2).compute(4, 1, &preds, &labels, &[]);
        assert!((acc_02 - 0.5).abs() < 1e-10);
    }

    #[test]
    fn weighted_accuracy_emphasizes_correct_samples() {
        // Sample 0: correct, sample 1: incorrect
        let preds = vec![0.9, 0.9]; // Both predict class 1
        let labels = vec![1.0, 0.0]; // sample 0 correct, sample 1 wrong

        // Unweighted: 50% accuracy
        let unweighted = Accuracy::default().compute(2, 1, &preds, &labels, &[]);
        assert!((unweighted - 0.5).abs() < 1e-10);

        // High weight on correct sample: higher accuracy
        let weights_correct = vec![10.0, 1.0];
        let weighted_correct = Accuracy::default().compute(2, 1, &preds, &labels, &weights_correct);
        assert!((weighted_correct - 10.0 / 11.0).abs() < 1e-10);

        // High weight on incorrect sample: lower accuracy
        let weights_incorrect = vec![1.0, 10.0];
        let weighted_incorrect = Accuracy::default().compute(2, 1, &preds, &labels, &weights_incorrect);
        assert!((weighted_incorrect - 1.0 / 11.0).abs() < 1e-10);
    }

    // =========================================================================
    // Multiclass Accuracy tests
    // =========================================================================

    #[test]
    fn multiclass_accuracy() {
        // 4 samples, 3 classes. Predictions are per-class scores in column-major order.
        // Intended predicted classes by argmax: [0, 1, 2, 1]
        // Row-major layout would be:
        //   sample 0: [3.0, 1.0, 0.0] -> class 0
        //   sample 1: [0.0, 3.0, 1.0] -> class 1
        //   sample 2: [1.0, 0.0, 3.0] -> class 2
        //   sample 3: [0.0, 3.0, 1.0] -> class 1
        // Column-major layout: all class 0 preds, then class 1, then class 2
        let preds = vec![
            3.0, 0.0, 1.0, 0.0, // class 0 for samples [0,1,2,3]
            1.0, 3.0, 0.0, 3.0, // class 1 for samples [0,1,2,3]
            0.0, 1.0, 3.0, 1.0, // class 2 for samples [0,1,2,3]
        ];
        let labels = vec![0.0, 1.0, 2.0, 0.0]; // True classes
        let acc = MulticlassAccuracy.compute(4, 3, &preds, &labels, &[]);
        assert!((acc - 0.75).abs() < 1e-10); // 3/4 correct
    }

    #[test]
    fn weighted_multiclass_accuracy() {
        // 4 samples, 3/4 correct unweighted
        // Intended predicted classes by argmax: [0, 1, 2, 1]
        // Column-major layout: all class 0 preds, then class 1, then class 2
        let preds = vec![
            3.0, 0.0, 1.0, 0.0, // class 0 for samples [0,1,2,3]
            1.0, 3.0, 0.0, 3.0, // class 1 for samples [0,1,2,3]
            0.0, 1.0, 3.0, 1.0, // class 2 for samples [0,1,2,3]
        ];
        let labels = vec![0.0, 1.0, 2.0, 0.0]; // True classes (last one wrong)

        let unweighted = MulticlassAccuracy.compute(4, 3, &preds, &labels, &[]);
        assert!((unweighted - 0.75).abs() < 1e-10);

        // High weight on the wrong sample: accuracy drops
        let weights = vec![1.0, 1.0, 1.0, 10.0];
        let weighted = MulticlassAccuracy.compute(4, 3, &preds, &labels, &weights);
        // (1+1+1+0) / (1+1+1+10) = 3/13
        assert!(
            (weighted - 3.0 / 13.0).abs() < 1e-10,
            "got {} expected {}",
            weighted,
            3.0 / 13.0
        );
    }

    // =========================================================================
    // AUC tests
    // =========================================================================

    #[test]
    fn auc_perfect() {
        // Perfect separation: all positives have higher scores
        let preds = vec![0.9, 0.8, 0.3, 0.2];
        let labels = vec![1.0, 1.0, 0.0, 0.0];
        let auc = Auc.compute(4, 1, &preds, &labels, &[]);
        assert!((auc - 1.0).abs() < 1e-10);
    }

    #[test]
    fn auc_random() {
        // Random: mixed ordering
        let preds = vec![0.5, 0.5, 0.5, 0.5];
        let labels = vec![1.0, 0.0, 1.0, 0.0];
        let auc = Auc.compute(4, 1, &preds, &labels, &[]);
        assert!((auc - 0.5).abs() < 1e-10);
    }

    #[test]
    fn auc_worst() {
        // Worst: all negatives have higher scores
        let preds = vec![0.2, 0.3, 0.8, 0.9];
        let labels = vec![1.0, 1.0, 0.0, 0.0];
        let auc = Auc.compute(4, 1, &preds, &labels, &[]);
        assert!(auc.abs() < 1e-10);
    }

    #[test]
    fn auc_matches_naive_implementation() {
        // Test that our optimized AUC matches the naive O(n²) implementation
        let preds = vec![0.1, 0.4, 0.35, 0.8, 0.7, 0.2, 0.5];
        let labels = vec![0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0];

        // Naive O(n²) implementation for verification
        let naive_auc = {
            let mut concordant = 0.0f64;
            let mut pairs = 0.0f64;
            for (_i, (&p_i, &l_i)) in preds.iter().zip(labels.iter()).enumerate() {
                if l_i < 0.5 {
                    continue; // Only consider positives in outer loop
                }
                for (&p_j, &l_j) in preds.iter().zip(labels.iter()) {
                    if l_j > 0.5 {
                        continue; // Only consider negatives in inner loop
                    }
                    pairs += 1.0;
                    if p_i > p_j {
                        concordant += 1.0;
                    } else if f32::abs(p_i - p_j) < 1e-10_f32 {
                        concordant += 0.5;
                    }
                }
            }
            concordant / pairs
        };

        let optimized_auc = Auc.compute(7, 1, &preds, &labels, &[]);

        assert!(
            (optimized_auc - naive_auc).abs() < 1e-10,
            "optimized {} != naive {}",
            optimized_auc,
            naive_auc
        );
    }

    #[test]
    fn weighted_auc_emphasizes_pairs() {
        // 2 positives, 2 negatives
        // Scores: pos=[0.8, 0.6], neg=[0.4, 0.2]
        // All pairs concordant → AUC = 1.0 unweighted
        let preds = vec![0.8, 0.6, 0.4, 0.2];
        let labels = vec![1.0, 1.0, 0.0, 0.0];

        let unweighted = Auc.compute(4, 1, &preds, &labels, &[]);
        assert!((unweighted - 1.0).abs() < 1e-10);

        // Equal weights should also give 1.0
        let weights = vec![1.0, 1.0, 1.0, 1.0];
        let weighted = Auc.compute(4, 1, &preds, &labels, &weights);
        assert!((weighted - 1.0).abs() < 1e-10);

        // Make one positive have 0 weight - pairs involving it don't contribute
        let weights_zero = vec![0.0, 1.0, 1.0, 1.0];
        let weighted_zero = Auc.compute(4, 1, &preds, &labels, &weights_zero);
        // Only pairs with pos[1]=0.6 count: (0.6,0.4) and (0.6,0.2) both concordant
        assert!((weighted_zero - 1.0).abs() < 1e-10);
    }

    // =========================================================================
    // Multiclass LogLoss tests
    // =========================================================================

    #[test]
    fn mlogloss_perfect() {
        // 3-class classification, predictions are probabilities (column-major layout)
        // Sample 0: true class 0, predictions [0.99, 0.005, 0.005]
        // Sample 1: true class 1, predictions [0.005, 0.99, 0.005]
        // Column-major: all class 0 preds, then class 1, then class 2
        let preds = vec![
            0.99, 0.005,  // class 0 for samples [0, 1]
            0.005, 0.99,  // class 1 for samples [0, 1]
            0.005, 0.005, // class 2 for samples [0, 1]
        ];
        let labels = vec![0.0, 1.0];
        let mlogloss = MulticlassLogLoss.compute(2, 3, &preds, &labels, &[]);
        assert!(mlogloss < 0.02); // Should be very small
    }

    #[test]
    fn mlogloss_uniform() {
        // Uniform predictions for 3 classes: -log(1/3) ≈ 1.099 (column-major layout)
        // Column-major: all class 0 preds, then class 1, then class 2
        let preds = vec![
            0.333, 0.333, 0.333, // class 0 for samples [0, 1, 2]
            0.333, 0.333, 0.333, // class 1 for samples [0, 1, 2]
            0.334, 0.334, 0.334, // class 2 for samples [0, 1, 2]
        ];
        let labels = vec![0.0, 1.0, 2.0];
        let mlogloss = MulticlassLogLoss.compute(3, 3, &preds, &labels, &[]);
        assert!((mlogloss - 1.099).abs() < 0.01);
    }

    #[test]
    fn weighted_mlogloss() {
        // 2 samples, 3 classes (column-major layout)
        // Sample 0: true class 0, pred=[0.9, 0.05, 0.05] → loss = -log(0.9)
        // Sample 1: true class 1, pred=[0.1, 0.8, 0.1] → loss = -log(0.8)
        // Column-major: all class 0 preds, then class 1, then class 2
        let preds = vec![
            0.9, 0.1,  // class 0 for samples [0, 1]
            0.05, 0.8, // class 1 for samples [0, 1]
            0.05, 0.1, // class 2 for samples [0, 1]
        ];
        let labels = vec![0.0, 1.0];

        let loss0 = -(0.9f64.ln());
        let loss1 = -(0.8f64.ln());

        // Unweighted: mean of two losses
        let unweighted = MulticlassLogLoss.compute(2, 3, &preds, &labels, &[]);
        let expected_unweighted = (loss0 + loss1) / 2.0;
        assert!(
            (unweighted - expected_unweighted).abs() < 1e-6,
            "got {} expected {}",
            unweighted,
            expected_unweighted
        );

        // Weighted: emphasize sample 0 (lower loss)
        let weights = vec![10.0, 1.0];
        let weighted = MulticlassLogLoss.compute(2, 3, &preds, &labels, &weights);
        let expected_weighted = (10.0 * loss0 + 1.0 * loss1) / 11.0;
        assert!(
            (weighted - expected_weighted).abs() < 1e-6,
            "got {} expected {}",
            weighted,
            expected_weighted
        );
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

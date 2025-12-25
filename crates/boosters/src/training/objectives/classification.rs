//! Classification objective functions.

use ndarray::{ArrayView2, ArrayViewMut2};

use super::{ObjectiveFn, TargetSchema, TaskKind};
use crate::data::TargetsView;
use crate::inference::PredictionKind;
use crate::training::GradsTuple;
use crate::data::WeightsView;

// =============================================================================
// Logistic Loss
// =============================================================================

/// Logistic loss (log loss / binary cross-entropy) for binary classification.
///
/// Expects labels in {0, 1} and outputs log-odds.
/// - Loss: `-y*log(σ(pred)) - (1-y)*log(1-σ(pred))` where σ is sigmoid
/// - Gradient: `σ(pred) - y`
/// - Hessian: `σ(pred) * (1 - σ(pred))`
#[derive(Debug, Clone, Copy, Default)]
pub struct LogisticLoss;

impl LogisticLoss {
    #[inline]
    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }
}

impl ObjectiveFn for LogisticLoss {
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
                let p = Self::sigmoid(pred);
                gh_row[i].grad = w * (p - target);
                gh_row[i].hess = (w * p * (1.0 - p)).max(HESS_MIN);
            }
        }
    }

    fn compute_base_score(
        &self,
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
            .fold((0.0f64, 0.0f64), |(pos, total), (p, w)| (pos + p, total + w));

        let p = (pos_weight / total_weight).clamp(1e-7, 1.0 - 1e-7);
        vec![(p / (1.0 - p)).ln() as f32]
    }

    fn name(&self) -> &'static str {
        "logistic"
    }

    fn task_kind(&self) -> TaskKind {
        TaskKind::BinaryClassification
    }

    fn target_schema(&self) -> TargetSchema {
        TargetSchema::Binary01
    }

    fn transform_predictions_inplace(&self, mut predictions: ArrayViewMut2<f32>) -> PredictionKind {
        for x in predictions.iter_mut() {
            *x = Self::sigmoid(*x);
        }
        PredictionKind::Probability
    }
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
#[derive(Debug, Clone, Copy, Default)]
pub struct HingeLoss;

impl ObjectiveFn for HingeLoss {
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

    fn compute_base_score(
        &self,
        _targets: TargetsView<'_>,
        _weights: WeightsView<'_>,
    ) -> Vec<f32> {
        vec![0.0]
    }

    fn name(&self) -> &'static str {
        "hinge"
    }

    fn task_kind(&self) -> TaskKind {
        TaskKind::BinaryClassification
    }

    fn target_schema(&self) -> TargetSchema {
        TargetSchema::BinarySigned
    }

    fn transform_predictions_inplace(&self, _predictions: ArrayViewMut2<f32>) -> PredictionKind {
        PredictionKind::Margin
    }
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
#[derive(Debug, Clone, Copy)]
pub struct SoftmaxLoss {
    pub n_classes: usize,
}

impl Default for SoftmaxLoss {
    fn default() -> Self {
        Self { n_classes: 2 }
    }
}

impl SoftmaxLoss {
    pub fn new(n_classes: usize) -> Self {
        debug_assert!(n_classes >= 2, "n_classes must be >= 2");
        Self { n_classes }
    }
}

impl ObjectiveFn for SoftmaxLoss {
    fn n_outputs(&self) -> usize {
        self.n_classes
    }

    fn compute_gradients_into(
        &self,
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

        // Process sample by sample (need to compute softmax across classes)
        // Use column views to avoid repeated double-indexing
        for (i, (&target, w)) in targets
            .iter()
            .zip(weights.iter(n_rows))
            .enumerate()
        {
            let label = target as usize;
            debug_assert!(label < n_outputs, "label {} >= n_classes {}", label, n_outputs);

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

    fn compute_base_score(
        &self,
        targets: TargetsView<'_>,
        weights: WeightsView<'_>,
    ) -> Vec<f32> {
        let targets = targets.output(0);
        let n_rows = targets.len();
        let n_outputs = self.n_classes;

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

    fn name(&self) -> &'static str {
        "softmax"
    }

    fn task_kind(&self) -> TaskKind {
        TaskKind::MulticlassClassification { n_classes: self.n_classes }
    }

    fn target_schema(&self) -> TargetSchema {
        TargetSchema::MulticlassIndex
    }

    fn transform_predictions_inplace(&self, mut predictions: ArrayViewMut2<f32>) -> PredictionKind {
        let (n_outputs, n_samples) = predictions.dim();
        if n_outputs <= 1 {
            return PredictionKind::Probability;
        }

        // Apply softmax column-by-column (each column is a sample)
        // Using column_mut for better performance than double-indexing
        for sample_idx in 0..n_samples {
            let mut col = predictions.column_mut(sample_idx);
            
            // Find max for numerical stability
            let max_val = col.iter().copied().fold(f32::NEG_INFINITY, f32::max);

            // Compute exp(x - max) and sum
            let mut sum = 0.0f32;
            for x in col.iter_mut() {
                let v = (*x - max_val).exp();
                *x = v;
                sum += v;
            }

            // Normalize
            if sum > 0.0 {
                let inv_sum = 1.0 / sum;
                for x in col.iter_mut() {
                    *x *= inv_sum;
                }
            }
        }
        PredictionKind::Probability
    }
}

// =============================================================================
// LambdaRank
// =============================================================================

/// LambdaRank objective for learning to rank.
///
/// Implements the LambdaMART algorithm which optimizes for NDCG-like ranking metrics.
/// Each query group has multiple documents with relevance labels.
///
/// # Data Format
///
/// - `predictions`: Raw scores for all documents
/// - `targets`: Relevance labels (e.g., 0, 1, 2, 3 for bad to perfect)
/// - `query_groups`: Indices marking the start of each query group
#[derive(Debug, Clone)]
pub struct LambdaRankLoss {
    /// Query group boundaries (indices into the data).
    pub query_groups: Vec<usize>,
    /// Sigma parameter for sigmoid (default: 1.0).
    pub sigma: f32,
}

impl LambdaRankLoss {
    pub fn new(query_groups: Vec<usize>) -> Self {
        debug_assert!(query_groups.len() >= 2, "query_groups must have at least 2 elements");
        Self { query_groups, sigma: 1.0 }
    }

    pub fn with_sigma(mut self, sigma: f32) -> Self {
        self.sigma = sigma;
        self
    }

    #[inline]
    fn gain(label: f32) -> f64 {
        (2.0f64).powf(label as f64) - 1.0
    }

    #[inline]
    fn discount(pos: usize) -> f64 {
        1.0 / (2.0 + pos as f64).log2()
    }

    fn compute_idcg(labels: &[f32]) -> f64 {
        if labels.is_empty() {
            return 0.0;
        }

        let mut sorted_labels: Vec<f32> = labels.to_vec();
        sorted_labels.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        sorted_labels
            .iter()
            .enumerate()
            .map(|(pos, &label)| Self::gain(label) * Self::discount(pos))
            .sum()
    }
}

impl ObjectiveFn for LambdaRankLoss {
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
        let (_, n_rows) = predictions.dim();
        let targets = targets.output(0);
        debug_assert_eq!(targets.len(), n_rows);
        debug_assert_eq!(grad_hess.dim(), (1, n_rows));

        let preds_row = predictions.row(0);
        let preds_slice = preds_row.as_slice().expect("predictions row should be contiguous");
        let targets_slice = targets.as_slice().expect("targets should be contiguous");

        // Initialize gradients and hessians to zero
        let gh_row = grad_hess.row_mut(0);
        let gh_slice = gh_row.into_slice().expect("grad_hess row should be contiguous");
        for gh in gh_slice.iter_mut() {
            *gh = GradsTuple::default();
        }

        let sigma = self.sigma as f64;

        // Process each query group
        for q in 0..(self.query_groups.len() - 1) {
            let start = self.query_groups[q];
            let end = self.query_groups[q + 1].min(n_rows);

            if end <= start + 1 {
                continue;
            }

            let group_len = end - start;
            let labels = &targets_slice[start..end];
            let preds = &preds_slice[start..end];

            let idcg = Self::compute_idcg(labels);
            if idcg <= 0.0 {
                continue;
            }

            // Sort by prediction to get current ranking
            let mut indices: Vec<usize> = (0..group_len).collect();
            indices.sort_by(|&a, &b| {
                preds[b].partial_cmp(&preds[a]).unwrap_or(std::cmp::Ordering::Equal)
            });

            // Get weights slice for this group
            let group_weights: Vec<f32> = weights.iter(n_rows)
                .skip(start)
                .take(group_len)
                .collect();

            // For each pair (i, j) where label[i] > label[j]
            for (pos_i, &idx_i) in indices.iter().enumerate() {
                for (pos_j, &idx_j) in indices.iter().enumerate() {
                    if pos_i == pos_j {
                        continue;
                    }

                    let label_i = labels[idx_i];
                    let label_j = labels[idx_j];

                    if label_i <= label_j {
                        continue;
                    }

                    let s_ij = (preds[idx_i] - preds[idx_j]) as f64;
                    let sigmoid = 1.0 / (1.0 + (-sigma * s_ij).exp());

                    let gain_i = Self::gain(label_i);
                    let gain_j = Self::gain(label_j);
                    let disc_i = Self::discount(pos_i);
                    let disc_j = Self::discount(pos_j);

                    let delta_ndcg = ((gain_i - gain_j) * (disc_i - disc_j) / idcg).abs();
                    let lambda = -sigma * (1.0 - sigmoid) * delta_ndcg;

                    let w = group_weights[idx_i] as f64;

                    let gi = start + idx_i;
                    let gj = start + idx_j;

                    // Re-borrow to avoid double-mutable-borrow issues
                    grad_hess[[0, gi]].grad += (w * lambda) as f32;
                    grad_hess[[0, gj]].grad -= (w * lambda) as f32;

                    let hess_val = (w * sigma * sigma * sigmoid * (1.0 - sigmoid) * delta_ndcg).max(1e-6);
                    grad_hess[[0, gi]].hess += hess_val as f32;
                    grad_hess[[0, gj]].hess += hess_val as f32;
                }
            }
        }
    }

    fn compute_base_score(
        &self,
        _targets: TargetsView<'_>,
        _weights: WeightsView<'_>,
    ) -> Vec<f32> {
        vec![0.0]
    }

    fn name(&self) -> &'static str {
        "lambdarank"
    }

    fn task_kind(&self) -> TaskKind {
        TaskKind::Ranking
    }

    fn target_schema(&self) -> TargetSchema {
        TargetSchema::Continuous
    }

    fn transform_predictions_inplace(&self, _predictions: ArrayViewMut2<f32>) -> PredictionKind {
        PredictionKind::RankScore
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
    use crate::data::WeightsView;
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
        let obj = LogisticLoss;
        let preds = make_preds(1, 1, &[0.0]);
        let targets = make_targets(&[1.0]);
        let mut gh = make_grad_hess(1, 1);

        obj.compute_gradients_into(preds.view(), TargetsView::new(targets.view()), WeightsView::None, gh.view_mut());

        // sigmoid(0) = 0.5, grad = 0.5 - 1 = -0.5
        assert_abs_diff_eq!(gh[[0, 0]].grad, -0.5, epsilon = DEFAULT_TOLERANCE);
        // hess = 0.5 * 0.5 = 0.25
        assert_abs_diff_eq!(gh[[0, 0]].hess, 0.25, epsilon = DEFAULT_TOLERANCE);
    }

    #[test]
    fn logistic_base_score() {
        let obj = LogisticLoss;
        let targets = make_targets(&[0.0, 0.0, 1.0, 1.0]);

        let output = obj.compute_base_score(TargetsView::new(targets.view()), WeightsView::None);

        // 50% positive: log-odds = log(0.5/0.5) = 0
        assert_eq!(output.len(), 1);
        assert_abs_diff_eq!(output[0], 0.0, epsilon = DEFAULT_TOLERANCE);
    }

    #[test]
    fn logistic_weighted() {
        let obj = LogisticLoss;
        let preds = make_preds(1, 2, &[0.0, 0.0]);
        let targets = make_targets(&[1.0, 0.0]);
        let weights = ndarray::array![2.0f32, 0.5];
        let mut gh = make_grad_hess(1, 2);

        obj.compute_gradients_into(preds.view(), TargetsView::new(targets.view()), WeightsView::from_array(weights.view()), gh.view_mut());

        // sigmoid(0) = 0.5
        // grad[0] = 2.0 * (0.5 - 1) = -1.0
        assert_abs_diff_eq!(gh[[0, 0]].grad, -1.0, epsilon = DEFAULT_TOLERANCE);
        // grad[1] = 0.5 * (0.5 - 0) = 0.25
        assert_abs_diff_eq!(gh[[0, 1]].grad, 0.25, epsilon = DEFAULT_TOLERANCE);
    }

    #[test]
    fn hinge_loss_margin() {
        let obj = HingeLoss;

        // Correctly classified with margin
        let preds = make_preds(1, 1, &[2.0]);
        let targets = make_targets(&[1.0]);
        let mut gh = make_grad_hess(1, 1);

        obj.compute_gradients_into(preds.view(), TargetsView::new(targets.view()), WeightsView::None, gh.view_mut());
        // margin = 1 * 2 = 2 >= 1, so grad = 0
        assert_abs_diff_eq!(gh[[0, 0]].grad, 0.0, epsilon = DEFAULT_TOLERANCE);

        // Misclassified
        let preds = make_preds(1, 1, &[-0.5]);
        obj.compute_gradients_into(preds.view(), TargetsView::new(targets.view()), WeightsView::None, gh.view_mut());
        // margin = 1 * -0.5 < 1, so grad = -y = -1
        assert_abs_diff_eq!(gh[[0, 0]].grad, -1.0, epsilon = DEFAULT_TOLERANCE);
    }

    #[test]
    fn softmax_gradient() {
        let obj = SoftmaxLoss::new(3);
        // 3 classes, 2 samples
        let preds = make_preds(3, 2, &[
            1.0, 0.0,  // class 0
            0.0, 1.0,  // class 1
            0.0, 0.0,  // class 2
        ]);
        // Class indices: sample 0 -> class 0, sample 1 -> class 1
        let targets = make_targets(&[0.0, 1.0]);
        let mut gh = make_grad_hess(3, 2);

        obj.compute_gradients_into(preds.view(), TargetsView::new(targets.view()), WeightsView::None, gh.view_mut());

        // For sample 0 (label=0): grad for class 0 should be negative (correct class)
        assert!(gh[[0, 0]].grad < 0.0);
        // Wrong classes should have positive gradient
        assert!(gh[[1, 0]].grad > 0.0);
    }

    #[test]
    fn softmax_base_score() {
        let obj = SoftmaxLoss::new(3);
        // Class indices
        let targets = make_targets(&[0.0, 0.0, 1.0, 2.0]);

        let outputs = obj.compute_base_score(TargetsView::new(targets.view()), WeightsView::None);

        // Class 0: 2/4, Class 1: 1/4, Class 2: 1/4
        assert_eq!(outputs.len(), 3);
        assert!(outputs[0] > outputs[1]); // Class 0 more frequent
        assert_abs_diff_eq!(outputs[1], outputs[2], epsilon = DEFAULT_TOLERANCE);
    }

    #[test]
    fn lambdarank_basic() {
        let query_groups = vec![0, 3];
        let obj = LambdaRankLoss::new(query_groups);

        let preds = make_preds(1, 3, &[2.0, 0.0, 1.0]);
        let targets = make_targets(&[2.0, 0.0, 1.0]);
        let mut gh = make_grad_hess(1, 3);

        obj.compute_gradients_into(preds.view(), TargetsView::new(targets.view()), WeightsView::None, gh.view_mut());

        // All hessians should be positive
        assert!(gh[[0, 0]].hess > 0.0);
        assert!(gh[[0, 1]].hess > 0.0);
        assert!(gh[[0, 2]].hess > 0.0);
    }

    #[test]
    fn lambdarank_two_queries() {
        let query_groups = vec![0, 2, 4];
        let obj = LambdaRankLoss::new(query_groups);

        let preds = make_preds(1, 4, &[1.0, 0.0, 0.5, 0.5]);
        let targets = make_targets(&[1.0, 0.0, 2.0, 1.0]);
        let mut gh = make_grad_hess(1, 4);

        obj.compute_gradients_into(preds.view(), TargetsView::new(targets.view()), WeightsView::None, gh.view_mut());

        // Both queries should contribute gradients
        assert!(gh[[0, 0]].hess > 0.0);
        assert!(gh[[0, 2]].hess > 0.0 || gh[[0, 3]].hess > 0.0);
    }

    #[test]
    fn lambdarank_base_score() {
        let query_groups = vec![0, 3];
        let obj = LambdaRankLoss::new(query_groups);
        let targets = make_targets(&[0.0, 0.0, 0.0]);

        let output = obj.compute_base_score(TargetsView::new(targets.view()), WeightsView::None);
        assert_eq!(output.len(), 1);
        assert_abs_diff_eq!(output[0], 0.0, epsilon = DEFAULT_TOLERANCE);
    }
}

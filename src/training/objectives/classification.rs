//! Classification objective functions.

use super::{validate_objective_inputs, weight_iter, Objective, TargetSchema, TaskKind};
use crate::inference::common::{sigmoid_inplace, softmax_rows, PredictionKind, PredictionOutput};
use crate::training::metrics::MetricKind;

// =============================================================================
// Logistic Loss
// =============================================================================

/// Logistic loss (log loss / binary cross-entropy) for binary classification.
///
/// Expects labels in {0, 1} and outputs log-odds.
/// - Loss: `-y*log(σ(pred)) - (1-y)*log(1-σ(pred))` where σ is sigmoid
/// - Gradient: `σ(pred) - y`
/// - Hessian: `σ(pred) * (1 - σ(pred))`
///
/// # Multi-Output
///
/// Supports multi-label classification where each output is an independent
/// binary classification task with its own target.
#[derive(Debug, Clone, Copy, Default)]
pub struct LogisticLoss;

impl LogisticLoss {
    #[inline]
    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }
}

impl Objective for LogisticLoss {
    fn compute_gradients(
        &self,
        n_rows: usize,
        n_outputs: usize,
        predictions: &[f32],
        targets: &[f32],
        weights: &[f32],
        gradients: &mut [f32],
        hessians: &mut [f32],
    ) {
        validate_objective_inputs(
            n_rows,
            n_outputs,
            predictions.len(),
            gradients.len(),
            hessians.len(),
            weights,
        );
        debug_assert!(targets.len() >= n_outputs * n_rows);

        const HESS_MIN: f32 = 1e-6;

        for out_idx in 0..n_outputs {
            let offset = out_idx * n_rows;
            let pred_slice = &predictions[offset..offset + n_rows];
            let target_slice = &targets[offset..offset + n_rows];
            let grad_slice = &mut gradients[offset..offset + n_rows];
            let hess_slice = &mut hessians[offset..offset + n_rows];

            for (i, w) in weight_iter(weights, n_rows).enumerate() {
                let p = Self::sigmoid(pred_slice[i]);
                grad_slice[i] = w * (p - target_slice[i]);
                hess_slice[i] = (w * p * (1.0 - p)).max(HESS_MIN);
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

        for out_idx in 0..n_outputs {
            let offset = out_idx * n_rows;
            let target_slice = &targets[offset..offset + n_rows];

            let (pos_weight, total_weight) = target_slice
                .iter()
                .zip(weight_iter(weights, n_rows))
                .fold((0.0f64, 0.0f64), |(pos, total), (&t, w)| {
                    (pos + t as f64 * w as f64, total + w as f64)
                });

            // Convert to log-odds
            let p = (pos_weight / total_weight).clamp(1e-7, 1.0 - 1e-7);
            outputs[out_idx] = (p / (1.0 - p)).ln() as f32;
        }
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

    fn default_metric(&self) -> MetricKind {
        MetricKind::LogLoss
    }

    fn transform_prediction_inplace(&self, raw: &mut PredictionOutput) -> PredictionKind {
        sigmoid_inplace(raw.as_mut_slice());
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
///
/// # Multi-Output
///
/// Supports multi-label classification where each output is an independent
/// binary classification task with its own target.
#[derive(Debug, Clone, Copy, Default)]
pub struct HingeLoss;

impl Objective for HingeLoss {
    fn compute_gradients(
        &self,
        n_rows: usize,
        n_outputs: usize,
        predictions: &[f32],
        targets: &[f32],
        weights: &[f32],
        gradients: &mut [f32],
        hessians: &mut [f32],
    ) {
        validate_objective_inputs(
            n_rows,
            n_outputs,
            predictions.len(),
            gradients.len(),
            hessians.len(),
            weights,
        );
        debug_assert!(targets.len() >= n_outputs * n_rows);

        for out_idx in 0..n_outputs {
            let offset = out_idx * n_rows;
            let pred_slice = &predictions[offset..offset + n_rows];
            let target_slice = &targets[offset..offset + n_rows];
            let grad_slice = &mut gradients[offset..offset + n_rows];
            let hess_slice = &mut hessians[offset..offset + n_rows];

            for (i, w) in weight_iter(weights, n_rows).enumerate() {
                // Convert {0, 1} to {-1, +1}
                let y = if target_slice[i] > 0.5 { 1.0 } else { -1.0 };
                let margin = y * pred_slice[i];

                if margin < 1.0 {
                    grad_slice[i] = -w * y;
                } else {
                    grad_slice[i] = 0.0;
                }
                hess_slice[i] = w;
            }
        }
    }

    fn compute_base_score(
        &self,
        _n_rows: usize,
        n_outputs: usize,
        _targets: &[f32],
        _weights: &[f32],
        outputs: &mut [f32],
    ) {
        // Start at zero for hinge loss
        outputs[..n_outputs].fill(0.0);
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

    fn default_metric(&self) -> MetricKind {
        MetricKind::MarginAccuracy
    }

    fn transform_prediction_inplace(&self, _raw: &mut PredictionOutput) -> PredictionKind {
        // Hinge is naturally evaluated in margin space.
        PredictionKind::Margin
    }
}

// =============================================================================
// Softmax Loss
// =============================================================================

/// Softmax cross-entropy loss for multiclass classification.
///
/// Expects labels as class indices (0 to num_classes-1) stored as f32.
/// Predictions are K raw logits per sample in column-major order.
///
/// # Layout
///
/// - `n_outputs` = num_classes
/// - `predictions`: column-major `[num_classes * n_rows]`
/// - `targets`: class indices `[n_rows]` (single target column)
/// - `gradients/hessians`: column-major `[num_classes * n_rows]`
///
/// # Example
///
/// ```ignore
/// use booste_rs::training::SoftmaxLoss;
///
/// // 3-class classification
/// let obj = SoftmaxLoss::new(3);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct SoftmaxLoss {
    /// Number of classes.
    pub num_classes: usize,
}

impl Default for SoftmaxLoss {
    fn default() -> Self {
        Self { num_classes: 2 }
    }
}

impl SoftmaxLoss {
    /// Create a new softmax loss for the given number of classes.
    pub fn new(num_classes: usize) -> Self {
        debug_assert!(num_classes >= 2, "num_classes must be >= 2");
        Self { num_classes }
    }
}

impl Objective for SoftmaxLoss {
    fn n_outputs(&self) -> usize {
        self.num_classes
    }

    fn compute_gradients(
        &self,
        n_rows: usize,
        n_outputs: usize,
        predictions: &[f32],
        targets: &[f32],
        weights: &[f32],
        gradients: &mut [f32],
        hessians: &mut [f32],
    ) {
        validate_objective_inputs(
            n_rows,
            n_outputs,
            predictions.len(),
            gradients.len(),
            hessians.len(),
            weights,
        );
        debug_assert!(targets.len() >= n_rows);

        let k = n_outputs;
        const HESS_MIN: f32 = 1e-6;

        for (i, w) in weight_iter(weights, n_rows).enumerate() {
            let label = targets[i] as usize;
            debug_assert!(label < k, "label {} >= num_classes {}", label, k);

            // Compute softmax probabilities (numerically stable)
            let mut max_logit = f32::NEG_INFINITY;
            for c in 0..k {
                max_logit = max_logit.max(predictions[c * n_rows + i]);
            }

            let mut exp_sum = 0.0f32;
            for c in 0..k {
                exp_sum += (predictions[c * n_rows + i] - max_logit).exp();
            }

            // Compute gradients and hessians for each class
            for c in 0..k {
                let p = (predictions[c * n_rows + i] - max_logit).exp() / exp_sum;
                let target_indicator = if c == label { 1.0 } else { 0.0 };

                gradients[c * n_rows + i] = w * (p - target_indicator);
                hessians[c * n_rows + i] = (w * p * (1.0 - p)).max(HESS_MIN);
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
        debug_assert!(outputs.len() >= n_outputs);

        if n_rows == 0 {
            outputs[..n_outputs].fill(0.0);
            return;
        }

        // Count class frequencies
        let mut class_weights = vec![0.0f64; n_outputs];
        let mut total_weight = 0.0f64;

        for (i, w) in weight_iter(weights, n_rows).enumerate() {
            let label = targets[i] as usize;
            if label < n_outputs {
                class_weights[label] += w as f64;
            }
            total_weight += w as f64;
        }

        // Convert to log-probabilities
        for c in 0..n_outputs {
            let p = (class_weights[c] / total_weight).clamp(1e-7, 1.0 - 1e-7);
            outputs[c] = p.ln() as f32;
        }
    }

    fn name(&self) -> &'static str {
        "softmax"
    }

    fn task_kind(&self) -> TaskKind {
        TaskKind::MulticlassClassification
    }

    fn target_schema(&self) -> TargetSchema {
        TargetSchema::MulticlassIndex
    }

    fn default_metric(&self) -> MetricKind {
        MetricKind::MulticlassLogLoss
    }

    fn transform_prediction_inplace(&self, raw: &mut PredictionOutput) -> PredictionKind {
        softmax_rows(raw);
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
///
/// Documents within a group are compared pairwise, with gradients weighted by
/// the change in NDCG that swapping the pair would cause.
///
/// # Example
///
/// ```ignore
/// // 2 queries: query 0 has docs 0-2, query 1 has docs 3-5
/// let query_groups = vec![0, 3, 6];
/// let targets = vec![2.0, 0.0, 1.0, 3.0, 1.0, 0.0]; // relevance labels
/// let obj = LambdaRankLoss::new(query_groups);
/// ```
#[derive(Debug, Clone)]
pub struct LambdaRankLoss {
    /// Query group boundaries (indices into the data).
    /// query_groups[i] is the start index of query i.
    /// query_groups.len() - 1 is the number of queries.
    pub query_groups: Vec<usize>,
    /// Sigma parameter for sigmoid (default: 1.0).
    pub sigma: f32,
}

impl LambdaRankLoss {
    /// Create a new LambdaRank objective.
    ///
    /// # Arguments
    ///
    /// * `query_groups` - Query group boundaries. Must have at least 2 elements
    ///   (start and end). Each element is the start index of a query.
    pub fn new(query_groups: Vec<usize>) -> Self {
        debug_assert!(
            query_groups.len() >= 2,
            "query_groups must have at least 2 elements"
        );
        Self { query_groups, sigma: 1.0 }
    }

    /// Set the sigma parameter.
    pub fn with_sigma(mut self, sigma: f32) -> Self {
        self.sigma = sigma;
        self
    }

    /// Compute DCG gain for a relevance label.
    #[inline]
    fn gain(label: f32) -> f64 {
        (2.0f64).powf(label as f64) - 1.0
    }

    /// Compute position discount.
    #[inline]
    fn discount(pos: usize) -> f64 {
        1.0 / (2.0 + pos as f64).log2()
    }

    /// Compute ideal DCG for a query group.
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

impl Objective for LambdaRankLoss {
    fn compute_gradients(
        &self,
        n_rows: usize,
        _n_outputs: usize,
        predictions: &[f32],
        targets: &[f32],
        weights: &[f32],
        gradients: &mut [f32],
        hessians: &mut [f32],
    ) {
        debug_assert!(predictions.len() >= n_rows);
        debug_assert!(targets.len() >= n_rows);
        debug_assert!(gradients.len() >= n_rows);
        debug_assert!(hessians.len() >= n_rows);

        // Initialize gradients and hessians to zero
        gradients[..n_rows].fill(0.0);
        hessians[..n_rows].fill(0.0);

        let sigma = self.sigma as f64;

        // Process each query group
        for q in 0..(self.query_groups.len() - 1) {
            let start = self.query_groups[q];
            let end = self.query_groups[q + 1].min(n_rows);

            if end <= start + 1 {
                continue; // Need at least 2 docs to form pairs
            }

            // Get labels for this query
            let labels = &targets[start..end];
            let preds = &predictions[start..end];

            // Compute IDCG
            let idcg = Self::compute_idcg(labels);
            if idcg <= 0.0 {
                continue;
            }

            // Sort by prediction to get current ranking
            let mut indices: Vec<usize> = (0..labels.len()).collect();
            indices.sort_by(|&a, &b| {
                preds[b]
                    .partial_cmp(&preds[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // For each pair (i, j) where label[i] > label[j]
            for (pos_i, &idx_i) in indices.iter().enumerate() {
                for (pos_j, &idx_j) in indices.iter().enumerate() {
                    if pos_i == pos_j {
                        continue;
                    }

                    let label_i = labels[idx_i];
                    let label_j = labels[idx_j];

                    // Only consider pairs where i is more relevant
                    if label_i <= label_j {
                        continue;
                    }

                    // Score difference
                    let s_ij = (preds[idx_i] - preds[idx_j]) as f64;

                    // Sigmoid and gradient
                    let sigmoid = 1.0 / (1.0 + (-sigma * s_ij).exp());

                    // Compute |ΔNDCG| if we swapped positions
                    let gain_i = Self::gain(label_i);
                    let gain_j = Self::gain(label_j);
                    let disc_i = Self::discount(pos_i);
                    let disc_j = Self::discount(pos_j);

                    // Change in DCG from swapping
                    let delta_ndcg = ((gain_i - gain_j) * (disc_i - disc_j) / idcg).abs();

                    // Lambda gradient
                    let lambda = -sigma * (1.0 - sigmoid) * delta_ndcg;

                    // Weight
                    let w = if weights.is_empty() {
                        1.0
                    } else {
                        weights[start + idx_i] as f64
                    };

                    // Update gradients
                    gradients[start + idx_i] += (w * lambda) as f32;
                    gradients[start + idx_j] -= (w * lambda) as f32;

                    // Update hessians (second derivative approximation)
                    let hess_val =
                        (w * sigma * sigma * sigmoid * (1.0 - sigmoid) * delta_ndcg).max(1e-6);
                    hessians[start + idx_i] += hess_val as f32;
                    hessians[start + idx_j] += hess_val as f32;
                }
            }
        }
    }

    fn compute_base_score(
        &self,
        _n_rows: usize,
        n_outputs: usize,
        _targets: &[f32],
        _weights: &[f32],
        outputs: &mut [f32],
    ) {
        // Start at zero for ranking
        outputs[..n_outputs].fill(0.0);
    }

    fn name(&self) -> &'static str {
        "lambdarank"
    }

    fn task_kind(&self) -> TaskKind {
        TaskKind::Ranking
    }

    fn target_schema(&self) -> TargetSchema {
        // LambdaRank supports graded relevance labels.
        TargetSchema::Continuous
    }

    fn default_metric(&self) -> MetricKind {
        // TODO: add NDCG metric; AUC is a placeholder.
        MetricKind::Auc
    }

    fn transform_prediction_inplace(&self, _raw: &mut PredictionOutput) -> PredictionKind {
        // Ranking objectives typically expose a score.
        PredictionKind::RankScore
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logistic_gradient_at_zero() {
        let obj = LogisticLoss;
        let preds = [0.0f32];
        let targets = [1.0f32];
        let mut grads = [0.0f32];
        let mut hess = [0.0f32];

        obj.compute_gradients(1, 1, &preds, &targets, &[], &mut grads, &mut hess);

        // sigmoid(0) = 0.5, grad = 0.5 - 1 = -0.5
        assert!((grads[0] - -0.5).abs() < 1e-6);
        // hess = 0.5 * 0.5 = 0.25
        assert!((hess[0] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn logistic_multi_output() {
        let obj = LogisticLoss;
        // 2 rows, 2 outputs (multi-label)
        let preds = [0.0f32, 0.0, 0.0, 0.0]; // all zero logits
        let targets = [1.0f32, 0.0, 0.0, 1.0]; // out0=[1,0], out1=[0,1]
        let mut grads = [0.0f32; 4];
        let mut hess = [0.0f32; 4];

        obj.compute_gradients(2, 2, &preds, &targets, &[], &mut grads, &mut hess);

        // All sigmoid(0) = 0.5
        // out0: grads = [0.5-1, 0.5-0] = [-0.5, 0.5]
        assert!((grads[0] - -0.5).abs() < 1e-6);
        assert!((grads[1] - 0.5).abs() < 1e-6);
        // out1: grads = [0.5-0, 0.5-1] = [0.5, -0.5]
        assert!((grads[2] - 0.5).abs() < 1e-6);
        assert!((grads[3] - -0.5).abs() < 1e-6);
    }

    #[test]
    fn logistic_base_score() {
        let obj = LogisticLoss;
        let targets = [0.0f32, 0.0, 1.0, 1.0];
        let mut output = [0.0f32];

        obj.compute_base_score(4, 1, &targets, &[], &mut output);

        // 50% positive: log-odds = log(0.5/0.5) = 0
        assert!(output[0].abs() < 1e-6);
    }

    #[test]
    fn hinge_loss_margin() {
        let obj = HingeLoss;

        // Correctly classified with margin
        let preds = [2.0f32];
        let targets = [1.0f32];
        let mut grads = [0.0f32];
        let mut hess = [0.0f32];

        obj.compute_gradients(1, 1, &preds, &targets, &[], &mut grads, &mut hess);
        // margin = 1 * 2 = 2 >= 1, so grad = 0
        assert!(grads[0].abs() < 1e-6);

        // Misclassified
        let preds = [-0.5f32];
        obj.compute_gradients(1, 1, &preds, &targets, &[], &mut grads, &mut hess);
        // margin = 1 * -0.5 < 1, so grad = -y = -1
        assert!((grads[0] - -1.0).abs() < 1e-6);
    }

    #[test]
    fn softmax_gradient() {
        let obj = SoftmaxLoss::new(3);
        let n_rows = 2;
        // Predictions: 3 classes, 2 rows - column major
        let preds = [
            1.0f32, 0.0, // class 0: [1.0, 0.0]
            0.0, 1.0, // class 1: [0.0, 1.0]
            0.0, 0.0, // class 2: [0.0, 0.0]
        ];
        let targets = [0.0f32, 1.0]; // sample 0 -> class 0, sample 1 -> class 1
        let mut grads = [0.0f32; 6];
        let mut hess = [0.0f32; 6];

        obj.compute_gradients(n_rows, 3, &preds, &targets, &[], &mut grads, &mut hess);

        // For sample 0 (label=0):
        // softmax([1,0,0]) ≈ [0.576, 0.212, 0.212]
        // grad for class 0 = p - 1 ≈ -0.424
        assert!(grads[0] < 0.0); // Should be negative for correct class
        assert!(grads[2] > 0.0); // Should be positive for wrong class
    }

    #[test]
    fn softmax_base_score() {
        let obj = SoftmaxLoss::new(3);
        let targets = [0.0f32, 0.0, 1.0, 2.0];
        let mut outputs = [0.0f32; 3];

        obj.compute_base_score(4, 3, &targets, &[], &mut outputs);

        // Class 0: 2/4, Class 1: 1/4, Class 2: 1/4
        assert!(outputs[0] > outputs[1]); // Class 0 more frequent
        assert!((outputs[1] - outputs[2]).abs() < 1e-6); // Same frequency
    }

    #[test]
    fn logistic_weighted() {
        let obj = LogisticLoss;
        let preds = [0.0f32, 0.0];
        let targets = [1.0f32, 0.0];
        let weights = [2.0f32, 0.5];
        let mut grads = [0.0f32; 2];
        let mut hess = [0.0f32; 2];

        obj.compute_gradients(2, 1, &preds, &targets, &weights, &mut grads, &mut hess);

        // sigmoid(0) = 0.5
        // grad[0] = 2.0 * (0.5 - 1) = -1.0
        assert!((grads[0] - -1.0).abs() < 1e-6);
        // grad[1] = 0.5 * (0.5 - 0) = 0.25
        assert!((grads[1] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn lambdarank_basic() {
        // One query with 3 docs: relevance [2, 0, 1]
        let query_groups = vec![0, 3];
        let obj = LambdaRankLoss::new(query_groups);

        // Predictions: doc0 scored highest (correct since label=2)
        let preds = [2.0f32, 0.0, 1.0];
        let targets = [2.0f32, 0.0, 1.0]; // relevance labels
        let mut grads = [0.0f32; 3];
        let mut hess = [0.0f32; 3];

        obj.compute_gradients(3, 1, &preds, &targets, &[], &mut grads, &mut hess);

        // All gradients and hessians should be computed
        // Can't easily verify exact values, but should be non-trivial
        assert!(hess[0] > 0.0);
        assert!(hess[1] > 0.0);
        assert!(hess[2] > 0.0);
    }

    #[test]
    fn lambdarank_two_queries() {
        // Two queries: query0 has docs 0-1, query1 has docs 2-3
        let query_groups = vec![0, 2, 4];
        let obj = LambdaRankLoss::new(query_groups);

        let preds = [1.0f32, 0.0, 0.5, 0.5];
        let targets = [1.0f32, 0.0, 2.0, 1.0];
        let mut grads = [0.0f32; 4];
        let mut hess = [0.0f32; 4];

        obj.compute_gradients(4, 1, &preds, &targets, &[], &mut grads, &mut hess);

        // Both queries should contribute gradients
        assert!(hess[0] > 0.0);
        assert!(hess[2] > 0.0 || hess[3] > 0.0);
    }

    #[test]
    fn lambdarank_base_score() {
        let query_groups = vec![0, 3];
        let obj = LambdaRankLoss::new(query_groups);
        let mut output = [1.0f32];

        obj.compute_base_score(3, 1, &[], &[], &mut output);
        assert!((output[0] - 0.0).abs() < 1e-6);
    }
}

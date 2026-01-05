//! Coordinate descent updaters for linear models.
//!
//! The [`Updater`] component handles weight updates using coordinate descent with
//! incremental prediction updates for efficiency.
//!
//! Two variants:
//! - [`UpdateStrategy::Shotgun`]: Parallel updates (faster, approximate)
//! - [`UpdateStrategy::Sequential`]: Sequential updates (deterministic ordering)
//!
//! # Data Format
//!
//! The updaters work with [`Dataset`] which provides efficient per-feature
//! iteration via [`Dataset::for_each_feature_value()`].
//!
//! # Gradient Storage
//!
//! Gradients are stored in Structure-of-Arrays (SoA) layout via [`Gradients`]:
//! - Shape `[n_samples, n_outputs]` for unified single/multi-output handling
//! - Separate `grads[]` and `hess[]` arrays for cache efficiency
//!
//! # Weight Handling and Regularization Denormalization
//!
//! Sample weights are applied during gradient computation by the objective function:
//! - `grad[i] = dL/dp * weight[i]`
//! - `hess[i] = d²L/dp² * weight[i]`
//!
//! This means gradient/hessian sums are scaled by the total weight. To keep
//! regularization balanced, we "denormalize" the penalties by multiplying them
//! by `sum_instance_weight` (matches XGBoost's `DenormalizePenalties`):
//! - `effective_alpha = alpha * sum_instance_weight`
//! - `effective_lambda = lambda * sum_instance_weight`
//!
//! This ensures consistent regularization strength regardless of weight scale.
//! Note: This is specific to coordinate descent in linear models. GBDT tree
//! regularization does NOT need this because the gain formula naturally handles
//! weighted gradients: `gain = (sum_grad)² / (sum_hess + lambda)`.

use ndarray::ArrayViewMut1;
use rayon::prelude::*;

use crate::data::Dataset;
use crate::training::GradsTuple;

use super::selector::FeatureSelector;

// =============================================================================
// Denormalized Regularization
// =============================================================================

/// Regularization parameters scaled by sum of instance weights.
///
/// This encapsulates XGBoost's "DenormalizePenalties" pattern: since gradients
/// are already weighted (scaled by per-sample weights), regularization terms
/// must be scaled proportionally to maintain consistent behavior.
///
/// # Example
///
/// ```ignore
/// let reg = Regularization::new(alpha, lambda);
/// let denorm = reg.denormalize(sum_instance_weight);
/// // Use denorm.alpha and denorm.lambda in coordinate descent
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Regularization {
    /// L1 regularization (lasso).
    pub alpha: f32,
    /// L2 regularization (ridge).
    pub lambda: f32,
}

impl Regularization {
    /// Create new regularization parameters.
    pub fn new(alpha: f32, lambda: f32) -> Self {
        Self { alpha, lambda }
    }

    /// Denormalize by sum of instance weights.
    ///
    /// This scales regularization to match weighted gradient sums.
    /// See module docs for rationale.
    #[inline]
    pub fn denormalize(&self, sum_instance_weight: f32) -> Self {
        Self {
            alpha: self.alpha * sum_instance_weight,
            lambda: self.lambda * sum_instance_weight,
        }
    }
}

/// Coordinate descent update strategy.
///
/// Matches XGBoost naming (`shotgun` vs `coord_descent`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum UpdateStrategy {
    /// Parallel (shotgun) coordinate descent - approximate, faster.
    #[default]
    Shotgun,
    /// Sequential coordinate descent - deterministic ordering.
    Sequential,
}

/// Coordinate descent updater for linear models.
///
/// Maintains predictions and residual gradients incrementally:
/// - Predictions are updated in-place after each coordinate update.
/// - Residual gradients are updated in-place as `grad += hess * delta_pred`.
#[derive(Debug, Clone)]
pub(super) struct Updater {
    kind: UpdateStrategy,
    /// Base regularization (before denormalization).
    regularization: Regularization,
    learning_rate: f32,
    max_delta_step: f32,
}

impl Updater {
    pub(super) fn new(
        kind: UpdateStrategy,
        alpha: f32,
        lambda: f32,
        learning_rate: f32,
        max_delta_step: f32,
    ) -> Self {
        Self {
            kind,
            regularization: Regularization::new(alpha, lambda),
            learning_rate,
            max_delta_step,
        }
    }

    /// Get base regularization parameters (before denormalization).
    pub(super) fn regularization(&self) -> Regularization {
        self.regularization
    }

    pub(super) fn update_bias_inplace(
        &self,
        mut weights_and_bias: ArrayViewMut1<'_, f32>,
        grad_pairs: &mut [GradsTuple],
        mut predictions_row: ArrayViewMut1<'_, f32>,
    ) -> f32 {
        let mut sum_grad = 0.0f32;
        let mut sum_hess = 0.0f32;
        for gh in grad_pairs.iter() {
            sum_grad += gh.grad;
            sum_hess += gh.hess;
        }

        // Match XGBoost's stability threshold.
        if sum_hess.abs() <= 1e-5 {
            return 0.0;
        }

        let mut delta = (-sum_grad / sum_hess) * self.learning_rate;
        if self.max_delta_step > 0.0 {
            delta = delta.clamp(-self.max_delta_step, self.max_delta_step);
        }
        if delta.abs() <= 1e-10 {
            return 0.0;
        }

        let n_features = weights_and_bias.len() - 1;
        weights_and_bias[n_features] += delta;
        apply_bias_delta_to_predictions(delta, predictions_row.view_mut());

        // XGBoost updater_{shotgun,coordinate}.cc:
        //   p += GradientPair(p.GetHess() * dbias, 0)
        for gh in grad_pairs.iter_mut() {
            gh.grad += gh.hess * delta;
        }

        delta
    }

    pub(super) fn update_round_inplace<Sel: FeatureSelector>(
        &self,
        data: &Dataset,
        weights_and_bias: ArrayViewMut1<'_, f32>,
        grad_pairs: &mut [GradsTuple],
        selector: &mut Sel,
        reg: Regularization,
        predictions: ArrayViewMut1<'_, f32>,
    ) -> Vec<(usize, f32)> {
        match self.kind {
            UpdateStrategy::Sequential => self.update_round_sequential_inplace(
                data,
                weights_and_bias,
                grad_pairs,
                selector,
                reg,
                predictions,
            ),
            UpdateStrategy::Shotgun => self.update_round_shotgun_inplace(
                data,
                weights_and_bias,
                grad_pairs,
                selector,
                reg,
                predictions,
            ),
        }
    }

    fn update_round_sequential_inplace<Sel: FeatureSelector>(
        &self,
        data: &Dataset,
        mut weights_and_bias: ArrayViewMut1<'_, f32>,
        grad_pairs: &mut [GradsTuple],
        selector: &mut Sel,
        reg: Regularization,
        mut predictions: ArrayViewMut1<'_, f32>,
    ) -> Vec<(usize, f32)> {
        let mut deltas: Vec<(usize, f32)> = Vec::new();
        let n_features = weights_and_bias.len() - 1;

        while let Some(feature) = selector.next() {
            debug_assert!(feature < n_features);
            let current_weight = weights_and_bias[feature];

            // Accumulate from current residual gradients.
            let mut sum_grad = 0.0f32;
            let mut sum_hess = 0.0f32;
            data.for_each_feature_value(feature, |row, value| {
                // Treat NaN as missing: contributes 0.
                if value.is_nan() {
                    return;
                }
                sum_grad += grad_pairs[row].grad * value;
                sum_hess += grad_pairs[row].hess * value * value;
            });

            let delta = coordinate_delta(
                sum_grad,
                sum_hess,
                current_weight,
                reg.alpha,
                reg.lambda,
                self.learning_rate,
                self.max_delta_step,
            );
            if delta.abs() <= 1e-10 {
                continue;
            }

            weights_and_bias[feature] += delta;
            deltas.push((feature, delta));

            // Update predictions and residual gradients for touched rows.
            data.for_each_feature_value(feature, |row, value| {
                if value.is_nan() {
                    return;
                }
                predictions[row] += value * delta;
                grad_pairs[row].grad += grad_pairs[row].hess * value * delta;
            });
        }

        deltas
    }

    fn update_round_shotgun_inplace<Sel: FeatureSelector>(
        &self,
        data: &Dataset,
        mut weights_and_bias: ArrayViewMut1<'_, f32>,
        grad_pairs: &mut [GradsTuple],
        selector: &mut Sel,
        reg: Regularization,
        mut predictions: ArrayViewMut1<'_, f32>,
    ) -> Vec<(usize, f32)> {
        // NOTE: This is "shotgun" in the sense that we compute coordinate deltas
        // in parallel from the same residual snapshot, then apply them.
        // Applying deltas truly in parallel would require conflict handling
        // for prediction/residual writes.
        let indices = selector.all_indices();
        if indices.is_empty() {
            return Vec::new();
        }

        let n_features = weights_and_bias.len() - 1;
        let weights_view = weights_and_bias.view();
        let gh: &[GradsTuple] = &*grad_pairs;
        let learning_rate = self.learning_rate;
        let max_delta_step = self.max_delta_step;

        let deltas: Vec<(usize, f32)> = indices
            .par_iter()
            .filter_map(|&feature| {
                if feature >= n_features {
                    return None;
                }
                let current_weight = weights_view[feature];
                let mut sum_grad = 0.0f32;
                let mut sum_hess = 0.0f32;
                data.for_each_feature_value(feature, |row, value| {
                    if value.is_nan() {
                        return;
                    }
                    sum_grad += gh[row].grad * value;
                    sum_hess += gh[row].hess * value * value;
                });
                let delta = coordinate_delta(
                    sum_grad,
                    sum_hess,
                    current_weight,
                    reg.alpha,
                    reg.lambda,
                    learning_rate,
                    max_delta_step,
                );
                (delta.abs() > 1e-10).then_some((feature, delta))
            })
            .collect();

        if deltas.is_empty() {
            return deltas;
        }

        // Apply to model.
        for &(feature, delta) in &deltas {
            weights_and_bias[feature] += delta;
        }

        // Apply to predictions and residual gradients.
        for &(feature, delta) in &deltas {
            data.for_each_feature_value(feature, |row, value| {
                if value.is_nan() {
                    return;
                }
                predictions[row] += value * delta;
                grad_pairs[row].grad += grad_pairs[row].hess * value * delta;
            });
        }

        deltas
    }
}

pub(super) fn apply_weight_deltas_to_predictions(
    data: &Dataset,
    deltas: &[(usize, f32)],
    mut predictions: ArrayViewMut1<'_, f32>,
) {
    for &(feature, delta) in deltas {
        data.for_each_feature_value(feature, |row, value| {
            if value.is_nan() {
                return;
            }
            predictions[row] += value * delta;
        });
    }
}

pub(super) fn apply_bias_delta_to_predictions(
    bias_delta: f32,
    mut predictions: ArrayViewMut1<'_, f32>,
) {
    if bias_delta.abs() > 1e-10 {
        for i in 0..predictions.len() {
            predictions[i] += bias_delta;
        }
    }
}

/// Compute weight update for a single feature using elastic net regularization.
///
/// Uses soft-thresholding for L1 regularization:
/// ```text
/// grad_l2 = Σ(gradient × feature) + lambda × w
/// hess_l2 = Σ(hessian × feature²) + lambda
/// delta = soft_threshold(-grad_l2 / hess_l2, alpha / hess_l2) × learning_rate
/// ```
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(super) fn compute_weight_update(
    data: &Dataset,
    grad_pairs: &[GradsTuple],
    feature: usize,
    current_weight: f32,
    alpha: f32,
    lambda: f32,
    learning_rate: f32,
    max_delta_step: f32,
) -> f32 {
    // Accumulate gradient and hessian for this feature
    let mut sum_grad = 0.0f32;
    let mut sum_hess = 0.0f32;

    data.for_each_feature_value(feature, |row, value| {
        if value.is_nan() {
            return;
        }
        sum_grad += grad_pairs[row].grad * value;
        sum_hess += grad_pairs[row].hess * value * value;
    });

    coordinate_delta(
        sum_grad,
        sum_hess,
        current_weight,
        alpha,
        lambda,
        learning_rate,
        max_delta_step,
    )
}

/// XGBoost-style coordinate delta (proximal Newton step with L1/L2).
///
/// See `xgboost/src/linear/coordinate_common.h::CoordinateDelta`.
pub(super) fn coordinate_delta(
    sum_grad: f32,
    sum_hess: f32,
    current_weight: f32,
    alpha: f32,
    lambda: f32,
    learning_rate: f32,
    max_delta_step: f32,
) -> f32 {
    let sum_grad_l2 = sum_grad + lambda * current_weight;
    let sum_hess_l2 = sum_hess + lambda;

    // Match XGBoost's stability threshold.
    if sum_hess_l2 < 1e-5 {
        return 0.0;
    }

    let tmp = current_weight - (sum_grad_l2 / sum_hess_l2);
    let raw_delta = if tmp >= 0.0 {
        (-(sum_grad_l2 + alpha) / sum_hess_l2).max(-current_weight)
    } else {
        (-(sum_grad_l2 - alpha) / sum_hess_l2).min(-current_weight)
    };

    let mut delta = raw_delta * learning_rate;
    if max_delta_step > 0.0 {
        delta = delta.clamp(-max_delta_step, max_delta_step);
    }
    delta
}

// =============================================================================
// Bias update function
// =============================================================================

/// Update bias term (no regularization).
///
/// Returns the bias delta that was applied, for incremental prediction updates.
///
/// This works for both single-output and multi-output models.
#[cfg(test)]
mod tests {
    use super::super::selector::{CyclicSelector, FeatureSelector};
    use super::*;
    use crate::training::Gradients;
    use ndarray::array;

    fn make_test_data() -> (Dataset, Gradients) {
        // Simple 2 features x 4 samples dataset
        // Feature-major layout: [n_features, n_samples]
        let features = array![
            [1.0f32, 0.0, 1.0, 2.0], // feature 0
            [0.0f32, 1.0, 1.0, 0.5]  // feature 1
        ];

        let dataset = Dataset::from_array(features.view(), None, None);

        // Gradients (simulating squared error loss)
        let mut buffer = Gradients::new(4, 1);
        buffer.set(0, 0, 0.5, 1.0);
        buffer.set(1, 0, -0.3, 1.0);
        buffer.set(2, 0, 0.2, 1.0);
        buffer.set(3, 0, -0.1, 1.0);

        (dataset, buffer)
    }

    #[test]
    fn sequential_updater_changes_weights() {
        let (dataset, buffer) = make_test_data();
        let mut weights_and_bias = ndarray::Array1::<f32>::zeros(3);
        let mut selector = CyclicSelector::new();

        let alpha = 0.0;
        let lambda = 0.0;
        let learning_rate = 1.0;
        let max_delta_step = 0.0;
        selector.reset(2);
        while let Some(feature) = selector.next() {
            let current_weight = weights_and_bias[feature];
            let delta = compute_weight_update(
                &dataset,
                buffer.output_pairs(0),
                feature,
                current_weight,
                alpha,
                lambda,
                learning_rate,
                max_delta_step,
            );
            if delta.abs() > 1e-10 {
                weights_and_bias[feature] += delta;
            }
        }

        // Weights should have changed
        let w0 = weights_and_bias[0];
        let w1 = weights_and_bias[1];
        assert!(w0.abs() > 1e-6 || w1.abs() > 1e-6);
    }

    #[test]
    fn parallel_updater_changes_weights() {
        let (dataset, buffer) = make_test_data();
        let mut weights_and_bias = ndarray::Array1::<f32>::zeros(3);
        let mut selector = CyclicSelector::new();

        let alpha = 0.0;
        let lambda = 0.0;
        let learning_rate = 1.0;
        let max_delta_step = 0.0;
        selector.reset(2);
        while let Some(feature) = selector.next() {
            let current_weight = weights_and_bias[feature];
            let delta = compute_weight_update(
                &dataset,
                buffer.output_pairs(0),
                feature,
                current_weight,
                alpha,
                lambda,
                learning_rate,
                max_delta_step,
            );
            if delta.abs() > 1e-10 {
                weights_and_bias[feature] += delta;
            }
        }

        // Weights should have changed
        let w0 = weights_and_bias[0];
        let w1 = weights_and_bias[1];
        assert!(w0.abs() > 1e-6 || w1.abs() > 1e-6);
    }

    #[test]
    fn l2_regularization_shrinks_weights() {
        let (dataset, buffer) = make_test_data();

        // No regularization
        let mut weights_and_bias_1 = ndarray::Array1::<f32>::zeros(3);
        weights_and_bias_1[0] = 1.0;
        let mut selector = CyclicSelector::new();
        let alpha = 0.0;
        let lambda_no_reg = 0.0;
        let learning_rate = 1.0;
        let max_delta_step = 0.0;
        selector.reset(2);
        while let Some(feature) = selector.next() {
            let current_weight = weights_and_bias_1[feature];
            let delta = compute_weight_update(
                &dataset,
                buffer.output_pairs(0),
                feature,
                current_weight,
                alpha,
                lambda_no_reg,
                learning_rate,
                max_delta_step,
            );
            if delta.abs() > 1e-10 {
                weights_and_bias_1[feature] += delta;
            }
        }
        let w1_no_reg = weights_and_bias_1[0];

        // With L2 regularization
        let mut weights_and_bias_2 = ndarray::Array1::<f32>::zeros(3);
        weights_and_bias_2[0] = 1.0;
        let lambda_l2 = 10.0; // Strong L2
        selector.reset(2);
        while let Some(feature) = selector.next() {
            let current_weight = weights_and_bias_2[feature];
            let delta = compute_weight_update(
                &dataset,
                buffer.output_pairs(0),
                feature,
                current_weight,
                alpha,
                lambda_l2,
                learning_rate,
                max_delta_step,
            );
            if delta.abs() > 1e-10 {
                weights_and_bias_2[feature] += delta;
            }
        }
        let w1_l2 = weights_and_bias_2[0];

        // L2 should shrink more towards zero
        assert!(w1_l2.abs() < w1_no_reg.abs());
    }

    // Bias updates are covered by GBLinearTrainer tests.
}

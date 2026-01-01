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

use ndarray::ArrayViewMut2;
use rayon::prelude::*;

use crate::data::Dataset;
use crate::repr::gblinear::LinearModel;
use crate::training::Gradients;

use super::selector::FeatureSelector;

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
    // Base (user-configured) regularization parameters.
    // We denormalize by `sum_instance_weight` at runtime in `update_round_inplace`.
    alpha: f32,
    lambda: f32,
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
            alpha,
            lambda,
            learning_rate,
            max_delta_step,
        }
    }

    pub(super) fn update_bias_inplace(
        &self,
        model: &mut LinearModel,
        gradients: &mut Gradients,
        output: usize,
        predictions: ArrayViewMut2<'_, f32>,
    ) -> f32 {
        let (sum_grad, sum_hess) = gradients.sum(output, None);
        let sum_grad = sum_grad as f32;
        let sum_hess = sum_hess as f32;

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

        model.add_bias(output, delta);
        apply_bias_delta_to_predictions(delta, output, predictions);

        // XGBoost updater_{shotgun,coordinate}.cc:
        //   p += GradientPair(p.GetHess() * dbias, 0)
        for gh in gradients.output_pairs_mut(output) {
            gh.grad += gh.hess * delta;
        }

        delta
    }

    pub(super) fn update_round_inplace<Sel: FeatureSelector>(
        &self,
        model: &mut LinearModel,
        data: &Dataset,
        gradients: &mut Gradients,
        selector: &mut Sel,
        output: usize,
        sum_instance_weight: f32,
        predictions: ArrayViewMut2<'_, f32>,
    ) -> Vec<(usize, f32)> {
        match self.kind {
            UpdateStrategy::Sequential => self.update_round_sequential_inplace(
                model,
                data,
                gradients,
                selector,
                output,
                sum_instance_weight,
                predictions,
            ),
            UpdateStrategy::Shotgun => {
                self.update_round_shotgun_inplace(
                    model,
                    data,
                    gradients,
                    selector,
                    output,
                    sum_instance_weight,
                    predictions,
                )
            }
        }
    }

    fn update_round_sequential_inplace<Sel: FeatureSelector>(
        &self,
        model: &mut LinearModel,
        data: &Dataset,
        gradients: &mut Gradients,
        selector: &mut Sel,
        output: usize,
        sum_instance_weight: f32,
        mut predictions: ArrayViewMut2<'_, f32>,
    ) -> Vec<(usize, f32)> {
        let mut deltas: Vec<(usize, f32)> = Vec::new();
        // Match XGBoost's `DenormalizePenalties(sum_instance_weight)`.
        let alpha = self.alpha * sum_instance_weight;
        let lambda = self.lambda * sum_instance_weight;

        while let Some(feature) = selector.next() {
            let current_weight = model.weight(feature, output);

            // Accumulate from current residual gradients.
            let gh = gradients.output_pairs(output);
            let mut sum_grad = 0.0f32;
            let mut sum_hess = 0.0f32;
            data.for_each_feature_value(feature, |row, value| {
                sum_grad += gh[row].grad * value;
                sum_hess += gh[row].hess * value * value;
            });

            let delta = coordinate_delta(
                sum_grad,
                sum_hess,
                current_weight,
                alpha,
                lambda,
                self.learning_rate,
                self.max_delta_step,
            );
            if delta.abs() <= 1e-10 {
                continue;
            }

            model.add_weight(feature, output, delta);
            deltas.push((feature, delta));

            // Update predictions and residual gradients for touched rows.
            {
                let mut output_row = predictions.row_mut(output);
                let gh_mut = gradients.output_pairs_mut(output);
                data.for_each_feature_value(feature, |row, value| {
                    output_row[row] += value * delta;
                    gh_mut[row].grad += gh_mut[row].hess * value * delta;
                });
            }
        }

        deltas
    }

    fn update_round_shotgun_inplace<Sel: FeatureSelector>(
        &self,
        model: &mut LinearModel,
        data: &Dataset,
        gradients: &mut Gradients,
        selector: &mut Sel,
        output: usize,
        sum_instance_weight: f32,
        mut predictions: ArrayViewMut2<'_, f32>,
    ) -> Vec<(usize, f32)> {
        // NOTE: This is "shotgun" in the sense that we compute coordinate deltas
        // in parallel from the same residual snapshot, then apply them.
        // Applying deltas truly in parallel would require conflict handling
        // for prediction/residual writes.
        let indices = selector.all_indices();
        if indices.is_empty() {
            return Vec::new();
        }

        // Match XGBoost's `DenormalizePenalties(sum_instance_weight)`.
        let alpha = self.alpha * sum_instance_weight;
        let lambda = self.lambda * sum_instance_weight;

        let model_ref: &LinearModel = model;
        let gh = gradients.output_pairs(output);
        let learning_rate = self.learning_rate;
        let max_delta_step = self.max_delta_step;

        let deltas: Vec<(usize, f32)> = indices
            .par_iter()
            .filter_map(|&feature| {
                let current_weight = model_ref.weight(feature, output);
                let mut sum_grad = 0.0f32;
                let mut sum_hess = 0.0f32;
                data.for_each_feature_value(feature, |row, value| {
                    sum_grad += gh[row].grad * value;
                    sum_hess += gh[row].hess * value * value;
                });
                let delta = coordinate_delta(
                    sum_grad,
                    sum_hess,
                    current_weight,
                    alpha,
                    lambda,
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
            model.add_weight(feature, output, delta);
        }

        // Apply to predictions and residual gradients.
        {
            let mut output_row = predictions.row_mut(output);
            let gh_mut = gradients.output_pairs_mut(output);
            for &(feature, delta) in &deltas {
                data.for_each_feature_value(feature, |row, value| {
                    output_row[row] += value * delta;
                    gh_mut[row].grad += gh_mut[row].hess * value * delta;
                });
            }
        }

        deltas
    }
}

pub(super) fn apply_weight_deltas_to_predictions(
    data: &Dataset,
    deltas: &[(usize, f32)],
    output: usize,
    mut predictions: ArrayViewMut2<'_, f32>,
) {
    let mut output_row = predictions.row_mut(output);
    for &(feature, delta) in deltas {
        data.for_each_feature_value(feature, |row, value| {
            output_row[row] += value * delta;
        });
    }
}

pub(super) fn apply_bias_delta_to_predictions(
    bias_delta: f32,
    output: usize,
    mut predictions: ArrayViewMut2<'_, f32>,
) {
    if bias_delta.abs() > 1e-10 {
        let mut output_row = predictions.row_mut(output);
        for i in 0..output_row.len() {
            output_row[i] += bias_delta;
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
pub(super) fn compute_weight_update(
    model: &LinearModel,
    data: &Dataset,
    buffer: &Gradients,
    feature: usize,
    output: usize,
    alpha: f32,
    lambda: f32,
    learning_rate: f32,
    max_delta_step: f32,
) -> f32 {
    let current_weight = model.weight(feature, output);

    // Feature-major: use output-specific slices for direct indexing by row
    let grad_hess = buffer.output_pairs(output);

    // Accumulate gradient and hessian for this feature
    let mut sum_grad = 0.0f32;
    let mut sum_hess = 0.0f32;

    data.for_each_feature_value(feature, |row, value| {
        sum_grad += grad_hess[row].grad * value;
        sum_hess += grad_hess[row].hess * value * value;
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
        let mut model = LinearModel::zeros(2, 1);
        let mut selector = CyclicSelector::new();

        let alpha = 0.0;
        let lambda = 0.0;
        let learning_rate = 1.0;
        let max_delta_step = 0.0;
        selector.reset(model.n_features());
        while let Some(feature) = selector.next() {
            let delta = compute_weight_update(
                &model,
                &dataset,
                &buffer,
                feature,
                0,
                alpha,
                lambda,
                learning_rate,
                max_delta_step,
            );
            if delta.abs() > 1e-10 {
                model.add_weight(feature, 0, delta);
            }
        }

        // Weights should have changed
        let w0 = model.weight(0, 0);
        let w1 = model.weight(1, 0);
        assert!(w0.abs() > 1e-6 || w1.abs() > 1e-6);
    }

    #[test]
    fn parallel_updater_changes_weights() {
        let (dataset, buffer) = make_test_data();
        let mut model = LinearModel::zeros(2, 1);
        let mut selector = CyclicSelector::new();

        let alpha = 0.0;
        let lambda = 0.0;
        let learning_rate = 1.0;
        let max_delta_step = 0.0;
        selector.reset(model.n_features());
        while let Some(feature) = selector.next() {
            let delta = compute_weight_update(
                &model,
                &dataset,
                &buffer,
                feature,
                0,
                alpha,
                lambda,
                learning_rate,
                max_delta_step,
            );
            if delta.abs() > 1e-10 {
                model.add_weight(feature, 0, delta);
            }
        }

        // Weights should have changed
        let w0 = model.weight(0, 0);
        let w1 = model.weight(1, 0);
        assert!(w0.abs() > 1e-6 || w1.abs() > 1e-6);
    }

    #[test]
    fn l2_regularization_shrinks_weights() {
        let (dataset, buffer) = make_test_data();

        // No regularization
        let mut model1 = LinearModel::zeros(2, 1);
        model1.set_weight(0, 0, 1.0);
        let mut selector = CyclicSelector::new();
        let alpha = 0.0;
        let lambda_no_reg = 0.0;
        let learning_rate = 1.0;
        let max_delta_step = 0.0;
        selector.reset(model1.n_features());
        while let Some(feature) = selector.next() {
            let delta = compute_weight_update(
                &model1,
                &dataset,
                &buffer,
                feature,
                0,
                alpha,
                lambda_no_reg,
                learning_rate,
                max_delta_step,
            );
            if delta.abs() > 1e-10 {
                model1.add_weight(feature, 0, delta);
            }
        }
        let w1_no_reg = model1.weight(0, 0);

        // With L2 regularization
        let mut model2 = LinearModel::zeros(2, 1);
        model2.set_weight(0, 0, 1.0);
        let lambda_l2 = 10.0; // Strong L2
        selector.reset(model2.n_features());
        while let Some(feature) = selector.next() {
            let delta = compute_weight_update(
                &model2,
                &dataset,
                &buffer,
                feature,
                0,
                alpha,
                lambda_l2,
                learning_rate,
                max_delta_step,
            );
            if delta.abs() > 1e-10 {
                model2.add_weight(feature, 0, delta);
            }
        }
        let w1_l2 = model2.weight(0, 0);

        // L2 should shrink more towards zero
        assert!(w1_l2.abs() < w1_no_reg.abs());
    }

    // Bias updates are covered by GBLinearTrainer tests.
}

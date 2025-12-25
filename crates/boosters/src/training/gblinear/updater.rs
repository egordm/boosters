//! Coordinate descent updaters for linear models.
//!
//! The [`Updater`] component handles weight updates using coordinate descent with
//! incremental prediction updates for efficiency.
//!
//! Two variants:
//! - [`UpdaterKind::Parallel`]: Parallel updates (faster, slight approximation)
//! - [`UpdaterKind::Sequential`]: Sequential updates (exact gradients)
//!
//! # Data Format
//!
//! The updaters require feature-major layout for efficient feature iteration:
//!
//! - [`FeaturesView`](crate::data::FeaturesView): Feature-major view `[n_features, n_samples]`
//!
//! # Gradient Storage
//!
//! Gradients are stored in Structure-of-Arrays (SoA) layout via [`Gradients`]:
//! - Shape `[n_samples, n_outputs]` for unified single/multi-output handling
//! - Separate `grads[]` and `hess[]` arrays for cache efficiency

use ndarray::ArrayViewMut2;
use rayon::prelude::*;

use crate::data::FeaturesView;
use crate::repr::gblinear::LinearModel;
use crate::training::Gradients;

use super::selector::FeatureSelector;

/// Coordinate descent updater selection.
///
/// Use [`UpdaterKind::Parallel`] (shotgun) for better performance on most workloads.
/// Use [`UpdaterKind::Sequential`] for exact gradient computation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum UpdaterKind {
    /// Sequential coordinate descent - exact gradients, slower
    Sequential,
    /// Parallel (shotgun) coordinate descent - approximate, faster
    #[default]
    Parallel,
}

/// Configuration for coordinate descent updates.
#[derive(Debug, Clone)]
pub struct UpdateConfig {
    /// L1 regularization strength (alpha).
    pub alpha: f32,
    /// L2 regularization strength (lambda).
    pub lambda: f32,
    /// Learning rate.
    pub learning_rate: f32,
}

impl Default for UpdateConfig {
    fn default() -> Self {
        Self {
            alpha: 0.0,
            lambda: 1.0,
            learning_rate: 0.5,
        }
    }
}

/// Coordinate descent updater for linear models.
///
/// Handles weight updates with incremental prediction maintenance for efficiency.
/// Instead of recomputing full predictions each round, it applies weight deltas
/// incrementally, achieving ~2-4x speedup over naive approaches.
///
/// # Example
///
/// ```ignore
/// use boosters::training::gblinear::{Updater, UpdaterKind, UpdateConfig};
///
/// let config = UpdateConfig {
///     alpha: 0.5,
///     lambda: 1.0,
///     learning_rate: 0.3,
/// };
/// let updater = Updater::new(UpdaterKind::Parallel, config);
/// ```
#[derive(Debug, Clone)]
pub struct Updater {
    kind: UpdaterKind,
    config: UpdateConfig,
}

impl Updater {
    /// Create a new updater with the specified kind and configuration.
    pub fn new(kind: UpdaterKind, config: UpdateConfig) -> Self {
        Self { kind, config }
    }

    /// Perform one round of coordinate descent updates with incremental prediction updates.
    ///
    /// Returns a vector of (feature_idx, delta) pairs for weight updates that were applied.
    /// These deltas can be used to incrementally update predictions instead of recomputing them.
    ///
    /// # Arguments
    ///
    /// * `model` - Linear model to update
    /// * `data` - Training data as FeaturesView `[n_features, n_samples]`
    /// * `buffer` - Gradient buffer with shape `[n_samples, n_outputs]`
    /// * `selector` - Feature selection strategy
    /// * `output` - Which output (group) to update (0 to n_outputs-1)
    ///
    /// # Returns
    ///
    /// Vector of (feature_idx, weight_delta) pairs for all non-zero updates
    pub fn update_round<Sel>(
        &self,
        model: &mut LinearModel,
        data: &FeaturesView<'_>,
        buffer: &Gradients,
        selector: &mut Sel,
        output: usize,
    ) -> Vec<(usize, f32)>
    where
        Sel: FeatureSelector,
    {
        match self.kind {
            UpdaterKind::Sequential => {
                sequential_update(model, data, buffer, selector, output, &self.config)
            }
            UpdaterKind::Parallel => {
                parallel_update(model, data, buffer, selector, output, &self.config)
            }
        }
    }

    /// Update bias term and return the delta for incremental prediction updates.
    ///
    /// Returns the bias delta that was applied, for incremental prediction updates.
    ///
    /// # Arguments
    ///
    /// * `model` - Linear model to update
    /// * `buffer` - Gradient buffer with shape `[n_samples, n_outputs]`
    /// * `output` - Which output to update (0 to n_outputs-1)
    ///
    /// # Returns
    ///
    /// The bias delta that was applied (0.0 if hessian too small)
    pub fn update_bias(
        &self,
        model: &mut LinearModel,
        buffer: &Gradients,
        output: usize,
    ) -> f32 {
        let (sum_grad, sum_hess) = buffer.sum(output, None);

        if sum_hess.abs() > 1e-10 {
            let delta = (-sum_grad / sum_hess) as f32 * self.config.learning_rate;
            model.add_bias(output, delta);
            delta
        } else {
            0.0
        }
    }

    /// Incrementally update predictions after weight changes.
    ///
    /// Instead of recomputing full predictions with `predict_col_major()`, this function
    /// applies weight deltas directly to the prediction buffer. This is much faster as it
    /// only touches rows where the feature is non-zero.
    ///
    /// # Arguments
    ///
    /// * `data` - Training data as FeaturesView `[n_features, n_samples]`
    /// * `deltas` - Weight deltas from coordinate descent: (feature_idx, delta) pairs
    /// * `output` - Which output group these deltas apply to
    /// * `predictions` - Prediction buffer `[n_outputs, n_samples]`
    pub fn apply_weight_deltas_to_predictions(
        &self,
        data: &FeaturesView<'_>,
        deltas: &[(usize, f32)],
        output: usize,
        mut predictions: ArrayViewMut2<'_, f32>,
    ) {
        let mut output_row = predictions.row_mut(output);
        for &(feature, delta) in deltas {
            let feature_values = data.feature(feature);
            for (row, &value) in feature_values.iter().enumerate() {
                output_row[row] += value * delta;
            }
        }
    }

    /// Incrementally update bias in predictions.
    ///
    /// Applies a bias delta to all predictions for a given output group.
    ///
    /// # Arguments
    ///
    /// * `bias_delta` - Change in bias value
    /// * `output` - Which output group to update
    /// * `predictions` - Prediction buffer `[n_outputs, n_samples]`
    pub fn apply_bias_delta_to_predictions(
        &self,
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
}

// =============================================================================
// Update implementations
// =============================================================================

/// Sequential coordinate descent update.
///
/// Updates features one at a time with exact gradient computation.
/// Returns deltas for incremental prediction updates.
fn sequential_update<Sel>(
    model: &mut LinearModel,
    data: &FeaturesView<'_>,
    buffer: &Gradients,
    selector: &mut Sel,
    output: usize,
    config: &UpdateConfig,
) -> Vec<(usize, f32)>
where
    Sel: FeatureSelector,
{
    selector.reset(model.n_features());
    let mut deltas = Vec::new();

    while let Some(feature) = selector.next() {
        let delta = compute_weight_update(model, data, buffer, feature, output, config);
        if delta.abs() > 1e-10 {
            model.add_weight(feature, output, delta);
            deltas.push((feature, delta));
        }
    }

    deltas
}

/// Parallel (shotgun) coordinate descent update.
///
/// Updates all features in parallel. Race conditions in residual updates
/// are tolerable with reasonable learning rates.
/// Returns deltas for incremental prediction updates.
fn parallel_update<Sel>(
    model: &mut LinearModel,
    data: &FeaturesView<'_>,
    buffer: &Gradients,
    selector: &mut Sel,
    output: usize,
    config: &UpdateConfig,
) -> Vec<(usize, f32)>
where
    Sel: FeatureSelector,
{
    selector.reset(model.n_features());
    let features = selector.all_indices();

    // Compute all deltas in parallel
    let deltas: Vec<(usize, f32)> = features
        .par_iter()
        .map(|&feature| {
            let delta = compute_weight_update(model, data, buffer, feature, output, config);
            (feature, delta)
        })
        .filter(|(_, delta)| delta.abs() > 1e-10)
        .collect();

    // Apply updates sequentially (thread-safe)
    for &(feature, delta) in &deltas {
        model.add_weight(feature, output, delta);
    }

    deltas
}

/// Compute weight update for a single feature using elastic net regularization.
///
/// Uses soft-thresholding for L1 regularization:
/// ```text
/// grad_l2 = Σ(gradient × feature) + lambda × w
/// hess_l2 = Σ(hessian × feature²) + lambda
/// delta = soft_threshold(-grad_l2 / hess_l2, alpha / hess_l2) × learning_rate
/// ```
fn compute_weight_update(
    model: &LinearModel,
    data: &FeaturesView<'_>,
    buffer: &Gradients,
    feature: usize,
    output: usize,
    config: &UpdateConfig,
) -> f32 {
    let current_weight = model.weight(feature, output);

    // Feature-major: use output-specific slices for direct indexing by row
    let grad_hess = buffer.output_pairs(output);

    // Accumulate gradient and hessian for this feature
    let mut sum_grad = 0.0f32;
    let mut sum_hess = 0.0f32;

    let feature_values = data.feature(feature);
    for (row, &value) in feature_values.iter().enumerate() {
        sum_grad += grad_hess[row].grad * value;
        sum_hess += grad_hess[row].hess * value * value;
    }

    // Add L2 regularization
    let grad_l2 = sum_grad + config.lambda * current_weight;
    let hess_l2 = sum_hess + config.lambda;

    // Avoid division by zero
    if hess_l2.abs() < 1e-10 {
        return 0.0;
    }

    // Compute raw update
    let raw_update = -grad_l2 / hess_l2;

    // Apply soft-thresholding for L1 (elastic net)
    let threshold = config.alpha / hess_l2;
    let thresholded = soft_threshold(raw_update, threshold);

    // Apply learning rate
    thresholded * config.learning_rate
}

/// Soft-thresholding operator for L1 regularization.
///
/// S(x, λ) = sign(x) × max(|x| - λ, 0)
#[inline]
fn soft_threshold(x: f32, threshold: f32) -> f32 {
    if x > threshold {
        x - threshold
    } else if x < -threshold {
        x + threshold
    } else {
        0.0
    }
}

// =============================================================================
// Bias update function
// =============================================================================

/// Update bias term (no regularization).
///
/// Returns the bias delta that was applied, for incremental prediction updates.
///
/// This works for both single-output and multi-output models.
///
/// # Arguments
///

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::FeaturesView;
    use ndarray::{array, Array2};
    use super::super::selector::CyclicSelector;

    fn make_test_data() -> (Array2<f32>, Gradients) {
        // Simple 2 features x 4 samples dataset (feature-major layout)
        let feature_data = array![
            [1.0f32, 0.0, 1.0, 2.0], // feature 0
            [0.0, 1.0, 1.0, 0.5],    // feature 1
        ];

        // Gradients (simulating squared error loss)
        let mut buffer = Gradients::new(4, 1);
        buffer.set(0, 0, 0.5, 1.0);
        buffer.set(1, 0, -0.3, 1.0);
        buffer.set(2, 0, 0.2, 1.0);
        buffer.set(3, 0, -0.1, 1.0);

        (feature_data, buffer)
    }

    #[test]
    fn soft_threshold_positive() {
        assert!((soft_threshold(1.0, 0.3) - 0.7).abs() < 1e-6);
    }

    #[test]
    fn soft_threshold_negative() {
        assert!((soft_threshold(-1.0, 0.3) - (-0.7)).abs() < 1e-6);
    }

    #[test]
    fn soft_threshold_within() {
        assert_eq!(soft_threshold(0.2, 0.3), 0.0);
        assert_eq!(soft_threshold(-0.2, 0.3), 0.0);
    }

    #[test]
    fn sequential_updater_changes_weights() {
        let (feature_data, buffer) = make_test_data();
        let data = FeaturesView::from_array(feature_data.view());
        let mut model = LinearModel::zeros(2, 1);
        let mut selector = CyclicSelector::new();

        let config = UpdateConfig {
            alpha: 0.0,
            lambda: 0.0,
            learning_rate: 1.0,
        };
        let updater = Updater::new(UpdaterKind::Sequential, config);

        updater.update_round(
            &mut model,
            &data,
            &buffer,
            &mut selector,
            0, // output
        );

        // Weights should have changed
        let w0 = model.weight(0, 0);
        let w1 = model.weight(1, 0);
        assert!(w0.abs() > 1e-6 || w1.abs() > 1e-6);
    }

    #[test]
    fn parallel_updater_changes_weights() {
        let (feature_data, buffer) = make_test_data();
        let data = FeaturesView::from_array(feature_data.view());
        let mut model = LinearModel::zeros(2, 1);
        let mut selector = CyclicSelector::new();

        let config = UpdateConfig {
            alpha: 0.0,
            lambda: 0.0,
            learning_rate: 1.0,
        };
        let updater = Updater::new(UpdaterKind::Parallel, config);

        updater.update_round(
            &mut model,
            &data,
            &buffer,
            &mut selector,
            0, // output
        );

        // Weights should have changed
        let w0 = model.weight(0, 0);
        let w1 = model.weight(1, 0);
        assert!(w0.abs() > 1e-6 || w1.abs() > 1e-6);
    }

    #[test]
    fn l2_regularization_shrinks_weights() {
        let (feature_data, buffer) = make_test_data();
        let data = FeaturesView::from_array(feature_data.view());

        // No regularization
        let mut model1 = LinearModel::zeros(2, 1);
        model1.set_weight(0, 0, 1.0);
        let mut selector = CyclicSelector::new();
        let config_no_reg = UpdateConfig {
            alpha: 0.0,
            lambda: 0.0,
            learning_rate: 1.0,
        };
        let updater1 = Updater::new(UpdaterKind::Sequential, config_no_reg);

        updater1.update_round(
            &mut model1,
            &data,
            &buffer,
            &mut selector,
            0, // output
        );
        let w1_no_reg = model1.weight(0, 0);

        // With L2 regularization
        let mut model2 = LinearModel::zeros(2, 1);
        model2.set_weight(0, 0, 1.0);
        let config_l2 = UpdateConfig {
            alpha: 0.0,
            lambda: 10.0, // Strong L2
            learning_rate: 1.0,
        };
        let updater2 = Updater::new(UpdaterKind::Sequential, config_l2);

        updater2.update_round(
            &mut model2,
            &data,
            &buffer,
            &mut selector,
            0, // output
        );
        let w1_l2 = model2.weight(0, 0);

        // L2 should shrink more towards zero
        assert!(w1_l2.abs() < w1_no_reg.abs());
    }

    #[test]
    fn update_bias_works() {
        let mut buffer = Gradients::new(3, 1);
        buffer.set(0, 0, 0.5, 1.0);
        buffer.set(1, 0, -0.3, 1.0);
        buffer.set(2, 0, 0.2, 1.0);

        let mut model = LinearModel::zeros(2, 1);
        
        let config = UpdateConfig {
            alpha: 0.0,
            lambda: 0.0,
            learning_rate: 1.0,
        };
        let updater = Updater::new(UpdaterKind::Sequential, config);
        updater.update_bias(&mut model, &buffer, 0);

        // Bias should have changed
        // sum_grad = 0.5 - 0.3 + 0.2 = 0.4
        // sum_hess = 3.0
        // delta = -0.4 / 3.0 ≈ -0.133
        let bias = model.bias(0);
        assert!((bias - (-0.4 / 3.0)).abs() < 1e-6);
    }

    #[test]
    fn update_bias_multiclass() {
        // 3 samples, 2 outputs
        let mut buffer = Gradients::new(3, 2);

        // Output 0: sum_grad = 1.0, sum_hess = 3.0 → delta = -1/3
        buffer.set(0, 0, 0.5, 1.0);
        buffer.set(1, 0, 0.3, 1.0);
        buffer.set(2, 0, 0.2, 1.0);

        // Output 1: sum_grad = -0.6, sum_hess = 3.0 → delta = 0.2
        buffer.set(0, 1, -0.1, 1.0);
        buffer.set(1, 1, -0.2, 1.0);
        buffer.set(2, 1, -0.3, 1.0);

        let mut model = LinearModel::zeros(2, 2);

        let config = UpdateConfig {
            alpha: 0.0,
            lambda: 0.0,
            learning_rate: 1.0,
        };
        let updater = Updater::new(UpdaterKind::Sequential, config);
        updater.update_bias(&mut model, &buffer, 0);
        updater.update_bias(&mut model, &buffer, 1);

        assert!((model.bias(0) - (-1.0 / 3.0)).abs() < 1e-6);
        assert!((model.bias(1) - 0.2).abs() < 1e-6);
    }
}

//! Coordinate descent updaters for linear models.
//!
//! Two variants:
//! - [`UpdaterKind::Parallel`]: Parallel updates (faster, slight approximation)
//! - [`UpdaterKind::Sequential`]: Sequential updates (exact gradients)
//!
//! # Data Format
//!
//! The updaters require column-major matrices for efficient column iteration:
//!
//! - [`ColMatrix`](crate::data::ColMatrix): Column-major dense matrix
//!
//! For row-major data, convert first:
//! ```ignore
//! let col_matrix: ColMatrix = row_matrix.to_layout();
//! ```
//!
//! # Gradient Storage
//!
//! Gradients are stored in Structure-of-Arrays (SoA) layout via [`Gradients`]:
//! - Shape `[n_samples, n_outputs]` for unified single/multi-output handling
//! - Separate `grads[]` and `hess[]` arrays for cache efficiency

use rayon::prelude::*;

use crate::data::ColMatrix;
use crate::inference::gblinear::LinearModel;
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

impl UpdaterKind {
    /// Perform one round of coordinate descent updates.
    ///
    /// # Type Parameters
    ///
    /// * `S` - Storage type for the column-major matrix
    ///
    /// # Arguments
    ///
    /// * `model` - Linear model to update
    /// * `data` - Training data (column-major matrix)
    /// * `buffer` - Gradient buffer with shape `[n_samples, n_outputs]`
    /// * `selector` - Feature selection strategy
    /// * `output` - Which output (group) to update (0 to n_outputs-1)
    /// * `config` - Update configuration (learning rate, regularization)
    pub fn update_round<S, Sel>(
        &self,
        model: &mut LinearModel,
        data: &ColMatrix<f32, S>,
        buffer: &Gradients,
        selector: &mut Sel,
        output: usize,
        config: &UpdateConfig,
    )
    where
        S: AsRef<[f32]> + Sync,
        Sel: FeatureSelector,
    {
        match self {
            UpdaterKind::Sequential => {
                sequential_update(model, data, buffer, selector, output, config)
            }
            UpdaterKind::Parallel => {
                parallel_update(model, data, buffer, selector, output, config)
            }
        }
    }
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

// =============================================================================
// Update implementations
// =============================================================================

/// Sequential coordinate descent update.
///
/// Updates features one at a time with exact gradient computation.
fn sequential_update<S, Sel>(
    model: &mut LinearModel,
    data: &ColMatrix<f32, S>,
    buffer: &Gradients,
    selector: &mut Sel,
    output: usize,
    config: &UpdateConfig,
) where
    S: AsRef<[f32]>,
    Sel: FeatureSelector,
{
    selector.reset(model.num_features());

    while let Some(feature) = selector.next() {
        let delta = compute_weight_update(model, data, buffer, feature, output, config);
        if delta.abs() > 1e-10 {
            model.add_weight(feature, output, delta);
        }
    }
}

/// Parallel (shotgun) coordinate descent update.
///
/// Updates all features in parallel. Race conditions in residual updates
/// are tolerable with reasonable learning rates.
fn parallel_update<S, Sel>(
    model: &mut LinearModel,
    data: &ColMatrix<f32, S>,
    buffer: &Gradients,
    selector: &mut Sel,
    output: usize,
    config: &UpdateConfig,
) where
    S: AsRef<[f32]> + Sync,
    Sel: FeatureSelector,
{
    selector.reset(model.num_features());
    let features = selector.all_indices();

    // Compute all deltas in parallel
    let deltas: Vec<(usize, f32)> = features
        .par_iter()
        .map(|&feature| {
            let delta = compute_weight_update(model, data, buffer, feature, output, config);
            (feature, delta)
        })
        .collect();

    // Apply updates sequentially (thread-safe)
    for (feature, delta) in deltas {
        if delta.abs() > 1e-10 {
            model.add_weight(feature, output, delta);
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
fn compute_weight_update<S: AsRef<[f32]>>(
    model: &LinearModel,
    data: &ColMatrix<f32, S>,
    buffer: &Gradients,
    feature: usize,
    output: usize,
    config: &UpdateConfig,
) -> f32 {
    let current_weight = model.weight(feature, output);

    // Column-major: use output-specific slices for direct indexing by row
    let grad_hess = buffer.output_pairs(output);

    // Accumulate gradient and hessian for this feature
    let mut sum_grad = 0.0f32;
    let mut sum_hess = 0.0f32;

    for (row, value) in data.column(feature) {
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
/// This works for both single-output and multi-output models.
///
/// # Arguments
///
/// * `model` - Linear model to update
/// * `buffer` - Gradient buffer with shape `[n_samples, n_outputs]`
/// * `output` - Which output to update (0 to n_outputs-1)
/// * `learning_rate` - Step size multiplier
pub fn update_bias(
    model: &mut LinearModel,
    buffer: &Gradients,
    output: usize,
    learning_rate: f32,
) {
    let (sum_grad, sum_hess) = buffer.sum(output, None);

    if sum_hess.abs() > 1e-10 {
        let delta = (-sum_grad / sum_hess) as f32 * learning_rate;
        model.add_bias(output, delta);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{ColMatrix, RowMatrix};
    use super::super::selector::CyclicSelector;

    fn make_test_data() -> (ColMatrix<f32>, Gradients) {
        // Simple 4x2 dataset
        let dense = RowMatrix::from_vec(
            vec![
                1.0, 0.0, // row 0
                0.0, 1.0, // row 1
                1.0, 1.0, // row 2
                2.0, 0.5, // row 3
            ],
            4,
            2,
        );
        let col_matrix: ColMatrix = dense.to_layout();

        // Gradients (simulating squared error loss)
        let mut buffer = Gradients::new(4, 1);
        buffer.set(0, 0, 0.5, 1.0);
        buffer.set(1, 0, -0.3, 1.0);
        buffer.set(2, 0, 0.2, 1.0);
        buffer.set(3, 0, -0.1, 1.0);

        (col_matrix, buffer)
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
        let (data, buffer) = make_test_data();
        let mut model = LinearModel::zeros(2, 1);
        let mut selector = CyclicSelector::new();

        let config = UpdateConfig {
            alpha: 0.0,
            lambda: 0.0,
            learning_rate: 1.0,
        };

        UpdaterKind::Sequential.update_round(
            &mut model,
            &data,
            &buffer,
            &mut selector,
            0, // output
            &config,
        );

        // Weights should have changed
        let w0 = model.weight(0, 0);
        let w1 = model.weight(1, 0);
        assert!(w0.abs() > 1e-6 || w1.abs() > 1e-6);
    }

    #[test]
    fn parallel_updater_changes_weights() {
        let (data, buffer) = make_test_data();
        let mut model = LinearModel::zeros(2, 1);
        let mut selector = CyclicSelector::new();

        let config = UpdateConfig {
            alpha: 0.0,
            lambda: 0.0,
            learning_rate: 1.0,
        };

        UpdaterKind::Parallel.update_round(
            &mut model,
            &data,
            &buffer,
            &mut selector,
            0, // output
            &config,
        );

        // Weights should have changed
        let w0 = model.weight(0, 0);
        let w1 = model.weight(1, 0);
        assert!(w0.abs() > 1e-6 || w1.abs() > 1e-6);
    }

    #[test]
    fn l2_regularization_shrinks_weights() {
        let (data, buffer) = make_test_data();

        // No regularization
        let mut model1 = LinearModel::zeros(2, 1);
        model1.set_weight(0, 0, 1.0);
        let mut selector = CyclicSelector::new();
        let config_no_reg = UpdateConfig {
            alpha: 0.0,
            lambda: 0.0,
            learning_rate: 1.0,
        };

        UpdaterKind::Sequential.update_round(
            &mut model1,
            &data,
            &buffer,
            &mut selector,
            0, // output
            &config_no_reg,
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

        UpdaterKind::Sequential.update_round(
            &mut model2,
            &data,
            &buffer,
            &mut selector,
            0, // output
            &config_l2,
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
        update_bias(&mut model, &buffer, 0, 1.0);

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

        update_bias(&mut model, &buffer, 0, 1.0);
        update_bias(&mut model, &buffer, 1, 1.0);

        assert!((model.bias(0) - (-1.0 / 3.0)).abs() < 1e-6);
        assert!((model.bias(1) - 0.2).abs() < 1e-6);
    }
}
